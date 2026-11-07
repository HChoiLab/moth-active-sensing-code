from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

try:
    from experiment_config import EnvironmentConfig, ExperimentConfig
    from model import AllocentricNectarySearchModel
    from nectary_allocentric_env import (
        AllocentricNectarSearchEnv,
        CanvasParams,
        NonStationaryParams,
    )
    from nectary_env import FlowerParams
    from training import _allocentric_vision_shape
    from training_uncertainty_gated import _allocentric_action_delta_bounds
    from uncertainty_gated_model import UncertaintyGatedAllocentricNectarySearchModel
except ImportError:
    from .experiment_config import EnvironmentConfig, ExperimentConfig
    from .model import AllocentricNectarySearchModel
    from .nectary_allocentric_env import (
        AllocentricNectarSearchEnv,
        CanvasParams,
        NonStationaryParams,
    )
    from .nectary_env import FlowerParams
    from .training import _allocentric_vision_shape
    from .training_uncertainty_gated import _allocentric_action_delta_bounds
    from .uncertainty_gated_model import UncertaintyGatedAllocentricNectarySearchModel


SUPPORTED_MODEL_TYPES = {
    "uncertainty_gated_allocentric",
    "allocentric_perception",
}

PRETRAINED_PERCEPTION_MODEL_CONFIG_FIELDS = (
    "mech_embedding_dim",
    "vis_embedding_dim",
    "fusion_dim",
    "latent_dim",
    "hidden_dim",
    "rnn_type",
    "visual_likelihood",
    "learn_decoder_variance",
    "posterior_fusion",
    "moment_matching",
    "modality_rnn_mode",
)

PRETRAINED_PERCEPTION_ENV_CONFIG_FIELDS = (
    "visual_field_size",
)

PERCEPTION_STATE_PREFIXES = (
    "mech_encoder.",
    "visual_encoder.",
    "fusion_layer.",
    "y2h.",
    "h2h.",
    "lstm_cell.",
    "mech_y2h.",
    "mech_h2h.",
    "vis_y2h.",
    "vis_h2h.",
    "encoder.",
    "enc_mu.",
    "enc_logvar.",
    "mech_posterior_encoder.",
    "mech_post_mu.",
    "mech_post_logvar.",
    "vis_posterior_encoder.",
    "vis_post_mu.",
    "vis_post_logvar.",
    "moe_weight_net.",
    "mech_decoder.",
    "vis_decoder.",
)


def _build_flower_params(env_cfg: EnvironmentConfig, c_value: Optional[float] = None) -> FlowerParams:
    return FlowerParams(
        r0=env_cfg.flower_r0,
        R=env_cfg.flower_R,
        L=env_cfg.flower_L,
        c=c_value or env_cfg.flower_c,
        k=env_cfg.flower_k,
        mu=env_cfg.flower_mu,
        depth_success=env_cfg.flower_depth_success,
    )


def _build_canvas_params(
    env_cfg: EnvironmentConfig,
    pattern_type: Optional[str] = None,
    cross_offset: Optional[tuple[float, float]] = None,
    stripe_orientation: Optional[str] = None,
) -> CanvasParams:
    return CanvasParams(
        width=env_cfg.canvas_width,
        height=env_cfg.canvas_height,
        background_value=env_cfg.background_value,
        flower_value=env_cfg.flower_value,
        hole_value=env_cfg.hole_value,
        pattern_type=pattern_type or env_cfg.pattern_type,
        pattern_value=env_cfg.pattern_value,
        cross_width=env_cfg.cross_width,
        cross_offset=env_cfg.cross_offset if cross_offset is None else tuple(cross_offset),
        stripe_width=env_cfg.stripe_width,
        stripe_offset=env_cfg.stripe_offset,
        stripe_orientation=stripe_orientation or env_cfg.stripe_orientation,
        circular_radius=env_cfg.circular_radius,
    )


def build_allocentric_env(
    env_cfg: EnvironmentConfig,
    c_value: Optional[float] = None,
    *,
    pattern_type: Optional[str] = None,
    cross_offset: Optional[tuple[float, float]] = None,
    stripe_orientation: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> AllocentricNectarSearchEnv:
    non_stationary_cfg = env_cfg.non_stationary
    return AllocentricNectarSearchEnv(
        flower=_build_flower_params(env_cfg, c_value),
        canvas=_build_canvas_params(
            env_cfg,
            pattern_type=pattern_type,
            cross_offset=cross_offset,
            stripe_orientation=stripe_orientation,
        ),
        dt=env_cfg.dt,
        max_steps=env_cfg.max_steps if max_steps is None else int(max_steps),
        max_press_depth=env_cfg.max_press_depth,
        min_depth=env_cfg.min_depth,
        max_depth=env_cfg.max_depth,
        proboscis_length=env_cfg.proboscis_length,
        visual_field_size=env_cfg.visual_field_size,
        non_stationary=NonStationaryParams(
            enabled=non_stationary_cfg.enabled,
            change_probability=non_stationary_cfg.change_probability,
            position_shift_radius=non_stationary_cfg.position_shift_radius,
            pattern_choices=tuple(non_stationary_cfg.pattern_choices),
            curvature_choices=tuple(non_stationary_cfg.curvature_choices),
            min_change_timestep=non_stationary_cfg.min_change_timestep,
            max_change_timestep=non_stationary_cfg.max_change_timestep,
        ),
    )


def build_training_envs(config: ExperimentConfig) -> List[AllocentricNectarSearchEnv]:
    return [
        build_allocentric_env(config.environment, c_value=c_val)
        for c_val in config.train.c_vals
    ]


def build_model(config: ExperimentConfig, env: AllocentricNectarSearchEnv):
    model_type = str(config.model.model_type).strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{config.model.model_type}'. "
            f"Expected one of {sorted(SUPPORTED_MODEL_TYPES)}."
        )

    if model_type == "allocentric_perception":
        return AllocentricNectarySearchModel(
            mech_obs_dim=env.observation_space["mech"].shape[0],
            vis_obs_shape=_allocentric_vision_shape(env),
            action_dim=env.action_space.shape[0],
            mech_embedding_dim=config.model.mech_embedding_dim,
            vis_embedding_dim=config.model.vis_embedding_dim,
            fusion_dim=config.model.fusion_dim,
            latent_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
            beta=config.model.beta,
            visual_likelihood=config.model.visual_likelihood,
            learn_decoder_variance=config.model.learn_decoder_variance,
            posterior_fusion=config.model.posterior_fusion,
            moment_matching=config.model.moment_matching,
            rnn_type=config.model.rnn_type,
            modality_rnn_mode=config.model.modality_rnn_mode,
        )

    action_low = env.action_space.low.copy()
    action_low[-1] = max(float(action_low[-1]), float(config.environment.min_depth))
    action_delta_low, action_delta_high = _allocentric_action_delta_bounds(env)
    return UncertaintyGatedAllocentricNectarySearchModel(
        mech_obs_dim=env.observation_space["mech"].shape[0],
        vis_obs_shape=_allocentric_vision_shape(env),
        action_dim=env.action_space.shape[0],
        mech_embedding_dim=config.model.mech_embedding_dim,
        vis_embedding_dim=config.model.vis_embedding_dim,
        fusion_dim=config.model.fusion_dim,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        beta=config.model.beta,
        visual_likelihood=config.model.visual_likelihood,
        learn_decoder_variance=config.model.learn_decoder_variance,
        posterior_fusion=config.model.posterior_fusion,
        moment_matching=config.model.moment_matching,
        uncertainty_metric=config.model.uncertainty_metric,
        rnn_type=config.model.rnn_type,
        modality_rnn_mode=config.model.modality_rnn_mode,
        action_low=action_low,
        action_high=env.action_space.high,
        action_delta_low=action_delta_low,
        action_delta_high=action_delta_high,
        rl_algorithm=config.model.rl_algorithm,
        exploration_objective=config.model.exploration_objective,
        gate_mode=config.model.gate_mode,
        fixed_gate_threshold=config.model.fixed_gate_threshold,
        gating_temperature=config.model.gating_temperature,
        action_selection_strategy=config.model.action_selection_strategy,
        epsilon_greedy_probability=config.model.epsilon_greedy_probability,
        fixed_switch_timestep=config.model.fixed_switch_timestep,
        random_switch_probability=config.model.random_switch_probability,
        information_gain_reward_coef=config.model.information_gain_reward_coef,
        gate_hidden_dims=config.model.gate_hidden_dims,
        gate_value_coef=config.model.gate_value_coef,
        gate_entropy_coef=config.model.gate_entropy_coef,
        exploration_action_delta_huber_beta=config.model.exploration_action_delta_huber_beta,
        exploration_action_l2=config.model.exploration_action_l2,
    )


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Checkpoint payload does not contain a usable state dict.")


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


def apply_pretrained_perception_construction_config(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location)

    experiment_config = payload.get("experiment_config")
    saved_model_config = {}
    saved_environment_config = {}
    if isinstance(experiment_config, dict):
        saved_model_config = experiment_config.get("model") or {}
        saved_environment_config = experiment_config.get("environment") or {}

    applied_model_overrides = {}
    for field_name in PRETRAINED_PERCEPTION_MODEL_CONFIG_FIELDS:
        if field_name in saved_model_config:
            field_value = saved_model_config[field_name]
        elif field_name in payload:
            field_value = payload[field_name]
        else:
            continue
        setattr(config.model, field_name, field_value)
        applied_model_overrides[field_name] = field_value

    applied_environment_overrides = {}
    for field_name in PRETRAINED_PERCEPTION_ENV_CONFIG_FIELDS:
        if field_name in saved_environment_config:
            field_value = saved_environment_config[field_name]
        elif field_name in payload:
            field_value = payload[field_name]
        else:
            continue
        setattr(config.environment, field_name, field_value)
        applied_environment_overrides[field_name] = field_value

    return {
        "checkpoint_path": str(checkpoint_path),
        "applied_model_overrides": applied_model_overrides,
        "applied_environment_overrides": applied_environment_overrides,
    }


def load_pretrained_perception_components(
    model,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    source_state_dict = _strip_module_prefix(_extract_state_dict(payload))
    perception_state_dict = {
        key: value
        for key, value in source_state_dict.items()
        if key.startswith(PERCEPTION_STATE_PREFIXES)
    }
    if not perception_state_dict:
        raise ValueError(
            f"No perception-component weights were found in checkpoint '{checkpoint_path}'."
        )

    try:
        original_threshold = model.uncertainty_threshold.clone()
        incompatible = model.load_state_dict(source_state_dict, strict=False)
        # make sure the gate threshold is still determined by the config
        model.uncertainty_threshold = original_threshold
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load pretrained perception components from '{checkpoint_path}'. "
            "This usually means the source and target perception architectures do not match."
        ) from exc

    return {
        "checkpoint_path": str(checkpoint_path),
        "num_loaded_keys": len(perception_state_dict),
        "loaded_keys": sorted(perception_state_dict.keys()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
