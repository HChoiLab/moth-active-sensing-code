from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from benchmark_suites import BenchmarkCondition
    from experiment_config import ExperimentConfig, ObservationCorruptionConfig
    from experiment_registry import build_allocentric_env
    from uncertainty_gated_model import (
        UncertaintyGatedAllocentricNectarySearchModel,
        apply_stateful_fixed_threshold_gate,
    )
except ImportError:
    from .benchmark_suites import BenchmarkCondition
    from .experiment_config import ExperimentConfig, ObservationCorruptionConfig
    from .experiment_registry import build_allocentric_env
    from .uncertainty_gated_model import (
        UncertaintyGatedAllocentricNectarySearchModel,
        apply_stateful_fixed_threshold_gate,
    )


def _policy_mode_to_force_mode(policy_mode: str) -> Optional[str]:
    normalized = str(policy_mode).strip().lower()
    if normalized in {"auto", "gated"}:
        return None
    if normalized in {"explore", "exploit"}:
        return normalized
    raise ValueError("policy_mode must be one of: auto, explore, exploit.")


def _obs_dict_to_tensors(obs: Dict[str, np.ndarray], device: torch.device):
    mech = torch.as_tensor(obs["mech"], dtype=torch.float32, device=device).reshape(1, -1)
    vis = torch.as_tensor(obs["vision"], dtype=torch.float32, device=device)
    if vis.ndim == 2:
        vis = vis.unsqueeze(0)
    vis = vis.unsqueeze(0)
    return mech, vis


def _allocentric_action_from_pose(env) -> np.ndarray:
    head_norm = env._normalize_canvas_xy(env.head_xy)
    tip_norm = env._normalize_canvas_xy(env.tip_xy)
    return np.array(
        [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], float(env.tip_depth)],
        dtype=np.float32,
    )


def _clamp_action(model: UncertaintyGatedAllocentricNectarySearchModel, action: torch.Tensor) -> torch.Tensor:
    action_low = model.action_low.to(action.device)
    action_high = model.action_high.to(action.device)
    return torch.max(torch.min(action, action_high), action_low)


def _apply_corruption(
    obs: Dict[str, np.ndarray],
    corruption: ObservationCorruptionConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    mech = np.asarray(obs["mech"], dtype=np.float32).copy()
    vision = np.asarray(obs["vision"], dtype=np.float32).copy()

    if corruption.blank_mech:
        mech[...] = 0.0
    else:
        if corruption.zero_mech_channels:
            mech[np.asarray(corruption.zero_mech_channels, dtype=int)] = 0.0
        if corruption.mech_noise_std > 0:
            mech += rng.normal(0.0, corruption.mech_noise_std, size=mech.shape).astype(np.float32)

    if corruption.blank_vision:
        vision[...] = 0.0
    elif corruption.vision_noise_std > 0:
        vision += rng.normal(0.0, corruption.vision_noise_std, size=vision.shape).astype(np.float32)
        vision = np.clip(vision, 0.0, 1.0)

    return {"mech": mech, "vision": vision}


def _distance_to_hole(env) -> float:
    local_xy = env._to_flower_xy(env.tip_xy)
    return float(np.hypot(local_xy[0], local_xy[1]))


@contextmanager
def _override_fixed_gate_threshold(
    model: UncertaintyGatedAllocentricNectarySearchModel,
    fixed_gate_threshold: Optional[float],
):
    if fixed_gate_threshold is None:
        yield float(model.uncertainty_threshold.detach().item())
        return

    if str(model.gate_mode).strip().lower() != "fixed_threshold":
        raise ValueError(
            "Per-condition fixed_gate_threshold overrides require "
            "model.gate_mode='fixed_threshold'."
        )

    original_threshold = model.uncertainty_threshold.detach().clone()
    with torch.no_grad():
        model.uncertainty_threshold.fill_(float(fixed_gate_threshold))
    try:
        yield float(model.uncertainty_threshold.detach().item())
    finally:
        with torch.no_grad():
            model.uncertainty_threshold.copy_(original_threshold)


def run_episode(
    model: UncertaintyGatedAllocentricNectarySearchModel,
    config: ExperimentConfig,
    condition: BenchmarkCondition,
    *,
    policy_mode: str,
    episode_seed: int,
    device: torch.device,
    record_steps: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env = build_allocentric_env(
        config.environment,
        condition.c_value,
        pattern_type=condition.pattern_type,
        cross_offset=condition.cross_offset,
        stripe_orientation=condition.stripe_orientation,
        max_steps=condition.max_steps,
    )
    obs, reset_info = env.reset(seed=episode_seed)
    a_prev_t = torch.as_tensor(
        _allocentric_action_from_pose(env),
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    h_prev = model.initial_hidden(batch_size=1, device=device, dtype=torch.float32)
    rng = np.random.default_rng(episode_seed)
    force_mode = _policy_mode_to_force_mode(policy_mode)
    gate_hysteresis = float(config.train.gate_hysteresis)
    gate_uncertainty_ema_alpha = config.train.gate_uncertainty_ema_alpha
    stateful_gate_requested = (
        gate_hysteresis > 0.0 or gate_uncertainty_ema_alpha is not None
    )
    if (
        stateful_gate_requested
        and force_mode is None
        and model.uses_uncertainty_gate
        and model.gate_uses_rl
    ):
        raise ValueError(
            "EMA/hysteresis gating is only supported with gate_mode='fixed_threshold'."
        )
    use_stateful_fixed_gate = (
        stateful_gate_requested
        and force_mode is None
        and model.uses_uncertainty_gate
        and not model.gate_uses_rl
    )
    gate_is_exploration = None
    gate_uncertainty_ema = None

    episode_return = 0.0
    explore_steps = 0
    contact_steps = 0
    fn_values = []
    slip_values = []
    mode_history: List[int] = []
    step_records: List[Dict[str, Any]] = []
    success = False
    time_to_success = None
    non_stationary_changed = False
    non_stationary_change_timestep = reset_info.get(
        "non_stationary_change_timestep"
    )

    with _override_fixed_gate_threshold(model, condition.fixed_gate_threshold) as effective_threshold:
        for step_index in range(env.max_steps):
            corrupted_obs = _apply_corruption(obs, condition.corruption, rng)
            x_mech_t, x_vis_t = _obs_dict_to_tensors(corrupted_obs, device)
            with torch.no_grad():
                outputs = model.act(
                    x_mech_t,
                    x_vis_t,
                    a_prev_t,
                    h_prev,
                    deterministic_exploitation=config.evaluation.deterministic_exploitation,
                    deterministic_exploration=config.evaluation.deterministic_exploration,
                    stochastic_gate=False,
                    force_mode=force_mode,
                    episode_timestep=step_index,
                )
            selection = outputs["action_selection"]
            if use_stateful_fixed_gate:
                (
                    selection,
                    gate_is_exploration,
                    gate_uncertainty_ema,
                ) = apply_stateful_fixed_threshold_gate(
                    selection,
                    previous_is_exploration=gate_is_exploration,
                    previous_smoothed_uncertainty=gate_uncertainty_ema,
                    uncertainty_ema_alpha=gate_uncertainty_ema_alpha,
                    hysteresis=gate_hysteresis,
                )
            env_action_t = _clamp_action(model, selection.action)
            env_action_np = env_action_t.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(env_action_np)
            executed_action_t = torch.as_tensor(
                _allocentric_action_from_pose(env),
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)

            is_explore = int(bool(selection.selected_is_exploration.reshape(()).item()))
            exploration_value = (
                math.nan
                if selection.exploration_value is None
                else float(selection.exploration_value.reshape(()).item())
            )
            exploitation_value = float(selection.exploitation_value.reshape(()).item())
            gate_value = (
                math.nan
                if selection.gate_value is None
                else float(selection.gate_value.reshape(()).item())
            )
            mode_history.append(is_explore)
            explore_steps += is_explore
            contact_steps += int(bool(info.get("contact", 0.0) > 0.0))
            fn_values.append(float(info.get("fn", 0.0)))
            slip_values.append(float(info.get("slip", 0.0)))

            episode_return += float(reward)
            success = bool(info.get("success", False))
            changed_this_step = bool(
                info.get("non_stationary_changed_this_step", False)
            )
            non_stationary_changed = non_stationary_changed or changed_this_step
            if success and time_to_success is None:
                time_to_success = step_index + 1

            if record_steps:
                step_records.append(
                    {
                        "suite_name": condition.suite_name,
                        "condition_name": condition.condition_name,
                        "policy_mode": policy_mode,
                        "seed": int(episode_seed),
                        "step": step_index,
                        "selected_mode": "explore" if is_explore else "exploit",
                        "reward": float(reward),
                        "success": float(success),
                        "contact": float(info.get("contact", 0.0)),
                        "fn": float(info.get("fn", 0.0)),
                        "slip": float(info.get("slip", 0.0)),
                        "uncertainty": float(selection.uncertainty.reshape(()).item()),
                        "gate_smoothed_uncertainty": (
                            float(selection.uncertainty.reshape(()).item())
                            if gate_uncertainty_ema is None
                            else float(gate_uncertainty_ema.reshape(()).item())
                        ),
                        "threshold": float(selection.threshold.reshape(()).item()),
                        "explore_probability": float(selection.explore_probability.reshape(()).item()),
                        "gate_value": gate_value,
                        "exploration_value": exploration_value,
                        "exploitation_value": exploitation_value,
                        "head_x": float(info["head_xy"][0]),
                        "head_y": float(info["head_xy"][1]),
                        "tip_x": float(info["tip_xy"][0]),
                        "tip_y": float(info["tip_xy"][1]),
                        "tip_depth": float(info["tip_depth"]),
                        "non_stationary_changed_this_step": float(
                            changed_this_step
                        ),
                        "flower_center_x": float(info["flower_center_xy"][0]),
                        "flower_center_y": float(info["flower_center_xy"][1]),
                        "flower_pattern_type": info["flower_pattern_type"],
                        "flower_curvature": float(info["flower_curvature"]),
                    }
                )

            obs = next_obs
            a_prev_t = executed_action_t
            if "c" in outputs["responses"]:
                h_prev = (outputs["responses"]["h"], outputs["responses"]["c"])
            else:
                h_prev = outputs["responses"]["h"]

            if terminated or truncated:
                break

    num_steps = len(mode_history)
    num_switches = sum(
        1 for prev_mode, next_mode in zip(mode_history[:-1], mode_history[1:]) if prev_mode != next_mode
    )
    episode_record = {
        "suite_name": condition.suite_name,
        "condition_name": condition.condition_name,
        "policy_mode": policy_mode,
        "seed": int(episode_seed),
        "success": float(success),
        "episode_return": float(episode_return),
        "episode_length": int(num_steps),
        "time_to_success": None if time_to_success is None else int(time_to_success),
        "explore_fraction": float(explore_steps / max(num_steps, 1)),
        "num_mode_switches": int(num_switches),
        "contact_fraction": float(contact_steps / max(num_steps, 1)),
        "mean_fn": float(np.mean(fn_values)) if fn_values else 0.0,
        "mean_slip": float(np.mean(slip_values)) if slip_values else 0.0,
        "final_distance_to_hole": _distance_to_hole(env),
        "final_tip_depth": float(env.tip_depth),
        "flower_c": float(condition.c_value),
        "pattern_type": condition.pattern_type or config.environment.pattern_type,
        "cross_offset": condition.cross_offset or config.environment.cross_offset,
        "max_steps": int(env.max_steps),
        "fixed_gate_threshold": float(effective_threshold),
        "gate_hysteresis": gate_hysteresis,
        "gate_uncertainty_ema_alpha": gate_uncertainty_ema_alpha,
        "non_stationary_change_scheduled": float(
            non_stationary_change_timestep is not None
        ),
        "non_stationary_change_timestep": non_stationary_change_timestep,
        "non_stationary_changed": float(non_stationary_changed),
        "final_flower_pattern_type": env.current_pattern_type,
        "final_flower_curvature": env.current_flower_c,
    }
    return episode_record, step_records


def run_suite(
    model: UncertaintyGatedAllocentricNectarySearchModel,
    config: ExperimentConfig,
    suite_conditions: List[BenchmarkCondition],
    *,
    policy_mode: str,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episode_records: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    trace_budget = max(int(config.evaluation.max_trace_episodes_per_condition), 0)

    model.eval()
    for condition in suite_conditions:
        for episode_index, episode_seed in enumerate(condition.seeds):
            record_steps = (
                config.evaluation.record_step_traces and episode_index < trace_budget
            )
            episode_record, episode_step_records = run_episode(
                model,
                config,
                condition,
                policy_mode=policy_mode,
                episode_seed=episode_seed,
                device=device,
                record_steps=record_steps,
            )
            episode_records.append(episode_record)
            step_records.extend(episode_step_records)
    return episode_records, step_records
