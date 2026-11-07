from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_force_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    if normalized not in {"explore", "exploit"}:
        raise ValueError("force_mode must be one of: auto, explore, exploit.")
    return normalized


def _normalize_gate_mode(value: Optional[str]) -> str:
    if value is None:
        return "fixed_threshold"
    normalized = str(value).strip().lower()
    aliases = {
        "threshold": "fixed_threshold",
        "fixed": "fixed_threshold",
        "rl": "rl_head",
        "learned": "rl_head",
        "learned_head": "rl_head",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"fixed_threshold", "rl_head"}:
        raise ValueError("gate_mode must be one of: fixed_threshold, rl_head.")
    return normalized


def _normalize_action_selection_strategy(value: Optional[str]) -> str:
    if value is None:
        return "uncertainty_gate"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gate": "uncertainty_gate",
        "gated": "uncertainty_gate",
        "uncertainty_gated": "uncertainty_gate",
        "epsilon": "epsilon_greedy",
        "eps_greedy": "epsilon_greedy",
        "random_exploration": "epsilon_greedy",
        "fixed_switch": "fixed_switching",
        "fixed": "fixed_switching",
        "random_switch": "random_switching",
        "stochastic_switching": "random_switching",
        "nonlexicographic": "non_lexicographic",
        "nonlex": "non_lexicographic",
        "combined_reward": "non_lexicographic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "uncertainty_gate",
        "epsilon_greedy",
        "fixed_switching",
        "random_switching",
        "non_lexicographic",
    }:
        raise ValueError(
            "action_selection_strategy must be one of: uncertainty_gate, "
            "epsilon_greedy, fixed_switching, random_switching, "
            "non_lexicographic."
        )
    return normalized


def _normalize_posterior_fusion(value: Optional[str]) -> str:
    if value is None:
        return "early"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "early_fusion": "early",
        "product_of_experts": "poe",
        "product": "poe",
        "moe": "weighted_moe",
        "mixture_of_experts": "weighted_moe",
        "mixture": "weighted_moe",
        "weighted_mixture_of_experts": "weighted_moe",
        "weighted_mixture": "weighted_moe",
        "moe_weighted": "weighted_moe",
        "uniform_mixture_of_experts": "uniform_moe",
        "uniform_mixture": "uniform_moe",
        "moe_uniform": "uniform_moe",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"early", "poe", "uniform_moe", "weighted_moe"}:
        raise ValueError(
            "posterior_fusion must be one of: early, poe, uniform_moe, weighted_moe."
        )
    return normalized


def _normalize_uncertainty_metric(value: Optional[str]) -> str:
    if value is None:
        return "aggregate"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "joint": "aggregate",
        "aggregate_posterior": "aggregate",
        "average": "average_unimodal",
        "mean_unimodal": "average_unimodal",
        "weighted": "weighted_unimodal",
        "moe_weighted": "weighted_unimodal",
        "mechanosensory": "mech",
        "visual": "vis",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "aggregate",
        "average_unimodal",
        "weighted_unimodal",
        "mech",
        "vis",
    }:
        raise ValueError(
            "uncertainty_metric must be one of: aggregate, average_unimodal, "
            "weighted_unimodal, mech, vis."
        )
    return normalized


def _normalize_bool(value: Optional[Any], default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _normalize_rnn_type(value: Optional[str]) -> str:
    if value is None:
        return "rnn"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "simple": "rnn",
        "vanilla": "rnn",
        "vanilla_rnn": "rnn",
        "lstm_cell": "lstm",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"rnn", "lstm"}:
        raise ValueError("rnn_type must be one of: rnn, lstm.")
    return normalized


def _normalize_modality_rnn_mode(value: Optional[str]) -> str:
    if value is None:
        return "shared"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "joint": "shared",
        "fused": "shared",
        "shared_fusion": "shared",
        "separate_modalities": "separate",
        "modality_separate": "separate",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"shared", "separate"}:
        raise ValueError("modality_rnn_mode must be one of: shared, separate.")
    return normalized


def _normalize_xy_offset(
    value: Optional[Any],
    *,
    field_name: str,
) -> tuple[float, float]:
    if value is None:
        return (0.0, 0.0)

    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a length-2 sequence, not a string.")

    try:
        values = tuple(float(component) for component in value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a length-2 sequence.") from exc

    if len(values) != 2:
        raise ValueError(f"{field_name} must contain exactly two values: (x_offset, y_offset).")

    return values


@dataclass
class NonStationaryConfig:
    enabled: bool = False
    change_probability: float = 0.0
    position_shift_radius: float = 0.0
    pattern_choices: List[str] = field(default_factory=list)
    curvature_choices: List[float] = field(default_factory=list)
    min_change_timestep: int = 1
    max_change_timestep: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "NonStationaryConfig":
        payload = dict(data or {})
        payload["enabled"] = _normalize_bool(payload.get("enabled"), default=False)

        change_probability = float(payload.get("change_probability", 0.0))
        if not 0.0 <= change_probability <= 1.0:
            raise ValueError("non_stationary.change_probability must be in [0, 1].")
        payload["change_probability"] = change_probability

        position_shift_radius = float(payload.get("position_shift_radius", 0.0))
        if position_shift_radius < 0.0:
            raise ValueError(
                "non_stationary.position_shift_radius must be non-negative."
            )
        payload["position_shift_radius"] = position_shift_radius

        pattern_choices = payload.get("pattern_choices") or []
        if isinstance(pattern_choices, str):
            pattern_choices = [pattern_choices]
        pattern_choices = [
            str(pattern).strip().lower()
            for pattern in pattern_choices
            if str(pattern).strip()
        ]
        supported_patterns = {"none", "cross", "stripes", "circular"}
        invalid_patterns = sorted(set(pattern_choices) - supported_patterns)
        if invalid_patterns:
            raise ValueError(
                "non_stationary.pattern_choices contains unsupported patterns: "
                + ", ".join(invalid_patterns)
            )
        payload["pattern_choices"] = pattern_choices

        curvature_choices = payload.get("curvature_choices") or []
        payload["curvature_choices"] = [
            float(curvature) for curvature in curvature_choices
        ]

        min_timestep = payload.get("min_change_timestep", 1)
        if (
            isinstance(min_timestep, bool)
            or int(min_timestep) != min_timestep
            or int(min_timestep) < 1
        ):
            raise ValueError(
                "non_stationary.min_change_timestep must be a positive integer."
            )
        payload["min_change_timestep"] = int(min_timestep)

        max_timestep = payload.get("max_change_timestep")
        if max_timestep is not None:
            if (
                isinstance(max_timestep, bool)
                or int(max_timestep) != max_timestep
                or int(max_timestep) < int(min_timestep)
            ):
                raise ValueError(
                    "non_stationary.max_change_timestep must be an integer greater "
                    "than or equal to min_change_timestep."
                )
            payload["max_change_timestep"] = int(max_timestep)

        if payload["enabled"] and change_probability > 0.0:
            if not pattern_choices:
                raise ValueError(
                    "non_stationary.pattern_choices must not be empty when changes "
                    "are enabled."
                )
            if not payload["curvature_choices"]:
                raise ValueError(
                    "non_stationary.curvature_choices must not be empty when changes "
                    "are enabled."
                )

        return cls(**payload)


@dataclass
class EnvironmentConfig:
    dt: float = 0.01
    max_steps: int = 100
    max_press_depth: float = 4.0
    min_depth: float = -2.0
    max_depth: float = 10.0
    proboscis_length: float = 25.0
    visual_field_size: float = 15.0
    canvas_width: float = 200.0
    canvas_height: float = 200.0
    background_value: float = 0.0
    flower_value: float = 1.0
    hole_value: float = 0.0
    pattern_type: str = "none"
    pattern_value: float = 0.5
    cross_width: float = 5.0
    cross_offset: tuple[float, float] = (0.0, 0.0)
    stripe_width: float = 5.0
    stripe_offset: float = 10.0
    stripe_orientation: str = "vertical"
    circular_radius: float = 10.0
    flower_r0: float = 3.0
    flower_R: float = 30.0
    flower_L: float = 25.0
    flower_k: float = 5.0
    flower_mu: float = 0.6
    flower_c: float = -1.0
    flower_depth_success: float = 3.0
    non_stationary: NonStationaryConfig = field(default_factory=NonStationaryConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvironmentConfig":
        payload = dict(data or {})
        payload["cross_offset"] = _normalize_xy_offset(
            payload.get("cross_offset"),
            field_name="cross_offset",
        )
        payload["non_stationary"] = NonStationaryConfig.from_dict(
            payload.get("non_stationary")
        )
        return cls(**payload)


@dataclass
class ModelConfig:
    model_type: str = "uncertainty_gated_allocentric"
    pretrained_perception_checkpoint: Optional[str] = None
    latent_dim: int = 1024
    hidden_dim: int = 2048
    mech_embedding_dim: Optional[int] = None
    vis_embedding_dim: int = 16
    fusion_dim: int = 256
    beta: float = 0.002
    visual_likelihood: str = "gaussian"
    learn_decoder_variance: bool = True
    posterior_fusion: str = "early"
    moment_matching: bool = True
    uncertainty_metric: str = "aggregate"
    rnn_type: str = "rnn"
    modality_rnn_mode: str = "shared"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    perception_lr: float = 3e-4
    exploration_lr: float = 1e-4
    exploitation_lr: float = 3e-4
    head_lr_scheduler: bool = False
    head_lr_scheduler_start_epoch: int = 0
    learning_rate: float = 3e-4
    rl_algorithm: str = "ppo"
    exploration_objective: str = "realized_information_gain"
    gate_mode: str = "fixed_threshold"
    fixed_gate_threshold: Optional[float] = None
    gating_temperature: float = 0.25
    action_selection_strategy: str = "uncertainty_gate"
    epsilon_greedy_probability: float = 0.1
    fixed_switch_timestep: Optional[int] = None
    random_switch_probability: float = 0.1
    information_gain_reward_coef: float = 1.0
    gate_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    gate_lr: Optional[float] = None
    gate_value_coef: Optional[float] = None
    gate_entropy_coef: Optional[float] = None
    exploration_action_delta_huber_beta: float = 0.05
    exploration_action_l2: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ModelConfig":
        payload = dict(data or {})
        payload["gate_mode"] = _normalize_gate_mode(payload.get("gate_mode"))
        payload["action_selection_strategy"] = _normalize_action_selection_strategy(
            payload.get("action_selection_strategy")
        )
        if payload.get("fixed_switch_timestep") is not None:
            value = payload["fixed_switch_timestep"]
            if (
                isinstance(value, bool)
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError("fixed_switch_timestep must be a non-negative integer.")
            payload["fixed_switch_timestep"] = int(value)
        if (
            payload["action_selection_strategy"] == "fixed_switching"
            and payload.get("fixed_switch_timestep") is None
        ):
            raise ValueError(
                "fixed_switch_timestep is required when "
                "action_selection_strategy='fixed_switching'."
            )
        if "random_switch_probability" in payload:
            probability = float(payload["random_switch_probability"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("random_switch_probability must be in [0, 1].")
            payload["random_switch_probability"] = probability
        payload["posterior_fusion"] = _normalize_posterior_fusion(
            payload.get("posterior_fusion")
        )
        payload["moment_matching"] = _normalize_bool(
            payload.get("moment_matching"),
            default=True,
        )
        payload["uncertainty_metric"] = _normalize_uncertainty_metric(
            payload.get("uncertainty_metric")
        )
        payload["rnn_type"] = _normalize_rnn_type(payload.get("rnn_type"))
        payload["modality_rnn_mode"] = _normalize_modality_rnn_mode(
            payload.get("modality_rnn_mode")
        )
        if "fixed_gate_threshold" not in payload and "initial_threshold" in payload:
            payload["fixed_gate_threshold"] = payload["initial_threshold"]
        payload.pop("initial_threshold", None)
        payload.pop("gate_loss_coef", None)
        gate_hidden_dims = payload.get("gate_hidden_dims")
        if gate_hidden_dims is not None:
            payload["gate_hidden_dims"] = [int(value) for value in gate_hidden_dims]
        return cls(**payload)


@dataclass
class TrainingConfig:
    enabled: bool = True
    num_epochs: int = 200
    batch_size: int = 8
    seq_len: Optional[int] = None
    print_every: int = 10
    seed: int = 2026
    resume_checkpoint: Optional[str] = None
    c_vals: List[float] = field(default_factory=lambda: [-1.0, -2.0, -100.0, -1000.0])
    force_mode: Optional[str] = None
    randomize_visual_pattern_each_epoch: bool = True
    visual_pattern_choices: Optional[List[str]] = None
    grad_clip_norm: Optional[float] = None
    detach_hidden_state: bool = True
    deterministic_exploitation: bool = False
    deterministic_exploration: bool = False
    exploration_warmup_epochs: int = 0
    gate_hysteresis: float = 0.0
    gate_uncertainty_ema_alpha: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        payload = dict(data or {})
        payload["force_mode"] = _normalize_force_mode(payload.get("force_mode"))
        if "gate_hysteresis" in payload:
            payload["gate_hysteresis"] = float(payload["gate_hysteresis"])
        if payload.get("gate_uncertainty_ema_alpha") is not None:
            payload["gate_uncertainty_ema_alpha"] = float(
                payload["gate_uncertainty_ema_alpha"]
            )
        payload.pop("sample_gate", None)
        return cls(**payload)


@dataclass
class ObservationCorruptionConfig:
    blank_vision: bool = False
    vision_noise_std: float = 0.0
    blank_mech: bool = False
    mech_noise_std: float = 0.0
    zero_mech_channels: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ObservationCorruptionConfig":
        return cls(**(data or {}))


@dataclass
class EvaluationConfig:
    enabled: bool = True
    suite_names: List[str] = field(
        default_factory=lambda: [
            "in_distribution",
            "curvature_generalization",
            "vision_degraded",
            "mech_degraded",
        ]
    )
    episodes_per_condition: int = 50
    base_seed: int = 5000
    policy_modes: List[str] = field(default_factory=lambda: ["auto"])
    record_step_traces: bool = True
    max_trace_episodes_per_condition: int = 3
    deterministic_exploitation: bool = True
    deterministic_exploration: bool = True
    default_corruption: ObservationCorruptionConfig = field(
        default_factory=ObservationCorruptionConfig
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvaluationConfig":
        payload = dict(data or {})
        payload.pop("sample_gate", None)
        payload["default_corruption"] = ObservationCorruptionConfig.from_dict(
            payload.get("default_corruption")
        )
        return cls(**payload)


@dataclass
class ExperimentConfig:
    run_name: Optional[str] = None
    output_root: str = "NectarySearch/results"
    notes: str = ""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        payload = dict(data)
        return cls(
            run_name=payload.get("run_name"),
            output_root=payload.get("output_root", "NectarySearch/results"),
            notes=payload.get("notes", ""),
            environment=EnvironmentConfig.from_dict(payload.get("environment")),
            model=ModelConfig.from_dict(payload.get("model")),
            train=TrainingConfig.from_dict(payload.get("train")),
            evaluation=EvaluationConfig.from_dict(payload.get("evaluation")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config = cls.from_dict(data)
        if config.run_name is None:
            config.run_name = path.stem
        return config

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["train"]["force_mode"] = self.train.force_mode
        return payload

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
