from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from experiment_config import ExperimentConfig, ObservationCorruptionConfig
except ImportError:
    from .experiment_config import ExperimentConfig, ObservationCorruptionConfig


@dataclass
class BenchmarkCondition:
    suite_name: str
    condition_name: str
    c_value: float
    episodes: int
    seeds: List[int]
    pattern_type: Optional[str] = None
    cross_offset: Optional[tuple[float, float]] = None
    stripe_orientation: Optional[str] = None
    corruption: ObservationCorruptionConfig = field(
        default_factory=ObservationCorruptionConfig
    )
    max_steps: Optional[int] = None
    fixed_gate_threshold: Optional[float] = None


def _condition_seeds(base_seed: int, suite_index: int, condition_index: int, episodes: int) -> List[int]:
    offset = base_seed + (suite_index * 10_000) + (condition_index * 100)
    return [offset + i for i in range(episodes)]


def _train_c_values(config: ExperimentConfig) -> List[float]:
    return list(config.train.c_vals) if config.train.c_vals else [-1.0]


def _condition_max_steps(
    max_steps_per_condition: Dict[str, Dict[str, int]],
    suite_name: str,
    condition_name: str,
) -> Optional[int]:
    max_steps = max_steps_per_condition.get(suite_name, {}).get(condition_name)
    if max_steps is None:
        return None

    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError(
            f"max_steps override for {suite_name}.{condition_name} must be positive."
        )
    return max_steps


def _condition_fixed_gate_threshold(
    fixed_gate_threshold_per_condition: Dict[str, Dict[str, float]],
    suite_name: str,
    condition_name: str,
) -> Optional[float]:
    threshold = fixed_gate_threshold_per_condition.get(suite_name, {}).get(condition_name)
    if threshold is None:
        return None
    return float(threshold)


def build_benchmark_suites(config: ExperimentConfig) -> Dict[str, List[BenchmarkCondition]]:
    episodes = int(config.evaluation.episodes_per_condition)
    train_c_values = _train_c_values(config)
    generalization_c_values = [-2.0, -5.0, -20.0, -50.0]
    generalization_c_values = [c for c in generalization_c_values if c not in set(train_c_values)]
    if not generalization_c_values:
        generalization_c_values = [-3.0]
    max_steps_per_condition = {
        "in_distribution": {},
        "curvature_generalization": {
            "c_-20": 100,
            "c_-50": 100,
        },
        "vision_degraded": {},
        "mech_degraded": {
            "drop_normals": 100,
            "blank_mech": 100,
        },
    }
    fixed_gate_threshold_per_condition = {
        "in_distribution": {},
        "curvature_generalization": {
            #"c_-2": 530,
            #"c_-5": 530,
            #"c_-20": 530,
            #"c_-50": 530,
        },
        "vision_degraded": {},
        "mech_degraded": {
            #"drop_normals": 530,
            #"blank_mech": 530,
        },
    }

    suite_builders = {
        "in_distribution": [],
        "curvature_generalization": [],
        "vision_degraded": [],
        "mech_degraded": [],
    }

    suite_names = list(suite_builders.keys())
    for suite_index, suite_name in enumerate(suite_names):
        if suite_name == "in_distribution":
            for condition_index, c_value in enumerate(train_c_values):
                condition_name = f"c_{c_value:g}"
                suite_builders[suite_name].append(
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name=condition_name,
                        c_value=float(c_value),
                        episodes=episodes,
                        seeds=_condition_seeds(
                            config.evaluation.base_seed,
                            suite_index,
                            condition_index,
                            episodes,
                        ),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            condition_name,
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            condition_name,
                        ),
                    )
                )
        elif suite_name == "curvature_generalization":
            for condition_index, c_value in enumerate(generalization_c_values):
                condition_name = f"c_{c_value:g}"
                suite_builders[suite_name].append(
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name=condition_name,
                        c_value=float(c_value),
                        episodes=episodes,
                        seeds=_condition_seeds(
                            config.evaluation.base_seed,
                            suite_index,
                            condition_index,
                            episodes,
                        ),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            condition_name,
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            condition_name,
                        ),
                    )
                )
        elif suite_name == "vision_degraded":
            base_c = float(train_c_values[0])
            shifted_cross_offset = (
                float(config.environment.stripe_offset),
                float(config.environment.stripe_offset),
            )
            suite_builders[suite_name].extend(
                [
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name="blank_vision",
                        c_value=base_c,
                        episodes=episodes,
                        seeds=_condition_seeds(config.evaluation.base_seed, suite_index, 0, episodes),
                        corruption=ObservationCorruptionConfig(blank_vision=True),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            "blank_vision",
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            "blank_vision",
                        ),
                    ),
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name="vision_noise_0.1",
                        c_value=base_c,
                        episodes=episodes,
                        seeds=_condition_seeds(config.evaluation.base_seed, suite_index, 1, episodes),
                        corruption=ObservationCorruptionConfig(vision_noise_std=0.1),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            "vision_noise_0.1",
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            "vision_noise_0.1",
                        ),
                    ),
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name="pattern_shift_cross",
                        c_value=base_c,
                        episodes=episodes,
                        seeds=_condition_seeds(config.evaluation.base_seed, suite_index, 2, episodes),
                        pattern_type="cross",
                        cross_offset=shifted_cross_offset,
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            "pattern_shift_cross",
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            "pattern_shift_cross",
                        ),
                    ),
                ]
            )
        elif suite_name == "mech_degraded":
            base_c = float(train_c_values[0])
            suite_builders[suite_name].extend(
                [
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name="blank_mech",
                        c_value=base_c,
                        episodes=episodes,
                        seeds=_condition_seeds(config.evaluation.base_seed, suite_index, 0, episodes),
                        corruption=ObservationCorruptionConfig(blank_mech=True),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            "blank_mech",
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            "blank_mech",
                        ),
                    ),
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name="mech_noise_0.1",
                        c_value=base_c,
                        episodes=episodes,
                        seeds=_condition_seeds(config.evaluation.base_seed, suite_index, 1, episodes),
                        corruption=ObservationCorruptionConfig(mech_noise_std=0.1),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            "mech_noise_0.1",
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            "mech_noise_0.1",
                        ),
                    ),
                    BenchmarkCondition(
                        suite_name=suite_name,
                        condition_name="drop_normals",
                        c_value=base_c,
                        episodes=episodes,
                        seeds=_condition_seeds(config.evaluation.base_seed, suite_index, 2, episodes),
                        corruption=ObservationCorruptionConfig(zero_mech_channels=[2, 3]),
                        max_steps=_condition_max_steps(
                            max_steps_per_condition,
                            suite_name,
                            "drop_normals",
                        ),
                        fixed_gate_threshold=_condition_fixed_gate_threshold(
                            fixed_gate_threshold_per_condition,
                            suite_name,
                            "drop_normals",
                        ),
                    ),
                ]
            )

    requested = []
    for suite_name in config.evaluation.suite_names:
        normalized = str(suite_name).strip().lower()
        if normalized not in suite_builders:
            raise ValueError(
                f"Unknown evaluation suite '{suite_name}'. "
                f"Expected one of {sorted(suite_builders)}."
            )
        requested.append(normalized)

    return {suite_name: suite_builders[suite_name] for suite_name in requested}
