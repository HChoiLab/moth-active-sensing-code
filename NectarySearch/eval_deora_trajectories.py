from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

try:
    from deora_data import (
        DEORA_DATA_ROOT,
        generate_matched_random_trajectory,
        load_deora_trajectories,
        replay_deora_trajectory,
    )
    from experiment_config import ExperimentConfig
    from experiment_registry import build_allocentric_env, build_model
except ImportError:
    from .deora_data import (
        DEORA_DATA_ROOT,
        generate_matched_random_trajectory,
        load_deora_trajectories,
        replay_deora_trajectory,
    )
    from .experiment_config import ExperimentConfig
    from .experiment_registry import build_allocentric_env, build_model


COMPARISON_METRICS = (
    "success",
    "reached_hole",
    "episode_return",
    "contact_fraction",
    "final_distance_to_hole",
    "min_distance_to_hole",
    "mean_realized_information_gain",
    "cumulative_realized_information_gain",
    "mean_posterior_entropy",
    "mean_moth_action_exploration_log_prob",
    "mean_moth_action_exploitation_log_prob",
)
HIGHER_IS_BETTER = {
    "success",
    "reached_hole",
    "episode_return",
    "contact_fraction",
    "mean_realized_information_gain",
    "cumulative_realized_information_gain",
}
LOWER_IS_BETTER = {
    "final_distance_to_hole",
    "min_distance_to_hole",
    "mean_posterior_entropy",
}


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_config(config_path: Optional[str], checkpoint_path: Optional[str]) -> ExperimentConfig:
    if config_path:
        return ExperimentConfig.from_json(config_path)

    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        experiment_config = payload.get("experiment_config")
        if isinstance(experiment_config, dict):
            return ExperimentConfig.from_dict(experiment_config)

        candidate = Path(checkpoint_path).parent.parent / "config.json"
        if candidate.exists():
            return ExperimentConfig.from_json(candidate)

    return ExperimentConfig()


def _load_model(checkpoint_path: Optional[str], config: ExperimentConfig, device: torch.device):
    env = build_allocentric_env(config.environment)
    model = build_model(config, env)
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(_strip_module_prefix(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def _write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _finite_metric_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = [_as_finite_float(row.get(metric)) for row in rows]
    return np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)


def _random_bound_for_trajectory(trajectory, mode: str) -> Optional[float]:
    if mode == "none":
        return None
    if mode == "flower":
        return 1.0
    if mode == "observed":
        return float(np.nanmax(trajectory.radius_normalized))
    raise ValueError("random baseline bound must be one of: observed, flower, none.")


def _comparison_row(
    biological_summary: dict[str, Any],
    random_summaries: list[dict[str, Any]],
    *,
    random_baseline_bound: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": biological_summary["name"],
        "visit": biological_summary["visit"],
        "source": biological_summary["source"],
        "c_value": biological_summary["c_value"],
        "moth_id": biological_summary["moth_id"],
        "num_input_points": biological_summary["num_input_points"],
        "num_replayed_steps": biological_summary["num_replayed_steps"],
        "random_baseline_bound": random_baseline_bound,
        "random_baselines": len(random_summaries),
    }
    for metric in COMPARISON_METRICS:
        observed = _as_finite_float(biological_summary.get(metric))
        random_values = _finite_metric_values(random_summaries, metric)
        prefix = metric
        row[f"biological_{prefix}"] = observed
        row[f"{prefix}_random_n"] = int(random_values.size)

        if random_values.size == 0 or not np.isfinite(observed):
            row[f"{prefix}_random_mean"] = float("nan")
            row[f"{prefix}_random_std"] = float("nan")
            row[f"{prefix}_random_p05"] = float("nan")
            row[f"{prefix}_random_p50"] = float("nan")
            row[f"{prefix}_random_p95"] = float("nan")
            row[f"{prefix}_biological_percentile"] = float("nan")
            row[f"{prefix}_z_score"] = float("nan")
            row[f"{prefix}_advantage"] = float("nan")
            row[f"{prefix}_better_than_random_fraction"] = float("nan")
            continue

        random_mean = float(np.mean(random_values))
        random_std = (
            float(np.std(random_values, ddof=1))
            if random_values.size > 1
            else 0.0
        )
        row[f"{prefix}_random_mean"] = random_mean
        row[f"{prefix}_random_std"] = random_std
        row[f"{prefix}_random_p05"] = float(np.percentile(random_values, 5))
        row[f"{prefix}_random_p50"] = float(np.percentile(random_values, 50))
        row[f"{prefix}_random_p95"] = float(np.percentile(random_values, 95))
        row[f"{prefix}_biological_percentile"] = float(np.mean(random_values <= observed))
        row[f"{prefix}_z_score"] = (
            float((observed - random_mean) / random_std)
            if random_std > 0.0
            else float("nan")
        )

        if metric in HIGHER_IS_BETTER:
            row[f"{prefix}_advantage"] = float(observed - random_mean)
            row[f"{prefix}_better_than_random_fraction"] = float(
                np.mean(random_values <= observed)
            )
        elif metric in LOWER_IS_BETTER:
            row[f"{prefix}_advantage"] = float(random_mean - observed)
            row[f"{prefix}_better_than_random_fraction"] = float(
                np.mean(random_values >= observed)
            )
        else:
            row[f"{prefix}_advantage"] = float("nan")
            row[f"{prefix}_better_than_random_fraction"] = float("nan")

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Deora et al. biological moth trajectories in the allocentric nectary environment."
    )
    parser.add_argument(
        "--checkpoint",
        default="NectarySearch/checkpoints/nectary_allocentric_2027_MoE_mechEmbed_sharedRNN_subsampling_noDecVar.pt", #NectarySearch/results/pretrained_perception/funnel_cross/mode_switching/1998/checkpoints/last.pt",
        help="Optional model checkpoint for posterior entropy/IG metrics."
    )
    parser.add_argument(
        "--config",
        default="NectarySearch/configs/experiments/mode_switch.json",
        help="Optional experiment config JSON. Defaults to checkpoint config if present."
    )
    parser.add_argument("--data-root", default=str(DEORA_DATA_ROOT))
    parser.add_argument(
        "--source",
        default="filtered",
        choices=("filtered", "final_compiled", "radius_and_angle"),
    )
    parser.add_argument(
        "--visits",
        nargs="+",
        default=["FirstVisit", "Later7thVisit", "Later20thVisit"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument(
        "--max-radius-normalized",
        type=float,
        default=None,
        help="Drop frames farther than this many Deora flower radii from the flower center.",
    )
    parser.add_argument(
        "--scale-mode",
        default="corolla_radius",
        choices=("corolla_radius", "flower_radius", "hole_radius", "raw"),
    )
    parser.add_argument(
        "--data-corolla-radius-normalized",
        type=float,
        default=None,
        help="Optional corolla radius in Deora normalized units. Defaults to 1 for --scale-mode=corolla_radius.",
    )
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--head-standoff", type=float, default=0.)
    parser.add_argument("--expected-information-gain-samples", type=int, default=0)
    parser.add_argument(
        "--record-policy-log-probs",
        dest="record_policy_log_probs",
        default=False,
        action="store_true",
        help=(
            "Add per-step log probabilities of each replayed action "
            "under the checkpoint's exploration and exploitation policies."
        ),
    )
    parser.add_argument(
        "--no-record-policy-log-probs",
        dest="record_policy_log_probs",
        action="store_false",
        help="Skip policy log-prob scoring, useful for model-free smoke tests.",
    )
    parser.add_argument(
        "--random-baselines-per-trajectory",
        type=int,
        default=20,
        help=(
            "Generate this many matched random-walk baselines per biological "
            "trajectory. Each baseline preserves start point, point count, "
            "frame indices, condition metadata, and empirical step lengths."
        ),
    )
    parser.add_argument(
        "--random-baseline-seed",
        type=int,
        default=20290503,
        help="Seed for matched random baseline generation.",
    )
    parser.add_argument(
        "--random-baseline-bound",
        default="observed",
        choices=("observed", "flower", "none"),
        help=(
            "Radial bound for matched random walks in Deora normalized units: "
            "'observed' uses each biological trajectory's max radius, 'flower' "
            "uses radius 1, and 'none' leaves the random walk unbounded."
        ),
    )
    parser.add_argument(
        "--record-random-baseline-steps",
        action="store_true",
        help="Also write per-step rows for matched random baselines.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--summary-csv", default="NectarySearch/results/deora_replay_summary_2030.csv")
    parser.add_argument("--steps-csv", default=None) #"NectarySearch/results/deora_replay_steps_2030.csv")
    parser.add_argument(
        "--comparison-csv",
        default="NectarySearch/results/deora_matched_random_comparison_2030.csv",
        help="One-row-per-biological-trajectory comparison against matched random baselines.",
    )
    args = parser.parse_args()

    if args.record_policy_log_probs and not args.checkpoint:
        parser.error("--record-policy-log-probs requires --checkpoint.")
    if args.random_baselines_per_trajectory < 0:
        parser.error("--random-baselines-per-trajectory must be non-negative.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = _load_config(args.config, args.checkpoint)
    model = _load_model(args.checkpoint, config, device) if args.checkpoint else None

    trajectories = load_deora_trajectories(
        args.data_root,
        visits=args.visits,
        source=args.source,
        max_radius_normalized=args.max_radius_normalized,
        min_points=args.min_points,
    )
    if args.limit is not None:
        trajectories = trajectories[: int(args.limit)]

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    random_seed_stream = np.random.default_rng(int(args.random_baseline_seed))
    for trajectory in trajectories:
        env = build_allocentric_env(
            config.environment,
            c_value=trajectory.metadata.c_value,
            max_steps=trajectory.num_points + 5,
        )
        summary, steps = replay_deora_trajectory(
            trajectory,
            env,
            model=model,
            device=device,
            scale_mode=args.scale_mode,
            data_corolla_radius_normalized=args.data_corolla_radius_normalized,
            head_standoff=args.head_standoff,
            depth=args.depth,
            expected_information_gain_samples=args.expected_information_gain_samples,
            record_policy_log_probs=args.record_policy_log_probs,
        )
        summary.update(
            {
                "trajectory_type": "biological",
                "matched_name": trajectory.metadata.name,
                "random_index": "",
                "baseline_seed": "",
            }
        )
        summary_rows.append(summary)
        if args.steps_csv:
            for step in steps:
                step.update(
                    {
                        "trajectory_type": "biological",
                        "matched_name": trajectory.metadata.name,
                        "random_index": "",
                        "baseline_seed": "",
                    }
                )
            step_rows.extend(steps)

        random_summaries: list[dict[str, Any]] = []
        random_bound = _random_bound_for_trajectory(
            trajectory,
            args.random_baseline_bound,
        )
        for random_index in range(int(args.random_baselines_per_trajectory)):
            baseline_seed = int(random_seed_stream.integers(0, np.iinfo(np.uint32).max))
            random_trajectory = generate_matched_random_trajectory(
                trajectory,
                rng=baseline_seed,
                random_index=random_index,
                max_radius_normalized=random_bound,
            )
            random_env = build_allocentric_env(
                config.environment,
                c_value=trajectory.metadata.c_value,
                max_steps=trajectory.num_points + 5,
            )
            random_summary, random_steps = replay_deora_trajectory(
                random_trajectory,
                random_env,
                model=model,
                device=device,
                scale_mode=args.scale_mode,
                data_corolla_radius_normalized=args.data_corolla_radius_normalized,
                head_standoff=args.head_standoff,
                depth=args.depth,
                expected_information_gain_samples=args.expected_information_gain_samples,
                record_policy_log_probs=args.record_policy_log_probs,
            )
            random_summary.update(
                {
                    "trajectory_type": "matched_random",
                    "matched_name": trajectory.metadata.name,
                    "random_index": int(random_index),
                    "baseline_seed": baseline_seed,
                }
            )
            random_summaries.append(random_summary)
            summary_rows.append(random_summary)

            if args.steps_csv and args.record_random_baseline_steps:
                for step in random_steps:
                    step.update(
                        {
                            "trajectory_type": "matched_random",
                            "matched_name": trajectory.metadata.name,
                            "random_index": int(random_index),
                            "baseline_seed": baseline_seed,
                        }
                    )
                step_rows.extend(random_steps)

        if random_summaries and args.comparison_csv:
            comparison_rows.append(
                _comparison_row(
                    summary,
                    random_summaries,
                    random_baseline_bound=args.random_baseline_bound,
                )
            )

    _write_rows(args.summary_csv, summary_rows)
    if args.steps_csv:
        _write_rows(args.steps_csv, step_rows)
    if args.comparison_csv:
        _write_rows(args.comparison_csv, comparison_rows)

    print(
        json.dumps(
            {
                "comparison_csv": args.comparison_csv,
                "num_comparison_rows": len(comparison_rows),
                "num_summary_rows": len(summary_rows),
                "num_trajectories": len(trajectories),
                "record_policy_log_probs": bool(args.record_policy_log_probs),
                "random_baselines_per_trajectory": int(args.random_baselines_per_trajectory),
                "summary_csv": args.summary_csv,
                "steps_csv": args.steps_csv,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
