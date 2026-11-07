from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from benchmark_suites import build_benchmark_suites
    from eval_metrics import summarize_by_suite, summarize_episode_records
    from eval_rollouts import run_suite
    from experiment_config import ExperimentConfig
    from experiment_io import write_json, write_rows_csv
    from experiment_registry import build_allocentric_env, build_model
except ImportError:
    from .benchmark_suites import build_benchmark_suites
    from .eval_metrics import summarize_by_suite, summarize_episode_records
    from .eval_rollouts import run_suite
    from .experiment_config import ExperimentConfig
    from .experiment_io import write_json, write_rows_csv
    from .experiment_registry import build_allocentric_env, build_model


def load_experiment_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    if "experiment_config" not in checkpoint:
        raise ValueError(
            "Checkpoint does not include an embedded experiment_config. "
            "This first-pass pipeline expects checkpoints saved by run_experiment.py."
        )
    config = ExperimentConfig.from_dict(checkpoint["experiment_config"])
    env = build_allocentric_env(config.environment, config.train.c_vals[0])
    model = build_model(config, env)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)
    return model, config, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    suite_names: Optional[List[str]] = None,
    policy_modes: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, List[Dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, checkpoint = load_experiment_checkpoint(checkpoint_path, device=device)
    if str(config.model.model_type).strip().lower() != "uncertainty_gated_allocentric":
        raise NotImplementedError(
            "Evaluation is currently implemented for uncertainty_gated_allocentric only."
        )

    if suite_names is not None:
        config.evaluation.suite_names = list(suite_names)
    if policy_modes is not None:
        config.evaluation.policy_modes = list(policy_modes)

    suites = build_benchmark_suites(config)
    all_episode_records = []
    all_step_records = []

    for policy_mode in config.evaluation.policy_modes:
        for suite_name, suite_conditions in suites.items():
            episode_records, step_records = run_suite(
                model,
                config,
                suite_conditions,
                policy_mode=policy_mode,
                device=device,
            )
            all_episode_records.extend(episode_records)
            all_step_records.extend(step_records)

    condition_summary = summarize_episode_records(all_episode_records)
    suite_summary = summarize_by_suite(all_episode_records)
    results = {
        "episode_records": all_episode_records,
        "step_records": all_step_records,
        "condition_summary": condition_summary,
        "suite_summary": suite_summary,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_rows_csv(output_dir / "episode_metrics.csv", all_episode_records)
        write_rows_csv(output_dir / "step_traces.csv", all_step_records)
        write_rows_csv(output_dir / "condition_summary.csv", condition_summary)
        write_rows_csv(output_dir / "suite_summary.csv", suite_summary)
        write_json(
            output_dir / "evaluation_manifest.json",
            {
                "checkpoint_path": str(checkpoint_path),
                "policy_modes": config.evaluation.policy_modes,
                "suite_names": list(suites.keys()),
                "saved_checkpoint_model_class": checkpoint.get("model_class"),
            },
        )
    return results

