from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from eval_models import evaluate_checkpoint
    from experiment_config import ExperimentConfig
    from experiment_io import (
        create_run_dir,
        save_config,
        save_experiment_checkpoint,
        save_git_metadata,
        write_json,
    )
    from experiment_registry import (
        apply_pretrained_perception_construction_config,
        build_model,
        build_training_envs,
        load_pretrained_perception_components,
    )
    from training import train_nectary_search_model
    from training_uncertainty_gated import train_uncertainty_gated_allocentric_model
except ImportError:
    from .eval_models import evaluate_checkpoint
    from .experiment_config import ExperimentConfig
    from .experiment_io import (
        create_run_dir,
        save_config,
        save_experiment_checkpoint,
        save_git_metadata,
        write_json,
    )
    from .experiment_registry import (
        apply_pretrained_perception_construction_config,
        build_model,
        build_training_envs,
        load_pretrained_perception_components,
    )
    from .training import train_nectary_search_model
    from .training_uncertainty_gated import train_uncertainty_gated_allocentric_model


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_optional_path(path_value: str | None, config_path: Path | None) -> Path | None:
    if not path_value:
        return None

    raw_path = Path(path_value)
    candidates = []
    if config_path is not None and not raw_path.is_absolute():
        candidates.append((config_path.parent / raw_path).resolve())
    candidates.append(raw_path.resolve() if raw_path.is_absolute() else raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


def _path_for_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_checkpoint_artifact_path(path_value: str | None, checkpoint_path: Path) -> Path | None:
    if not path_value:
        return None

    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return raw_path

    run_dir = checkpoint_path.parent.parent
    candidates = [
        (Path.cwd() / raw_path).resolve(),
        (run_dir / raw_path).resolve(),
        (run_dir / raw_path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_last_completed_epoch(metrics_csv_path: Path) -> int | None:
    if not metrics_csv_path.exists():
        return None

    last_epoch = None
    with metrics_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_value = row.get("epoch")
            if epoch_value in {None, ""}:
                continue
            last_epoch = int(float(epoch_value))
    return last_epoch


def _infer_completed_epochs(checkpoint_payload: Dict[str, Any], checkpoint_path: Path) -> int:
    training_state = checkpoint_payload.get("training_state")
    if isinstance(training_state, dict) and training_state.get("completed_epochs") is not None:
        return int(training_state["completed_epochs"])

    metrics_csv_path = _resolve_checkpoint_artifact_path(
        checkpoint_payload.get("train_metrics_csv"),
        checkpoint_path,
    )
    if metrics_csv_path is not None:
        inferred_epoch = _read_last_completed_epoch(metrics_csv_path)
        if inferred_epoch is not None:
            return inferred_epoch

    checkpoint_config = checkpoint_payload.get("experiment_config")
    if isinstance(checkpoint_config, dict):
        train_cfg = checkpoint_config.get("train")
        if isinstance(train_cfg, dict) and train_cfg.get("num_epochs") is not None:
            return int(train_cfg["num_epochs"])

    return 0


def _prepare_resume_state(
    checkpoint_payload: Dict[str, Any],
    checkpoint_path: Path,
) -> tuple[Dict[str, Any], bool]:
    training_state = checkpoint_payload.get("training_state")
    if training_state is not None and not isinstance(training_state, dict):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has an invalid training_state payload."
        )

    resume_state = dict(training_state or {})
    resume_state.setdefault(
        "completed_epochs",
        _infer_completed_epochs(checkpoint_payload, checkpoint_path),
    )
    has_optimizer_state = (
        (resume_state.get("optimizer_state_dict") is not None)
        or (resume_state.get("optimizer_state_dicts") is not None)
    )
    return resume_state, has_optimizer_state


def _train(
    config: ExperimentConfig,
    device: torch.device,
    config_path: Path | None = None,
    resume_checkpoint_override: str | None = None,
) -> Path:
    resume_checkpoint = _resolve_optional_path(
        resume_checkpoint_override or config.train.resume_checkpoint,
        config_path,
    )
    pretrained_perception_checkpoint = None
    pretrained_construction_info = None
    if resume_checkpoint is not None:
        run_dir = resume_checkpoint.parent.parent
        config.train.resume_checkpoint = str(resume_checkpoint)
        config.run_name = run_dir.name
        config.output_root = _path_for_display(run_dir.parent)
    else:
        run_dir = create_run_dir(config)
        pretrained_perception_checkpoint = _resolve_optional_path(
            config.model.pretrained_perception_checkpoint,
            config_path,
        )
        if pretrained_perception_checkpoint is not None:
            pretrained_construction_info = apply_pretrained_perception_construction_config(
                config,
                pretrained_perception_checkpoint,
                map_location="cpu",
            )
    save_config(run_dir, config)
    save_git_metadata(run_dir)

    envs = build_training_envs(config)
    model = build_model(config, envs[0])
    pretrained_init_info = None
    resume_state = None
    resumed_from_epoch = 0
    resumed_with_optimizer_state = False

    if resume_checkpoint is not None:
        checkpoint_payload = torch.load(resume_checkpoint, map_location="cpu")
        if "model_state_dict" not in checkpoint_payload:
            raise ValueError(
                f"Checkpoint '{resume_checkpoint}' does not contain a model_state_dict."
            )
        model.load_state_dict(_strip_module_prefix(checkpoint_payload["model_state_dict"]))
        resume_state, resumed_with_optimizer_state = _prepare_resume_state(
            checkpoint_payload,
            resume_checkpoint,
        )
        resumed_from_epoch = int(resume_state.get("completed_epochs", 0))
        if resumed_from_epoch >= int(config.train.num_epochs):
            raise ValueError(
                f"Checkpoint '{resume_checkpoint}' is already at epoch {resumed_from_epoch}. "
                f"Set train.num_epochs higher than {resumed_from_epoch} to continue training."
            )
        print(
            f"Resuming training from {_path_for_display(resume_checkpoint)} at epoch "
            f"{resumed_from_epoch}."
        )
        if not resumed_with_optimizer_state:
            print(
                "Resume checkpoint does not include optimizer state; continuing with "
                "freshly initialized optimizer(s)."
            )
        if config.model.pretrained_perception_checkpoint:
            print(
                "Ignoring pretrained_perception_checkpoint because a resume checkpoint "
                "was provided."
            )
    else:
        if pretrained_perception_checkpoint is not None:
            pretrained_init_info = load_pretrained_perception_components(
                model,
                pretrained_perception_checkpoint,
                map_location=device,
            )
            if pretrained_construction_info is not None:
                pretrained_init_info["construction_overrides"] = pretrained_construction_info
            write_json(run_dir / "pretrained_perception_init.json", pretrained_init_info)

    train_metrics_path = run_dir / "train_metrics.csv"
    model_type = str(config.model.model_type).strip().lower()
    if model_type == "uncertainty_gated_allocentric":
        trained_model, training_state = train_uncertainty_gated_allocentric_model(
            envs=envs,
            model=model,
            num_epochs=config.train.num_epochs,
            batch_size=config.train.batch_size,
            seq_len=config.train.seq_len,
            perception_learning_rate=config.model.perception_lr,
            exploration_learning_rate=config.model.exploration_lr,
            exploitation_learning_rate=config.model.exploitation_lr,
            gate_learning_rate=config.model.gate_lr,
            head_lr_scheduler=config.model.head_lr_scheduler,
            head_lr_scheduler_start_epoch=config.model.head_lr_scheduler_start_epoch,
            print_every=config.train.print_every,
            min_depth=config.environment.min_depth,
            gamma=config.model.gamma,
            gae_lambda=config.model.gae_lambda,
            exploration_warmup_epochs=config.train.exploration_warmup_epochs,
            deterministic_exploitation=config.train.deterministic_exploitation,
            deterministic_exploration=config.train.deterministic_exploration,
            detach_hidden_state=config.train.detach_hidden_state,
            grad_clip_norm=config.train.grad_clip_norm,
            metrics_csv_path=str(train_metrics_path),
            append_metrics_csv=train_metrics_path.exists() and resumed_from_epoch > 0,
            exploration_objective=config.model.exploration_objective,
            visual_likelihood=config.model.visual_likelihood,
            learn_decoder_variance=config.model.learn_decoder_variance,
            posterior_fusion=config.model.posterior_fusion,
            moment_matching=config.model.moment_matching,
            uncertainty_metric=config.model.uncertainty_metric,
            rnn_type=config.model.rnn_type,
            modality_rnn_mode=config.model.modality_rnn_mode,
            action_selection_strategy=config.model.action_selection_strategy,
            epsilon_greedy_probability=config.model.epsilon_greedy_probability,
            fixed_switch_timestep=config.model.fixed_switch_timestep,
            random_switch_probability=config.model.random_switch_probability,
            information_gain_reward_coef=config.model.information_gain_reward_coef,
            parallel=False,
            device=device,
            force_mode=config.train.force_mode,
            gate_hysteresis=config.train.gate_hysteresis,
            gate_uncertainty_ema_alpha=config.train.gate_uncertainty_ema_alpha,
            resume_state=resume_state,
            return_training_state=True,
        )
    else:
        trained_model, training_state = train_nectary_search_model(
            envs=envs,
            model=model,
            num_epochs=config.train.num_epochs,
            batch_size=config.train.batch_size,
            seq_len=config.train.seq_len,
            learning_rate=config.model.learning_rate,
            print_every=config.train.print_every,
            allocentric=True,
            min_depth=config.environment.min_depth,
            parallel=False,
            device=device,
            wrap_ddp=False,
            allocentric_trajectory_mode="smooth_modes",
            randomize_visual_pattern_each_epoch=config.train.randomize_visual_pattern_each_epoch,
            visual_pattern_choices=config.train.visual_pattern_choices,
            visual_likelihood=config.model.visual_likelihood,
            learn_decoder_variance=config.model.learn_decoder_variance,
            posterior_fusion=config.model.posterior_fusion,
            moment_matching=config.model.moment_matching,
            rnn_type=config.model.rnn_type,
            modality_rnn_mode=config.model.modality_rnn_mode,
            metrics_csv_path=str(train_metrics_path),
            append_metrics_csv=train_metrics_path.exists() and resumed_from_epoch > 0,
            resume_state=resume_state,
            return_training_state=True,
        )

    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    save_experiment_checkpoint(
        checkpoint_path,
        trained_model,
        config,
        extra={
            "train_metrics_csv": str(train_metrics_path),
            "pretrained_perception_init": pretrained_init_info,
            "training_state": training_state,
        },
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "run_name": config.run_name,
            "model_type": config.model.model_type,
            "force_mode": config.train.force_mode,
            "pretrained_perception_checkpoint": (
                None
                if pretrained_perception_checkpoint is None
                else str(pretrained_perception_checkpoint)
            ),
            "resume_checkpoint": (
                None if resume_checkpoint is None else str(resume_checkpoint)
            ),
            "resumed_from_epoch": resumed_from_epoch,
            "resumed_with_optimizer_state": resumed_with_optimizer_state,
            "checkpoint_path": str(checkpoint_path),
            "train_metrics_csv": str(train_metrics_path),
        },
    )
    return checkpoint_path


def _evaluate(checkpoint_path: Path, device: torch.device, suite_names=None, policy_modes=None):
    output_dir = checkpoint_path.parent.parent / "eval"
    results = evaluate_checkpoint(
        checkpoint_path,
        output_dir=output_dir,
        suite_names=suite_names,
        policy_modes=policy_modes,
        device=device,
    )
    print(f"Saved evaluation outputs to {output_dir}")
    for row in results["suite_summary"]:
        print(
            f"{row['suite_name']} [{row['policy_mode']}]: "
            f"success={row['success_rate']:.3f}, "
            f"return={row['mean_return']:.3f}, "
            f"len={row['mean_episode_length']:.2f}, "
            f"explore={row['mean_explore_fraction']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment runner for NectarySearch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from an experiment config.")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to resume. train.num_epochs is treated as the total target epoch.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved experiment checkpoint.")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--device", type=str, default=None)
    eval_parser.add_argument("--suites", nargs="*", default=None)
    eval_parser.add_argument("--policy-modes", nargs="*", default=None)

    train_eval_parser = subparsers.add_parser(
        "train-eval",
        help="Train from config and immediately run evaluation.",
    )
    train_eval_parser.add_argument("--config", type=str, required=True)
    train_eval_parser.add_argument("--device", type=str, default=None)
    train_eval_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to resume. train.num_epochs is treated as the total target epoch.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(getattr(args, "device", None))

    if args.command == "train":
        config_path = Path(args.config)
        config = ExperimentConfig.from_json(config_path)
        if args.seed is not None:
            config.train.seed = args.seed
            config.run_name = str(args.seed)
        checkpoint_path = _train(
            config,
            device,
            config_path=config_path,
            resume_checkpoint_override=args.resume_checkpoint,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
        return

    if args.command == "eval":
        _evaluate(
            Path(args.checkpoint),
            device=device,
            suite_names=args.suites,
            policy_modes=args.policy_modes,
        )
        return

    if args.command == "train-eval":
        config_path = Path(args.config)
        config = ExperimentConfig.from_json(config_path)
        checkpoint_path = _train(
            config,
            device,
            config_path=config_path,
            resume_checkpoint_override=args.resume_checkpoint,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
        _evaluate(checkpoint_path, device=device)
        return


if __name__ == "__main__":
    main()
