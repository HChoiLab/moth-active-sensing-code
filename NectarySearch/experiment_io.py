from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, Optional

import torch

try:
    from experiment_config import ExperimentConfig
except ImportError:
    from .experiment_config import ExperimentConfig


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_run_name(config: ExperimentConfig) -> str:
    if config.run_name:
        return config.run_name
    model_tag = config.model.model_type.replace("_", "-")
    force_tag = config.train.force_mode or "auto"
    return f"{_timestamp()}_{model_tag}_{force_tag}_seed{config.train.seed}"


def create_run_dir(config: ExperimentConfig) -> Path:
    run_name = ensure_run_name(config)
    run_dir = Path(config.output_root) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    config.run_name = run_name
    return run_dir


def save_config(run_dir: str | Path, config: ExperimentConfig) -> Path:
    path = Path(run_dir) / "config.json"
    config.to_json(path)
    return path


def save_git_metadata(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "git_metadata.json"
    metadata: Dict[str, Any] = {"commit": None, "status": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
        metadata["status"] = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
        )
    except Exception as exc:  # pragma: no cover - best-effort metadata
        metadata["error"] = str(exc)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata_path


def save_experiment_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    config: ExperimentConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "experiment_config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "saved_at": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def write_rows_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("")
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
