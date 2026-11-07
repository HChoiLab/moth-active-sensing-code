from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def summarize_episode_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["suite_name"],
            record["condition_name"],
            record["policy_mode"],
        )
        grouped[key].append(record)

    summary_rows = []
    for (suite_name, condition_name, policy_mode), group in grouped.items():
        summary_rows.append(
            {
                "suite_name": suite_name,
                "condition_name": condition_name,
                "policy_mode": policy_mode,
                "num_episodes": len(group),
                "success_rate": _safe_mean(record["success"] for record in group),
                "mean_return": _safe_mean(record["episode_return"] for record in group),
                "mean_episode_length": _safe_mean(record["episode_length"] for record in group),
                "mean_time_to_success": _safe_mean(
                    record["time_to_success"]
                    for record in group
                    if record["time_to_success"] is not None
                ),
                "mean_explore_fraction": _safe_mean(record["explore_fraction"] for record in group),
                "mean_num_mode_switches": _safe_mean(
                    record["num_mode_switches"] for record in group
                ),
                "mean_contact_fraction": _safe_mean(record["contact_fraction"] for record in group),
                "mean_final_distance_to_hole": _safe_mean(
                    record["final_distance_to_hole"] for record in group
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (row["suite_name"], row["condition_name"], row["policy_mode"])
    )
    return summary_rows


def summarize_by_suite(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for record in records:
        key = (record["suite_name"], record["policy_mode"])
        grouped[key].append(record)

    rows = []
    for (suite_name, policy_mode), group in grouped.items():
        rows.append(
            {
                "suite_name": suite_name,
                "policy_mode": policy_mode,
                "num_episodes": len(group),
                "success_rate": _safe_mean(record["success"] for record in group),
                "mean_return": _safe_mean(record["episode_return"] for record in group),
                "mean_episode_length": _safe_mean(record["episode_length"] for record in group),
                "mean_explore_fraction": _safe_mean(record["explore_fraction"] for record in group),
                "mean_num_mode_switches": _safe_mean(
                    record["num_mode_switches"] for record in group
                ),
            }
        )
    rows.sort(key=lambda row: (row["suite_name"], row["policy_mode"]))
    return rows

