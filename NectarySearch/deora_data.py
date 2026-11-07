from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from nectary_allocentric_env import AllocentricNectarSearchEnv
    from uncertainty_reduction import compute_realized_information_gain, estimate_expected_information_gain
except ImportError:
    from .nectary_allocentric_env import AllocentricNectarSearchEnv
    from .uncertainty_reduction import compute_realized_information_gain, estimate_expected_information_gain


DEORA_DATA_ROOT = Path("./data/deora_et_al/ProboscisTracking_dataFolders")
VISIT_NAMES = ("FirstVisit", "Later7thVisit", "Later20thVisit")
TRACK_NAME_RE = re.compile(r"^(?P<key>c-(?P<c_value>\d+)_m(?P<moth_id>\d+)(?P<cropped>_cropped)?)")


@dataclass(frozen=True)
class CircleParameters:
    name: str
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class DeoraTrackMetadata:
    name: str
    circle_key: str
    c_value: float
    moth_id: int
    cropped: bool
    visit: str
    source: str
    path: Path


@dataclass(frozen=True)
class DeoraTrajectory:
    metadata: DeoraTrackMetadata
    circle: CircleParameters
    xy_pixels: np.ndarray
    xy_flower_normalized: np.ndarray
    frame_indices: np.ndarray

    @property
    def radius_normalized(self) -> np.ndarray:
        return np.linalg.norm(self.xy_flower_normalized, axis=-1)

    @property
    def num_points(self) -> int:
        return int(self.xy_flower_normalized.shape[0])


def load_circle_parameters(
    data_root: str | Path = DEORA_DATA_ROOT,
) -> dict[str, CircleParameters]:
    """Load the per-video center/radius table used by Deora et al. tracks."""
    csv_path = Path(data_root) / "CircleParameters" / "circle_parameters.csv"
    parameters: dict[str, CircleParameters] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = str(row["name"]).strip()
            parameters[name] = CircleParameters(
                name=name,
                center_x=float(row["circ_x"]),
                center_y=float(row["circ_y"]),
                radius=float(row["circ_radii"]),
            )
    return parameters


def _strip_known_suffixes(stem: str) -> str:
    for suffix in ("_RadiusAndAngle", "_FirstVisit", "_Later7thVisit", "_Later20thVisit"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_track_metadata(
    path: str | Path,
    *,
    source: Optional[str] = None,
    circle_parameters: Optional[Mapping[str, CircleParameters]] = None,
) -> DeoraTrackMetadata:
    path = Path(path)
    stem = _strip_known_suffixes(path.stem)
    match = TRACK_NAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Could not parse Deora track name from '{path.name}'.")

    key = match.group("key")
    circle_key = key
    if circle_parameters is not None and circle_key not in circle_parameters:
        cropped_key = f"{key}_cropped"
        if cropped_key in circle_parameters:
            circle_key = cropped_key

    visit = next((part for part in path.parts if part in VISIT_NAMES), "")
    if not visit:
        visit = next((visit_name for visit_name in VISIT_NAMES if visit_name in path.stem), "")
    if not visit:
        raise ValueError(f"Could not infer visit name from '{path}'.")

    if source is None:
        path_parts = set(path.parts)
        if "RadiusAndAngle" in path_parts:
            source = "radius_and_angle"
        elif "FinalCompiledTracks" in path_parts:
            source = "final_compiled"
        elif "FilteredTracks" in path_parts:
            source = "filtered"
        else:
            source = "unknown"

    raw_c_value = match.group("c_value")
    c_value = -100.0 if raw_c_value == "10" else -float(raw_c_value)

    return DeoraTrackMetadata(
        name=key,
        circle_key=circle_key,
        c_value=c_value,
        moth_id=int(match.group("moth_id")),
        cropped=bool(match.group("cropped")) or circle_key.endswith("_cropped"),
        visit=visit,
        source=str(source),
        path=path,
    )


def iter_deora_track_paths(
    data_root: str | Path = DEORA_DATA_ROOT,
    *,
    visits: Optional[Iterable[str]] = None,
    source: str = "filtered",
    include_all_video_names: bool = False,
) -> list[Path]:
    """Return track CSV paths for one of: filtered, final_compiled, radius_and_angle."""
    data_root = Path(data_root)
    visits = tuple(visits or VISIT_NAMES)
    source = str(source).strip().lower()
    if source == "filtered":
        root = data_root / "v3" / "FilteredTracks"
    elif source == "final_compiled":
        root = data_root / "v3" / "FilteredTracks" / "FinalCompiledTracks"
    elif source == "radius_and_angle":
        root = data_root / "v3" / "RadiusAndAngle"
    else:
        raise ValueError("source must be one of: filtered, final_compiled, radius_and_angle.")

    paths: list[Path] = []
    for visit in visits:
        visit_root = root / visit
        if not visit_root.exists():
            continue
        for path in sorted(visit_root.glob("*.csv")):
            if path.name == "AllVideoNames.csv" and not include_all_video_names:
                continue
            paths.append(path)
    return paths


def _finite_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _row_frame_index(row: Mapping[str, str], fallback_index: int) -> int:
    for key in ("frame", "Frame", "index", "Unnamed: 0", ""):
        if key in row:
            value = _finite_float(row[key])
            if np.isfinite(value):
                return int(value)
    return int(fallback_index)


def load_deora_trajectory(
    path: str | Path,
    circle_parameters: Optional[Mapping[str, CircleParameters]] = None,
    *,
    data_root: str | Path = DEORA_DATA_ROOT,
    max_radius_normalized: Optional[float] = None,
    min_points: int = 2,
    y_axis: str = "image_down",
) -> DeoraTrajectory:
    """
    Load one Deora track and express xy in flower-centered, flower-radius units.

    The Deora CSVs store either raw image xy (`x`, `y`) or centered image xy
    (`x_centered`, `y_centered`). The returned `xy_flower_normalized` uses
    right-handed flower coordinates: +x right, +y up, and one unit equals the
    flower radius from `circle_parameters.csv`. Values with radius > 1 are
    valid off-flower tip locations unless filtered with `max_radius_normalized`.
    """
    if circle_parameters is None:
        circle_parameters = load_circle_parameters(data_root)
    metadata = parse_track_metadata(
        path,
        circle_parameters=circle_parameters,
    )
    if metadata.circle_key not in circle_parameters:
        raise KeyError(
            f"No circle parameters for '{metadata.circle_key}' while loading '{path}'."
        )
    circle = circle_parameters[metadata.circle_key]
    if circle.radius <= 0.0:
        raise ValueError(f"Circle radius must be positive for '{metadata.circle_key}'.")

    xy_pixels: list[tuple[float, float]] = []
    xy_normalized: list[tuple[float, float]] = []
    frame_indices: list[int] = []
    y_axis = str(y_axis).strip().lower()
    if y_axis not in {"image_down", "y_up"}:
        raise ValueError("y_axis must be either 'image_down' or 'y_up'.")

    with Path(path).open(newline="") as f:
        for fallback_index, row in enumerate(csv.DictReader(f)):
            if "x_centered" in row or "y_centered" in row:
                x_centered = _finite_float(row.get("x_centered"))
                y_centered = _finite_float(row.get("y_centered"))
                x_pixel = circle.center_x + x_centered if np.isfinite(x_centered) else float("nan")
                y_pixel = circle.center_y + y_centered if np.isfinite(y_centered) else float("nan")
            else:
                x_pixel = _finite_float(row.get("x"))
                y_pixel = _finite_float(row.get("y"))
                x_centered = x_pixel - circle.center_x
                y_centered = y_pixel - circle.center_y

            if not np.isfinite(x_centered) or not np.isfinite(y_centered):
                continue

            x_norm = x_centered / circle.radius
            y_norm = y_centered / circle.radius
            if y_axis == "image_down":
                y_norm = -y_norm

            radius_norm = float(np.hypot(x_norm, y_norm))
            if max_radius_normalized is not None and radius_norm > float(max_radius_normalized):
                continue

            xy_pixels.append((x_pixel, y_pixel))
            xy_normalized.append((x_norm, y_norm))
            frame_indices.append(_row_frame_index(row, fallback_index))

    if len(xy_normalized) < int(min_points):
        raise ValueError(
            f"Track '{path}' has only {len(xy_normalized)} usable points after filtering."
        )

    return DeoraTrajectory(
        metadata=metadata,
        circle=circle,
        xy_pixels=np.asarray(xy_pixels, dtype=np.float32),
        xy_flower_normalized=np.asarray(xy_normalized, dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
    )


def load_deora_trajectories(
    data_root: str | Path = DEORA_DATA_ROOT,
    *,
    visits: Optional[Iterable[str]] = None,
    source: str = "filtered",
    max_radius_normalized: Optional[float] = None,
    min_points: int = 2,
) -> list[DeoraTrajectory]:
    circle_parameters = load_circle_parameters(data_root)
    trajectories: list[DeoraTrajectory] = []
    for path in iter_deora_track_paths(data_root, visits=visits, source=source):
        try:
            trajectories.append(
                load_deora_trajectory(
                    path,
                    circle_parameters,
                    data_root=data_root,
                    max_radius_normalized=max_radius_normalized,
                    min_points=min_points,
                )
            )
        except ValueError:
            continue
    return trajectories


def deora_xy_to_env_flower_xy(
    trajectory: DeoraTrajectory,
    env: AllocentricNectarSearchEnv,
    *,
    scale_mode: str = "corolla_radius",
    data_corolla_radius_normalized: Optional[float] = None,
) -> np.ndarray:
    """
    Convert normalized Deora flower xy to the environment's flower coordinates.

    The default treats the circle radius in Deora's table as the flower radius,
    so one Deora normalized unit maps to `env.flower.R`.
    """
    scale_mode = str(scale_mode).strip().lower()
    if scale_mode in {"corolla_radius", "flower_radius"}:
        if data_corolla_radius_normalized is None:
            data_corolla_radius_normalized = 1.0
        if data_corolla_radius_normalized <= 0:
            raise ValueError(
                "data_corolla_radius_normalized must be positive when scale_mode='corolla_radius'."
            )
        scale = float(env.flower.R) / float(data_corolla_radius_normalized)
    elif scale_mode == "hole_radius":
        scale = float(env.flower.r0)
    elif scale_mode == "raw":
        scale = 1.0
    else:
        raise ValueError(
            "scale_mode must be one of: corolla_radius, flower_radius, hole_radius, raw."
        )
    return trajectory.xy_flower_normalized.astype(np.float32) * np.float32(scale)


def generate_matched_random_trajectory(
    trajectory: DeoraTrajectory,
    *,
    rng: Optional[np.random.Generator | int] = None,
    random_index: Optional[int] = None,
    max_radius_normalized: Optional[float] = None,
    max_attempts: int = 128,
) -> DeoraTrajectory:
    """
    Build a matched random-walk trajectory for a biological Deora trajectory.

    The synthetic trajectory preserves the first tip location, number of points,
    frame indices, metadata/condition, and empirical step-length sequence. Only
    the heading of each step is randomized. When ``max_radius_normalized`` is
    supplied, headings are resampled to keep the path inside that radial bound;
    a final radial clip is used as a deterministic fallback for very large steps.
    """
    if trajectory.num_points < 1:
        raise ValueError("trajectory must contain at least one point.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    max_radius = None
    if max_radius_normalized is not None:
        max_radius = float(max_radius_normalized)
        if not np.isfinite(max_radius) or max_radius <= 0.0:
            raise ValueError("max_radius_normalized must be positive and finite.")

    xy = np.empty_like(trajectory.xy_flower_normalized, dtype=np.float32)
    xy[0] = trajectory.xy_flower_normalized[0]

    if max_radius is not None:
        start_radius = float(np.linalg.norm(xy[0]))
        max_radius = max(max_radius, start_radius, 1e-6)

    step_lengths = np.linalg.norm(
        np.diff(trajectory.xy_flower_normalized.astype(np.float64), axis=0),
        axis=-1,
    )
    for point_index, step_length in enumerate(step_lengths, start=1):
        if not np.isfinite(step_length) or step_length <= 0.0:
            xy[point_index] = xy[point_index - 1]
            continue

        candidate = None
        previous = xy[point_index - 1].astype(np.float64)
        for _ in range(int(max_attempts)):
            angle = float(rng.uniform(-np.pi, np.pi))
            delta = step_length * np.array([np.cos(angle), np.sin(angle)])
            proposal = previous + delta
            if max_radius is None or float(np.linalg.norm(proposal)) <= max_radius:
                candidate = proposal
                break

        if candidate is None:
            angle = float(rng.uniform(-np.pi, np.pi))
            delta = step_length * np.array([np.cos(angle), np.sin(angle)])
            candidate = previous + delta
            radius = float(np.linalg.norm(candidate))
            if max_radius is not None and radius > max_radius:
                candidate = candidate * (max_radius / max(radius, 1e-12))

        xy[point_index] = candidate.astype(np.float32)

    xy_pixels = np.empty_like(trajectory.xy_pixels, dtype=np.float32)
    xy_pixels[:, 0] = trajectory.circle.center_x + xy[:, 0] * trajectory.circle.radius
    xy_pixels[:, 1] = trajectory.circle.center_y - xy[:, 1] * trajectory.circle.radius

    suffix = "random" if random_index is None else f"random_{int(random_index)}"
    metadata = replace(
        trajectory.metadata,
        name=f"{trajectory.metadata.name}_{suffix}",
    )
    return DeoraTrajectory(
        metadata=metadata,
        circle=trajectory.circle,
        xy_pixels=xy_pixels,
        xy_flower_normalized=xy,
        frame_indices=trajectory.frame_indices.copy(),
    )


def _outward_directions(local_xy: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(local_xy, axis=-1, keepdims=True)
    directions = np.divide(
        local_xy,
        radii,
        out=np.zeros_like(local_xy, dtype=np.float32),
        where=radii > 1e-6,
    )
    if len(directions) > 1:
        motion = np.diff(local_xy, axis=0, prepend=local_xy[:1])
        motion_norm = np.linalg.norm(motion, axis=-1, keepdims=True)
        motion_dir = np.divide(
            motion,
            motion_norm,
            out=np.zeros_like(motion, dtype=np.float32),
            where=motion_norm > 1e-6,
        )
        zero_rows = np.linalg.norm(directions, axis=-1) <= 1e-6
        directions[zero_rows] = -motion_dir[zero_rows]
    still_zero = np.linalg.norm(directions, axis=-1) <= 1e-6
    directions[still_zero] = np.array([1.0, 0.0], dtype=np.float32)
    return directions.astype(np.float32)


def configure_env_for_deora_replay(
    env: AllocentricNectarSearchEnv,
    trajectory: DeoraTrajectory,
    *,
    flower_center_xy: Optional[Sequence[float]] = None,
    scale_mode: str = "corolla_radius",
    data_corolla_radius_normalized: Optional[float] = None,
    head_standoff: Optional[float] = None,
    depth: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Place the env's flower and initialize pose at the first trajectory point."""
    del trajectory  # The env is configured by the first generated replay action below.
    if flower_center_xy is None:
        flower_center_xy = (0.5 * env.canvas.width, 0.5 * env.canvas.height)
    env.flower_center_xy = np.asarray(flower_center_xy, dtype=np.float32)
    env.canvas_image = env.get_canvas_image()
    env._step_count = 0
    env.tip_depth = float(depth)
    env.prev_xy = env.flower_center_xy.copy()

    return {
        "flower_center_xy": env._normalize_canvas_xy(env.flower_center_xy),
        "scale_mode": scale_mode,
        "data_corolla_radius_normalized": (
            float("nan")
            if data_corolla_radius_normalized is None
            else float(data_corolla_radius_normalized)
        ),
        "head_standoff": (
            float("nan") if head_standoff is None else float(head_standoff)
        ),
    }


def deora_actions_for_env(
    trajectory: DeoraTrajectory,
    env: AllocentricNectarSearchEnv,
    *,
    scale_mode: str = "corolla_radius",
    data_corolla_radius_normalized: Optional[float] = None,
    head_standoff: Optional[float] = None,
    depth: float | Sequence[float] = 1.0,
) -> np.ndarray:
    """Convert a Deora trajectory to allocentric environment actions."""
    local_tip_xy = deora_xy_to_env_flower_xy(
        trajectory,
        env,
        scale_mode=scale_mode,
        data_corolla_radius_normalized=data_corolla_radius_normalized,
    )
    if head_standoff is None:
        head_standoff = 0.8 * float(env.proboscis_length)
    directions = _outward_directions(local_tip_xy)
    local_head_xy = local_tip_xy + directions * np.float32(head_standoff)

    tip_canvas_xy = env._to_canvas_xy(local_tip_xy)
    head_canvas_xy = env._to_canvas_xy(local_head_xy)

    if np.isscalar(depth):
        depths = np.full((trajectory.num_points,), float(depth), dtype=np.float32)
    else:
        depths = np.asarray(depth, dtype=np.float32)
        if depths.shape != (trajectory.num_points,):
            raise ValueError("depth sequence must have one value per trajectory point.")

    actions = np.zeros((trajectory.num_points, 5), dtype=np.float32)
    for idx, (head_xy, tip_xy, tip_depth) in enumerate(
        zip(head_canvas_xy, tip_canvas_xy, depths)
    ):
        head_norm = env._normalize_canvas_xy(env._constrain_head_to_canvas(head_xy))
        tip_norm = env._normalize_canvas_xy(tip_xy)
        actions[idx] = np.array(
            [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], float(tip_depth)],
            dtype=np.float32,
        )
    return np.clip(actions, env.action_space.low, env.action_space.high)


def _obs_dict_to_tensors(obs: Mapping[str, np.ndarray], device: torch.device):
    mech = torch.as_tensor(obs["mech"], dtype=torch.float32, device=device).reshape(1, -1)
    vis = torch.as_tensor(obs["vision"], dtype=torch.float32, device=device)
    if vis.ndim == 2:
        vis = vis.unsqueeze(0)
    vis = vis.unsqueeze(0)
    return mech, vis


def _recurrent_state_from_responses(responses: Mapping[str, torch.Tensor]):
    if "c" in responses:
        return responses["h"], responses["c"]
    return responses["h"]


def _action_from_env_pose(env: AllocentricNectarSearchEnv) -> np.ndarray:
    head_norm = env._normalize_canvas_xy(env.head_xy)
    tip_norm = env._normalize_canvas_xy(env.tip_xy)
    return np.array(
        [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], float(env.tip_depth)],
        dtype=np.float32,
    )


def _nanmean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def _policy_action_in_support(
    head: torch.nn.Module,
    action: torch.Tensor,
    prev_action: torch.Tensor,
) -> Optional[torch.Tensor]:
    actor = getattr(head, "actor", None)
    if actor is None or not hasattr(actor, "action_bounds"):
        return None

    action_low, action_high = actor.action_bounds(prev_action)
    return torch.logical_and(action > action_low, action < action_high).all(dim=-1)


def _evaluate_policy_action_log_prob(
    head: Optional[torch.nn.Module],
    belief_features: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
) -> float:
    if head is None or not hasattr(head, "evaluate_actions"):
        return float("nan")

    in_support = _policy_action_in_support(head, action, prev_action)
    if in_support is not None and not bool(in_support.reshape(()).item()):
        return float("-inf")

    log_prob, *_ = head.evaluate_actions(belief_features, action, prev_action)
    return float(log_prob.reshape(()).item())


def _moth_action_policy_log_probs(
    model: torch.nn.Module,
    responses: Mapping[str, torch.Tensor],
    prev_action: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, float]:
    result = {
        "moth_action_exploration_log_prob": float("nan"),
        "moth_action_exploitation_log_prob": float("nan"),
    }
    if not hasattr(model, "belief_features"):
        return result

    with torch.no_grad():
        post_mu = responses["post_mu"]
        post_logvar = responses["post_logvar"]
        h_t = responses["h"]
        if hasattr(model, "_planning_state"):
            planning_state = model._planning_state(  # noqa: SLF001 - matches policy feature construction.
                post_mu,
                post_logvar,
                s_t=responses.get("s"),
            )
        else:
            planning_state = responses.get("s", post_mu)
        belief_features = model.belief_features(post_mu, post_logvar, h_t, planning_state)

        if bool(getattr(model, "exploration_uses_rl", False)):
            result["moth_action_exploration_log_prob"] = _evaluate_policy_action_log_prob(
                getattr(model, "exploration_head", None),
                belief_features,
                action,
                prev_action,
            )
        result["moth_action_exploitation_log_prob"] = _evaluate_policy_action_log_prob(
            getattr(model, "exploitation_head", None),
            belief_features,
            action,
            prev_action,
        )
    return result


def replay_deora_trajectory(
    trajectory: DeoraTrajectory,
    env: AllocentricNectarSearchEnv,
    *,
    model: Optional[torch.nn.Module] = None,
    device: str | torch.device = "cpu",
    scale_mode: str = "corolla_radius",
    data_corolla_radius_normalized: Optional[float] = None,
    flower_center_xy: Optional[Sequence[float]] = None,
    head_standoff: Optional[float] = None,
    depth: float | Sequence[float] = 1.0,
    expected_information_gain_samples: int = 0,
    record_policy_log_probs: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Replay a biological trajectory through the allocentric environment.

    If `model` is supplied, each step also reports posterior entropy and
    realized information gain from the model's latent posterior. Set
    `record_policy_log_probs=True` to additionally score each biological
    action under the model's exploration and exploitation policies.
    """
    device = torch.device(device)
    configure_env_for_deora_replay(
        env,
        trajectory,
        flower_center_xy=flower_center_xy,
        scale_mode=scale_mode,
        data_corolla_radius_normalized=data_corolla_radius_normalized,
        head_standoff=head_standoff,
        depth=float(depth[0]) if not np.isscalar(depth) else float(depth),
    )
    actions = deora_actions_for_env(
        trajectory,
        env,
        scale_mode=scale_mode,
        data_corolla_radius_normalized=data_corolla_radius_normalized,
        head_standoff=head_standoff,
        depth=depth,
    )

    first_action = actions[0]
    env.head_xy = env._denormalize_canvas_xy(first_action[:2])
    env.tip_xy = env._constrain_tip_to_head(
        env._denormalize_canvas_xy(first_action[2:4]),
        env.head_xy,
    )
    env.prev_xy = env.tip_xy.copy()
    env.tip_depth = float(first_action[4])
    obs = env._get_obs()
    prev_action_np = _action_from_env_pose(env)

    model_was_training = False
    responses = None
    prev_action_t = None
    if model is not None:
        model_was_training = bool(model.training)
        model.eval()
        x_mech_t, x_vis_t = _obs_dict_to_tensors(obs, device)
        prev_action_t = torch.as_tensor(prev_action_np, dtype=torch.float32, device=device).reshape(1, -1)
        h_prev = model.initial_hidden(batch_size=1, device=device, dtype=torch.float32)
        with torch.no_grad():
            responses, _ = model.forward_t(x_mech_t, x_vis_t, prev_action_t, h_prev)

    step_records: list[dict[str, Any]] = []
    success = False
    episode_return = 0.0
    contact_count = 0
    information_gain_values: list[float] = []
    entropy_values: list[float] = []
    exploration_policy_log_probs: list[float] = []
    exploitation_policy_log_probs: list[float] = []
    min_distance_to_hole = float(np.linalg.norm(env._to_flower_xy(env.tip_xy)))
    reached_hole = bool(min_distance_to_hole <= float(env.flower.r0))

    for step_index, action in enumerate(actions[1:], start=1):
        current_responses = responses
        current_entropy = float("nan")
        expected_ig = float("nan")
        policy_log_probs: dict[str, float] = {}
        action_t = (
            torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
            if model is not None and current_responses is not None
            else None
        )
        if model is not None and current_responses is not None:
            current_entropy = float(
                torch.sum(
                    0.5
                    * (
                        1.0
                        + np.log(2.0 * np.pi)
                        + current_responses["post_logvar"]
                    ),
                    dim=-1,
                ).mean().item()
            )
            entropy_values.append(current_entropy)
            if record_policy_log_probs:
                policy_log_probs = _moth_action_policy_log_probs(
                    model,
                    current_responses,
                    prev_action_t,
                    action_t,
                )
                exploration_policy_log_probs.append(
                    policy_log_probs["moth_action_exploration_log_prob"]
                )
                exploitation_policy_log_probs.append(
                    policy_log_probs["moth_action_exploitation_log_prob"]
                )
            if expected_information_gain_samples > 0:
                with torch.no_grad():
                    expected_stats = estimate_expected_information_gain(
                        model=model,
                        post_logvar=current_responses["post_logvar"],
                        h_t=current_responses["h"],
                        c_t=current_responses.get("c"),
                        planning_state=current_responses["s"],
                        action=action_t,
                        num_samples=int(expected_information_gain_samples),
                    )
                expected_ig = float(expected_stats["information_gain"].reshape(()).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        success = success or bool(info.get("success", False))
        contact_count += int(bool(info.get("contact", 0.0) > 0.0))

        realized_ig = float("nan")
        next_entropy = float("nan")
        if model is not None and current_responses is not None:
            x_mech_next, x_vis_next = _obs_dict_to_tensors(next_obs, device)
            with torch.no_grad():
                responses, _ = model.forward_t(
                    x_mech_next,
                    x_vis_next,
                    action_t,
                    _recurrent_state_from_responses(current_responses),
                )
                realized_stats = compute_realized_information_gain(
                    post_logvar=current_responses["post_logvar"],
                    next_post_logvar=responses["post_logvar"],
                )
            realized_ig = float(realized_stats["information_gain"].reshape(()).item())
            next_entropy = float(realized_stats["next_entropy"].reshape(()).item())
            information_gain_values.append(realized_ig)

        local_tip_xy = env._to_flower_xy(env.tip_xy)
        local_radius = float(np.linalg.norm(local_tip_xy))
        min_distance_to_hole = min(min_distance_to_hole, local_radius)
        reached_hole = reached_hole or bool(local_radius <= float(env.flower.r0))
        step_record = {
            "name": trajectory.metadata.name,
            "visit": trajectory.metadata.visit,
            "source": trajectory.metadata.source,
            "c_value": trajectory.metadata.c_value,
            "moth_id": trajectory.metadata.moth_id,
            "frame": int(trajectory.frame_indices[step_index]),
            "step": int(step_index),
            "reward": float(reward),
            "success": float(bool(info.get("success", False))),
            "contact": float(info.get("contact", 0.0)),
            "fn": float(info.get("fn", 0.0)),
            "slip": float(info.get("slip", 0.0)),
            "tip_depth": float(info["tip_depth"]),
            "tip_x": float(info["tip_xy"][0]),
            "tip_y": float(info["tip_xy"][1]),
            "head_x": float(info["head_xy"][0]),
            "head_y": float(info["head_xy"][1]),
            "local_radius": local_radius,
            "deora_radius_normalized": float(trajectory.radius_normalized[step_index]),
            "posterior_entropy": current_entropy,
            "next_posterior_entropy": next_entropy,
            "realized_information_gain": realized_ig,
            "expected_information_gain": expected_ig,
        }
        if record_policy_log_probs:
            step_record.update(policy_log_probs)
        step_records.append(step_record)

        if action_t is not None:
            prev_action_t = action_t
        if terminated or truncated:
            break

    if model is not None and model_was_training:
        model.train()

    final_radius = float(np.linalg.norm(env._to_flower_xy(env.tip_xy)))
    summary = {
        "name": trajectory.metadata.name,
        "visit": trajectory.metadata.visit,
        "source": trajectory.metadata.source,
        "c_value": trajectory.metadata.c_value,
        "moth_id": trajectory.metadata.moth_id,
        "num_input_points": trajectory.num_points,
        "num_replayed_steps": len(step_records),
        "success": float(success),
        "reached_hole": float(reached_hole),
        "episode_return": float(episode_return),
        "contact_fraction": float(contact_count / max(len(step_records), 1)),
        "final_distance_to_hole": final_radius,
        "min_distance_to_hole": float(min_distance_to_hole),
        "mean_realized_information_gain": (
            float(np.nanmean(information_gain_values)) if information_gain_values else float("nan")
        ),
        "cumulative_realized_information_gain": (
            float(np.nansum(information_gain_values)) if information_gain_values else float("nan")
        ),
        "mean_posterior_entropy": (
            float(np.nanmean(entropy_values)) if entropy_values else float("nan")
        ),
    }
    if record_policy_log_probs:
        summary.update(
            {
                "mean_moth_action_exploration_log_prob": _nanmean_or_nan(
                    exploration_policy_log_probs
                ),
                "mean_moth_action_exploitation_log_prob": _nanmean_or_nan(
                    exploitation_policy_log_probs
                ),
            }
        )
    return summary, step_records
