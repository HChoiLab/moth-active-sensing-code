import csv
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from nectary_env import NectarSearchEnv
from nectary_allocentric_env import AllocentricNectarSearchEnv
from model import AllocentricNectarySearchModel, NectarySearchModel

ALLOCENTRIC_VISUAL_PATTERN_TYPES = ("none", "cross", "stripes", "circular")
ALLOCENTRIC_STRIPE_ORIENTATIONS = ("vertical", "horizontal")
ALLOCENTRIC_VISUAL_FLOWER_VALUES = (1.0, 0.5)


def _allocentric_vision_shape(env):
    vision_shape = tuple(env.observation_space["vision"].shape)
    if len(vision_shape) == 2:
        return (1, *vision_shape)
    return vision_shape


def _obs_dict_to_arrays(obs):
    mech = np.asarray(obs["mech"], dtype=np.float32)
    vis = np.asarray(obs["vision"], dtype=np.float32)
    if vis.ndim == 2:
        vis = vis[None, ...]
    return mech, vis


def _resolve_allocentric_visual_pattern_choices(visual_pattern_choices=None):
    if visual_pattern_choices is None:
        return ALLOCENTRIC_VISUAL_PATTERN_TYPES

    if isinstance(visual_pattern_choices, str):
        visual_pattern_choices = [visual_pattern_choices]

    choices = tuple(str(choice).strip().lower() for choice in visual_pattern_choices if str(choice).strip())
    if not choices:
        raise ValueError("visual_pattern_choices must contain at least one pattern.")

    invalid = sorted(set(choices) - set(ALLOCENTRIC_VISUAL_PATTERN_TYPES))
    if invalid:
        raise ValueError(
            "visual_pattern_choices contains unsupported pattern types: "
            + ", ".join(invalid)
        )
    return choices


def _configure_allocentric_visual_pattern(
    env,
    *,
    pattern_type=None,
    stripe_orientation=None,
):
    if not isinstance(env, AllocentricNectarSearchEnv):
        return None, None

    if pattern_type is not None:
        env.canvas.pattern_type = str(pattern_type).lower()

    active_pattern = str(env.canvas.pattern_type).lower()
    active_orientation = None
    if active_pattern == "stripes":
        if stripe_orientation is None:
            orientation_idx = int(env.np_random.integers(len(ALLOCENTRIC_STRIPE_ORIENTATIONS)))
            stripe_orientation = ALLOCENTRIC_STRIPE_ORIENTATIONS[orientation_idx]
        env.canvas.stripe_orientation = str(stripe_orientation).lower()
        active_orientation = env.canvas.stripe_orientation

    return active_pattern, active_orientation


def _sample_allocentric_visual_pattern_for_epoch(env, visual_pattern_choices=None):
    if not isinstance(env, AllocentricNectarSearchEnv):
        return None, None

    pattern_choices = _resolve_allocentric_visual_pattern_choices(visual_pattern_choices)
    pattern_idx = int(env.np_random.integers(len(pattern_choices)))
    pattern_type = pattern_choices[pattern_idx]
    return _configure_allocentric_visual_pattern(env, pattern_type=pattern_type)


def _configure_allocentric_visual_colors(env, *, flower_value=None):
    if not isinstance(env, AllocentricNectarSearchEnv):
        return None

    if flower_value is not None:
        flower_value = float(flower_value)
        if flower_value not in ALLOCENTRIC_VISUAL_FLOWER_VALUES:
            raise ValueError(
                "flower_value must be one of: "
                + ", ".join(str(value) for value in ALLOCENTRIC_VISUAL_FLOWER_VALUES)
            )
        env.canvas.flower_value = flower_value
        env.canvas.pattern_value = 0.5 if flower_value == 1.0 else 1.0

    return float(env.canvas.flower_value)


def _sample_allocentric_visual_colors_for_epoch(env):
    if not isinstance(env, AllocentricNectarSearchEnv):
        return None

    value_idx = int(env.np_random.integers(len(ALLOCENTRIC_VISUAL_FLOWER_VALUES)))
    return _configure_allocentric_visual_colors(
        env,
        flower_value=ALLOCENTRIC_VISUAL_FLOWER_VALUES[value_idx],
    )


class AllocentricTrajectorySampler:
    """
    Stateful trajectory sampler for allocentric pretraining.

    The sampler switches between a small set of behaviour modes and updates the
    absolute head/tip targets smoothly over time instead of redrawing them
    independently at every step. This produces training rollouts with both
    on-flower and off-flower observations while preserving temporal coherence.
    """

    def __init__(self, min_depth=None):
        self.min_depth = min_depth
        self.mode = None
        self.mode_steps_remaining = 0

        self.tip_velocity = np.zeros(2, dtype=np.float32)
        self.head_velocity = np.zeros(2, dtype=np.float32)
        self.depth_velocity = 0.0

        self.offflower_target_xy = None
        self.offflower_extension = 0.0

        self.approach_angle = 0.0
        self.approach_tip_radius = 0.0
        self.approach_depth = 0.0
        self.next_onflower_mode = "surface_scan"

        self.scan_radius = 0.0
        self.scan_ang_velocity = 0.0
        self.scan_depth = 0.0

        self.probe_angle = 0.0
        self.probe_target_radius = 0.0
        self.probe_depth = 0.0

        self.head_standoff = 0.0

    def reset(self, env):
        self.mode = None
        self.mode_steps_remaining = 0
        self.tip_velocity.fill(0.0)
        self.head_velocity.fill(0.0)
        self.depth_velocity = 0.0
        self._enter_mode(self._sample_initial_mode(env), env)

    def sample_action(self, env):
        if self.mode is None:
            self.reset(env)

        if self._should_switch_mode(env):
            self._enter_mode(self._sample_next_mode(env), env)

        action = self._sample_action_for_mode(env)
        self.mode_steps_remaining -= 1
        return action

    def _sample_initial_mode(self, env):
        mode_idx = env.np_random.choice(3, p=np.array([0.35, 0.45, 0.20]))
        return ("offflower_cruise", "surface_scan", "center_probe")[int(mode_idx)]

    def _sample_next_mode(self, env):
        if self.mode == "offflower_cruise":
            return "approach_rim" if env.np_random.uniform() < 0.75 else "offflower_cruise"
        if self.mode == "approach_rim":
            return self.next_onflower_mode
        if self.mode == "surface_scan":
            return "center_probe" if env.np_random.uniform() < 0.35 else "offflower_cruise"
        if self.mode == "center_probe":
            if self._tip_radius(env) <= env.flower.r0 + 0.5:
                return "offflower_cruise"
            return "surface_scan" if env.np_random.uniform() < 0.45 else "offflower_cruise"
        return "surface_scan"

    def _should_switch_mode(self, env):
        if self.mode is None or self.mode_steps_remaining <= 0:
            return True

        if self.mode == "offflower_cruise":
            if self.offflower_target_xy is None:
                return True
            close_to_target = np.linalg.norm(env.head_xy - self.offflower_target_xy) < 10.0
            return close_to_target and self.mode_steps_remaining < 12

        if self.mode == "approach_rim":
            return self._tip_radius(env) <= env.flower.R * 0.98

        if self.mode == "surface_scan":
            r = self._tip_radius(env)
            return (r < env.flower.r0 + 2.0) or (r > env.flower.R * 1.05)

        if self.mode == "center_probe":
            return (
                self._tip_radius(env) <= max(env.flower.r0 * 0.8, 0.25)
                and env.tip_depth >= min(self.probe_depth, env.flower.depth_success)
            )

        return False

    def _enter_mode(self, mode, env):
        self.mode = mode
        self.head_standoff = env.np_random.uniform(
            0.45 * env.proboscis_length,
            0.75 * env.proboscis_length,
        )

        if mode == "offflower_cruise":
            self.mode_steps_remaining = int(env.np_random.integers(25, 80))
            self.offflower_target_xy = self._sample_offflower_target(env)
            self.offflower_extension = float(
                env.np_random.uniform(0.15, 0.45) * env.proboscis_length
            )
            return

        if mode == "approach_rim":
            self.mode_steps_remaining = int(env.np_random.integers(12, 35))
            self.approach_angle = self._current_angle(env)
            self.approach_tip_radius = float(
                env.np_random.uniform(0.78 * env.flower.R, 0.95 * env.flower.R)
            )
            self.approach_depth = self._sample_onflower_depth(env, low_bias=0.65, high_bias=1.10)
            self.next_onflower_mode = (
                "surface_scan" if env.np_random.uniform() < 0.7 else "center_probe"
            )
            return

        if mode == "surface_scan":
            self.mode_steps_remaining = int(env.np_random.integers(35, 120))
            min_radius = max(env.flower.r0 + 2.0, 0.20 * env.flower.R)
            max_radius = 0.92 * env.flower.R
            current_radius = self._tip_radius(env)
            if min_radius <= current_radius <= max_radius:
                self.scan_radius = current_radius
            else:
                self.scan_radius = float(env.np_random.uniform(min_radius, max_radius))
            self.scan_ang_velocity = float(
                env.np_random.choice([-1.0, 1.0]) * env.np_random.uniform(0.05, 0.14)
            )
            self.scan_depth = self._sample_onflower_depth(env, low_bias=0.9, high_bias=1.35)
            return

        if mode == "center_probe":
            self.mode_steps_remaining = int(env.np_random.integers(18, 60))
            self.probe_angle = self._current_angle(env)
            self.probe_target_radius = float(env.np_random.uniform(0.0, 0.6 * env.flower.r0))
            self.probe_depth = float(
                env.np_random.uniform(
                    max(self._min_onflower_depth(env), env.flower.depth_success * 0.85),
                    min(env.max_depth, env.flower.depth_success + 2.0),
                )
            )
            return

        raise ValueError(f"Unknown trajectory mode '{mode}'.")

    def _sample_action_for_mode(self, env):
        if self.mode == "offflower_cruise":
            return self._sample_offflower_action(env)
        if self.mode == "approach_rim":
            return self._sample_approach_action(env)
        if self.mode == "surface_scan":
            return self._sample_surface_scan_action(env)
        if self.mode == "center_probe":
            return self._sample_center_probe_action(env)
        raise ValueError(f"Unknown trajectory mode '{self.mode}'.")

    def _sample_offflower_action(self, env):
        head_target = self.offflower_target_xy
        head_xy, self.head_velocity = self._smooth_move(
            current=env.head_xy,
            target=head_target,
            velocity=self.head_velocity,
            rng=env.np_random,
            max_step=5.0,
            noise_scale=0.45,
            attraction=0.18,
            inertia=0.82,
        )

        outward_dir = self._unit_vector(env._to_flower_xy(head_xy), fallback_angle=self._current_angle(env))
        tip_target = head_xy - outward_dir * self.offflower_extension

        # Keep the tip outside the corolla during off-flower cruising so the
        # pretraining batch contains genuine no-contact observations.
        tip_local = env._to_flower_xy(tip_target)
        tip_radius = float(np.linalg.norm(tip_local))
        if tip_radius < env.flower.R + 2.0:
            tip_target = env._to_canvas_xy(outward_dir * (env.flower.R + 2.0))

        tip_xy, self.tip_velocity = self._smooth_move(
            current=env.tip_xy,
            target=tip_target,
            velocity=self.tip_velocity,
            rng=env.np_random,
            max_step=4.5,
            noise_scale=0.35,
            attraction=0.20,
            inertia=0.78,
        )

        depth_target = float(env.np_random.uniform(max(env.min_depth, -1.0), 0.0))
        depth, self.depth_velocity = self._smooth_scalar(
            current=float(env.tip_depth),
            target=depth_target,
            velocity=self.depth_velocity,
            rng=env.np_random,
            max_step=0.35,
            noise_scale=0.05,
            attraction=0.22,
            inertia=0.70,
        )
        return self._compose_action(env, head_xy, tip_xy, depth)

    def _sample_approach_action(self, env):
        direction = np.array(
            [np.cos(self.approach_angle), np.sin(self.approach_angle)],
            dtype=np.float32,
        )
        tip_target = env._to_canvas_xy(direction * self.approach_tip_radius)
        head_target = env._to_canvas_xy(direction * (self.approach_tip_radius + self.head_standoff))

        head_xy, self.head_velocity = self._smooth_move(
            current=env.head_xy,
            target=head_target,
            velocity=self.head_velocity,
            rng=env.np_random,
            max_step=4.0,
            noise_scale=0.25,
            attraction=0.22,
            inertia=0.78,
        )
        tip_xy, self.tip_velocity = self._smooth_move(
            current=env.tip_xy,
            target=tip_target,
            velocity=self.tip_velocity,
            rng=env.np_random,
            max_step=3.5,
            noise_scale=0.20,
            attraction=0.24,
            inertia=0.78,
        )
        depth, self.depth_velocity = self._smooth_scalar(
            current=float(env.tip_depth),
            target=self.approach_depth,
            velocity=self.depth_velocity,
            rng=env.np_random,
            max_step=0.30,
            noise_scale=0.04,
            attraction=0.25,
            inertia=0.72,
        )
        return self._compose_action(env, head_xy, tip_xy, depth)

    def _sample_surface_scan_action(self, env):
        current_angle = self._current_angle(env)
        self.scan_radius = float(
            np.clip(
                0.96 * self.scan_radius + env.np_random.normal(scale=0.25),
                max(env.flower.r0 + 2.0, 0.20 * env.flower.R),
                0.92 * env.flower.R,
            )
        )
        target_angle = current_angle + self.scan_ang_velocity + env.np_random.normal(scale=0.03)

        radial_dir = np.array([np.cos(target_angle), np.sin(target_angle)], dtype=np.float32)
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)

        tip_target = env._to_canvas_xy(radial_dir * self.scan_radius)
        head_target = env._to_canvas_xy(
            radial_dir * (self.scan_radius + self.head_standoff)
            + tangent_dir * (0.10 * self.head_standoff * np.sign(self.scan_ang_velocity))
        )

        head_xy, self.head_velocity = self._smooth_move(
            current=env.head_xy,
            target=head_target,
            velocity=self.head_velocity,
            rng=env.np_random,
            max_step=3.0,
            noise_scale=0.18,
            attraction=0.24,
            inertia=0.80,
        )
        tip_xy, self.tip_velocity = self._smooth_move(
            current=env.tip_xy,
            target=tip_target,
            velocity=self.tip_velocity,
            rng=env.np_random,
            max_step=2.5,
            noise_scale=0.16,
            attraction=0.28,
            inertia=0.80,
        )
        depth, self.depth_velocity = self._smooth_scalar(
            current=float(env.tip_depth),
            target=self.scan_depth,
            velocity=self.depth_velocity,
            rng=env.np_random,
            max_step=0.22,
            noise_scale=0.03,
            attraction=0.22,
            inertia=0.76,
        )
        return self._compose_action(env, head_xy, tip_xy, depth)

    def _sample_center_probe_action(self, env):
        current_radius = self._tip_radius(env)
        self.probe_angle = float(self.probe_angle + env.np_random.normal(scale=0.04))
        target_radius = max(
            self.probe_target_radius,
            0.78 * current_radius + 0.22 * self.probe_target_radius,
        )

        radial_dir = np.array(
            [np.cos(self.probe_angle), np.sin(self.probe_angle)],
            dtype=np.float32,
        )
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)

        tip_target = env._to_canvas_xy(radial_dir * target_radius)
        head_target = env._to_canvas_xy(
            radial_dir * (target_radius + self.head_standoff)
            + tangent_dir * env.np_random.normal(scale=0.06 * self.head_standoff)
        )

        head_xy, self.head_velocity = self._smooth_move(
            current=env.head_xy,
            target=head_target,
            velocity=self.head_velocity,
            rng=env.np_random,
            max_step=2.8,
            noise_scale=0.16,
            attraction=0.26,
            inertia=0.78,
        )
        tip_xy, self.tip_velocity = self._smooth_move(
            current=env.tip_xy,
            target=tip_target,
            velocity=self.tip_velocity,
            rng=env.np_random,
            max_step=2.2,
            noise_scale=0.14,
            attraction=0.30,
            inertia=0.78,
        )
        depth, self.depth_velocity = self._smooth_scalar(
            current=float(env.tip_depth),
            target=self.probe_depth,
            velocity=self.depth_velocity,
            rng=env.np_random,
            max_step=0.28,
            noise_scale=0.03,
            attraction=0.24,
            inertia=0.76,
        )
        return self._compose_action(env, head_xy, tip_xy, depth)

    def _sample_offflower_target(self, env):
        max_radius = max(1.15 * env.flower.R, 0.35 * min(env.canvas.width, env.canvas.height))
        for _ in range(64):
            radius = env.np_random.uniform(1.10 * env.flower.R, max_radius)
            angle = env.np_random.uniform(-np.pi, np.pi)
            target_xy = env._to_canvas_xy(
                np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
            )
            if (
                5.0 <= target_xy[0] <= env.canvas.width - 5.0
                and 5.0 <= target_xy[1] <= env.canvas.height - 5.0
            ):
                return target_xy.astype(np.float32)

        fallback_angle = self._current_angle(env)
        fallback_radius = min(max_radius, 1.20 * env.flower.R)
        return env._to_canvas_xy(
            np.array(
                [
                    fallback_radius * np.cos(fallback_angle),
                    fallback_radius * np.sin(fallback_angle),
                ],
                dtype=np.float32,
            )
        )

    def _sample_onflower_depth(self, env, low_bias=0.8, high_bias=1.2):
        min_depth = self._min_onflower_depth(env)
        lower = max(min_depth, 0.3 * low_bias)
        upper = min(env.max_press_depth, max(lower + 0.2, env.max_press_depth * 0.75 * high_bias))
        if upper <= lower:
            return float(np.clip(lower, env.min_depth, env.max_depth))
        return float(env.np_random.uniform(lower, upper))

    def _min_onflower_depth(self, env):
        return 0.0 if self.min_depth is None else float(self.min_depth)

    def _compose_action(self, env, head_xy, tip_xy, depth):
        head_xy = self._clip_canvas_xy(env, head_xy)
        tip_xy = self._clip_canvas_xy(env, tip_xy)
        tip_xy = self._clip_canvas_xy(env, env._constrain_tip_to_head(tip_xy, head_xy))
        depth = float(np.clip(depth, env.min_depth, env.max_depth))

        head_norm = env._normalize_canvas_xy(head_xy)
        tip_norm = env._normalize_canvas_xy(tip_xy)
        return np.array(
            [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], depth],
            dtype=np.float32,
        )

    def _tip_radius(self, env):
        return float(np.linalg.norm(env._to_flower_xy(env.tip_xy)))

    def _current_angle(self, env):
        local_tip = env._to_flower_xy(env.tip_xy)
        radius = float(np.linalg.norm(local_tip))
        if radius < 1e-6:
            return float(env.np_random.uniform(-np.pi, np.pi))
        return float(np.arctan2(local_tip[1], local_tip[0]))

    def _clip_canvas_xy(self, env, xy):
        xy = np.asarray(xy, dtype=np.float32).copy()
        xy[0] = np.clip(xy[0], 0.0, env.canvas.width)
        xy[1] = np.clip(xy[1], 0.0, env.canvas.height)
        return xy

    def _unit_vector(self, vec, fallback_angle=0.0):
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return np.array(
                [np.cos(fallback_angle), np.sin(fallback_angle)],
                dtype=np.float32,
            )
        return (vec / norm).astype(np.float32)

    def _smooth_move(
        self,
        current,
        target,
        velocity,
        rng,
        max_step,
        noise_scale,
        attraction,
        inertia,
    ):
        current = np.asarray(current, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        velocity = np.asarray(velocity, dtype=np.float32)

        velocity = inertia * velocity + attraction * (target - current)
        velocity = velocity + rng.normal(scale=noise_scale, size=2).astype(np.float32)

        speed = float(np.linalg.norm(velocity))
        if speed > max_step:
            velocity = velocity * (max_step / (speed + 1e-8))

        next_xy = current + velocity
        return next_xy.astype(np.float32), velocity.astype(np.float32)

    def _smooth_scalar(
        self,
        current,
        target,
        velocity,
        rng,
        max_step,
        noise_scale,
        attraction,
        inertia,
    ):
        velocity = float(inertia * velocity + attraction * (target - current))
        velocity += float(rng.normal(scale=noise_scale))
        velocity = float(np.clip(velocity, -max_step, max_step))
        return float(current + velocity), velocity


def collect_data_random(env, batch_size, seq_len, min_depth=None):

    seeds = random.sample(range(1000 * batch_size), batch_size)

    X = np.zeros(
        (batch_size, seq_len + 1, env.observation_space.shape[0]),
        dtype=np.float32,
    )
    A = np.zeros(
        (batch_size, seq_len, env.action_space.shape[0]),
        dtype=np.float32,
    )

    xyd = np.zeros((batch_size, seq_len + 1, 3), dtype=np.float32)

    for s in range(batch_size):
        X[s, 0] = env.reset(seed=seeds[s])[0]
        xyd[s, 0, :2] = env.tip_xy
        xyd[s, 0, -1] = env.tip_depth
        for t in range(seq_len):
            A[s, t] = env.action_space.sample()
            if min_depth is not None:
                A[s, t, -1] = max(min_depth, A[s, t, -1])
            X[s, t + 1] = env.step(A[s, t])[0]
            xyd[s, t + 1, :2] = env.tip_xy
            xyd[s, t + 1, -1] = env.tip_depth

    X = torch.as_tensor(X, dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.float32)

    return X, A, xyd


def collect_data_random_allocentric(
    env,
    batch_size,
    seq_len,
    min_depth=None,
    trajectory_mode="smooth_modes",
):

    seeds = random.sample(range(1000 * batch_size), batch_size)
    mech_dim = env.observation_space["mech"].shape[0]
    vis_shape = _allocentric_vision_shape(env)

    X_mech = np.zeros((batch_size, seq_len, mech_dim), dtype=np.float32)
    X_vis = np.zeros((batch_size, seq_len, *vis_shape), dtype=np.float32)
    A = np.zeros((batch_size, seq_len, env.action_space.shape[0]), dtype=np.float32)
    pose = np.zeros((batch_size, seq_len, 5), dtype=np.float32)

    for s in range(batch_size):
        env.reset(seed=seeds[s])
        sampler = None
        if trajectory_mode == "smooth_modes":
            sampler = AllocentricTrajectorySampler(min_depth=min_depth)
            sampler.reset(env)
        elif trajectory_mode != "random":
            raise ValueError(
                "trajectory_mode must be one of: 'smooth_modes', 'random'."
            )

        for t in range(seq_len):
            if trajectory_mode == "smooth_modes":
                A[s, t] = sampler.sample_action(env)
            else:
                A[s, t] = sample_allocentric_flower_action(env)
                if min_depth is not None:
                    A[s, t, -1] = max(min_depth, A[s, t, -1])
            obs, _, _, _, _ = env.step(A[s, t])
            mech_obs, vis_obs = _obs_dict_to_arrays(obs)
            X_mech[s, t] = mech_obs
            X_vis[s, t] = vis_obs
            pose[s, t, :2] = env.head_xy
            pose[s, t, 2:4] = env.tip_xy
            pose[s, t, 4] = env.tip_depth

    X_mech = torch.as_tensor(X_mech, dtype=torch.float32)
    X_vis = torch.as_tensor(X_vis, dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.float32)

    return X_mech, X_vis, A, pose


def sample_allocentric_flower_action(env):
    radius = env.flower.R * np.sqrt(env.np_random.uniform())
    angle = env.np_random.uniform(-np.pi, np.pi)
    flower_xy = np.array(
        [radius * np.cos(angle), radius * np.sin(angle)],
        dtype=np.float32,
    )
    tip_xy = env._to_canvas_xy(flower_xy)

    head_angle = env.np_random.uniform(-np.pi, np.pi)
    head_radius = env.proboscis_length * np.sqrt(env.np_random.uniform())
    head_offset = np.array(
        [head_radius * np.cos(head_angle), head_radius * np.sin(head_angle)],
        dtype=np.float32,
    )
    head_xy = env._constrain_head_to_canvas(tip_xy + head_offset)

    head_norm = env._normalize_canvas_xy(head_xy)
    tip_norm = env._normalize_canvas_xy(tip_xy)
    depth = env.np_random.uniform(
        env.action_space.low[4],
        env.action_space.high[4],
    )

    return np.array(
        [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], depth],
        dtype=np.float32,
    )


def _build_default_model(
    env,
    allocentric,
    visual_likelihood="gaussian",
    learn_decoder_variance=True,
    posterior_fusion="early",
    moment_matching=True,
    rnn_type="rnn",
    modality_rnn_mode="shared",
):
    if not allocentric:
        return NectarySearchModel(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            latent_dim=5,
            beta=0.1,
        )

    return AllocentricNectarySearchModel(
        mech_obs_dim=env.observation_space["mech"].shape[0],
        vis_obs_shape=_allocentric_vision_shape(env),
        action_dim=env.action_space.shape[0],
        latent_dim=5,
        beta=0.1,
        visual_likelihood=visual_likelihood,
        learn_decoder_variance=learn_decoder_variance,
        posterior_fusion=posterior_fusion,
        moment_matching=moment_matching,
        rnn_type=rnn_type,
        modality_rnn_mode=modality_rnn_mode,
    )


def _resolve_training_device(device=None, local_rank=None):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        if local_rank is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{local_rank}")

    return torch.device("cpu")


def _distributed_ready():
    return dist.is_available() and dist.is_initialized()


def _reduce_scalar(value, device, world_size):
    if not _distributed_ready() or world_size <= 1:
        return float(value)

    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor.item()


def _unwrap_training_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _resolve_beta_for_epoch(
    epoch,
    num_epochs,
    base_beta,
    beta_anneal=False,
    beta_start=0.0,
    beta_end=None,
    beta_anneal_epochs=None,
):
    if not beta_anneal:
        return float(base_beta)

    target_beta = float(base_beta if beta_end is None else beta_end)
    start_beta = float(beta_start)

    if beta_anneal_epochs is None:
        beta_anneal_epochs = num_epochs

    anneal_epochs = max(1, int(beta_anneal_epochs))
    if anneal_epochs == 1:
        return target_beta

    progress = min(max(epoch, 0), anneal_epochs - 1) / (anneal_epochs - 1)
    return start_beta + (target_beta - start_beta) * progress


def _initialize_metrics_csv(
    metrics_csv_path,
    fieldnames,
    append=False,
):
    csv_path = Path(metrics_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = (not append) or (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    mode = "a" if append else "w"
    with csv_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()


def _append_metrics_csv_row(
    metrics_csv_path,
    fieldnames,
    row,
):
    csv_path = Path(metrics_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_nectary_search_model(
    envs,
    model,
    num_epochs=10,
    batch_size=4,
    seq_len=200,
    learning_rate=1e-3,
    print_every=50,
    allocentric=False,
    min_depth=None,
    parallel=False,
    rank=0,
    world_size=1,
    local_rank=None,
    device=None,
    wrap_ddp=False,
    allocentric_trajectory_mode="smooth_modes",
    beta_anneal=False,
    beta_start=0.0,
    beta_end=None,
    beta_anneal_epochs=None,
    randomize_visual_pattern_each_epoch=False,
    randomize_visual_colors_each_epoch=False,
    visual_pattern_choices=None,
    visual_likelihood="gaussian",
    learn_decoder_variance=True,
    posterior_fusion="early",
    moment_matching=True,
    rnn_type="rnn",
    modality_rnn_mode="shared",
    metrics_csv_path=None,
    append_metrics_csv=False,
    resume_state=None,
    return_training_state=False,
):

    if envs is None:
        envs = [NectarSearchEnv()] if not allocentric else [AllocentricNectarSearchEnv()]

    if model is None:
        model = _build_default_model(
            envs[0],
            allocentric,
            visual_likelihood=visual_likelihood,
            learn_decoder_variance=learn_decoder_variance,
            posterior_fusion=posterior_fusion,
            moment_matching=moment_matching,
            rnn_type=rnn_type,
            modality_rnn_mode=modality_rnn_mode,
        )

    device = _resolve_training_device(device=device, local_rank=local_rank)
    model.to(device)

    train_model = model
    if parallel and wrap_ddp and world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        if device.type == "cuda":
            train_model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
        else:
            train_model = DDP(model)

    optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    beta_model = _unwrap_training_model(train_model)
    base_beta = getattr(beta_model, "beta", None)
    if base_beta is None:
        raise AttributeError("Model must define a 'beta' attribute for training.")
    resume_state = dict(resume_state or {})
    start_epoch = int(resume_state.get("completed_epochs", 0))
    if start_epoch < 0:
        raise ValueError("resume_state.completed_epochs must be non-negative.")
    if start_epoch > int(num_epochs):
        raise ValueError(
            f"Cannot resume from epoch {start_epoch} when num_epochs={num_epochs}. "
            "Increase num_epochs to continue training."
        )
    optimizer_state_dict = resume_state.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        _move_optimizer_state_to_device(optimizer, device)

    csv_fieldnames = [
        "epoch",
        "total_loss",
        "recon_loss",
        "kl_loss",
        "beta",
        "flower_c",
        "flower_value",
        "pattern_type",
        "stripe_orientation",
    ]
    should_write_metrics_csv = metrics_csv_path is not None and (not parallel or rank == 0)
    if should_write_metrics_csv:
        _initialize_metrics_csv(
            metrics_csv_path=metrics_csv_path,
            fieldnames=csv_fieldnames,
            append=append_metrics_csv,
        )

    for epoch in range(start_epoch, num_epochs):
        train_model.train()
        beta_model.beta = _resolve_beta_for_epoch(
            epoch=epoch,
            num_epochs=num_epochs,
            base_beta=base_beta,
            beta_anneal=beta_anneal,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_anneal_epochs=beta_anneal_epochs,
        )
        active_env = envs[rank % len(envs)] if parallel else random.choice(envs)
        active_pattern = None
        active_orientation = None
        active_flower_value = None

        if allocentric and randomize_visual_pattern_each_epoch:
            active_pattern, active_orientation = _sample_allocentric_visual_pattern_for_epoch(
                active_env,
                visual_pattern_choices=visual_pattern_choices,
            )
        elif allocentric:
            active_pattern, active_orientation = _configure_allocentric_visual_pattern(active_env)

        if allocentric and randomize_visual_colors_each_epoch:
            active_flower_value = _sample_allocentric_visual_colors_for_epoch(active_env)
        elif allocentric:
            active_flower_value = _configure_allocentric_visual_colors(active_env)

        if not allocentric:
            x_batch, a_batch, _ = collect_data_random(
                active_env, batch_size, seq_len, min_depth
            )
            x_batch = x_batch.to(device)
            a_batch = a_batch.to(device)
            _, losses = train_model(x_batch, a_batch)
        else:
            x_mech_batch, x_vis_batch, a_batch, _ = collect_data_random_allocentric(
                active_env,
                batch_size,
                seq_len,
                min_depth,
                trajectory_mode=allocentric_trajectory_mode,
            )
            x_mech_batch = x_mech_batch.to(device)
            x_vis_batch = x_vis_batch.to(device)
            a_batch = a_batch.to(device)
            _, losses = train_model(x_mech_batch, x_vis_batch, a_batch)

        loss = losses["total"]
        mse_loss = losses["mse"]
        kl_div = losses["kl"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        avg_loss = loss.item() / batch_size
        avg_mse = mse_loss.item() / batch_size
        avg_kl = kl_div.item() / batch_size

        reduced_loss = _reduce_scalar(avg_loss, device, world_size)
        reduced_mse = _reduce_scalar(avg_mse, device, world_size)
        reduced_kl = _reduce_scalar(avg_kl, device, world_size)

        should_log = (epoch % print_every == 0) and (not parallel or rank == 0)
        if should_write_metrics_csv:
            _append_metrics_csv_row(
                metrics_csv_path=metrics_csv_path,
                fieldnames=csv_fieldnames,
                row={
                    "epoch": float(epoch + 1),
                    "total_loss": float(reduced_loss),
                    "recon_loss": float(reduced_mse),
                    "kl_loss": float(reduced_kl),
                    "beta": float(beta_model.beta),
                    "flower_c": float(active_env.flower.c),
                    "flower_value": "" if active_flower_value is None else float(active_flower_value),
                    "pattern_type": active_pattern or "",
                    "stripe_orientation": active_orientation or "",
                },
            )
        if should_log:
            mode_label = "Global" if parallel else "Loss"
            pattern_msg = ""
            if allocentric and active_pattern is not None:
                pattern_msg = f", pattern = {active_pattern}"
                if active_orientation is not None:
                    pattern_msg += f" ({active_orientation})"
            flower_value_msg = ""
            if allocentric and active_flower_value is not None:
                flower_value_msg = f", flower_value = {active_flower_value:.1f}"
            print(
                f"Epoch {epoch + 1}: {mode_label} = {reduced_loss:.4f}, "
                f"MSE = {reduced_mse:.4f}, "
                f"KL = {reduced_kl:.4f}, "
                f"beta = {beta_model.beta:.4f}, "
                f"c = {active_env.flower.c}"
                f"{pattern_msg}"
                f"{flower_value_msg}"
            )

    training_state = {
        "completed_epochs": int(num_epochs),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if return_training_state:
        return model, training_state
    return model
