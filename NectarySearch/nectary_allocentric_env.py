from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    from nectary_env import FlowerParams
except ImportError:
    from .nectary_env import FlowerParams


@dataclass
class CanvasParams:
    """Allocentric image/canvas configuration.

    Units are the same as FlowerParams, so a 300x300 canvas is interpreted as
    300 by 300 mm/pixels in the same coordinate system as the flower geometry.
    """

    width: float = 300.0
    height: float = 300.0
    background_value: float = 0.0
    flower_value: float = 1.0
    hole_value: float = 0.0
    pattern_type: str = "cross"  # one of: none, cross, stripes, circular
    pattern_value: float = 0.25
    cross_width: float = 4.0
    cross_offset: tuple[float, float] = (0.0, 0.0)
    stripe_width: float = 6.0
    stripe_offset: float = 10.0
    stripe_orientation: str = "vertical"  # one of: vertical, horizontal
    circular_radius: float = 12.0


@dataclass
class NonStationaryParams:
    """Configuration for a one-time, randomized change within an episode."""

    enabled: bool = False
    change_probability: float = 0.0
    position_shift_radius: float = 0.0
    pattern_choices: tuple[str, ...] = ()
    curvature_choices: tuple[float, ...] = ()
    min_change_timestep: int = 1
    max_change_timestep: int | None = None


class AllocentricNectarSearchEnv(gym.Env):
    """
    Nectary search on an allocentric image/canvas.

    The flower is placed at a random canvas location on reset, while contact
    mechanics and mechanosensory observations match the flower-centered
    NectarSearchEnv geometry. Actions are normalized absolute head and
    proboscis-tip targets:

        action = [head_x_norm, head_y_norm, tip_x_norm, tip_y_norm, depth]

    xy coordinates are allocentric image/canvas coordinates in [0, 1]. Depth
    is the absolute insertion/pressing depth used by the same contact model as
    the egocentric environment. The tip is projected into the reachable disk
    around the head when a proposed action exceeds proboscis_length.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        flower: FlowerParams | None = None,
        canvas: CanvasParams | None = None,
        dt: float = 0.02,
        max_steps: int = 500,
        max_press_depth: float = 2.0,
        min_depth: float = -2.0,
        max_depth: float = 10.0,
        proboscis_length: float = 25.0,
        visual_field_size: float = 50.0,
        visual_patch_shape: tuple[int, int] | None = None,
        non_stationary: NonStationaryParams | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.flower = flower or FlowerParams()
        self.canvas = canvas or CanvasParams()
        self.dt = dt
        self.max_steps = max_steps
        self.max_press_depth = max_press_depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.proboscis_length = proboscis_length
        self.visual_field_size = visual_field_size
        self.visual_patch_shape = tuple(visual_patch_shape) if visual_patch_shape is not None else (
            int(round(visual_field_size)),
            int(round(visual_field_size)),
        )
        self.non_stationary = non_stationary or NonStationaryParams()
        self.np_random = np.random.default_rng(seed)

        if self.canvas.width <= 0 or self.canvas.height <= 0:
            raise ValueError("Canvas width and height must be positive.")
        if self.flower.R <= 0:
            raise ValueError("Flower radius R must be positive.")
        if self.canvas.width < 2 * self.flower.R or self.canvas.height < 2 * self.flower.R:
            raise ValueError("Canvas must be large enough to contain the full flower diameter.")
        if self.proboscis_length <= 0:
            raise ValueError("proboscis_length must be positive.")
        if self.visual_field_size <= 0:
            raise ValueError("visual_field_size must be positive.")
        if len(self.visual_patch_shape) != 2 or min(self.visual_patch_shape) <= 0:
            raise ValueError("visual_patch_shape must contain two positive integers.")
        self._validate_non_stationary_params()

        self.max_lateral_delta = 50 #math.hypot(self.canvas.width, self.canvas.height)

        # Action: normalized absolute head x/y, normalized absolute tip x/y, absolute depth.
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, self.min_depth], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, self.max_depth], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )

        # Observation contains mechanosensory features plus a head-centered visual crop.
        self.observation_space = spaces.Dict(
            {
                "mech": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
                "vision": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=tuple(self.visual_patch_shape),
                    dtype=np.float32,
                ),
            }
        )

        self._step_count = 0
        self.flower_center_xy = np.zeros(2, dtype=np.float32)
        self.head_xy = np.zeros(2, dtype=np.float32)  # allocentric canvas coordinates
        self.tip_xy = np.zeros(2, dtype=np.float32)  # allocentric canvas coordinates
        self.prev_xy = np.zeros(2, dtype=np.float32)
        self.tip_depth = 0.0
        self.canvas_image = None      # will be initialized upon reset
        self._active_pattern_type: str | None = None
        self._active_flower_c: float | None = None
        self._non_stationary_change_timestep: int | None = None
        self._non_stationary_change_applied = False
        self._non_stationary_position_shift = np.zeros(2, dtype=np.float32)

    def _validate_non_stationary_params(self) -> None:
        cfg = self.non_stationary
        cfg.enabled = bool(cfg.enabled)
        cfg.change_probability = float(cfg.change_probability)
        cfg.position_shift_radius = float(cfg.position_shift_radius)
        cfg.pattern_choices = tuple(
            str(pattern).strip().lower() for pattern in cfg.pattern_choices
        )
        cfg.curvature_choices = tuple(
            float(curvature) for curvature in cfg.curvature_choices
        )
        cfg.min_change_timestep = int(cfg.min_change_timestep)
        if cfg.max_change_timestep is not None:
            cfg.max_change_timestep = int(cfg.max_change_timestep)

        if not 0.0 <= cfg.change_probability <= 1.0:
            raise ValueError("non_stationary.change_probability must be in [0, 1].")
        if cfg.position_shift_radius < 0.0:
            raise ValueError("non_stationary.position_shift_radius must be non-negative.")
        if cfg.min_change_timestep < 1:
            raise ValueError("non_stationary.min_change_timestep must be positive.")
        if (
            cfg.max_change_timestep is not None
            and cfg.max_change_timestep < cfg.min_change_timestep
        ):
            raise ValueError(
                "non_stationary.max_change_timestep must be greater than or equal "
                "to min_change_timestep."
            )

        supported_patterns = {"none", "cross", "stripes", "circular"}
        invalid_patterns = sorted(set(cfg.pattern_choices) - supported_patterns)
        if invalid_patterns:
            raise ValueError(
                "non_stationary.pattern_choices contains unsupported patterns: "
                + ", ".join(invalid_patterns)
            )
        if cfg.enabled and cfg.change_probability > 0.0:
            if not cfg.pattern_choices:
                raise ValueError(
                    "non_stationary.pattern_choices must not be empty when changes "
                    "are enabled."
                )
            if not cfg.curvature_choices:
                raise ValueError(
                    "non_stationary.curvature_choices must not be empty when changes "
                    "are enabled."
                )
            maximum = self.max_steps
            if cfg.max_change_timestep is not None:
                maximum = min(maximum, cfg.max_change_timestep)
            if cfg.min_change_timestep > maximum:
                raise ValueError(
                    "non_stationary.min_change_timestep must not exceed the episode's "
                    "maximum change timestep."
                )

    @property
    def current_pattern_type(self) -> str:
        if self._active_pattern_type is not None:
            return self._active_pattern_type
        return str(self.canvas.pattern_type).strip().lower()

    @property
    def current_flower_c(self) -> float:
        if self._active_flower_c is not None:
            return float(self._active_flower_c)
        return float(self.flower.c)

    # ------------------------
    # Coordinate transforms
    # ------------------------
    def _to_flower_xy(self, canvas_xy):
        """
        Note: we flip the y coordinate to be consistent with y-down convention
        """
        canvas_xy = np.asarray(canvas_xy, dtype=np.float32)
        center_xy = self.flower_center_xy.astype(np.float32)
        local_xy = canvas_xy - center_xy
        local_xy[..., 1] *= -1.0
        return local_xy.astype(np.float32)
    
    def _to_canvas_xy(self, flower_xy):
        """
        Note: we flip the y coordinate to be consistent with y-down convention
        """
        flower_xy = np.asarray(flower_xy, dtype=np.float32)
        center_xy = self.flower_center_xy.astype(np.float32)
        canvas_xy = flower_xy.copy()
        canvas_xy[..., 1] *= -1.0
        canvas_xy += center_xy
        return canvas_xy.astype(np.float32)

    def _normalize_canvas_xy(self, canvas_xy: np.ndarray) -> np.ndarray:
        canvas_xy = np.asarray(canvas_xy, dtype=np.float32)
        return np.array(
            [canvas_xy[0] / self.canvas.width, canvas_xy[1] / self.canvas.height],
            dtype=np.float32,
        )

    def _denormalize_canvas_xy(self, norm_xy: np.ndarray) -> np.ndarray:
        norm_xy = np.asarray(norm_xy, dtype=np.float32)
        return np.array(
            [norm_xy[0] * self.canvas.width, norm_xy[1] * self.canvas.height],
            dtype=np.float32,
        )

    def _constrain_head_to_canvas(self, head_xy: np.ndarray) -> np.ndarray:
        head_xy = np.asarray(head_xy, dtype=np.float32).copy()
        head_xy[0] = np.clip(head_xy[0], 0.0, self.canvas.width)
        head_xy[1] = np.clip(head_xy[1], 0.0, self.canvas.height)
        return head_xy

    def _constrain_tip_to_head(self, tip_xy: np.ndarray, head_xy: np.ndarray) -> np.ndarray:
        tip_xy = np.asarray(tip_xy, dtype=np.float32).copy()
        head_xy = np.asarray(head_xy, dtype=np.float32)
        v = tip_xy - head_xy
        dist = float(np.linalg.norm(v))
        if dist <= self.proboscis_length:
            return tip_xy
        return (head_xy + v / (dist + 1e-8) * self.proboscis_length).astype(np.float32)

    # ------------------------
    # Geometry: surface z(r)
    # ------------------------
    def _z_surface(self, r: float) -> float:
        """Axisymmetric corolla height z as function of flower-centered radius r."""
        f = self.flower
        flower_c = self.current_flower_c
        if np.isinf(flower_c):
            return 0.0

        u = (r - f.r0) / f.R
        u = float(np.clip(u, 0.0, 1.0))
        p = float(np.exp(flower_c))
        return float(f.L * (u**p))

    def _dz_dr(self, r: float) -> float:
        """Derivative dz/dr for analytic normal."""
        f = self.flower
        flower_c = self.current_flower_c
        if np.isinf(flower_c):
            return 0.0

        u = (r - f.r0) / f.R
        if u <= 0.0 or u >= 1.0:
            return 0.0

        p = float(np.exp(flower_c))
        return float((f.L / f.R) * p * (float(u) ** (p - 1.0)))

    def _surface_normal_xy(self, canvas_xy: np.ndarray) -> np.ndarray:
        """
        Return normal projection onto xy-plane in canvas axes.

        Translation from flower-centered to canvas coordinates does not rotate
        the axes, so the normal vector components are unchanged.
        """
        local_xy = self._to_flower_xy(canvas_xy)
        x, y = float(local_xy[0]), float(local_xy[1])
        r = np.hypot(x, y)
        if r < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        dzdr = self._dz_dr(r)
        dzdx = dzdr * x / r
        dzdy = dzdr * y / r
        nxy = np.array([-dzdx, -dzdy], dtype=np.float32)
        n = np.linalg.norm(nxy) + 1e-8
        return nxy / n

    # ------------------------
    # Mechanics
    # ------------------------
    def _inside_hole(self, canvas_xy: np.ndarray) -> bool:
        local_xy = self._to_flower_xy(canvas_xy)
        return np.hypot(float(local_xy[0]), float(local_xy[1])) <= self.flower.r0

    def _step_contact(self, proposed_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Quasi-static contact on the randomly placed flower.

        proposed_xy is allocentric/canvas-space. Positions outside the flower
        rim are valid canvas locations with no contact.
        """
        f = self.flower
        proposed_xy = np.asarray(proposed_xy, dtype=np.float32).copy()
        proposed_xy[0] = np.clip(proposed_xy[0], 0.0, self.canvas.width)
        proposed_xy[1] = np.clip(proposed_xy[1], 0.0, self.canvas.height)

        local_xy = self._to_flower_xy(proposed_xy)
        r = np.hypot(float(local_xy[0]), float(local_xy[1]))

        if r > f.R:
            return proposed_xy, dict(contact=0.0, fn=0.0, nx=0.0, ny=0.0, slip=0.0)

        if self._inside_hole(proposed_xy):
            return proposed_xy, dict(contact=0.0, fn=0.0, nx=0.0, ny=0.0, slip=0.0)

        if float(self.tip_depth) <= 0:
            return proposed_xy, dict(contact=0.0, fn=0.0, nx=0.0, ny=0.0, slip=0.0)

        nxy = self._surface_normal_xy(proposed_xy)
        penetration = max(0.0, float(self.tip_depth))
        penetration = min(penetration, self.max_press_depth)
        fn = f.k * penetration
        fn = fn / (f.k * self.max_press_depth)
        fn = np.log1p(fn)

        delta = proposed_xy - self.tip_xy
        tang = delta - (delta @ nxy) * nxy if np.linalg.norm(nxy) > 0 else delta

        max_slip = float(self.max_lateral_delta / (self.dt + 1e-8))
        slip = float(np.linalg.norm(tang) / (self.dt + 1e-8))
        slip = slip / max_slip
        slip = np.log1p(slip)

        return proposed_xy, dict(contact=1.0, fn=fn, nx=float(nxy[0]), ny=float(nxy[1]), slip=slip)

    # ------------------------
    # Canvas helpers
    # ------------------------
    def _sample_flower_center(self) -> np.ndarray:
        f = self.flower
        x = self.np_random.uniform(f.R, self.canvas.width - f.R)
        y = self.np_random.uniform(f.R, self.canvas.height - f.R)
        return np.array([x, y], dtype=np.float32)

    def _sample_start_tip(self) -> np.ndarray:
        f = self.flower
        r_start = self.np_random.uniform(0.7 * f.R, 0.95 * f.R)
        ang = self.np_random.uniform(-np.pi, np.pi)
        local_xy = np.array([r_start * np.cos(ang), r_start * np.sin(ang)], dtype=np.float32)
        return self._to_canvas_xy(local_xy)

    def _sample_start_head(self, tip_xy: np.ndarray) -> np.ndarray:
        local_tip_xy = self._to_flower_xy(tip_xy)
        norm = float(np.linalg.norm(local_tip_xy))
        if norm < 1e-8:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = local_tip_xy / norm
        head_xy = tip_xy + direction * (0.8 * self.proboscis_length)
        return self._constrain_head_to_canvas(head_xy)

    def _schedule_non_stationary_change(self) -> None:
        self._non_stationary_change_timestep = None
        self._non_stationary_change_applied = False
        self._non_stationary_position_shift = np.zeros(2, dtype=np.float32)
        self._active_pattern_type = None
        self._active_flower_c = None

        cfg = self.non_stationary
        if not cfg.enabled or self.np_random.random() >= cfg.change_probability:
            return

        maximum = self.max_steps
        if cfg.max_change_timestep is not None:
            maximum = min(maximum, cfg.max_change_timestep)
        if cfg.min_change_timestep > maximum:
            raise ValueError(
                "non_stationary.min_change_timestep must not exceed the episode's "
                "maximum change timestep."
            )
        self._non_stationary_change_timestep = int(
            self.np_random.integers(cfg.min_change_timestep, maximum + 1)
        )

    def _sample_different_pattern(self) -> str:
        choices = [
            pattern
            for pattern in self.non_stationary.pattern_choices
            if pattern != self.current_pattern_type
        ]
        if not choices:
            choices = list(self.non_stationary.pattern_choices)
        return choices[int(self.np_random.integers(len(choices)))]

    def _sample_different_curvature(self) -> float:
        choices = [
            curvature
            for curvature in self.non_stationary.curvature_choices
            if curvature != self.current_flower_c
        ]
        if not choices:
            choices = list(self.non_stationary.curvature_choices)
        return float(choices[int(self.np_random.integers(len(choices)))])

    def _apply_non_stationary_change(self) -> None:
        cfg = self.non_stationary
        angle = self.np_random.uniform(-np.pi, np.pi)
        radius = cfg.position_shift_radius * np.sqrt(self.np_random.random())
        requested_shift = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)],
            dtype=np.float32,
        )

        old_center = self.flower_center_xy.copy()
        lower = np.array([self.flower.R, self.flower.R], dtype=np.float32)
        upper = np.array(
            [self.canvas.width - self.flower.R, self.canvas.height - self.flower.R],
            dtype=np.float32,
        )
        self.flower_center_xy = np.clip(
            old_center + requested_shift, lower, upper
        ).astype(np.float32)
        self._non_stationary_position_shift = self.flower_center_xy - old_center
        self._active_pattern_type = self._sample_different_pattern()
        self._active_flower_c = self._sample_different_curvature()
        self.canvas_image = self.get_canvas_image()
        self._non_stationary_change_applied = True

    def _non_stationary_info(self, *, changed_this_step: bool) -> dict:
        return {
            "non_stationary_change_scheduled": (
                self._non_stationary_change_timestep is not None
            ),
            "non_stationary_change_timestep": self._non_stationary_change_timestep,
            "non_stationary_changed_this_step": bool(changed_this_step),
            "non_stationary_has_changed": bool(
                self._non_stationary_change_applied
            ),
            "non_stationary_position_shift_xy": (
                self._non_stationary_position_shift.copy()
            ),
            "flower_pattern_type": self.current_pattern_type,
            "flower_curvature": self.current_flower_c,
        }

    def get_canvas_image(self) -> np.ndarray:
        """
        Return a top-down grayscale canvas image with a visual flower pattern.

        Supported pattern_type values are "none", "cross", "stripes", and
        "circular". Stripes are rendered as two bands on opposite sides of
        the nectary, circular is rendered as one centered disk, and the cross
        can be shifted relative to the nectary via canvas.cross_offset. Shape
        is (height, width).
        """
        width = int(round(self.canvas.width))
        height = int(round(self.canvas.height))
        image = np.full((height, width), self.canvas.background_value, dtype=np.float32)

        yy, xx = np.ogrid[:height, :width]
        canvas_xy = np.stack(
            np.broadcast_arrays(xx.astype(np.float32), yy.astype(np.float32)),
            axis=-1,
        )
        flower_xy = self._to_flower_xy(canvas_xy)
        local_x = flower_xy[..., 0]
        local_y = flower_xy[..., 1]
        rr = np.sqrt(local_x ** 2 + local_y ** 2)

        flower_mask = rr <= self.flower.R
        image[flower_mask] = self.canvas.flower_value

        pattern_type = self.current_pattern_type
        if pattern_type not in {"none", "cross", "stripes", "circular"}:
            raise ValueError("pattern_type must be one of: none, cross, stripes, circular.")

        if pattern_type == "cross":
            half_width = max(0.5, float(self.canvas.cross_width) / 2.0)
            cross_offset = np.asarray(self.canvas.cross_offset, dtype=np.float32)
            if cross_offset.shape != (2,):
                raise ValueError(
                    "cross_offset must contain exactly two values: (x_offset, y_offset)."
                )
            pattern_mask = (
                (np.abs(local_x - float(cross_offset[0])) <= half_width)
                | (np.abs(local_y - float(cross_offset[1])) <= half_width)
            )
            image[flower_mask & pattern_mask] = self.canvas.pattern_value
        elif pattern_type == "stripes":
            stripe_width = max(1.0, float(self.canvas.stripe_width))
            stripe_orientation = str(self.canvas.stripe_orientation).lower()
            stripe_loc = self.flower.r0 + self.canvas.stripe_offset
            if stripe_orientation not in {"vertical", "horizontal"}:
                raise ValueError("stripe_orientation must be one of: vertical, horizontal.")

            if stripe_orientation == "vertical":
                pattern_mask = (np.abs(local_x) >= stripe_loc) & (
                    np.abs(local_x) <= stripe_loc + stripe_width
                )
            else:
                pattern_mask = (np.abs(local_y) >= stripe_loc) & (
                    np.abs(local_y) <= stripe_loc + stripe_width
                )
            image[flower_mask & pattern_mask] = self.canvas.pattern_value
        elif pattern_type == "circular":
            circular_radius = max(0.0, float(self.canvas.circular_radius))
            pattern_mask = rr <= circular_radius
            image[flower_mask & pattern_mask] = self.canvas.pattern_value

        image[rr <= self.flower.r0] = self.canvas.hole_value
        return image

    def get_visual_patch(self, head_xy: np.ndarray | None = None) -> np.ndarray:
        """
        Return a head-centered visual crop, padding out-of-canvas samples with
        canvas.background_value.
        """
        if head_xy is None:
            head_xy = self.head_xy
        head_xy = np.asarray(head_xy, dtype=np.float32)
        canvas_img = self.canvas_image
        patch_h, patch_w = self.visual_patch_shape
        half_size = float(self.visual_field_size) / 2.0

        xs = head_xy[0] - half_size + (np.arange(patch_w, dtype=np.float32) + 0.5) * (
            self.visual_field_size / patch_w
        )
        ys = head_xy[1] - half_size + (np.arange(patch_h, dtype=np.float32) + 0.5) * (
            self.visual_field_size / patch_h
        )
        xi = np.floor(xs).astype(int)
        yi = np.floor(ys).astype(int)

        patch = np.full((patch_h, patch_w), self.canvas.background_value, dtype=np.float32)
        valid_x = (xi >= 0) & (xi < canvas_img.shape[1])
        valid_y = (yi >= 0) & (yi < canvas_img.shape[0])
        if np.any(valid_x) and np.any(valid_y):
            patch[np.ix_(valid_y, valid_x)] = canvas_img[np.ix_(yi[valid_y], xi[valid_x])]
        return patch

    def render(self):
        image = self.get_canvas_image()
        rgb = np.repeat(image[..., None], 3, axis=-1)
        return rgb

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._step_count = 0
        self.flower_center_xy = self._sample_flower_center()
        self.tip_xy = self._sample_start_tip()
        self.head_xy = self._sample_start_head(self.tip_xy)
        self.tip_xy = self._constrain_tip_to_head(self.tip_xy, self.head_xy)
        self.prev_xy = self.tip_xy.copy()
        self.tip_depth = 1.0
        self._schedule_non_stationary_change()
        self.canvas_image = self.get_canvas_image()

        obs = self._get_obs()
        info = {
            "flower_center_xy": self._normalize_canvas_xy(self.flower_center_xy),
            "head_xy": self._normalize_canvas_xy(self.head_xy),
            "tip_xy": self._normalize_canvas_xy(self.tip_xy),
            "tip_depth": float(self.tip_depth),
            **self._non_stationary_info(changed_this_step=False),
        }
        return obs, info

    def _get_mech_obs(self) -> np.ndarray:
        local_xy = self._to_flower_xy(self.tip_xy)
        r = float(np.hypot(local_xy[0], local_xy[1]))

        in_rim = r <= self.flower.R
        in_hole = self._inside_hole(self.tip_xy)
        pressing = float(self.tip_depth) > 0
        contact = in_rim and (not in_hole) and pressing

        if not contact:
            return np.zeros(5, dtype=np.float32)

        nxy = self._surface_normal_xy(self.tip_xy)
        penetration = max(0.0, float(self.tip_depth))
        penetration = min(penetration, self.max_press_depth)
        fn = self.flower.k * penetration
        fn = fn / (self.flower.k * self.max_press_depth)
        fn = np.log1p(fn)

        delta = self.tip_xy - self.prev_xy
        tang = delta - (delta @ nxy) * nxy if np.linalg.norm(nxy) > 0 else delta
        max_slip = float(self.max_lateral_delta / (self.dt + 1e-8))
        slip = float(np.linalg.norm(tang) / (self.dt + 1e-8)) / max_slip
        slip = np.log1p(slip)

        return np.array([1.0, fn, float(nxy[0]), float(nxy[1]), slip], dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "mech": self._get_mech_obs(),
            "vision": self.get_visual_patch(self.head_xy),
        }

    def step(self, action: np.ndarray):
        self._step_count += 1
        changed_this_step = (
            not self._non_stationary_change_applied
            and self._non_stationary_change_timestep == self._step_count
        )
        if changed_this_step:
            self._apply_non_stationary_change()

        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self.prev_xy = self.tip_xy.copy()

        self.head_xy = self._constrain_head_to_canvas(self._denormalize_canvas_xy(a[:2]))
        proposed_xy = self._denormalize_canvas_xy(a[2:4])
        proposed_xy = self._constrain_tip_to_head(proposed_xy, self.head_xy)
        self.tip_depth = float(np.clip(a[4], self.min_depth, self.max_depth))
        new_xy, contact_info = self._step_contact(proposed_xy)
        new_xy = self._constrain_tip_to_head(new_xy, self.head_xy)
        self.tip_xy = new_xy

        success = self._inside_hole(self.tip_xy) and (self.tip_depth >= self.flower.depth_success)
        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps)
        reward = 10.0 if success else -0.01

        obs = self._get_obs()
        info = {
            "success": success,
            **contact_info,
            "flower_center_xy": self._normalize_canvas_xy(self.flower_center_xy),
            "head_xy": self._normalize_canvas_xy(self.head_xy),
            "tip_xy": self._normalize_canvas_xy(self.tip_xy),
            "tip_depth": float(self.tip_depth),
            **self._non_stationary_info(changed_this_step=changed_this_step),
        }
        return obs, reward, terminated, truncated, info

    def peek_obs(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return the observation that would result from taking `action`,
        without advancing time or changing flower placement.
        """
        tip_xy0 = self.tip_xy.copy()
        head_xy0 = self.head_xy.copy()
        prev_xy0 = self.prev_xy.copy()
        tip_depth0 = float(self.tip_depth)
        step_count0 = int(self._step_count)

        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self.prev_xy = self.tip_xy.copy()
        self.head_xy = self._constrain_head_to_canvas(self._denormalize_canvas_xy(a[:2]))
        proposed_xy = self._denormalize_canvas_xy(a[2:4])
        proposed_xy = self._constrain_tip_to_head(proposed_xy, self.head_xy)
        self.tip_depth = float(np.clip(a[4], self.min_depth, self.max_depth))
        new_xy, _ = self._step_contact(proposed_xy)
        new_xy = self._constrain_tip_to_head(new_xy, self.head_xy)
        self.tip_xy = new_xy

        obs = self._get_obs()

        self.tip_xy = tip_xy0
        self.head_xy = head_xy0
        self.prev_xy = prev_xy0
        self.tip_depth = tip_depth0
        self._step_count = step_count0

        return obs


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import matplotlib.pyplot as plt
    from utils.plotting import interactive_allocentric_observation_viewer


    fp = FlowerParams(c=-1000)
    env = AllocentricNectarSearchEnv(flower=fp, seed=2026)
    obs, info = env.reset()
    print("obs:", obs)
    print("flower_center_xy:", info["flower_center_xy"])
    print("tip_xy:", info["tip_xy"])
    print("canvas image shape:", env.get_canvas_image().shape)

    canvas_img = env.get_canvas_image()
    tip_xy = env._denormalize_canvas_xy(info['tip_xy'])
    head_xy = env._denormalize_canvas_xy(info['head_xy'])

    plt.imshow(canvas_img, cmap='gray')
    plt.plot(tip_xy[0], tip_xy[1], 'rx')
    plt.plot(head_xy[0], head_xy[1], 'bo')
    plt.show()

    interactive_allocentric_observation_viewer(env)

    print("Done!")
