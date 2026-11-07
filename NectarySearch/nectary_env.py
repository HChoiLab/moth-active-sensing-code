from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


@dataclass
class FlowerParams:
    # JEB-inspired geometry parameters (mm)
    r0: float = 2.0      # hole radius
    R: float = 25.0      # corolla radius
    L: float = 25.0      # flower length scale
    c: float = -1.0      # curvature: -1 funnel, -2 near-funnel, -3 near-flat, np.inf flat

    # Contact / friction
    k: float = 5.0       # spring constant for penetration -> force (arbitrary units)
    mu: float = 0.6      # friction coefficient

    # Success
    depth_success: float = 3.0  # required insertion depth once in hole (mm)

class NectarSearchEnv(gym.Env):
    """
    Simplified nectary search: proboscis tip explores a corolla surface to find a small hole (nectary).
    Units are millimeters. The agent is assumed to be hover-stabilized; only tip motion is modeled.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        flower: FlowerParams | None = None,
        dt: float = 0.02,
        max_steps: int = 500,
        action_scale_xy: float = 1.0,   # mm per step
        action_scale_d: float = 1.0,    # mm per step
        max_press_depth: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.flower = flower or FlowerParams()
        self.dt = dt
        self.max_steps = max_steps
        self.action_scale_xy = action_scale_xy
        self.action_scale_d = action_scale_d
        self.np_random = np.random.default_rng(seed)

        # Constraints on maximum insertion depth and maximum displacement
        self.max_press_depth = max_press_depth
        self.max_lateral_delta = math.sqrt(2) * self.action_scale_xy

        # Action: dx, dy, dd in [-1, 1] scaled to mm.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation
        # mech: contact, fn, nx, ny, slip
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self._step_count = 0
        self.tip_xy = np.zeros(2, dtype=np.float32)
        self.tip_depth = 0.0  # insertion depth along local normal-ish direction (mm), simplified
        self.prev_xy = np.zeros(2, dtype=np.float32)

    # ------------------------
    # Geometry: surface z(r)
    # ------------------------
    def _z_surface(self, r: float) -> float:
        """Axisymmetric corolla height z as function of radius r."""

        f = self.flower
        if np.isinf(f.c):
            return 0.0  # flat surface

        # u in [0,1]
        u = (r - f.r0) / f.R
        u = float(np.clip(u, 0.0, 1.0))         # Flower surface model only applies when within flower/corolla radius 

        p = float(np.exp(f.c))  # exponent

        # z = L * u^p
        return float(f.L * (u ** p))


    def _dz_dr(self, r: float) -> float:
        """Derivative dz/dr for analytic normal."""
        f = self.flower
        if np.isinf(f.c):
            return 0.0  # flat surface

        # u in [0,1]
        u = (r - f.r0) / f.R

        # Outside the active corolla region, z is constant -> slope 0
        if u <= 0.0 or u >= 1.0:
            return 0.0

        u = float(u)
        p = float(np.exp(f.c))

        # dz/dr = (L/R) * p * u^(p-1)
        # Guard: if p < 1, u^(p-1) can blow up near u->0; but we already returned 0 for u<=0.
        return float((f.L / f.R) * p * (u ** (p - 1.0)))


    def _surface_normal_xy(self, xy: np.ndarray) -> np.ndarray:
        """
        Return normal projection onto xy-plane: (nx, ny) normalized in xy.
        For a heightfield z(r), normal ~ (-dz/dx, -dz/dy, 1).
        """
        x, y = float(xy[0]), float(xy[1])
        r = np.hypot(x, y)
        if r < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        dzdr = self._dz_dr(r)
        # dz/dx = dz/dr * x/r ; dz/dy = dz/dr * y/r
        dzdx = dzdr * x / r
        dzdy = dzdr * y / r
        # xy components of normal: (-dzdx, -dzdy)
        nxy = np.array([-dzdx, -dzdy], dtype=np.float32)
        n = np.linalg.norm(nxy) + 1e-8
        return nxy / n

    # ------------------------
    # Mechanics
    # ------------------------
    def _inside_hole(self, xy: np.ndarray) -> bool:
        return (np.hypot(float(xy[0]), float(xy[1])) <= self.flower.r0)

    def _step_contact(self, proposed_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Quasi-static contact: constrain to surface if penetrating; compute forces & slip proxy.
        We treat tip as a point that can slide on the surface.
        """
        f = self.flower
        r = np.hypot(float(proposed_xy[0]), float(proposed_xy[1]))

        # Keep within corolla radius
        if r > f.R:
            proposed_xy = proposed_xy * (f.R / (r + 1e-8))
            r = f.R

        # If inside hole region, no surface contact (tip can insert)
        if self._inside_hole(proposed_xy):
            return proposed_xy, dict(contact=0.0, fn=0.0, nx=0.0, ny=0.0, slip=0.0)

        # Not pressing the surface: no contact.
        if float(self.tip_depth) <= 0:
            return proposed_xy, dict(contact=0.0, fn=0.0, nx=0.0, ny=0.0, slip=0.0)

        # Otherwise: contact with surface
        nxy = self._surface_normal_xy(proposed_xy)

        # Penetration proxy: if agent tries to "go below" surface via depth, we penalize.
        # In this simplified 2.5D, depth should be near 0 outside hole; positive depth means pushing.
        penetration = max(0.0, float(self.tip_depth))  # pushing into surface
        penetration = min(float(self.tip_depth), self.max_press_depth)
        fn = f.k * penetration
        fn = fn / (f.k * self.max_press_depth)     # TODO: what do we need k for then?
        fn = np.log1p(fn)

        # Slip proxy: tangential speed relative to surface normal direction
        delta = proposed_xy - self.tip_xy
        # Tangential component magnitude (remove component along nxy)
        if np.linalg.norm(nxy) > 0:
            tang = delta - (delta @ nxy) * nxy
        else:
            tang = delta

        max_slip = float(self.max_lateral_delta / (self.dt + 1e-8))
        slip = float(np.linalg.norm(tang) / (self.dt + 1e-8))
        slip = slip / max_slip
        slip = np.log1p(slip)

        # Friction: optionally damp tangential movement if fn large
        # Simple: if tangential "demand" exceeds mu*fn, scale it down
        """
        tang_norm = np.linalg.norm(tang) + 1e-8
        max_tang = f.mu * fn
        if tang_norm > 0 and max_tang > 0:
            scale = min(1.0, max_tang / tang_norm)
            proposed_xy = self.tip_xy + (delta - tang + tang * scale)
        """

        return proposed_xy, dict(contact=1.0, fn=fn, nx=float(nxy[0]), ny=float(nxy[1]), slip=slip)

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._step_count = 0

        # Start near rim (like initial contact near outer corolla)
        f = self.flower
        r_start = self.np_random.uniform(0.7 * f.R, 0.95 * f.R)
        ang = self.np_random.uniform(-np.pi, np.pi)
        self.tip_xy = np.array([r_start * np.cos(ang), r_start * np.sin(ang)], dtype=np.float32)
        self.prev_xy = self.tip_xy.copy()

        # Start with no insertion depth
        self.tip_depth = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:

        # mech obs computed “on current state” (no new contact computation):
        # Treat as contact if not inside hole and within corolla

        f = self.flower
        r = float(np.hypot(self.tip_xy[0], self.tip_xy[1]))

        # Contact condition must match physics:
        in_rim = (r <= f.R)
        in_hole = self._inside_hole(self.tip_xy)
        pressing = (float(self.tip_depth) > 0)
        contact = (in_rim and (not in_hole) and pressing)

        if not contact:
            mech = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            nxy = self._surface_normal_xy(self.tip_xy)
            penetration = max(0.0, float(self.tip_depth))
            penetration = min(float(self.tip_depth), self.max_press_depth)
            fn = self.flower.k * penetration
            fn = fn / (self.flower.k * self.max_press_depth)
            fn = np.log1p(fn)
            # slip based on previous movement
            delta = self.tip_xy - self.prev_xy
            tang = delta - (delta @ nxy) * nxy if np.linalg.norm(nxy) > 0 else delta
            max_slip = float(self.max_lateral_delta / (self.dt + 1e-8))
            slip = float(np.linalg.norm(tang) / (self.dt + 1e-8)) / max_slip
            slip = np.log1p(slip)
            mech = np.array([1.0, fn, float(nxy[0]), float(nxy[1]), slip], dtype=np.float32)

        return mech

    def step(self, action: np.ndarray):
        self._step_count += 1
        a = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        dx = float(a[0] * self.action_scale_xy)
        dy = float(a[1] * self.action_scale_xy)
        dd = float(a[2] * self.action_scale_d)

        self.prev_xy = self.tip_xy.copy()

        # Update depth
        # Outside hole: depth tends to be rejected (interpreted as penetration -> force)
        # Inside hole: depth corresponds to insertion.
        self.tip_depth = float(np.clip(self.tip_depth + dd, -2.0, 10.0))

        proposed_xy = self.tip_xy + np.array([dx, dy], dtype=np.float32)

        # Contact resolution
        new_xy, contact_info = self._step_contact(proposed_xy)
        self.tip_xy = new_xy

        # Success criteria: inside hole and sufficient depth
        success = self._inside_hole(self.tip_xy) and (self.tip_depth >= self.flower.depth_success)

        terminated = bool(success)
        truncated = bool(self._step_count >= self.max_steps)

        # Reward: keep it generic; your agent may ignore it if training via uncertainty.
        # Provide a small step cost + big success bonus.
        reward = 10.0 if success else -0.01

        obs = self._get_obs()
        info = {
            "success": success,
            **contact_info,
            "tip_xy": self.tip_xy.copy(),
            "tip_depth": float(self.tip_depth),
        }
        return obs, reward, terminated, truncated, info
    
    # ----------------------------------------------
    # Helper functions for testing perception models
    # ----------------------------------------------
    def peek_obs(self, action: np.ndarray) -> np.ndarray:
        """
        Return the observation that would result from taking `action`,
        without advancing time (no persistent state change).
        """
        # Save state
        tip_xy0 = self.tip_xy.copy()
        prev_xy0 = self.prev_xy.copy()
        tip_depth0 = float(self.tip_depth)
        step_count0 = int(self._step_count)

        a = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        dx = float(a[0] * self.action_scale_xy)
        dy = float(a[1] * self.action_scale_xy)
        dd = float(a[2] * self.action_scale_d)

        # Apply as if a step, but restore afterwards
        self.prev_xy = self.tip_xy.copy()
        self.tip_depth = float(np.clip(self.tip_depth + dd, -2.0, 10.0))
        proposed_xy = self.tip_xy + np.array([dx, dy], dtype=np.float32)

        # Use the same contact resolution you use in step()
        new_xy, _ = self._step_contact(proposed_xy)
        self.tip_xy = new_xy

        obs = self._get_obs()

        # Restore state
        self.tip_xy = tip_xy0
        self.prev_xy = prev_xy0
        self.tip_depth = tip_depth0
        self._step_count = step_count0

        return obs



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from helpers import animate_proboscis_trajectory


    x = np.arange(-50, 50)
    y = np.arange(-50, 50)
    x, y = np.meshgrid(x, y)
    r = np.hypot(x, y)
    
    fp = FlowerParams()
    fp.c = -3.0
    env = NectarSearchEnv(flower=fp)

    #u = np.clip((r - fp.r0) / fp.R, 0.0, 1.0)
    #z = fp.L * u ** np.exp(fp.c)

    z = np.array([[env._z_surface(ry) for ry in rx] for rx in r])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()

    obs = []
    act = []
    tip_xy = []
    tip_depth = []

    obs.append(env.reset(seed=99)[0])
    tip_xy.append(env.tip_xy)
    tip_depth.append(env.tip_depth)
    for i in range(1, 200):
        action = env.action_space.sample()
        obs.append(env.step(action)[0])
        act.append(action)
        tip_xy.append(env.tip_xy)
        tip_depth.append(env.tip_depth)

    obs = np.array(obs)
    act = np.array(act)
    tip_xy = np.array(tip_xy)

    anim = animate_proboscis_trajectory(
        xy=tip_xy,  # (T,2)
        obs={
            "contact": obs[:, 0],  # (T,)
            "fn": obs[:, 1],
            "nx": obs[:, 2],
            "ny": obs[:, 3],
            "slip": obs[:, 4],  # optional
        },
        r0=fp.r0,
        R=fp.R,
        dt=0.02,
        save_path="demo_flat.mp4"
    )

    print(".")



