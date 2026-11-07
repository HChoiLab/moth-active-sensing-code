import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def straight_trajectory(R: float, T: int, start_frac: float = 0.95, end_r: float = 0.5, angle: float = 0.0):
    """
    Return xy trajectory from r=start_frac*R down to r=end_r along direction `angle`.
    """
    r_start = start_frac * R
    rs = np.linspace(r_start, end_r, T)
    xs = rs * np.cos(angle)
    ys = rs * np.sin(angle)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def animate_proboscis_trajectory(
    xy,                      # (T,2) tip positions in mm
    obs,                     # dict of arrays length T: contact, fn, nx, ny, slip (optional)
    r0=1.0,                  # nectary hole radius (mm)
    R=25.0,                  # flower rim radius (mm)
    dt=0.02,                 # seconds per step
    save_path=None,          # e.g. "traj.mp4" or "traj.gif" (optional)
    fps=30,
    trail=300,               # how many past points to show in the trajectory trail
    show=True                # plt.show() at end
):
    """
    Animated visualization:
      - Left: top-down flower with hole + rim, trajectory trail, current tip position
      - Right: observation time-series (contact, fn, |n_xy|, nx, ny, ft, slip) with moving time cursor

    Inputs
    ------
    xy: np.ndarray (T,2) tip x,y positions (mm), flower-centered frame
    obs: dict mapping str->np.ndarray (T,)
         Required: "contact", "fn", "nx", "ny"
         Optional: "ft" (tangential force magnitude), "slip"
    """

    xy = np.asarray(xy, dtype=float)
    assert xy.ndim == 2 and xy.shape[1] == 2, "xy must be (T,2)"
    T = xy.shape[0]

    # Required channels
    for k in ["contact", "fn", "nx", "ny"]:
        if k not in obs:
            raise ValueError(f"obs must contain key '{k}' (length T).")
        obs[k] = np.asarray(obs[k], dtype=float)
        assert obs[k].shape[0] == T, f"obs['{k}'] must have length T."

    # Optional channels
    for k in ["ft", "slip"]:
        if k in obs:
            obs[k] = np.asarray(obs[k], dtype=float)
            assert obs[k].shape[0] == T, f"obs['{k}'] must have length T."

    t = np.arange(T) * dt

    # Derived: normal direction angle and magnitude in xy
    nxy_mag = np.sqrt(obs["nx"]**2 + obs["ny"]**2)

    # Prepare the list of channels to plot on the right
    channels = [
        ("contact", obs["contact"]),
        ("fn", obs["fn"]),
        ("|n_xy|", nxy_mag),
        ("nx", obs["nx"]),
        ("ny", obs["ny"]),
    ]
    if "ft" in obs:
        channels.append(("ft", obs["ft"]))
    if "slip" in obs:
        channels.append(("slip", obs["slip"]))

    nrows = len(channels)

    # ---- Figure layout ----
    fig = plt.figure(figsize=(12, 2.2 * nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=2, width_ratios=[1.1, 2.0], wspace=0.25, hspace=0.65)

    # Left axis (trajectory) occupies all rows
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_title("Proboscis tip trajectory (top-down)")
    ax_traj.set_xlabel("x (mm)")
    ax_traj.set_ylabel("y (mm)")
    ax_traj.set_xlim(-1.1 * R, 1.1 * R)
    ax_traj.set_ylim(-1.1 * R, 1.1 * R)

    # Draw flower rim and hole
    rim = plt.Circle((0, 0), R, fill=False)
    hole = plt.Circle((0, 0), r0, fill=False)
    ax_traj.add_patch(rim)
    ax_traj.add_patch(hole)

    # Trajectory artists
    (traj_line,) = ax_traj.plot([], [], linewidth=1)
    (tip_point,) = ax_traj.plot([], [], marker="o", markersize=6, linestyle="None")

    # Normal vector arrow (quiver) at tip
    # We'll scale it for visibility; direction from nx,ny
    normal_quiv = ax_traj.quiver(
        [0], [0], [0], [0],
        angles="xy", scale_units="xy", scale=1.0, width=0.008
    )

    # Right axes: one per channel
    axes_ts = []
    lines_ts = []
    cursor_lines = []

    for i, (name, y) in enumerate(channels):
        ax = fig.add_subplot(gs[i, 1])
        ax.set_title(name)
        ax.plot(t, y, linewidth=1)  # full series (static)
        ax.set_xlim(t[0], t[-1])
        # y-limits: add a small padding
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        if np.isfinite(ymin) and np.isfinite(ymax):
            pad = 0.05 * (ymax - ymin + 1e-9)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_ylabel(name)
        if i == nrows - 1:
            ax.set_xlabel("time (s)")

        # moving cursor
        cursor = ax.axvline(t[0], linewidth=1)
        axes_ts.append(ax)
        cursor_lines.append(cursor)

    # Helper: update quiver arrow for normal
    def _set_normal_arrow(i):
        # Draw a short arrow at the tip in direction (nx,ny).
        # If contact==0, hide arrow.
        x, y = xy[i]
        if obs["contact"][i] <= 0.0 or (obs["nx"][i] == 0 and obs["ny"][i] == 0):
            # zero-length arrow
            normal_quiv.set_offsets([[x, y]])
            normal_quiv.set_UVC([0.0], [0.0])
            return

        # Scale arrow length for visibility relative to flower size.
        # Use a fixed length; direction comes from nx,ny.
        arrow_len = 0.1 * R
        u = arrow_len * obs["nx"][i]
        v = arrow_len * obs["ny"][i]
        normal_quiv.set_offsets([[x, y]])
        normal_quiv.set_UVC([u], [v])

    # Init
    def init():
        traj_line.set_data([], [])
        tip_point.set_data([], [])
        _set_normal_arrow(0)
        for cur in cursor_lines:
            cur.set_xdata([t[0], t[0]])
        return [traj_line, tip_point, normal_quiv, *cursor_lines]

    # Update frame
    def update(frame):
        i = int(frame)
        start = max(0, i - trail)

        # Update trajectory trail and current point
        traj_line.set_data(xy[start:i+1, 0], xy[start:i+1, 1])
        tip_point.set_data([xy[i, 0]], [xy[i, 1]])

        # Update normal arrow
        _set_normal_arrow(i)

        # Update time cursors
        ti = t[i]
        for cur in cursor_lines:
            cur.set_xdata([ti, ti])

        return [traj_line, tip_point, normal_quiv, *cursor_lines]

    interval_ms = int(1000 * dt)
    anim = FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=interval_ms, blit=True, repeat=False
    )

    # Save if requested
    if save_path is not None:
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)

    if show:
        plt.show()

    return anim
