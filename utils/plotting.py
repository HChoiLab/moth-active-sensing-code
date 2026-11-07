import numpy as np
import matplotlib.pyplot as plt
from utils.analysis import compute_population_stats


def interactive_allocentric_observation_viewer(
    env,
    depth=None,
    max_points: int = 400,
    trace_cursor: bool = True,
    show: bool = True,
):
    """
    Visualize allocentric observations interactively from cursor position.

    Move the cursor over the canvas to append proboscis-tip observations to the
    time-series. Hold the left mouse button to use the depth slider value;
    release it to sample with depth zero. Right-click to move the head position
    used for visual sampling. When trace_cursor is True, slip is computed from
    the previous cursor sample to the current one, while the environment state
    is still restored after every sample.

    Parameters
    ----------
    env:
        Reset AllocentricNectarSearchEnv instance to visualize.
    depth:
        Initial insertion/pressing depth used while the left mouse button is
        pressed. Defaults to the env's current depth.
    max_points:
        Maximum number of cursor samples retained in the time-series.
    trace_cursor:
        If True, interpret cursor motion as a temporary trajectory for slip.
        If False, each cursor sample uses env.peek_obs independently.
    show:
        If True, call matplotlib.pyplot.show().

    Returns
    -------
    tuple
        (fig, state) where state stores sampled xy positions, depths, and
        observation arrays.
    """
    from matplotlib.widgets import Button, Slider

    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    if depth is None:
        depth = float(env.tip_depth)
    depth = float(np.clip(depth, env.min_depth, env.max_depth))

    obs_names = ("contact", "fn", "nx", "ny", "slip")
    canvas_img = env.get_canvas_image()
    initial_vision = env.get_visual_patch()

    state = {
        "xy": [],
        "head_xy": env.head_xy.copy(),
        "depth": [],
        "obs": [],
        "vision": initial_vision.copy(),
        "last_xy": env.tip_xy.copy(),
        "left_pressed": False,
        "widgets": {},
    }

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        nrows=len(obs_names),
        ncols=3,
        width_ratios=[1.0, 0.8, 1.5],
        hspace=0.35,
        wspace=0.28,
        bottom=0.16,
    )

    ax_canvas = fig.add_subplot(gs[:, 0])
    ax_canvas.imshow(
        canvas_img,
        cmap="gray",
        origin="upper",
        extent=(0.0, env.canvas.width, env.canvas.height, 0.0),
    )
    ax_canvas.set_title("Move cursor over canvas")
    ax_canvas.set_xlabel("canvas x")
    ax_canvas.set_ylabel("canvas y")
    ax_canvas.set_xlim(0.0, env.canvas.width)
    ax_canvas.set_ylim(env.canvas.height, 0.0)
    ax_canvas.set_aspect("equal", adjustable="box")

    (tip_point,) = ax_canvas.plot([], [], "ro", markersize=5, linestyle="None", label="tip")
    (head_point,) = ax_canvas.plot(
        [state["head_xy"][0]],
        [state["head_xy"][1]],
        "bo",
        markersize=5,
        linestyle="None",
        label="head",
    )
    (cursor_trace,) = ax_canvas.plot([], [], "r-", linewidth=1, alpha=0.6)
    ax_canvas.legend(loc="upper right")

    ax_visual = fig.add_subplot(gs[:2, 1])
    visual_img = ax_visual.imshow(
        initial_vision,
        cmap="gray",
        origin="upper",
        vmin=float(np.nanmin(canvas_img)),
        vmax=float(np.nanmax(canvas_img)),
    )
    ax_visual.set_title("Head-centered visual patch")
    ax_visual.set_xticks([])
    ax_visual.set_yticks([])

    ts_axes = []
    ts_lines = []
    for i, name in enumerate(obs_names):
        ax = fig.add_subplot(gs[i, 2], sharex=ts_axes[0] if ts_axes else None)
        (line,) = ax.plot([], [], linewidth=1.2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_title("Observation time series")
        if i == len(obs_names) - 1:
            ax.set_xlabel("sample")
        ts_axes.append(ax)
        ts_lines.append(line)

    ax_depth = fig.add_axes([0.15, 0.07, 0.58, 0.035])
    depth_slider = Slider(
        ax=ax_depth,
        label="depth",
        valmin=float(env.min_depth),
        valmax=float(env.max_depth),
        valinit=depth,
    )

    ax_clear = fig.add_axes([0.78, 0.055, 0.1, 0.055])
    clear_button = Button(ax_clear, "Clear")

    status = fig.text(
        0.15,
        0.025,
        "Hold left click to use slider depth; right-click to move head.",
    )

    def _trim_history():
        if len(state["obs"]) <= max_points:
            return
        drop = len(state["obs"]) - max_points
        del state["xy"][:drop]
        del state["depth"][:drop]
        del state["obs"][:drop]

    def _restore_env(head_xy, tip_xy, prev_xy, tip_depth, step_count):
        env.head_xy = head_xy
        env.tip_xy = tip_xy
        env.prev_xy = prev_xy
        env.tip_depth = tip_depth
        env._step_count = step_count

    def _sample_obs(canvas_xy, depth_value):
        head_xy0 = env.head_xy.copy()
        tip_xy0 = env.tip_xy.copy()
        prev_xy0 = env.prev_xy.copy()
        tip_depth0 = float(env.tip_depth)
        step_count0 = int(env._step_count)

        try:
            head_norm = env._normalize_canvas_xy(state["head_xy"])
            tip_norm = env._normalize_canvas_xy(canvas_xy)
            action = np.array(
                [
                    head_norm[0],
                    head_norm[1],
                    tip_norm[0],
                    tip_norm[1],
                    depth_value,
                ],
                dtype=np.float32,
            )

            if not trace_cursor:
                obs = env.peek_obs(action)
                env.tip_depth = float(np.clip(depth_value, env.min_depth, env.max_depth))
                env.head_xy = env._constrain_head_to_canvas(env._denormalize_canvas_xy(action[:2]))
                proposed_xy = env._denormalize_canvas_xy(action[2:4])
                proposed_xy = env._constrain_tip_to_head(proposed_xy, env.head_xy)
                realized_xy, _ = env._step_contact(proposed_xy)
                realized_xy = env._constrain_tip_to_head(realized_xy, env.head_xy)
                return obs, realized_xy.copy()

            env.head_xy = np.asarray(state["head_xy"], dtype=np.float32).copy()
            env.tip_xy = np.asarray(state["last_xy"], dtype=np.float32).copy()
            env.prev_xy = env.tip_xy.copy()
            env.tip_depth = float(np.clip(depth_value, env.min_depth, env.max_depth))

            proposed_xy = env._denormalize_canvas_xy(action[2:4])
            proposed_xy = env._constrain_tip_to_head(proposed_xy, env.head_xy)
            new_xy, _ = env._step_contact(proposed_xy)
            new_xy = env._constrain_tip_to_head(new_xy, env.head_xy)
            env.tip_xy = new_xy
            obs = env._get_obs()
            return obs, new_xy.copy()
        finally:
            _restore_env(head_xy0, tip_xy0, prev_xy0, tip_depth0, step_count0)

    def _refresh_plot():
        if state["obs"]:
            xy = np.asarray(state["xy"], dtype=float)
            obs = np.asarray(state["obs"], dtype=float)
            x = np.arange(obs.shape[0])

            tip_point.set_data([xy[-1, 0]], [xy[-1, 1]])
            cursor_trace.set_data(xy[:, 0], xy[:, 1])
            visual_img.set_data(state["vision"])

            for i, line in enumerate(ts_lines):
                line.set_data(x, obs[:, i])

            for i, ax in enumerate(ts_axes):
                y = obs[:, i]
                ymin = float(np.nanmin(y))
                ymax = float(np.nanmax(y))
                pad = 0.05 * (ymax - ymin + 1e-9)
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.set_xlim(max(0, len(state["obs"]) - max_points), max(1, len(state["obs"]) - 1))

            depth_value = state["depth"][-1]
            status.set_text(
                f"head=({state['head_xy'][0]:.1f}, {state['head_xy'][1]:.1f}), "
                f"tip=({xy[-1, 0]:.1f}, {xy[-1, 1]:.1f}), depth={depth_value:.2f}, "
                f"mech={obs[-1].round(3).tolist()}"
            )
        else:
            tip_point.set_data([], [])
            cursor_trace.set_data([], [])
            visual_img.set_data(state["vision"])
            for line in ts_lines:
                line.set_data([], [])
            for ax in ts_axes:
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.05, 1.05)
            status.set_text(
                "Hold left click to use slider depth; right-click to move head."
            )

        head_point.set_data([state["head_xy"][0]], [state["head_xy"][1]])
        fig.canvas.draw_idle()

    def _event_canvas_xy(event):
        if event.inaxes is not ax_canvas or event.xdata is None or event.ydata is None:
            return None

        canvas_xy = np.array([event.xdata, event.ydata], dtype=np.float32)
        canvas_xy[0] = np.clip(canvas_xy[0], 0.0, env.canvas.width)
        canvas_xy[1] = np.clip(canvas_xy[1], 0.0, env.canvas.height)
        return canvas_xy

    def _append_sample(canvas_xy):
        active_depth = float(depth_slider.val) if state["left_pressed"] else 0.0
        obs, realized_xy = _sample_obs(canvas_xy, active_depth)
        state["xy"].append(realized_xy)
        state["depth"].append(active_depth)
        state["obs"].append(obs["mech"] if isinstance(obs, dict) else obs)
        state["vision"] = obs["vision"].copy() if isinstance(obs, dict) else env.get_visual_patch(state["head_xy"])
        state["last_xy"] = realized_xy.copy()
        _trim_history()
        _refresh_plot()

    def _on_motion(event):
        canvas_xy = _event_canvas_xy(event)
        if canvas_xy is None:
            return
        _append_sample(canvas_xy)

    def _on_button_press(event):
        canvas_xy = _event_canvas_xy(event)
        if canvas_xy is None:
            return
        if event.button == 3:
            state["head_xy"] = env._constrain_head_to_canvas(canvas_xy)
            state["vision"] = env.get_visual_patch(state["head_xy"])
            _append_sample(state["last_xy"])
            return
        if event.button != 1:
            return
        state["left_pressed"] = True
        _append_sample(canvas_xy)

    def _on_button_release(event):
        if event.button != 1:
            return
        state["left_pressed"] = False
        canvas_xy = _event_canvas_xy(event)
        if canvas_xy is not None:
            _append_sample(canvas_xy)

    def _on_clear(event):
        state["xy"].clear()
        state["depth"].clear()
        state["obs"].clear()
        state["vision"] = env.get_visual_patch(state["head_xy"])
        state["last_xy"] = env.tip_xy.copy()
        state["left_pressed"] = False
        _refresh_plot()

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_press_event", _on_button_press)
    fig.canvas.mpl_connect("button_release_event", _on_button_release)
    clear_button.on_clicked(_on_clear)
    state["widgets"] = {"depth_slider": depth_slider, "clear_button": clear_button}
    _refresh_plot()

    if show:
        plt.show()

    return fig, state


def half_violinplot(ax, data, positions, side='right', color='b', alpha=1, width=0.6):
    """
    Creates a half violin plot on the given axis 'ax'.
    'data' is the array of data points.
    'positions' is the positions on the x-axis.
    'side' specifies which side to plot the violin ('left' or 'right').
    """
    # Create a full violin plot but display only half by modifying the paths
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=True
    )
    for b in vp['bodies']:
        # Get the center of the violin
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify the vertices to display only half of the violin
        if side == 'right':
            b.get_paths()[0].vertices[:, 0] = np.clip(
                b.get_paths()[0].vertices[:, 0], m, np.inf
            )
        elif side == 'left':
            b.get_paths()[0].vertices[:, 0] = np.clip(
                b.get_paths()[0].vertices[:, 0], -np.inf, m
            )
        else:
            raise ValueError("side should be 'left' or 'right'")
        b.set_facecolor(color)
        b.set_edgecolor('none')
        b.set_alpha(alpha)

def plot_raincloud(*arrays, labels=None, y_label=None, title=None, ylim=None, figsz=None, fliers=False):
    """
    Plot a raincloud plot with half violin and half box plot for each array provided.

    Parameters:
    *arrays: multiple arrays, each of shape (N,)
    """
    n_arrays = len(arrays)
    fig, ax = plt.subplots(figsize=figsz if figsz is not None else (n_arrays * 3.5, 5))
    for i, data in enumerate(arrays):
        x_pos = i + 1  # Position along the x-axis
        data = np.array(data)
        color = f"C{i % 10}"  # Cycle through the default color cycle
        violin_offset = -0.15  # Shift the violin plot to the left
        box_offset = 0.15      # Shift the box plot to the right

        # Half violin plot on the left
        half_violinplot(
            ax,
            data,
            positions=[x_pos + violin_offset],
            side='left',
            color=color,
            alpha=0.5,
            width=0.6
        )

        # Box plot on the right
        ax.boxplot(
            data,
            positions=[x_pos + box_offset],
            showmeans=True,
            widths=0.1,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='white', color='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(color='black', markeredgecolor='black'),
            showfliers=fliers
        )

        # Scatter plot of individual data points (raindrops)
        jitter = np.random.uniform(-0.02, 0.02, size=len(data))
        ax.scatter(
            np.full_like(data, x_pos) + jitter,
            data,
            color=color,
            alpha=0.6,
            s=10
        )

    ax.set_xticks(range(1, n_arrays + 1))
    if labels is None:
        ax.set_xticklabels([f'Dataset {i+1}' for i in range(n_arrays)])
    else:
        ax.set_xticklabels([f'{label}' for label in labels])
    ax.set_ylabel('Value' if y_label is None else y_label)

    if ylim is not None:
        plt.ylim(ylim)

    if title is not None:
        plt.title(title)
    plt.tight_layout()

    return fig, ax

def plot_confidence_intervals(*arrays, labels=None, y_label=None, title=None, ylim=None, alpha=0.05):

    n_arrays = len(arrays)

    fig, ax = plt.subplots(figsize=(n_arrays * 2., 4))

    for i, data in enumerate(arrays):
        x_pos = i + 1  # Position along the x-axis
        data = np.array(data)
        color = f"C{i % 10}"  # Cycle through the default color cycle
        
        mean, err = compute_population_stats(data, alpha=alpha)

        ax.errorbar([x_pos], [mean], yerr=err, fmt='o', markersize=8, color=color,
                    ecolor=color, capsize=0, elinewidth=2)
        
    plt.xlim([0.5, n_arrays+0.5])
    ax.set_xticks(range(1, n_arrays+1))
    if labels is None:
        ax.set_xticklabels([f'Dataset {i+1}' for i in range(n_arrays)])
    else:
        ax.set_xticklabels([f'{label}' for label in labels])
    ax.set_ylabel('Value' if y_label is None else y_label)

    if ylim is not None:
        plt.ylim(ylim)

    if title is not None:
        plt.title(title)
    plt.tight_layout()

    return fig, ax

# Example usage:
if __name__ == "__main__":
    data1 = np.random.normal(0, 1, 200)
    data2 = np.random.normal(2, 0.5, 200)
    data3 = np.random.normal(-1, 0.8, 200)

    plot_raincloud(data1, data2, data3)
