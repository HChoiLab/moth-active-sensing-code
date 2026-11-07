import numpy as np
import torch

try:
    from uncertainty_reduction import (
        estimate_expected_information_gain,
        gaussian_entropy_from_logvar,
        infer_observation_posterior,
    )
except ImportError:
    from .uncertainty_reduction import (
        estimate_expected_information_gain,
        gaussian_entropy_from_logvar,
        infer_observation_posterior,
    )


def compute_entropy(logvar_z):
    return gaussian_entropy_from_logvar(logvar_z)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_sem(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def _weighted_variance(means, counts):
    means = np.asarray(means, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    keep = np.isfinite(means) & np.isfinite(counts) & (counts > 0)
    if keep.sum() <= 1:
        return 0.0
    means = means[keep]
    counts = counts[keep]
    center = np.sum(counts * means) / np.sum(counts)
    return float(np.sum(counts * (means - center) ** 2) / np.sum(counts))


def summarize_ur_heatmaps(
    heatmaps,
    times,
    X,
    Y,
    valid_mask,
    center_xy=None,
    n_radial_bins=8,
    n_angle_bins=16,
    eps=1e-12,
):
    """
    Summarize UR heatmaps as compact per-timestep statistics.

    Returns one record per heatmap with:
      mean_ur: average uncertainty reduction over valid map pixels
      radial_anisotropy: variance explained by radius-binned means
      angular_anisotropy: variance explained by angle-binned means

    The anisotropy scores are normalized by the full valid-pixel variance, so
    values near zero indicate weak structure along that coordinate.
    """
    H = _as_numpy(heatmaps).astype(np.float64, copy=False)
    X = _as_numpy(X).astype(np.float64, copy=False)
    Y = _as_numpy(Y).astype(np.float64, copy=False)
    valid_mask = _as_numpy(valid_mask).astype(bool, copy=False)
    times = list(times)

    if H.ndim != 3:
        raise ValueError("heatmaps must have shape (time, y, x).")
    if H.shape[1:] != X.shape or X.shape != Y.shape or X.shape != valid_mask.shape:
        raise ValueError("heatmaps, X, Y, and valid_mask shapes are inconsistent.")

    if center_xy is None:
        center_x = float(np.nanmean(X))
        center_y = float(np.nanmean(Y))
    else:
        center_x, center_y = map(float, center_xy)

    dx = X - center_x
    dy = Y - center_y
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    base_mask = valid_mask & np.isfinite(radius) & np.isfinite(angle)
    if not np.any(base_mask):
        return []

    r_values = radius[base_mask]
    r_edges = np.linspace(float(np.nanmin(r_values)), float(np.nanmax(r_values)), n_radial_bins + 1)
    if np.isclose(r_edges[0], r_edges[-1]):
        r_edges[-1] = r_edges[0] + 1.0
    theta_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)

    r_bin = np.digitize(radius, r_edges[1:-1], right=False)
    theta_bin = np.digitize(angle, theta_edges[1:-1], right=False)
    r_bin = np.clip(r_bin, 0, n_radial_bins - 1)
    theta_bin = np.clip(theta_bin, 0, n_angle_bins - 1)

    records = []
    for idx, Ht in enumerate(H):
        mask = base_mask & np.isfinite(Ht)
        values = Ht[mask]
        if values.size == 0:
            mean_ur = np.nan
            peak_ur = np.nan
            total_var = np.nan
            radial_anisotropy = np.nan
            angular_anisotropy = np.nan
        else:
            mean_ur = float(np.nanmean(values))
            peak_ur = float(np.nanmax(values))
            total_var = float(np.nanvar(values))
            if total_var <= eps:
                radial_anisotropy = 0.0
                angular_anisotropy = 0.0
            else:
                radial_means = np.full(n_radial_bins, np.nan, dtype=np.float64)
                radial_counts = np.zeros(n_radial_bins, dtype=np.float64)
                angular_means = np.full(n_angle_bins, np.nan, dtype=np.float64)
                angular_counts = np.zeros(n_angle_bins, dtype=np.float64)

                for bin_i in range(n_radial_bins):
                    bin_mask = mask & (r_bin == bin_i)
                    radial_counts[bin_i] = int(np.count_nonzero(bin_mask))
                    if radial_counts[bin_i] > 0:
                        radial_means[bin_i] = float(np.nanmean(Ht[bin_mask]))

                for bin_i in range(n_angle_bins):
                    bin_mask = mask & (theta_bin == bin_i)
                    angular_counts[bin_i] = int(np.count_nonzero(bin_mask))
                    if angular_counts[bin_i] > 0:
                        angular_means[bin_i] = float(np.nanmean(Ht[bin_mask]))

                radial_anisotropy = _weighted_variance(radial_means, radial_counts) / total_var
                angular_anisotropy = _weighted_variance(angular_means, angular_counts) / total_var

        records.append(
            {
                "time": int(times[idx]) if idx < len(times) else int(idx),
                "mean_ur": mean_ur,
                "peak_ur": peak_ur,
                "radial_anisotropy": float(radial_anisotropy),
                "angular_anisotropy": float(angular_anisotropy),
                "valid_pixel_count": int(np.count_nonzero(mask)),
            }
        )
    return records


def aggregate_summary_records(records, group_keys, value_key):
    """
    Aggregate list-of-dict summary records without requiring pandas.
    """
    grouped = {}
    for record in records:
        key = tuple(record[k] for k in group_keys)
        grouped.setdefault(key, []).append(record.get(value_key, np.nan))

    summaries = []
    for key, values in grouped.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        out = {name: key_i for name, key_i in zip(group_keys, key)}
        out["n"] = int(values.size)
        out[f"{value_key}_mean"] = float(np.nanmean(values)) if values.size else np.nan
        out[f"{value_key}_sem"] = _safe_sem(values)
        summaries.append(out)
    return summaries


def run_radial_with_ur_heatmaps(
    env,
    model,
    T=200,
    grid_n=50,
    depth_fixed=0.5,
    radial_step_mm=1.0,
    angle_step_rad=1 / (2 * np.pi),
    angle=0.0,
    dd_action=0.0,
    start_r_frac=0.95,
    end_r=2.0,
    compute_heatmap_every=1,
    path="radial",
    device="cuda",
):
    """
    Returns:
      H: (K, grid_n, grid_n) heatmaps computed at selected timesteps
      times: list of t indices corresponding to heatmaps
      traj_xy: (T,2) realized trajectory
      traj_obs: list of observations along the realized trajectory
    """

    f = env.flower
    R, r0 = float(f.R), float(f.r0)

    xs = np.linspace(-R, R, grid_n, dtype=np.float32)
    ys = np.linspace(-R, R, grid_n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rr = np.sqrt(X**2 + Y**2)
    valid_mask = (rr <= R) & (rr > r0)

    _, _ = env.reset()
    env.tip_depth = float(depth_fixed)

    if path == "circular":
        angle = 0.0
    r_start = start_r_frac * R
    env.tip_xy = np.array([r_start * np.cos(angle), r_start * np.sin(angle)], dtype=np.float32)
    env.prev_xy = env.tip_xy.copy()
    x = env._get_obs().reshape(1, -1)

    traj_xy = np.zeros((T, 2), dtype=np.float32)
    traj_obs = []

    heatmaps = []
    times = []

    x = torch.tensor(x).reshape(1, -1).to(device)
    traj_xy[0] = env.tip_xy.copy()
    traj_obs.append(x)
    responses0, _ = model.forward_t(x, None, torch.zeros(1, model.hidden_dim).to(x), t=0)
    h_m_1 = responses0["h"]

    for t in range(1, T):
        if ((t - 1) % compute_heatmap_every) == 0:
            Ht = torch.full((grid_n, grid_n), torch.nan, dtype=torch.float32).to(device)

            for i in range(grid_n):
                for j in range(grid_n):
                    if not valid_mask[i, j]:
                        continue

                    target_xy = np.array([X[i, j], Y[i, j]], dtype=np.float32)
                    dxdy = target_xy - env.tip_xy

                    ax = float(dxdy[0] / env.action_scale_xy)
                    ay = float(dxdy[1] / env.action_scale_xy)
                    ad = 0.0
                    a = np.array([ax, ay, ad], dtype=np.float32)

                    x_peek = env.peek_obs(a)

                    a = torch.tensor(a).reshape(1, -1).to(device)
                    x_peek = torch.tensor(x_peek).reshape(1, -1).to(device)
                    prior_mu, prior_logvar = model.prior_t(h_m_1, a)
                    post_mu, post_logvar = model.encode_t(x_peek, a, h_m_1)

                    Ht[i, j] = (
                        compute_entropy(prior_logvar) - compute_entropy(post_logvar)
                    ).sum(-1).mean().item()

            heatmaps.append(Ht)
            times.append(t)

        r = float(np.hypot(env.tip_xy[0], env.tip_xy[1]))
        theta = np.arctan2(env.tip_xy[1], env.tip_xy[0])
        if r <= max(end_r, r0 + 1e-3):
            break
        if theta >= 2 * np.pi:
            break

        if path == "radial":
            toward_center = -env.tip_xy / (r + 1e-8)
            step_vec = toward_center * radial_step_mm
        else:
            tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
            step_vec = tangent * (r * angle_step_rad)

        ax = float(step_vec[0] / env.action_scale_xy)
        ay = float(step_vec[1] / env.action_scale_xy)
        ad = float(dd_action / env.action_scale_d)
        a_star = np.array([ax, ay, ad], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(a_star)
        traj_obs.append(obs)
        traj_xy[t] = env.tip_xy.copy()

        obs = torch.tensor(obs).reshape(1, -1).to(device)
        a_star = torch.tensor(a_star).reshape(1, -1).to(device)
        responses, _ = model.forward_t(obs, a_star, h_m_1)
        h_m_1 = responses["h"]

        if terminated or truncated:
            break

    H = torch.stack(heatmaps, dim=0) if heatmaps else torch.zeros((0, grid_n, grid_n), dtype=torch.float32)
    return H, times, traj_xy[: len(traj_obs) + 1], traj_obs, (X, Y, valid_mask)


def _allocentric_obs_to_tensors(
    obs,
    device,
    blank_mechanosensory=False,
    blank_visual=False,
):
    mech = torch.as_tensor(obs["mech"], dtype=torch.float32, device=device).reshape(1, -1)
    vis = torch.as_tensor(obs["vision"], dtype=torch.float32, device=device)
    if vis.ndim == 2:
        vis = vis.unsqueeze(0)
    vis = vis.unsqueeze(0)
    if blank_mechanosensory:
        mech = torch.zeros_like(mech)
    if blank_visual:
        vis = torch.zeros_like(vis)
    return mech, vis


def _allocentric_action_from_xy(env, head_xy, tip_xy, depth):
    head_norm = env._normalize_canvas_xy(head_xy)
    tip_norm = env._normalize_canvas_xy(tip_xy)
    return np.array(
        [head_norm[0], head_norm[1], tip_norm[0], tip_norm[1], depth],
        dtype=np.float32,
    )


def _allocentric_counterfactual_ur(
    model,
    obs,
    action,
    h_m_1,
    device,
    blank_mechanosensory=False,
    blank_visual=False,
    modality='joint'
):
    x_mech_t, x_vis_t = _allocentric_obs_to_tensors(
        obs,
        device,
        blank_mechanosensory=blank_mechanosensory,
        blank_visual=blank_visual,
    )
    a_t = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    _, post_logvar, _ = infer_observation_posterior(
        model,
        x_mech_t,
        x_vis_t,
        a_t,
        h_m_1,
        modality=modality
    )
    prior_logvar = torch.zeros_like(post_logvar)
    return (compute_entropy(prior_logvar) - compute_entropy(post_logvar)).sum(-1).mean()

def _expected_ur(
    model,
    responses,
    action,
    device,
    blank_mechanosensory=False,
    blank_visual=False,
):
    a_t = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
    return estimate_expected_information_gain(
        model=model,
        post_logvar=responses['post_logvar'],
        h_t=responses['h'],
        c_t=responses.get('c'),
        planning_state=responses['s'],
        action=a_t,
        num_samples=50,
        action_l2=0.0,
        blank_mechanosensory=blank_mechanosensory,
        blank_visual=blank_visual,
    )['information_gain']

def run_allocentric_with_ur_heatmaps(
    env,
    model,
    T=200,
    grid_n=50,
    heatmap_square_size=None,
    depth_fixed=1.0,
    radial_step_mm=1.0,
    angle_step_rad=1 / (2 * np.pi),
    angle=0.0,
    circular_radius=None,
    circular_max_angle=2 * np.pi,
    start_r_frac=0.95,
    end_r=2.0,
    compute_heatmap_every=1,
    path="radial",
    mask_to_flower=False,
    device="cuda",
    expected=False,
    blank_mechanosensory=False,
    blank_visual=False,
    modality='joint'
):
    """
    Compute UR heatmaps for AllocentricNectarSearchEnv.

    The heatmap is still 2D over candidate tip locations. The current head-tip
    offset is preserved for both counterfactual and realized actions so visual
    and mechanosensory observations stay coupled under the new action space.

    Set blank_mechanosensory and/or blank_visual to make the agent and
    counterfactual heatmap posterior updates receive zeros for that modality.
    The returned trajectory xy positions are flower-local coordinates. The
    returned trajectory observations remain the raw environment observations.
    times[k] is the zero-based index of the conditioning observation: H[k]
    is computed after observing traj_xy[times[k]], before the next movement.
    """

    f = env.flower
    R, r0 = float(f.R), float(f.r0)
    if path not in {"radial", "circular"}:
        raise ValueError("path must be either 'radial' or 'circular'.")
    if heatmap_square_size is None:
        heatmap_square_size = 2.0 * R
    if heatmap_square_size <= 0:
        raise ValueError("heatmap_square_size must be positive.")

    _, _ = env.reset()
    env.tip_depth = float(depth_fixed)
    initial_head_tip_offset = 0 #env.head_xy - env.tip_xy

    r_start = float(circular_radius) if circular_radius is not None else start_r_frac * R
    if not (r0 < r_start <= R):
        raise ValueError("The starting/circular radius must lie on the corolla annulus: r0 < radius <= R.")

    start_flower_xy = np.array(
        [r_start * np.cos(angle), r_start * np.sin(angle)],
        dtype=np.float32,
    )
    start_tip_xy = env._to_canvas_xy(start_flower_xy)
    start_head_xy = env._constrain_head_to_canvas(start_tip_xy + initial_head_tip_offset)
    env.head_xy = start_head_xy
    env.tip_xy = env._constrain_tip_to_head(start_tip_xy, env.head_xy)
    env.prev_xy = env.tip_xy.copy()

    center_xy = env.flower_center_xy.astype(np.float32)
    half_size = float(heatmap_square_size) / 2.0
    xs = np.linspace(center_xy[0] - half_size, center_xy[0] + half_size, grid_n, dtype=np.float32)
    ys = np.linspace(center_xy[1] - half_size, center_xy[1] + half_size, grid_n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    valid_mask = (
        (X >= 0.0)
        & (X <= float(env.canvas.width))
        & (Y >= 0.0)
        & (Y <= float(env.canvas.height))
    )
    if mask_to_flower:
        canvas_xy = np.stack((X, Y), axis=-1)
        flower_xy = env._to_flower_xy(canvas_xy)
        rr = np.sqrt(flower_xy[..., 0] ** 2 + flower_xy[..., 1] ** 2)
        valid_mask = valid_mask & (rr <= R) & (rr > r0)

    traj_xy = np.zeros((T, 2), dtype=np.float32)
    traj_obs = []
    heatmaps = []
    traj_uncertainties = []
    times = []

    obs0 = env._get_obs()
    a0 = _allocentric_action_from_xy(env, env.head_xy, env.tip_xy, depth_fixed)
    x_mech_0, x_vis_0 = _allocentric_obs_to_tensors(
        obs0,
        device,
        blank_mechanosensory=blank_mechanosensory,
        blank_visual=blank_visual,
    )
    a0_t = torch.as_tensor(a0, dtype=torch.float32, device=device).reshape(1, -1)
    traj_xy[0] = env._to_flower_xy(env.tip_xy).astype(np.float32, copy=False)
    traj_obs.append(obs0)

    responses, _ = model.forward_t(
        x_mech_0,
        x_vis_0,
        a0_t,
        model.initial_hidden(batch_size=1, device=device, dtype=x_mech_0.dtype),
    )

    if modality == 'joint':
        modality_post_logvar = responses['post_logvar']
    elif modality == 'vis':
        modality_post_logvar = responses['vis_post_logvar']
    else:
        modality_post_logvar = responses['mech_post_logvar']
    current_uncertainty = compute_entropy(modality_post_logvar).sum(-1).mean()
    traj_uncertainties.append(current_uncertainty)

    path_angle = float(angle)
    circular_angle_traveled = 0.0

    for t in range(1, T):
        head_tip_offset = env.head_xy - env.tip_xy

        if ((t - 1) % compute_heatmap_every) == 0:
            Ht = torch.full((grid_n, grid_n), torch.nan, dtype=torch.float32, device=device)

            for i in range(grid_n):
                for j in range(grid_n):
                    if not valid_mask[i, j]:
                        continue

                    target_tip_xy = np.array([X[i, j], Y[i, j]], dtype=np.float32)
                    target_head_xy = env._constrain_head_to_canvas(target_tip_xy + head_tip_offset)
                    action = _allocentric_action_from_xy(env, target_head_xy, target_tip_xy, depth_fixed)

                    if not expected:
                        obs_peek = env.peek_obs(action)
                        Ht[i, j] = _allocentric_counterfactual_ur(
                            model,
                            obs_peek,
                            action,
                            (responses['h'], responses['c']) if 'c' in responses else responses['h'],
                            device,
                            blank_mechanosensory=blank_mechanosensory,
                            blank_visual=blank_visual,
                            modality=modality
                        ).item()
                    else:
                        Ht[i, j] = _expected_ur(
                            model,
                            responses,
                            action,
                            device,
                            blank_mechanosensory=blank_mechanosensory,
                            blank_visual=blank_visual,
                        ).item()

            heatmaps.append(Ht)
            times.append(t - 1)

        local_xy = env._to_flower_xy(env.tip_xy)
        r = float(np.hypot(local_xy[0], local_xy[1]))
        theta = np.arctan2(local_xy[1], local_xy[0])

        if path == "radial":
            if r <= max(end_r, r0 + 1e-3):
                break
            path_angle = float(theta)
            next_r = max(r - float(radial_step_mm), float(end_r), r0 + 1e-3)
            target_flower_xy = np.array(
                [next_r * np.cos(path_angle), next_r * np.sin(path_angle)],
                dtype=np.float32,
            )
        else:
            if circular_max_angle is not None and circular_angle_traveled >= abs(float(circular_max_angle)):
                break
            step_angle = float(angle_step_rad)
            path_angle += step_angle
            circular_angle_traveled += abs(step_angle)
            target_flower_xy = np.array(
                [r_start * np.cos(path_angle), r_start * np.sin(path_angle)],
                dtype=np.float32,
            )

        target_tip_xy = env._to_canvas_xy(target_flower_xy)
        target_head_xy = env._constrain_head_to_canvas(target_tip_xy + head_tip_offset)
        a_star = _allocentric_action_from_xy(env, target_head_xy, target_tip_xy, depth_fixed)

        obs, reward, terminated, truncated, info = env.step(a_star)
        traj_obs.append(obs)
        traj_xy[t] = env._to_flower_xy(env.tip_xy).astype(np.float32, copy=False)

        x_mech_t, x_vis_t = _allocentric_obs_to_tensors(
            obs,
            device,
            blank_mechanosensory=blank_mechanosensory,
            blank_visual=blank_visual,
        )
        a_star_t = torch.as_tensor(a_star, dtype=torch.float32, device=device).reshape(1, -1)
        h_prev = (responses['h'], responses['c']) if 'c' in responses else responses['h']
        responses, _ = model.forward_t(x_mech_t, x_vis_t, a_star_t, h_prev)

        if modality == 'joint':
            modality_post_logvar = responses['post_logvar']
        elif modality == 'vis':
            modality_post_logvar = responses['vis_post_logvar']
        else:
            modality_post_logvar = responses['mech_post_logvar']
        current_uncertainty = compute_entropy(modality_post_logvar).sum(-1).mean()
        traj_uncertainties.append(current_uncertainty)

        if terminated or truncated:
            break

    H = torch.stack(heatmaps, dim=0) if heatmaps else torch.zeros((0, grid_n, grid_n), dtype=torch.float32)
    traj_uncertainties = torch.stack(traj_uncertainties).detach()
    return H, times, traj_xy[: len(traj_obs)], traj_uncertainties, traj_obs, (X, Y, valid_mask)


def run_allocentric_ur_map_suite(
    model,
    env_factory,
    conditions,
    paths=("radial", "circular"),
    modalities=("mech", "vis"),
    radial_angles=(0.0,),
    circular_radii=(None,),
    T=64,
    grid_n=31,
    heatmap_square_size=None,
    depth_fixed=2.0,
    radial_step_mm=1.0,
    angle_step_rad=0.25 * np.pi,
    circular_max_angle=2 * np.pi,
    start_r_frac=0.95,
    end_r=2.0,
    compute_heatmap_every=8,
    mask_to_flower=True,
    expected=False,
    device="cuda",
    seed_base=20260503,
    include_maps=True,
    n_radial_bins=8,
    n_angle_bins=16,
):
    """
    Run a small factorial suite of allocentric UR maps.

    env_factory receives (condition, T, seed) and must return an
    AllocentricNectarSearchEnv. Each condition is a dict; common keys are
    "geometry", "c", and "pattern", but any metadata keys are preserved.

    Returns:
      runs: one record per trajectory/map run, optionally including map arrays
      summaries: one record per run and heatmap timestep
    """
    runs = []
    summaries = []
    run_i = 0

    for condition in conditions:
        condition = dict(condition)
        for modality in modalities:
            for path in paths:
                if path == "radial":
                    replicates = [
                        {"angle": float(angle), "radius": np.nan, "replicate": rep_i}
                        for rep_i, angle in enumerate(radial_angles)
                    ]
                elif path == "circular":
                    replicates = [
                        {"angle": 0.0, "radius": None if radius is None else float(radius), "replicate": rep_i}
                        for rep_i, radius in enumerate(circular_radii)
                    ]
                else:
                    raise ValueError("paths entries must be either 'radial' or 'circular'.")

                for replicate in replicates:
                    seed = int(seed_base + run_i)
                    env = env_factory(condition, T, seed)
                    radius = replicate["radius"]
                    H, times, traj_xy, traj_uncertainties, traj_obs, grid = run_allocentric_with_ur_heatmaps(
                        env,
                        model,
                        T=T,
                        grid_n=grid_n,
                        heatmap_square_size=heatmap_square_size,
                        depth_fixed=depth_fixed,
                        radial_step_mm=radial_step_mm,
                        angle_step_rad=angle_step_rad,
                        angle=replicate["angle"],
                        circular_radius=None if radius is None or not np.isfinite(radius) else radius,
                        circular_max_angle=circular_max_angle,
                        start_r_frac=start_r_frac,
                        end_r=end_r,
                        compute_heatmap_every=compute_heatmap_every,
                        path=path,
                        mask_to_flower=mask_to_flower,
                        device=device,
                        expected=expected,
                        blank_mechanosensory=False,
                        blank_visual=False,
                        modality=modality,
                    )
                    X, Y, valid_mask = grid
                    metadata = {
                        **condition,
                        "modality": modality,
                        "path": path,
                        "angle": replicate["angle"],
                        "radius": np.nan if radius is None else float(radius),
                        "replicate": int(replicate["replicate"]),
                        "seed": seed,
                    }
                    run_record = dict(metadata)
                    run_record.update(
                        {
                            "times": np.asarray(times, dtype=np.int64),
                            "traj_xy": np.asarray(traj_xy, dtype=np.float32),
                            "traj_uncertainties": np.asarray(traj_uncertainties, dtype=np.float32)
                        }
                    )
                    if include_maps:
                        run_record.update(
                            {
                                "H": _as_numpy(H).astype(np.float32, copy=False),
                                "X": np.asarray(X, dtype=np.float32),
                                "Y": np.asarray(Y, dtype=np.float32),
                                "valid_mask": np.asarray(valid_mask, dtype=bool),
                            }
                        )
                    runs.append(run_record)

                    center_xy = getattr(env, "flower_center_xy", None)
                    for summary in summarize_ur_heatmaps(
                        H,
                        times,
                        X,
                        Y,
                        valid_mask,
                        center_xy=center_xy,
                        n_radial_bins=n_radial_bins,
                        n_angle_bins=n_angle_bins,
                    ):
                        summaries.append({**metadata, **summary})

                    run_i += 1

    return runs, summaries
