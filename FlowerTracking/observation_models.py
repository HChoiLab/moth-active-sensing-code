import numpy as np
import matplotlib.pyplot as plt

def generate_proboscis_observations_1d(
    x_f: np.ndarray,
    x_m: np.ndarray,
    dt: float = 1e-3,
    *,
    L_prob: float = 0.07,      # m   proboscis reach
    k_p: float = 0.2,          # N/m spring stiffness
    d_sat: float = 8e-2,       # m   saturation compression
    tau_p: float = 2e-3,       # s   low-pass τ  (≈80 Hz cut-off)
    alpha: float = 200.0,      # spikes·s⁻¹·N⁻¹  gain
    F_thr: float = 0.01,       # N   firing threshold
    sigma_frac: float = 0.05,  # 7 % relative noise
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    1-D proboscis mechanosensory observation model.

    Parameters
    ----------
    x_f, x_m : (B, T) float
        Flower and moth positions along ONE axis (same units, metres preferred).
    dt : float
        Sampling interval in seconds.
    Returns
    -------
    y : (B, T) float
        Noisy firing-rate observation sequences.
    """
    if x_f.shape != x_m.shape or x_f.ndim != 2:
        raise ValueError("x_f and x_m must both be (B, T) arrays of 1-D positions")

    B, T = x_f.shape
    if rng is None:
        rng = np.random.default_rng()

    # State buffer for the first-order low-pass filter, one per batch element
    F_p = np.zeros(B, dtype=x_f.dtype)
    y   = np.empty((B, T), dtype=x_f.dtype)

    for t in range(T):
        # Positive scalar distance (flower – moth) along the tracking axis
        d = np.abs(x_f[:, t] - x_m[:, t]) - L_prob        # signed gap to proboscis tip

        # Spring contact force with compression saturation
        F_raw = k_p * np.clip(-d, 0.0, d_sat)             # (B,)
        # Forward-Euler update of first-order low-pass sensor dynamics
        F_p += (dt / tau_p) * (F_raw - F_p)

        # Force-to-rate transduction with rectification
        r_prob = alpha * np.maximum(F_p - F_thr, 0.0)

        # Add multiplicative/relative Gaussian noise
        noise_sigma = sigma_frac * np.where(r_prob > 0, r_prob, 1.0)
        y[:, t] = r_prob + rng.normal(scale=noise_sigma)

    return y


if __name__ == '__main__':

    # Example batch: 32 trials, 2000 time steps (2 s at 1 kHz)
    B, T = 32, 2000
    dt   = 1e-3

    # Dummy trajectories (replace with your real data)
    t_axis = np.arange(T) * dt
    x_f = 0.01 * np.sin(2*np.pi*1.2*t_axis)[None, :] + 0.15       # flower oscillates
    x_f = np.repeat(x_f, B, axis=0)   # broadcast to batch

    x_m = 0.15 * np.ones_like(x_f)                                # moth hovers

    obs = generate_proboscis_observations_1d(x_f, x_m, dt)
    
    i = 0
    
    plt.subplot(3, 1, 1)
    plt.plot(x_f[i])
    #plt.subplot(3, 1, 2)
    plt.plot(x_m[i])
    plt.subplot(3, 1, 2)
    plt.plot(obs[i])
    plt.show()

