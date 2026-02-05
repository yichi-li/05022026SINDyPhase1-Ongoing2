"""
Visualization helpers for comparing ground truth and learned models.
"""

import numpy as np
import matplotlib.pyplot as plt


def _make_axes(n_panels: int):
    """Create a subplot grid sized to the number of states."""
    if n_panels <= 2:
        rows, cols = 1, n_panels
    else:
        cols = 2
        rows = (n_panels + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = np.atleast_1d(axes).flatten()
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)
    return fig, axes[:n_panels]


def plot_trajectory_comparison(
    t,
    X_true,
    X_obs,
    X_hat,
    noise_level,
    state_names=["S", "E", "I", "R"],
    extra_series=None,
    r_note=None,
):
    """
    Compare ground truth, noisy input, and learned trajectory.

    Args:
        t: time array
        X_true: ground-truth trajectory (T, 4)
        X_obs: noisy observations used as SINDy input (T, 4)
        X_hat: learned trajectory (T, 4), teacher-forced 1-step Euler
        noise_level: noise level used to create X_obs
        state_names: state variable names
        extra_series: optional list of dicts with keys:
            "noise", "X_obs", "X_hat" for additional noise levels
    """
    fig, axes = _make_axes(len(state_names))
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        # Plot ground truth, observed input, and SINDy reconstruction.
        ax.plot(t, X_true[:, i], 'k-', label='Ground Truth', linewidth=1.4, alpha=0.9)
        if noise_level > 0:
            ax.plot(t, X_obs[:, i], 'b--', label=f'Observed (noise={noise_level})',
                   linewidth=1, alpha=0.6)
        ax.plot(t, X_hat[:, i], 'r:', label=f'SINDy (teacher-forced, noise={noise_level})', linewidth=2, alpha=0.8)
        if extra_series:
            for series in extra_series:
                nl = series["noise"]
                x_obs = series["X_obs"]
                x_hat = series["X_hat"]
                if nl > 0:
                    ax.plot(t, x_obs[:, i], '--', linewidth=1.4, alpha=0.6, label=f'Observed (noise={nl})')
                ax.plot(t, x_hat[:, i], ':', linewidth=1.6, alpha=0.7, label=f'SINDy (teacher-forced, noise={nl})')
        
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel(f'{name} (fraction)', fontsize=11)
        ax.set_title(f'{name}(t): Ground Truth vs Learned', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    title = "Trajectory Comparison (teacher-forced 1-step: observed state + model-predicted derivatives)"
    if r_note:
        title += f"\n{r_note}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.94])
    return fig


def plot_input_data_verification(t, X_true, X_obs, noise_level, extra_obs=None, state_names=None, r_note=None):
    """
    Verify input data: compare ground truth vs noisy observations.

    Args:
        t: time array
        X_true: ground-truth trajectory (T, 4)
        X_obs: noisy observations used as SINDy input (T, 4)
        noise_level: noise level used to create X_obs
        extra_obs: optional list of dicts with keys "noise", "X_obs"
    """
    if state_names is None:
        state_names = ["S", "E", "I", "R"]
    fig, axes = _make_axes(len(state_names))
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        # Plot ground truth and observed input.
        ax.plot(t, X_true[:, i], 'b-', label='Ground Truth', linewidth=1.4, alpha=0.9)
        ax.plot(t, X_obs[:, i], 'r--', label=f'Input to SINDy (noise={noise_level})',
               linewidth=1.8, alpha=0.85)
        if extra_obs:
            for series in extra_obs:
                nl = series["noise"]
                x_obs = series["X_obs"]
                ax.plot(t, x_obs[:, i], '--', linewidth=1.4, alpha=0.6, label=f'Input to SINDy (noise={nl})')
        
        # If noise=0, show the max absolute difference.
        if noise_level == 0.0:
            diff = np.abs(X_true[:, i] - X_obs[:, i])
            max_diff = np.max(diff)
            ax.text(0.05, 0.95, f'Max diff: {max_diff:.2e}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel(f'{name} (fraction)', fontsize=11)
        ax.set_title(f'{name}(t): Input Data Verification', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    title = 'Input Data Verification (ground truth vs noisy inputs)'
    if r_note:
        title += f"\n{r_note}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.94])
    return fig


def plot_error_over_time(t, X_true, X_hat, state_names=None, extra_hats=None, r_note=None):
    """
    Plot error over time for each state: error = ground truth - learned.

    Args:
        t: time array
        X_true: ground-truth trajectory (T, 4)
        X_hat: learned trajectory (T, 4), teacher-forced 1-step Euler
        state_names: state variable names
        extra_hats: optional list of dicts with keys "noise", "X_hat"
    """
    if state_names is None:
        state_names = ["S", "E", "I", "R"]
    fig, axes = _make_axes(len(state_names))
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        error = X_true[:, i] - X_hat[:, i]
        ax.plot(t, error, 'r-', linewidth=1.8, label='Error (noise=0.0)')
        if extra_hats:
            for series in extra_hats:
                nl = series["noise"]
                x_hat = series["X_hat"]
                err_nl = X_true[:, i] - x_hat[:, i]
                ax.plot(t, err_nl, '--', linewidth=1.2, alpha=0.6, label=f'Error (noise={nl})')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(t, error, 0, alpha=0.3, color='red')
        
        rmse = np.sqrt(np.mean(error**2))
        ax.text(0.05, 0.95, f'RMSE over time (noise=0.0): {rmse:.2e}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='lightblue', alpha=0.5))
        ax.legend(fontsize=8)
        
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel(f'Error ({name})', fontsize=11)
        ax.set_title(f'{name}(t): Error = Ground Truth - Learned', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    title = 'Error Over Time (teacher-forced 1-step)'
    if r_note:
        title += f"\n{r_note}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.94])
    return fig


def plot_sir_scenario_comparison(t, scenario_series, note=None):
    """
    Plot SIR ground-truth trajectories for multiple scenarios.

    Args:
        t: time array
        scenario_series: list of dicts with keys:
            "name", "X", "beta", "gamma", "R0", "t_peak", "i_peak"
        note: optional note string to display on the figure
    """
    fig, axes = _make_axes(3)
    state_names = ["S", "I", "R"]
    line_styles = ["-", "--", ":", "-."]

    for idx, series in enumerate(scenario_series):
        name = series["name"]
        X = series["X"]
        beta = series["beta"]
        gamma = series["gamma"]
        r0 = series["R0"]
        t_peak = series["t_peak"]
        i_peak = series["i_peak"]
        label = (
            f"{name}: beta={beta:.2f}, gamma={gamma:.2f}, "
            f"R0={r0:.2f}, t_peak={t_peak:.1f}d, I_peak={i_peak:.2f}"
        )
        style = line_styles[idx % len(line_styles)]
        for i, ax in enumerate(axes):
            ax.plot(t, X[:, i], style, linewidth=1.8, label=label)

    for i, ax in enumerate(axes):
        name = state_names[i]
        ax.set_xlabel("Time (days)", fontsize=11)
        ax.set_ylabel(f"{name} (fraction)", fontsize=11)
        ax.set_title(f"{name}(t): Ground Truth by Scenario", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    title = "SIR Ground Truth Trajectories by Scenario"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.94])
    return fig


def plot_r_consistency(
    t,
    r_cons,
    r_dyn,
    r_true=None,
    title_suffix="",
):
    """Compare R from algebraic conservation vs dynamic integration.

    r_cons: R from R = 1 - S - I (algebraic constraint)
    r_dyn:  R from integrating dR/dt = -(dS/dt + dI/dt)
    r_true: optional ground-truth R for reference
    """
    t = np.asarray(t)
    r_cons = np.asarray(r_cons)
    r_dyn = np.asarray(r_dyn)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(t, r_cons, "k-", linewidth=2.0, label="R (1 - S - I)")
    ax.plot(t, r_dyn, "r--", linewidth=1.8, label="R from dR/dt = -(dS/dt + dI/dt)")
    if r_true is not None:
        ax.plot(t, np.asarray(r_true), "b:", linewidth=1.6, label="R ground truth")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("R (fraction)")
    ax.set_title("R Consistency Check" + (f" {title_suffix}" if title_suffix else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_free_rollout_comparison(
    t,
    X_true,
    X_roll,
    noise_level=0.0,
    state_names=None,
    abort_reason=None,
    extra_rolls=None,
    r_note=None,
):
    """Ground truth vs SINDy free rollout comparison.

    Parameters
    ----------
    t : array-like
        Full time grid corresponding to X_true.
    X_true : ndarray
        Ground-truth trajectory, shape (T, 4).
    X_roll : ndarray
        Trajectory generated ONLY from the learned model (no teacher forcing),
        shape (T', 4) where T' <= T.
    extra_rolls : list of dicts
        Optional list with keys "noise", "X_roll" for additional noise levels.
    abort_reason : str or None
        If rollout was aborted early, this is a short reason string.

    Behavior
    --------
    - Always produces a figure (even if rollout aborted early).
    - If X_roll is shorter than t, we plot it on the prefix t[:len(X_roll)].
    """

    t = np.asarray(t)
    X_true = np.asarray(X_true)
    X_roll = np.asarray(X_roll)

    if state_names is None:
        state_names = ("S", "E", "I", "R")
    fig, axes = _make_axes(len(state_names))

    t_roll = t[: len(X_roll)]
    for i, ax in enumerate(axes):
        ax.plot(t, X_true[:, i], label="Ground Truth", linewidth=2.2)
        ax.plot(t_roll, X_roll[:, i], "--", label=f"SINDy Free Rollout (noise={noise_level})", linewidth=2)
        if extra_rolls:
            for series in extra_rolls:
                nl = series["noise"]
                x_r = np.asarray(series["X_roll"])
                t_r = t[: len(x_r)]
                ax.plot(t_r, x_r[:, i], "--", linewidth=1.5, alpha=0.6, label=f"SINDy Free Rollout (noise={nl})")
        ax.set_title(f"{state_names[i]}(t)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(f"{state_names[i]} (fraction)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    title = "FREE ROLLOUT Comparison (model-only integration; no observed states)"
    if abort_reason:
        title += "\nABORTED: " + str(abort_reason)
    if r_note:
        title += "\n" + str(r_note)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.94])
    return fig


def plot_hybrid_rollout_comparison(
    t,
    X_true,
    X_hybrid,
    t_switch: float,
    state_names=None,
    r_note=None,
):
    """Ground truth vs hybrid rollout (teacher-forced then free)."""
    if state_names is None:
        state_names = ("S", "E", "I", "R")
    fig, axes = _make_axes(len(state_names))
    for i, ax in enumerate(axes):
        ax.plot(t, X_true[:, i], label="Ground Truth", linewidth=2.0)
        ax.plot(t, X_hybrid[:, i], "--", label="Hybrid rollout", linewidth=2.0)
        ax.axvline(t_switch, color="k", linestyle=":", linewidth=1.0, label="Switch time")
        ax.set_title(f"{state_names[i]}(t)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(f"{state_names[i]} (fraction)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    title = f"Hybrid Rollout (teacher-forced until t={t_switch:.1f}, then free rollout)"
    if r_note:
        title += f"\n{r_note}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.94])
    return fig
