"""
Entry point: Phase 1 recovery experiment.

You can run:
    python run_phase1.py

What you should expect:
- A plot: noise level vs MSE (recovery gets worse as noise increases)
- A printed learned model at one noise level
"""

from datetime import datetime
from pathlib import Path
import os
import sys

import numpy as np  # Numerical arrays.
import matplotlib.pyplot as plt  # Plotting.

# Import SEIR/SIR simulators to generate ground-truth trajectories.
from src.seir_sim import SEIRParams, SIRParams, ForcingParams, simulate_seir, simulate_sir
# Core Phase 1 recovery experiment across noise levels.
from src.phase1_recovery_pysindy import recovery_grid_experiment


def safe_free_rollout(model, x0, t, method="rk4", domain_check=True, reduced_coords=False):
    """Free rollout (NO teacher forcing) with a pure-Python explicit integrator.

    Why this exists
    --------------
    In Phase 1, a learned SINDy model can be numerically unstable.
    If we call `pysindy.SINDy.simulate(...)`, some solvers can crash
    hard when the RHS blows up (especially with unstable polynomials).

    Here we do a simple explicit integration ourselves (Euler/RK4) and:
    - stop early when the rollout becomes non-finite or obviously diverges
    - return a short abort reason
    - NEVER throw, so the whole script can still finish

    Returns
    -------
    X_roll : np.ndarray
        Trajectory with shape (T', n_states). May be shorter than `t`.
    abort_reason : str | None
        None if finished full horizon, else a short diagnostic string.
    """
    import numpy as np

    x0 = np.asarray(x0, dtype=float)
    t = np.asarray(t, dtype=float)

    X = np.zeros((len(t), len(x0)), dtype=float)
    X[0] = x0
    abort_reason = None

    # Right-hand side wrapper: model.predict expects shape (n_samples, n_states)
    def rhs(x):
        dx = model.predict(np.asarray(x, dtype=float).reshape(1, -1))[0]
        if not np.all(np.isfinite(dx)):
            raise FloatingPointError("non-finite dx")
        return dx

    # Very lightweight sanity checks (do NOT enforce SEIR constraints; just avoid nonsense)
    def check_domain(x):
        if not domain_check:
            return
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("non-finite state")
        # You are using fractions (divided by N), so values should be O(1)
        if np.max(np.abs(x)) > 50:
            raise FloatingPointError("state magnitude exploded (>50)")
        if not reduced_coords:
            s = float(np.sum(x))
            # In true SEIR fractions, sum should be ~1; allow slack because model may not conserve
            if abs(s - 1.0) > 2.0:
                raise FloatingPointError(f"mass conservation badly broken (sum={s:.3g})")

    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        xi = X[i]
        try:
            check_domain(xi)
            if method.lower() == "euler":
                x_next = xi + dt * rhs(xi)
            else:
                # RK4: more stable than Euler for the same dt
                k1 = rhs(xi)
                k2 = rhs(xi + 0.5 * dt * k1)
                k3 = rhs(xi + 0.5 * dt * k2)
                k4 = rhs(xi + dt * k3)
                x_next = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            check_domain(x_next)
            X[i + 1] = x_next
        except Exception as e:
            abort_reason = f"step={i}, t={t[i]:.2f}, {type(e).__name__}: {e}"
            X = X[: i + 1]
            break

    return X, abort_reason


def handle_figure(fig, name, output_dir, show=True, close_after=False):
    """Save a figure with a stable name and optionally show it non-blocking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{name}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    try:
        fig.canvas.manager.set_window_title(name)
    except Exception:
        pass
    if show:
        plt.show(block=False)
        plt.pause(0.1)
    if close_after:
        plt.close(fig)


class Tee:
    """Write output to both console and a file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def _env_str(key, default):
    return os.getenv(key, default)


def _env_float(key, default):
    val = os.getenv(key)
    return default if val is None else float(val)


def _env_int(key, default):
    val = os.getenv(key)
    return default if val is None else int(val)


def _env_bool(key, default):
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y")


def _env_float_list(key):
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [float(x) for x in raw.split(",") if x.strip()]

def _env_str_list(key):
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

def _reconstruct_sir_from_si(X_si):
    """Reconstruct R from S and I via R = 1 - S - I (SIR only)."""
    X_si = np.asarray(X_si, dtype=float)
    s = X_si[:, 0]
    i = X_si[:, 1]
    r = 1.0 - s - i
    return np.stack([s, i, r], axis=1)

def _integrate_r_from_si_derivatives(t, r0, dS_dt, dI_dt):
    """Integrate dR/dt = -(dS/dt + dI/dt) with Euler; returns R(t)."""
    t = np.asarray(t, dtype=float)
    dS_dt = np.asarray(dS_dt, dtype=float)
    dI_dt = np.asarray(dI_dt, dtype=float)
    r = np.zeros(len(t), dtype=float)
    r[0] = float(r0)
    for k in range(len(t) - 1):
        dt = float(t[k + 1] - t[k])
        dR_dt = -(dS_dt[k] + dI_dt[k])
        r[k + 1] = r[k] + dt * dR_dt
    return r

def _simplify_derived_r_expr(expr_str):
    """Best-effort simplification of the derived R expression (optional)."""
    try:
        import sympy as sp
        x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")
        expr = sp.sympify(expr_str, locals={"x0": x0, "x1": x1, "x2": x2, "x3": x3})
        return str(sp.simplify(expr))
    except Exception:
        return expr_str


def main():
    # Output folder for plots (timestamped).
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    tag = _env_str("PHASE1_TAG", "").strip()
    output_dir = Path("resultsSAVE") / (f"{timestamp}_{tag}" if tag else timestamp)
    show_plots = _env_bool("PHASE1_SHOW_PLOTS", True)  # non-blocking display
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "terminal_output.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(">>> run_phase1 started")  # Run start marker.
    print("="*70)

    # 1) Generate a ground-truth trajectory (baseline model, no forcing).
    # === USER SWITCHES (MODEL STRUCTURE) ===
    # model_type controls the ground-truth simulator and state dimension:
    # - "seir" -> states are [S, E, I, R]
    # - "sir"  -> states are [S, I, R]
    model_type = _env_str("PHASE1_MODEL_TYPE", "sir")  # "seir" or "sir"
    reduced_coords = _env_bool("PHASE1_REDUCED_COORDS", False)
    optimizer_type = _env_str("PHASE1_OPTIMIZER", "stlsq")
    # SR3-specific knobs (only used when optimizer_type="sr3").
    # These are NOT the same as STLSQ's threshold; keep them separate.
    sr3_reg_weight_lam = os.getenv("PHASE1_SR3_LAM")
    sr3_reg_weight_lam = None if sr3_reg_weight_lam is None else float(sr3_reg_weight_lam)
    sr3_relax_coeff_nu = os.getenv("PHASE1_SR3_NU")
    sr3_relax_coeff_nu = None if sr3_relax_coeff_nu is None else float(sr3_relax_coeff_nu)
    sr3_regularizer = _env_str("PHASE1_SR3_REG", "")
    sr3_regularizer = None if sr3_regularizer == "" else sr3_regularizer

    # === USER SWITCHES (SIMULATION/FIT WINDOW) ===
    # t_end_sim: how long to simulate the ground-truth system (days).
    # t_end_fit: how many days to use for fitting SINDy (<= t_end_sim).
    # Example: simulate 200 days, fit only first 40 days.
    t_end_sim = _env_float("PHASE1_T_END_SIM", 90.0)
    t_end_fit = _env_float("PHASE1_T_END_FIT", 90.0)
    # t_switch: hybrid rollout switch time. Use observed data up to t_switch,
    # then free rollout from that point. Default ties to t_end_fit.
    t_switch = _env_float("PHASE1_T_SWITCH", t_end_fit)
    t_switch_alt_list = _env_float_list("PHASE1_T_SWITCH_ALT")

    # === USER SWITCHES (PARAMETER SCENARIO PRESETS) ===
    # Choose a preset to control epidemic speed within the fit window.
    # - "fast":  faster epidemic (higher beta)
    # - "medium": moderate epidemic speed
    # - "slow": longer epidemic tail (lower beta)
    scenario = _env_str("PHASE1_SCENARIO", "slow")  # "fast", "medium", "slow"

    # === PARAMETER SCENARIOS (ADJUST IF EPIDEMIC TOO FAST/SLOW) ===
    # You can add more scenarios to test different epidemic speeds.
    scenario = scenario.lower().strip()
    sir_beta_map = None
    if model_type == "sir":
        beta_map = {
            "slow": 0.33,    # R0=1.65 with gamma=0.2
            "medium": 0.40,  # R0=2.0
            "fast": 0.60,    # R0=3.0
        }
        sir_beta_map = beta_map
        beta0 = beta_map.get(scenario, beta_map["medium"])
        p = SIRParams(beta0=beta0, gamma=1 / 5, N=1_000_000)
        x0 = np.array([p.N - 10.0, 10.0, 0.0])  # [S0, I0, R0]
        f = None
    else:
        beta_map = {
            "slow": 0.33,    # R0≈1.65 with gamma=0.2
            "medium": 0.40,  # R0≈2.0
            "fast": 0.60,    # R0≈3.0
        }
        beta0 = beta_map.get(scenario, beta_map["medium"])
        p = SEIRParams(beta0=beta0, sigma=1 / 8, gamma=1 / 5, N=1_000_000)
        # Forcing disabled for Phase 1 (no seasonality/term effects).
        f = ForcingParams(period=365.0, a_cos=0.0, b_sin=0.0, term_amp=0.0)
        x0 = np.array([p.N - 10.0, 5.0, 5.0, 0.0])

    t_end = t_end_sim
    # Output sampling step for saved trajectories.
    dt = _env_float("PHASE1_DT", 0.2)
    # Simulate to obtain ground-truth states for noisy observations and evaluation.
    if model_type == "sir":
        t, X_true = simulate_sir(t_end, dt, x0, p)
    else:
        t, X_true = simulate_seir(t_end, dt, x0, p, f)

    # Normalize to fractions (S/N, E/N, I/N, R/N). SINDy is fit on these fractions.
    X_true = X_true / p.N
    x0 = x0 / p.N

    # Plot SIR ground-truth trajectories across scenarios (for context).
    if model_type == "sir" and sir_beta_map:
        try:
            scenario_order = ["fast", "medium", "slow"]
            scenario_series = []
            t_plot = None
            for sc in scenario_order:
                if sc not in sir_beta_map:
                    continue
                beta0_sc = sir_beta_map[sc]
                p_sc = SIRParams(beta0=beta0_sc, gamma=p.gamma, N=p.N)
                x0_sc = np.array([p_sc.N - 10.0, 10.0, 0.0])
                t_sc, X_sc = simulate_sir(t_end, dt, x0_sc, p_sc)
                X_sc = X_sc / p_sc.N
                i_sc = X_sc[:, 1]
                peak_idx = int(np.argmax(i_sc))
                scenario_series.append({
                    "name": sc,
                    "X": X_sc,
                    "beta": beta0_sc,
                    "gamma": p_sc.gamma,
                    "R0": beta0_sc / p_sc.gamma if p_sc.gamma != 0 else np.nan,
                    "t_peak": float(t_sc[peak_idx]),
                    "i_peak": float(i_sc[peak_idx]),
                })
                if t_plot is None:
                    t_plot = t_sc
            if scenario_series:
                from src.visualization import plot_sir_scenario_comparison
                note = "Higher R0 => faster spread and earlier peak."
                fig0 = plot_sir_scenario_comparison(t_plot, scenario_series, note=note)
                handle_figure(fig0, "fig0_sir_ground_truth_scenarios", output_dir, show=show_plots)
        except Exception as e:
            print(f"[WARN] scenario ground-truth plot skipped: {e}")

    # Fit window selection (use a shorter window if desired).
    t_end_fit = min(t_end_fit, t_end_sim)
    fit_mask = t <= t_end_fit
    t_fit = t[fit_mask]
    X_fit = X_true[fit_mask]

    # Input summary for SINDy (useful for Phase 2 when states may differ).
    n_states = X_true.shape[1]
    if n_states == 4:
        state_names_full = ["S", "E", "I", "R"]
    elif n_states == 3:
        state_names_full = ["S", "I", "R"]
    else:
        state_names_full = [f"x{i}" for i in range(n_states)]

    # Reduced coords currently supports SIR only (S, I).
    if reduced_coords and model_type != "sir":
        raise ValueError("PHASE1_REDUCED_COORDS currently supports only model_type='sir'.")

    if reduced_coords:
        sindy_state_names = ["S", "I"]
        plot_state_names = ["S", "I", "R"]
    else:
        sindy_state_names = list(state_names_full)
        plot_state_names = list(state_names_full)

    print(f"SINDy input states: {sindy_state_names}")
    print(f"SINDy input shape: X_fit={X_fit.shape}, t_fit={t_fit.shape}")
    if reduced_coords:
        print(f"SINDy reduced input shape: X_fit_used={(X_fit[:, [0, 1]]).shape}")
        print("Note: R is reconstructed from R = 1 - S - I (not learned).")
    print(f"SINDy fit time range: t_fit[0]={t_fit[0]:.2f}, t_fit[-1]={t_fit[-1]:.2f}, dt={dt}")

    # Ground-truth SEIR equations in normalized (fraction) form.
    print("\nGround-truth model (fractions, no forcing):")
    print("="*70)
    beta = p.beta0
    if model_type == "sir":
        gamma = p.gamma
        print(f"dS/dt = -{beta:.3f} * S * I")
        print(f"dI/dt =  {beta:.3f} * S * I - {gamma:.3f} * I")
        print(f"dR/dt =  {gamma:.3f} * I")
        if reduced_coords:
            print("Note: reduced coords -> R is not learned by SINDy.")
            print("      We derive dR/dt = -(dS/dt + dI/dt) from learned S/I.")
    else:
        sigma = p.sigma
        gamma = p.gamma
        print(f"dS/dt = -{beta:.3f} * S * I")
        print(f"dE/dt =  {beta:.3f} * S * I - {sigma:.3f} * E")
        print(f"dI/dt =  {sigma:.3f} * E - {gamma:.3f} * I")
        print(f"dR/dt =  {gamma:.3f} * I")
    print("="*70)


    # 2) Recovery experiment: add Gaussian noise to X_true and fit SINDy.
    # SINDy input per noise level: X_obs is (T, 4) state fractions; t is (T,).
    # SINDy estimates derivatives internally from X_obs and t.
    # Noise model: per-state Gaussian noise with std = noise_level * std(state).
    # No reporting delays, under-reporting, or weekly aggregation are applied here.
    # === USER SWITCHES (CANDIDATE LIBRARY) ===
    # library_mode:
    # - "restricted": use a mechanism-informed library (see restricted_set below)
    # - "polynomial": use a full polynomial library (degree set by poly_degree)
    library_mode = _env_str("PHASE1_LIBRARY_MODE", "polynomial")  # "polynomial" or "restricted"

    # restricted_set is only used when library_mode="restricted":
    # - For SEIR:
    #   * "basic"    = {S*I, E, I}
    #   * "extended" = {S*I, E, I, S*E}
    # - For SIR:
    #   * "basic"    = {S*I, I}
    #   * "extended" = {S*I, I, S}
    restricted_set = _env_str("PHASE1_RESTRICTED_SET", "extended")

    # poly_degree is only used when library_mode="polynomial":
    # - degree=1 -> linear terms only
    # - degree=2 -> linear + quadratic terms
    # - degree=3 -> linear + quadratic + cubic terms, etc.
    poly_degree = _env_int("PHASE1_POLY_DEGREE", 2)
    threshold = _env_float("PHASE1_THRESHOLD", 0.035)
    print(
        "Active model_type="
        f"{model_type}, reduced_coords={reduced_coords}, optimizer={optimizer_type}, "
        f"library_mode={library_mode}, restricted_set={restricted_set}, "
        f"poly_degree={poly_degree}, threshold={threshold}"
    )
    if optimizer_type.lower() == "sr3":
        print(
            "SR3 settings: "
            f"reg_weight_lam={sr3_reg_weight_lam}, "
            f"relax_coeff_nu={sr3_relax_coeff_nu}, "
            f"regularizer={sr3_regularizer or 'L0'}"
        )
    # SINDy learns equations from X_fit and t_fit (not from full t_end_sim).
    results = recovery_grid_experiment(
        X_fit,  # Ground-truth state fractions used to create noisy observations.
        t_fit,  # Time grid used for fitting (subset of full simulation)
        noise_levels = (0.0, 0.001, 0.005, 0.01, 0.05),  # Relative noise grid (as fraction of state std).
        poly_degree=poly_degree,  # Polynomial library degree.
        threshold=threshold,  # STLSQ sparsity threshold.
        # Threshold trade-off: too small keeps many terms; too large can zero out equations.
        seed=0,  # Random seed for reproducibility.

        # ---------- Library selection (choose one mode) ----------
        # 1) Restricted library (choose a restricted_set below):

        # library_mode="restricted",
        # restricted_set="basic",   # SEIR: {S*I, E, I} | SIR: {S*I, I}
        # restricted_set="extended",  # SEIR: +S*E | SIR: +S

        # 2) Polynomial library:

        library_mode=library_mode,
        restricted_set=restricted_set,
        model_type=model_type,
        reduced_coords=reduced_coords,
        optimizer_type=optimizer_type,
        sr3_reg_weight_lam=sr3_reg_weight_lam,
        sr3_relax_coeff_nu=sr3_relax_coeff_nu,
        sr3_regularizer=sr3_regularizer,
    )
    print("Phase 1 evaluates derivative recovery, not long-term rollouts.")
    if reduced_coords:
        print("MSE is computed only on dS/dt and dI/dt (R is derived, not learned).")
    else:
        print("MSE is computed on dX/dt at every time point (true vs model-predicted).")

    # Extract noise levels and MSE for plotting.
    noise = [r["noise"] for r in results]
    mse = [r["mse"] for r in results]

    # 3) Plot 1: noise level vs derivative-fit MSE (aggregated over all time points).
    fig_mse = plt.figure()
    print("Plot 1: MSE of dX/dt vs noise level (all time points aggregated).")
    plt.plot(noise, mse, marker="o")
    plt.xlabel("Relative state noise level")
    plt.ylabel("MSE of dX/dt (true vs model prediction)")  # MSE is computed on derivatives.
    plt.title("Phase 1: Derivative-fit MSE vs noise level")
    handle_figure(fig_mse, "fig1_mse_vs_noise", output_dir, show=show_plots)

    # 4) Print the actual feature library used for fitting.
    print("\n" + "="*70)
    print("Feature Library (Actual Library Used For Fitting):")
    print("="*70)
    try:
        # Read feature names from the first fitted model.
        model0 = results[0]["model"]
        lib0 = model0.feature_library
        try:
            feature_names = lib0.get_feature_names(input_features=sindy_state_names)
        except Exception:
            feature_names = lib0.get_feature_names()
        print(f"Feature library contains {len(feature_names)} candidate terms:")
        # Check whether S*I (core epidemic interaction) is included.
        has_SI = any(
            "S*I" in name
            or "S I" in name
            or ("x0" in name and ("x1" in name or "x2" in name))
            for name in feature_names
        )
        print(f"Contains S*I (or x0*x2) term: {'yes' if has_SI else 'no'}")
        if has_SI:
            si_features = [
                name
                for name in feature_names
                if "S*I" in name
                or "S I" in name
                or ("x0" in name and ("x1" in name or "x2" in name) and "*" in name)
            ]
            if si_features:
                print(f"S*I-related features (first 10): {si_features[:10]}")
        print("\nFirst 30 feature examples:")
        for i, name in enumerate(feature_names[:30]):
            print(f"  {i+1:3d}. {name}")
        if len(feature_names) > 30:
            print(f"  ... {len(feature_names) - 30} more (total {len(feature_names)})")
        if sindy_state_names == ["S", "I"]:
            print("\nNote: reduced coords used; R is derived from 1 - S - I")
            print("Note: x0=S, x1=I")
        elif sindy_state_names == ["S", "I", "R"]:
            print("\nNote: x0=S, x1=I, x2=R")
        else:
            print("\nNote: x0=S, x1=E, x2=I, x3=R")
    except (AttributeError, TypeError) as e:
        # Fallback when get_feature_names is unavailable.
        print(f"Could not read feature names (error: {e})")
        print("Trying to read the fitted model's library info...")
        # Try reading the model's library metadata directly.
        if results:
            model = results[0]["model"]
            try:
                # Attempt to access the feature library.
                lib = model.feature_library
                print(f"Library type: {type(lib)}")
                print(f"Polynomial degree: {lib.degree if hasattr(lib, 'degree') else 'unknown'}")
                print(f"Include interactions: {lib.include_interaction if hasattr(lib, 'include_interaction') else 'unknown'}")
            except:
                print("Could not read detailed library info from the model.")
        print("\nBased on settings, the library should be one of:")
        print("  - Restricted: {S*I, E, I} (optional + constant)")
    print("="*70)
    
    # 5) Print learned models for all noise levels (so every fit is visible).
    for r in results:
        nl = r["noise"]
        print(f"\nLearned SINDy model (noise level={nl}):")
        print("="*70)
        r["model"].print()
        if reduced_coords:
            # Also print the derived R ODE from conservation for clarity.
            # This does not come from SINDy directly; it is implied by S + I + R = 1.
            print("Derived: dR/dt = -(dS/dt + dI/dt)")
            try:
                eqs = r["model"].equations()
                if len(eqs) >= 2:
                    raw = f"-({eqs[0]}) - ({eqs[1]})"
                    simplified = _simplify_derived_r_expr(raw)
                    print(f"Derived: dR/dt = -(dS/dt + dI/dt) = {simplified}")
                else:
                    print("Derived: dR/dt = -(dS/dt + dI/dt)")
            except Exception:
                print("Derived: dR/dt = -(dS/dt + dI/dt)")
        print("="*70)
    
    # 6) Visualization: short 40-day window, teacher-forced one-step Euler for X_hat_viz.
    print("\nGenerating comparison plots...")
    if reduced_coords:
        print("Note: reduced coords -> plots show R reconstructed from R = 1 - S - I.")
        print("      A separate R-consistency plot compares algebraic R vs integrated dR/dt.")
    r_note = "R reconstructed via R = 1 - S - I (not learned)" if reduced_coords else None
    from src.visualization import (
        plot_trajectory_comparison,
        plot_input_data_verification,
        plot_error_over_time,
        plot_free_rollout_comparison,
        plot_hybrid_rollout_comparison,
        plot_r_consistency,
    )
    
    # Use the noise=0.0 case as the main reference; overlay other noise levels.
    if 0.0 in noise:
        idx_viz = noise.index(0.0)
        result_viz = results[idx_viz]

        coord_idx = [0, 1] if reduced_coords else None
        X_obs_full = result_viz.get("X_obs_full", result_viz.get("X_obs", X_fit))
        if coord_idx is None:
            X_obs_used = X_obs_full
        else:
            X_obs_used = result_viz.get("X_obs_used", X_obs_full[:, coord_idx])
        model_viz = result_viz["model"]

        # Visualization window: follow the full fit window unless you change t_end_fit.
        t_viz = t_fit
        X_true_viz = X_fit
        X_obs_used = X_obs_used[: len(t_viz)]

        # For reduced coords, we still plot R but reconstruct it from S/I
        # using the algebraic constraint R = 1 - S - I (no R-ODE involved).
        if reduced_coords:
            X_obs_viz = _reconstruct_sir_from_si(X_obs_used)
        else:
            X_obs_viz = X_obs_full[: len(t_viz)]

        # Teacher-forced 1-step Euler: uses observed state at each step.
        X_hat_used = np.zeros_like(X_obs_used)
        X_hat_used[0] = X_obs_used[0]
        dx_pred = model_viz.predict(X_obs_used)
        for i in range(len(t_viz) - 1):
            dt_step = t_viz[i + 1] - t_viz[i]
            X_hat_used[i + 1] = X_obs_used[i] + dt_step * dx_pred[i]
        if not np.all(np.isfinite(X_hat_used)):
            X_hat_viz = None
        else:
            # If reduced coords, reconstruct R algebraically for visualization only.
            X_hat_viz = _reconstruct_sir_from_si(X_hat_used) if reduced_coords else X_hat_used

        # If reduced coords, we also build R via dynamic conservation:
        # dR/dt = -(dS/dt + dI/dt), using model-predicted derivatives at X_hat_used.
        r_cons_viz = None
        r_dyn_viz = None
        if reduced_coords and X_hat_viz is not None:
            r_cons_viz = 1.0 - X_hat_used[:, 0] - X_hat_used[:, 1]
            dx_hat = model_viz.predict(X_hat_used)
            r_dyn_viz = _integrate_r_from_si_derivatives(
                t_viz,
                r0=r_cons_viz[0],
                dS_dt=dx_hat[:, 0],
                dI_dt=dx_hat[:, 1],
            )
        
        # Prepare overlays for other noise levels.
        n_viz = len(t_viz)
        extra_obs = []
        extra_series = []
        extra_hats = []
        for r in results:
            if r["noise"] == 0.0:
                continue
            x_obs_full_n = r.get("X_obs_full", r.get("X_obs", X_fit))
            x_obs_full_n = x_obs_full_n[:n_viz]
            if coord_idx is None:
                x_obs_used_n = x_obs_full_n
            else:
                x_obs_used_n = r.get("X_obs_used", x_obs_full_n[:, coord_idx])
            dx_pred_n = r["model"].predict(x_obs_used_n)
            x_hat_used_n = np.zeros_like(x_obs_used_n)
            x_hat_used_n[0] = x_obs_used_n[0]
            for i in range(len(t_viz) - 1):
                dt_step = t_viz[i + 1] - t_viz[i]
                x_hat_used_n[i + 1] = x_obs_used_n[i] + dt_step * dx_pred_n[i]
            if reduced_coords:
                x_obs_plot_n = _reconstruct_sir_from_si(x_obs_used_n)
                x_hat_plot_n = _reconstruct_sir_from_si(x_hat_used_n)
            else:
                x_obs_plot_n = x_obs_full_n
                x_hat_plot_n = x_hat_used_n
            extra_obs.append({"noise": r["noise"], "X_obs": x_obs_plot_n})
            extra_series.append({"noise": r["noise"], "X_obs": x_obs_plot_n, "X_hat": x_hat_plot_n})
            extra_hats.append({"noise": r["noise"], "X_hat": x_hat_plot_n})

        # Plot 2: input data verification (ground truth vs noisy inputs).
        print("  - Plotting input data verification...")
        if reduced_coords:
            print("    Note: R curves in fig2 are reconstructed via R = 1 - S - I (not learned).")
        fig1_clean = plot_input_data_verification(
            t_viz,
            X_true_viz,
            X_obs_viz,
            noise_level=0.0,
            extra_obs=None,
            state_names=plot_state_names,
            r_note=r_note,
        )
        handle_figure(fig1_clean, "fig2a_input_data_verification_clean", output_dir, show=show_plots)

        fig1 = plot_input_data_verification(
            t_viz,
            X_true_viz,
            X_obs_viz,
            noise_level=0.0,
            extra_obs=extra_obs,
            state_names=plot_state_names,
            r_note=r_note,
        )
        handle_figure(fig1, "fig2_input_data_verification", output_dir, show=show_plots)
        
        if X_hat_viz is None:
            print("  - Trajectory simulation failed (numerical instability); skipping comparison/error plots.")
        else:
            # Plot 3: trajectory comparison (teacher-forced 1-step).
            print("  - Plotting trajectory comparison...")
            fig2_clean = plot_trajectory_comparison(
                t_viz,
                X_true_viz,
                X_obs_viz,
                X_hat_viz,
                noise_level=0.0,
                extra_series=None,
                state_names=plot_state_names,
                r_note=r_note,
            )
            handle_figure(fig2_clean, "fig3a_teacher_forced_trajectory_clean", output_dir, show=show_plots)

            fig2 = plot_trajectory_comparison(
                t_viz,
                X_true_viz,
                X_obs_viz,
                X_hat_viz,
                noise_level=0.0,
                extra_series=extra_series,
                state_names=plot_state_names,
                r_note=r_note,
            )
            handle_figure(fig2, "fig3_teacher_forced_trajectory", output_dir, show=show_plots)
            
            # Plot 4: error over time (teacher-forced 1-step).
            print("  - Plotting error over time...")
            fig3 = plot_error_over_time(
                t_viz,
                X_true_viz,
                X_hat_viz,
                extra_hats=extra_hats,
                state_names=plot_state_names,
                r_note=r_note,
            )
            handle_figure(fig3, "fig4_error_over_time", output_dir, show=show_plots)

            # Plot 4a: R consistency check (only for reduced coords).
            if reduced_coords and r_cons_viz is not None and r_dyn_viz is not None:
                print("  - Plotting R consistency check (algebraic vs dynamic)...")
                fig3a = plot_r_consistency(
                    t_viz,
                    r_cons=r_cons_viz,
                    r_dyn=r_dyn_viz,
                    r_true=X_true_viz[:, 2],
                    title_suffix="(teacher-forced, noise=0.0)",
                )
                handle_figure(fig3a, "fig4a_r_consistency", output_dir, show=show_plots)

            # Plot 4b: hybrid rollout (teacher-forced until t_switch, then free rollout).
            print("  - Plotting hybrid rollout (teacher-forced → free)...")
            # Teacher-forced part uses fit data; free-rollout continues on full t.
            t_switch = min(t_switch, t_fit[-1])
            switch_idx_full = int(np.searchsorted(t, t_switch, side="right")) - 1
            if switch_idx_full < 0:
                switch_idx_full = 0
            # Align with teacher-forced trajectory length (t_fit starts at t[0]).
            switch_idx_tf = min(switch_idx_full, len(X_hat_viz) - 1)
            if reduced_coords:
                X_hybrid_used = np.zeros((len(t), 2), dtype=float)
                X_hybrid_used[: switch_idx_tf + 1] = X_hat_used[: switch_idx_tf + 1]
                X_roll_hybrid, _ = safe_free_rollout(
                    model_viz,
                    x0=X_hybrid_used[switch_idx_tf],
                    t=t[switch_idx_tf:],
                    method="rk4",
                    domain_check=True,
                    reduced_coords=True,
                )
                if X_roll_hybrid is not None:
                    n_roll = min(len(X_roll_hybrid), len(X_hybrid_used) - switch_idx_tf)
                    X_hybrid_used[switch_idx_tf : switch_idx_tf + n_roll] = X_roll_hybrid[:n_roll]
                # Reconstruct R algebraically for plotting.
                X_hybrid = _reconstruct_sir_from_si(X_hybrid_used)
            else:
                X_hybrid = np.zeros_like(X_true)
                X_hybrid[: switch_idx_tf + 1] = X_hat_viz[: switch_idx_tf + 1]
                X_roll_hybrid, _ = safe_free_rollout(
                    model_viz,
                    x0=X_hybrid[switch_idx_tf],
                    t=t[switch_idx_tf:],
                    method="rk4",
                    domain_check=True,
                )
                if X_roll_hybrid is not None:
                    n_roll = min(len(X_roll_hybrid), len(X_hybrid) - switch_idx_tf)
                    X_hybrid[switch_idx_tf : switch_idx_tf + n_roll] = X_roll_hybrid[:n_roll]
            fig3b = plot_hybrid_rollout_comparison(
                t,
                X_true,
                X_hybrid,
                t_switch=t_switch,
                state_names=plot_state_names,
                r_note=r_note,
            )
            handle_figure(fig3b, "fig5_hybrid_rollout", output_dir, show=show_plots)

            # Plot 4c: additional hybrid rollouts with custom switch times.
            for t_switch_alt in t_switch_alt_list:
                t_switch_alt = min(t_switch_alt, t_fit[-1])
                switch_idx_full = int(np.searchsorted(t, t_switch_alt, side="right")) - 1
                if switch_idx_full < 0:
                    switch_idx_full = 0
                switch_idx_tf = min(switch_idx_full, len(X_hat_viz) - 1)
                if reduced_coords:
                    X_hybrid_alt_used = np.zeros((len(t), 2), dtype=float)
                    X_hybrid_alt_used[: switch_idx_tf + 1] = X_hat_used[: switch_idx_tf + 1]
                    X_roll_alt, _ = safe_free_rollout(
                        model_viz,
                        x0=X_hybrid_alt_used[switch_idx_tf],
                        t=t[switch_idx_tf:],
                        method="rk4",
                        domain_check=True,
                        reduced_coords=True,
                    )
                    if X_roll_alt is not None:
                        n_roll = min(len(X_roll_alt), len(X_hybrid_alt_used) - switch_idx_tf)
                        X_hybrid_alt_used[switch_idx_tf : switch_idx_tf + n_roll] = X_roll_alt[:n_roll]
                    # Reconstruct R algebraically for plotting.
                    X_hybrid_alt = _reconstruct_sir_from_si(X_hybrid_alt_used)
                else:
                    X_hybrid_alt = np.zeros_like(X_true)
                    X_hybrid_alt[: switch_idx_tf + 1] = X_hat_viz[: switch_idx_tf + 1]
                    X_roll_alt, _ = safe_free_rollout(
                        model_viz,
                        x0=X_hybrid_alt[switch_idx_tf],
                        t=t[switch_idx_tf:],
                        method="rk4",
                        domain_check=True,
                    )
                    if X_roll_alt is not None:
                        n_roll = min(len(X_roll_alt), len(X_hybrid_alt) - switch_idx_tf)
                        X_hybrid_alt[switch_idx_tf : switch_idx_tf + n_roll] = X_roll_alt[:n_roll]
                fig3c = plot_hybrid_rollout_comparison(
                    t,
                    X_true,
                    X_hybrid_alt,
                    t_switch=t_switch_alt,
                    state_names=plot_state_names,
                    r_note=r_note,
                )
                handle_figure(
                    fig3c,
                    f"fig5b_hybrid_rollout_switch_{int(t_switch_alt)}",
                    output_dir,
                    show=show_plots,
                )

        # Plot 5: FREE ROLLOUT (model-only integration, no observed states).
        print("\nPlot 5a: FREE ROLLOUT (noise=0.0 only; model-only integration).")
        x0_roll = X_obs_used[0] if reduced_coords else X_obs_viz[0]
        X_roll_used, abort_reason = safe_free_rollout(
            model_viz,
            x0=x0_roll,
            t=t,
            method="rk4",
            domain_check=True,
            reduced_coords=reduced_coords,
        )
        # In reduced coords, free-rollout produces S/I only; R is reconstructed for plotting.
        X_roll = _reconstruct_sir_from_si(X_roll_used) if reduced_coords else X_roll_used

        # If reduced coords, also check R consistency over the full simulation horizon.
        # This uses the model-only free rollout (no teacher forcing).
        if reduced_coords and X_roll_used is not None:
            t_roll = t[: len(X_roll_used)]
            r_cons_roll = 1.0 - X_roll_used[:, 0] - X_roll_used[:, 1]
            dx_roll = model_viz.predict(X_roll_used)
            r_dyn_roll = _integrate_r_from_si_derivatives(
                t_roll,
                r0=r_cons_roll[0],
                dS_dt=dx_roll[:, 0],
                dI_dt=dx_roll[:, 1],
            )
            fig6a = plot_r_consistency(
                t_roll,
                r_cons=r_cons_roll,
                r_dyn=r_dyn_roll,
                r_true=X_true[: len(t_roll), 2],
                title_suffix="(free rollout, full horizon)",
            )
            handle_figure(fig6a, "fig6a_r_consistency_free", output_dir, show=show_plots)

        extra_rolls = []
        for r in results:
            if r["noise"] == 0.0:
                continue
            if reduced_coords:
                x0_r = r.get("X_obs_used", r["X_obs"][:, coord_idx])[0]
                x_roll_n, _ = safe_free_rollout(
                    r["model"],
                    x0=x0_r,
                    t=t,
                    method="rk4",
                    domain_check=True,
                    reduced_coords=True,
                )
                x_roll_plot = _reconstruct_sir_from_si(x_roll_n)
            else:
                x0_r = r["X_obs"][0]
                x_roll_plot, _ = safe_free_rollout(
                    r["model"],
                    x0=x0_r,
                    t=t,
                    method="rk4",
                    domain_check=True,
                )
            extra_rolls.append({"noise": r["noise"], "X_roll": x_roll_plot})
        if abort_reason is None:
            print("[OK] Free rollout completed full horizon")
        else:
            print("[WARN] Free rollout aborted early:", abort_reason)
        fig4a = plot_free_rollout_comparison(
            t=t,
            X_true=X_true,
            X_roll=X_roll,
            noise_level=0.0,
            abort_reason=abort_reason,
            extra_rolls=None,
            state_names=plot_state_names,
            r_note=r_note,
        )
        handle_figure(fig4a, "fig6_free_rollout_noise0", output_dir, show=show_plots)

        print("\nPlot 5b: FREE ROLLOUT (all noise levels; model-only integration).")
        fig4 = plot_free_rollout_comparison(
            t=t,
            X_true=X_true,
            X_roll=X_roll,
            noise_level=0.0,
            abort_reason=abort_reason,
            extra_rolls=extra_rolls,
            state_names=plot_state_names,
            r_note=r_note,
        )
        handle_figure(fig4, "fig7_free_rollout_all_noise", output_dir, show=show_plots)
    
    print("\n" + "="*70)
    print(">>> run_phase1 finished")  # Run end marker.
    print("="*70)
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()


if __name__ == "__main__":  # Script entry point.
    main()
