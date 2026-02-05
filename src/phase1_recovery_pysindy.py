"""
Phase 1: Recovery experiments with PySINDy.

Phase 1.5 adjustment:
- Restrict the SINDy library to {S*I, E, I} (+ optional constant)
  so we can test whether the simplest mechanism set recovers the
  true SEIR equations.
"""

import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error


# ----------------------------
# Phase 1.5: restricted library
# ----------------------------

def _build_phase15_library(
    restricted_set: str = "basic",
    include_bias: bool = False,
    model_type: str = "seir",
):
    """Build a restricted mechanism library (+ optional constant).

    Supported restricted_set values (SEIR):
    - "basic":    {S*I, E, I}
    - "extended": {S*I, E, I, S*E}

    Supported restricted_set values (SIR):
    - "basic":    {S*I, I}
    - "extended": {S*I, I, S}
    """
    if model_type.lower() == "sir":
        def _si(s, i, r):
            return s * i

        def _i(s, i, r):
            return i

        def _s(s, i, r):
            return s

        if restricted_set == "extended":
            functions = [_si, _i, _s]
            name_funcs = [
                lambda *_: "S*I",
                lambda *_: "I",
                lambda *_: "S",
            ]
        else:
            functions = [_si, _i]
            name_funcs = [
                lambda *_: "S*I",
                lambda *_: "I",
            ]
    else:
        def _si(s, e, i, r):
            return s * i

        def _se(s, e, i, r):
            return s * e

        def _e(s, e, i, r):
            return e

        def _i(s, e, i, r):
            return i

        if restricted_set == "extended":
            functions = [_si, _e, _i, _se]
            name_funcs = [
                lambda *_: "S*I",
                lambda *_: "E",
                lambda *_: "I",
                lambda *_: "S*E",
            ]
        else:
            # Default restricted set (3 terms total).
            functions = [_si, _e, _i]
            name_funcs = [
                lambda *_: "S*I",
                lambda *_: "E",
                lambda *_: "I",
            ]
    if include_bias:
        def _const(*args):
            return np.ones_like(args[0])

        functions.append(_const)
        name_funcs.append(lambda *_: "1")

    try:
        return ps.CustomLibrary(library_functions=functions, function_names=name_funcs)
    except Exception:
        pass

    try:
        return ps.CustomLibrary(library_functions=functions)
    except Exception:
        return None


def _phase15_keep_feature_indices(
    model,
    input_features=("S", "E", "I", "R"),
    include_bias: bool = False,
    model_type: str = "seir",
    restricted_set: str = "basic",
):
    """Return indices of features to keep for masking."""
    try:
        names = model.feature_library.get_feature_names(input_features=list(input_features))
    except Exception:
        try:
            names = model.feature_library.get_feature_names()
        except Exception:
            names = None

    if names is None:
        return None, None

    keep = set()
    for idx, name in enumerate(names):
        n = name.replace("  ", " ").strip()
        if include_bias and n in ("1", "bias", "const"):
            keep.add(idx)
            continue
        if model_type.lower() == "sir":
            if n in ("I", "x1"):
                keep.add(idx)
                continue
            if restricted_set == "extended" and n in ("S", "x0"):
                keep.add(idx)
                continue
            if n in ("S I", "S*I", "x0 x1", "x0*x1"):
                keep.add(idx)
                continue
        else:
            if n in ("E", "x1"):
                keep.add(idx)
                continue
            if n in ("I", "x2"):
                keep.add(idx)
                continue
            if n in ("S I", "S*I", "x0 x2", "x0*x2"):
                keep.add(idx)
                continue
            if restricted_set == "extended" and n in ("S E", "S*E", "x0 x1", "x0*x1"):
                keep.add(idx)
                continue

    return sorted(list(keep)), names


def _phase15_apply_coefficient_mask(model, keep_idx):
    """Zero-out coefficients for features not in keep_idx (in-place)."""
    if keep_idx is None:
        return False
    try:
        coef = model.coefficients()
    except Exception:
        return False
    if coef is None:
        return False

    mask = np.ones(coef.shape[1], dtype=bool)
    mask[keep_idx] = False
    coef[:, mask] = 0.0

    # Write back in the most compatible way
    try:
        model.optimizer.coef_ = coef
        return True
    except Exception:
        pass
    try:
        model._coefficients = coef
        return True
    except Exception:
        return False


# ----------------------------
# Core API used by run_phase1
# ----------------------------

def fit_sindy(
    X: np.ndarray,
    t: np.ndarray,
    poly_degree: int = 2,
    threshold: float = 0.02,
    library_mode: str = "restricted",
    restricted_set: str = "basic",
    model_type: str = "seir",
    optimizer_type: str = "stlsq",
    sr3_reg_weight_lam: float = None,
    sr3_relax_coeff_nu: float = None,
    sr3_regularizer: str = None,
) -> ps.SINDy:
    """
    Fit continuous-time SINDy on provided states (full or reduced).

    Args:
        X: (T, 4) observed states [S,E,I,R]
        t: (T,) time points
        poly_degree: polynomial library degree (used only when library_mode="polynomial")
        threshold: sparsity threshold in STLSQ
        library_mode: "restricted" or "polynomial"
        restricted_set: which restricted term set to use when library_mode="restricted"
        model_type: "seir" or "sir"
        optimizer_type: "stlsq", "sr3", or "constrained_sr3"
        sr3_reg_weight_lam: SR3 sparsity weight (not the same as STLSQ threshold)
        sr3_relax_coeff_nu: SR3 relaxation coefficient (trade-off between fit and sparsity)
        sr3_regularizer: SR3 regularizer ("L0" or "L1")

    Returns:
        fitted ps.SINDy model
    """
    # Inputs: X is (T, 4) state fractions [S, E, I, R]; t is (T,) time grid.
    # Derivatives are NOT provided; they are estimated internally from X and t.
    diff = ps.SmoothedFiniteDifference()

    using_mask_fallback = False
    if library_mode == "polynomial":
        lib = ps.PolynomialLibrary(degree=poly_degree, include_interaction=True, include_bias=False)
    else:
        # Try to use the truly restricted library first (3 terms total).
        lib = _build_phase15_library(
            restricted_set=restricted_set,
            include_bias=False,
            model_type=model_type,
        )
        if lib is None:
            # Fallback only if CustomLibrary is incompatible:
            # fit with full polynomial library, then zero-out all coefficients
            # except S*I, E, I. This keeps the effective model sparse even if
            # the printed candidate list from run_phase1 shows 14 polynomial terms.
            lib = ps.PolynomialLibrary(degree=poly_degree, include_interaction=True, include_bias=False)
            using_mask_fallback = True

    # Regression step: choose optimizer type (default: STLSQ).
    # Note: SR3 can be more stable under collinearity in some settings.
    optimizer_type = (optimizer_type or "stlsq").lower()
    if optimizer_type == "sr3":
        # NOTE: PySINDy SR3 API differs by version.
        # Older versions used "threshold"/"nu"; newer (>=2.0) uses
        # reg_weight_lam (sparsity weight) and relax_coeff_nu.
        # We map the user "threshold" to reg_weight_lam ONLY if no SR3-specific
        # parameter is provided. This keeps backward compatibility but is not
        # semantically identical, so prefer sr3_reg_weight_lam in new runs.
        reg_weight_lam = threshold if sr3_reg_weight_lam is None else sr3_reg_weight_lam
        relax_coeff_nu = 1.0 if sr3_relax_coeff_nu is None else sr3_relax_coeff_nu
        regularizer = "L0" if sr3_regularizer is None else sr3_regularizer
        try:
            opt = ps.SR3(
                threshold=reg_weight_lam,
                nu=relax_coeff_nu,
                normalize_columns=True,
            )
        except TypeError:
            opt = ps.SR3(
                reg_weight_lam=reg_weight_lam,
                regularizer=regularizer,
                relax_coeff_nu=relax_coeff_nu,
                normalize_columns=True,
            )
    elif optimizer_type == "constrained_sr3":
        raise NotImplementedError("constrained_sr3 is planned but not implemented yet.")
    else:
        opt = ps.STLSQ(threshold=threshold, alpha=1e-6, normalize_columns=True)
    model = ps.SINDy(
        feature_library=lib,
        optimizer=opt,
        differentiation_method=diff,
    )
    model.fit(X, t=t)

    if using_mask_fallback and library_mode != "polynomial":
        if model_type.lower() == "sir":
            input_features = ["S", "I", "R"]
        else:
            input_features = ["S", "E", "I", "R"]
        keep_idx, _ = _phase15_keep_feature_indices(
            model,
            input_features=input_features,
            include_bias=False,
            model_type=model_type,
            restricted_set=restricted_set,
        )
        _phase15_apply_coefficient_mask(model, keep_idx)

    return model


def simulate_model(model: ps.SINDy, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Simulate the learned SINDy model from x0."""
    return model.simulate(x0, t)


def safe_simulate_model(model: ps.SINDy, x0: np.ndarray, t: np.ndarray):
    """Simulate with basic safety checks; return None on failure."""
    try:
        X_hat = model.simulate(x0, t)
    except Exception:
        return None
    if not np.all(np.isfinite(X_hat)):
        return None
    return X_hat


def recovery_grid_experiment(
    X_true: np.ndarray,
    t: np.ndarray,
    noise_levels=(0.0, 0.01, 0.05, 0.1),
    poly_degree: int = 2,
    threshold: float = 0.02,
    seed: int = 0,
    simulate: bool = False,
    simulate_t: np.ndarray = None,
    library_mode: str = "restricted",
    restricted_set: str = "basic",
    model_type: str = "seir",
    reduced_coords: bool = False,
    optimizer_type: str = "stlsq",
    sr3_reg_weight_lam: float = None,
    sr3_relax_coeff_nu: float = None,
    sr3_regularizer: str = None,
):
    """
    Add Gaussian noise to the true states and refit SINDy.

    Noise model:
    - For each state dimension, add N(0, (noise_level * std(state))^2).
    - No reporting delays, under-reporting, or aggregation are applied here.

    Returns:
        list of dicts with fields: noise, mse, model, X_obs_full, X_obs_used
    """
    # X_true should already be normalized fractions (not absolute counts).
    rng = np.random.default_rng(seed)
    scale = np.std(X_true, axis=0, keepdims=True)
    results = []

    # Reduced-coordinates: keep only S and I (SIR only for now).
    # This enforces the invariant manifold at the coordinate level.
    coord_idx = None
    if reduced_coords:
        if model_type.lower() != "sir":
            raise ValueError("reduced_coords currently supports only model_type='sir'.")
        coord_idx = [0, 1]  # S, I

    for nl in noise_levels:
        # Gaussian noise per state: std = nl * std(state).
        X_obs = X_true + rng.normal(0, nl, size=X_true.shape) * scale

        # Use reduced coordinates if requested.
        X_obs_used = X_obs if coord_idx is None else X_obs[:, coord_idx]

        model = fit_sindy(
            X_obs_used,
            t,
            poly_degree=poly_degree,
            threshold=threshold,
            library_mode=library_mode,
            restricted_set=restricted_set,
            model_type=model_type,
            optimizer_type=optimizer_type,
            sr3_reg_weight_lam=sr3_reg_weight_lam,
            sr3_relax_coeff_nu=sr3_relax_coeff_nu,
            sr3_regularizer=sr3_regularizer,
        )

        X_hat = None
        if simulate:
            t_sim = t if simulate_t is None else simulate_t
            x0_sim = X_obs_used[0]
            X_hat = safe_simulate_model(model, x0_sim, t_sim)

        # Evaluation compares derivatives: true dX/dt from X_true vs model.predict(X_obs).
        Xdot_true = np.gradient(X_true, t, axis=0)
        Xdot_true_used = Xdot_true if coord_idx is None else Xdot_true[:, coord_idx]
        Xdot_pred = model.predict(X_obs_used)
        mse = mean_squared_error(Xdot_true_used, Xdot_pred)

        results.append({
            "noise": nl,
            "mse": float(mse),
            "model": model,
            # Keep both full and reduced observations for downstream plotting.
            "X_obs": X_obs.copy(),  # backward-compatible: full noisy states
            "X_obs_full": X_obs.copy(),
            "X_obs_used": X_obs_used.copy(),
            "X_hat": None if X_hat is None else X_hat.copy(),
            "coord_idx": None if coord_idx is None else list(coord_idx),
        })

    return results
