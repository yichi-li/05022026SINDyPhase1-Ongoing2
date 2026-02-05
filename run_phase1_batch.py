"""
Batch runner for Phase 1 experiments.

Set grids below and run:
    python run_phase1_batch.py
"""

import itertools
import os
import subprocess
import sys


def fmt_float(val: float) -> str:
    s = f"{val:.3g}"
    return s.replace(".", "p")


def main():
    # ====== EDIT GRIDS HERE ======================
    model_types = ["sir"]  # "sir", "seir"
    scenarios = ["fast"]   # "fast", "medium", "slow"
    t_end_sim_list = [120.0]  # "fast" = 120, "medium" = 120, "slow" = 120
    t_end_fit_list = [100.0]
    dt_list = [1.0]  # output sampling step (days)
    t_switch_list = [5]  # optional extra hybrid switches (e.g., [5, 10, 15])
    # Optional scenario-specific simulation horizon override.
    # Set to {} to disable.
    scenario_t_end_sim = {"slow": 120.0}
    library_modes = ["polynomial"]    # "restricted", "polynomial"
    restricted_sets = ["basic"]   #   "basic", "extended"
    poly_degrees = [2]
    # thresholds = [0.045, 0.035]
    thresholds = [0.055, 0.050, 0.045, 0.040, 0.035, 0.030, 0.025] #  - "slow" & "fast" & "medium" works
    #thresholds = [1.38, 1.49, 1.52, 1.61, 1.12, 1.04, 0.96]
    reduced_coords_list = [True]  # False or True. -  True enforces S,I only with R=1-S-I (SIR only)
    optimizers = ["stlsq"]  # "stlsq", "sr3"

    # SR3-only hyperparameters (used only when optimizer == "sr3")
    # reg_weight_lam: sparsity weight (NOT the same as STLSQ threshold)
    # relax_coeff_nu: relaxation coefficient (trade-off between fit and sparsity)
    # regularizer: "L0" (harder sparsity) or "L1" (softer sparsity)
    sr3_lams = [0.01, 0.05, 0.1]
    sr3_nus = [0.1, 1.0, 10.0]
    sr3_regularizers = ["L0", "L1"]
    # With sr3_lams=[0.01,0.05,0.1], sr3_nus=[0.1,1.0,10.0], sr3_regularizers=["L0","L1"]:
    # all 18 combinations performed poorly in my tests.
    # =================================================

    # If SR3 is used, we do NOT sweep STLSQ thresholds (they are not applicable).
    if "sr3" in optimizers:
        thresholds = [None]
    # If not using SR3, keep a single placeholder so the SR3 grid still works.
    if "sr3" not in optimizers:
        sr3_lams = [None]
        sr3_nus = [None]
        sr3_regularizers = [None]

    combos = list(itertools.product(
        model_types,
        scenarios,
        t_end_sim_list,
        t_end_fit_list,
        dt_list,
        library_modes,
        restricted_sets,
        poly_degrees,
        thresholds,
        reduced_coords_list,
        optimizers,
        sr3_lams,
        sr3_nus,
        sr3_regularizers,
    ))

    for (model_type, scenario, t_end_sim, t_end_fit, dt,
         library_mode, restricted_set, poly_degree, threshold,
         reduced_coords, optimizer_type, sr3_lam, sr3_nu, sr3_reg) in combos:
        t_end_sim = scenario_t_end_sim.get(scenario, t_end_sim)
        # Skip SR3-only combinations when optimizer is not SR3.
        if optimizer_type != "sr3" and any(v is not None for v in (sr3_lam, sr3_nu, sr3_reg)):
            continue
        if library_mode == "polynomial":
            # restricted_set not used, keep a short placeholder
            rs_tag = "na"
        else:
            rs_tag = restricted_set

        tag = (
            f"mt{model_type}_sc{scenario[0]}_ts{int(t_end_sim)}"
            f"_tf{int(t_end_fit)}_lm{library_mode[0]}_rs{rs_tag}"
            f"_pd{poly_degree}_th{fmt_float(threshold) if threshold is not None else 'na'}"
            f"_rc{int(bool(reduced_coords))}_opt{optimizer_type}"
            f"_lam{fmt_float(sr3_lam) if sr3_lam is not None else 'na'}"
            f"_nu{fmt_float(sr3_nu) if sr3_nu is not None else 'na'}"
            f"_reg{sr3_reg if sr3_reg is not None else 'na'}"
        )
        env = os.environ.copy()
        env.update({
            "PHASE1_MODEL_TYPE": model_type,
            "PHASE1_SCENARIO": scenario,
            "PHASE1_T_END_SIM": str(t_end_sim),
            "PHASE1_T_END_FIT": str(t_end_fit),
            "PHASE1_DT": str(dt),
            "PHASE1_LIBRARY_MODE": library_mode,
            "PHASE1_RESTRICTED_SET": restricted_set,
            "PHASE1_POLY_DEGREE": str(poly_degree),
            "PHASE1_TAG": tag,
            "PHASE1_SHOW_PLOTS": "0",
            "PHASE1_REDUCED_COORDS": "1" if reduced_coords else "0",
            "PHASE1_OPTIMIZER": optimizer_type,
        })
        if threshold is not None:
            env["PHASE1_THRESHOLD"] = str(threshold)
        if optimizer_type == "sr3":
            if sr3_lam is not None:
                env["PHASE1_SR3_LAM"] = str(sr3_lam)
            if sr3_nu is not None:
                env["PHASE1_SR3_NU"] = str(sr3_nu)
            if sr3_reg is not None:
                env["PHASE1_SR3_REG"] = str(sr3_reg)
        if t_switch_list:
            # Additional hybrid rollouts in run_phase1.py (comma-separated list).
            env["PHASE1_T_SWITCH_ALT"] = ",".join(str(v) for v in t_switch_list)

        print(f">>> Running: {tag}")
        result = subprocess.run([sys.executable, "run_phase1.py"], env=env)
        if result.returncode != 0:
            print(f"[WARN] run failed: {tag} (code={result.returncode})")


if __name__ == "__main__":
    main()
