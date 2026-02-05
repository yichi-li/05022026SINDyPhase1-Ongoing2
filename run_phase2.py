"""
Not yet fully developed.

Entry point: Phase 2 demo (PySINDy) with "slightly different" mechanisms.

This script demonstrates:
- true world includes seasonality + term forcing
- observation includes reporting + delay + NegBin noise + weekly aggregation
- SINDy is given forcing bases U(t) and can select terms involving U(t)

Run:
    python run_phase2.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src.seir_sim import SEIRParams, ForcingParams, simulate_seir
from src.observation import (
    incidence_from_states,
    apply_reporting_delay,
    negbin_noise,
    aggregate,
)
from src.phase2_slightly_different_pysindy import build_forcing, fit_sindy_with_forcing


def main():
    p = SEIRParams(beta0=2.8, sigma=1 / 8, gamma=1 / 5, N=1_000_000)

    # TRUE hidden mechanism: seasonality + term forcing
    f_true = ForcingParams(period=365.0, a_cos=0.15, b_sin=0.05, term_amp=0.25)

    x0 = np.array([p.N - 10.0, 5.0, 5.0, 0.0])
    t_end, dt = 365.0 * 3, 1.0
    t, X = simulate_seir(t_end, dt, x0, p, f_true)

    # ---- observation process (what you'd see in real data) ----
    inc = incidence_from_states(X, sigma=p.sigma, dt=dt)
    mean_cases = apply_reporting_delay(inc, rho=0.25, delay_steps=7)
    cases = negbin_noise(mean_cases, overdisp=30.0)
    weekly_cases = aggregate(cases, window=7)
    weekly_t = t[::7][: len(weekly_cases)]

    # Plot observed weekly cases
    plt.figure()
    plt.plot(weekly_t, weekly_cases)
    plt.xlabel("t (days)")
    plt.ylabel("Weekly reported cases")
    plt.title("Observed cases (synthetic): reporting+delay+NegBin+weekly aggregation")
    plt.show()

    # ---- mechanism library for Phase 2 ----
    # Provide forcing bases U(t) to SINDy and let it select terms
    U = build_forcing(t, period=365.0, use_term=True)
    model = fit_sindy_with_forcing(X, t, U, poly_degree=3, threshold=0.2)

    print("\nRecovered augmented SINDy model (states + forcing):")
    model.print()

    print("\nTip:")
    print("- The learned equations include terms involving forcing bases (sin/cos/term) if they help.")
    print("- In a real project, you'd compare model variants with/without forcing and test forecasting.")


if __name__ == "__main__":
    main()
