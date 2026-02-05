"""
SEIR simulator (synthetic data generator).

This file is the "physics engine" for the toy epidemic system.

Key idea:
- Phase 1 needs a *known ground truth* to test whether SINDy/SR can recover it.
- We generate SEIR trajectories under controllable mechanisms (e.g., seasonality).

Notation:
S: susceptible
E: exposed (latent)
I: infectious
R: removed/recovered
"""

from dataclasses import dataclass  # Data class decorator for parameter containers.
import numpy as np  # Numerical arrays and math.
from scipy.integrate import solve_ivp  # ODE solver for numerical integration.


@dataclass
class SEIRParams:
    """Core SEIR parameters."""
    beta0: float = 2.8       # Baseline transmission rate (R0≈2.8 at gamma=1/5).
    sigma: float = 1 / 8.0   # Incubation -> infectious rate (mean 8 days).
    gamma: float = 1 / 5.0   # Recovery rate (mean infectious period 5 days).
    N: float = 1_000_000.0   # Total population size for mass-action scaling.


@dataclass
class SIRParams:
    """Core SIR parameters."""
    beta0: float = 2.8       # Baseline transmission rate.
    gamma: float = 1 / 5.0   # Recovery rate.
    N: float = 1_000_000.0   # Total population size.

@dataclass
class ForcingParams:
    """
    Optional forcing parameters.

    - seasonal forcing uses sin/cos with period (e.g., 365 days).
    - term forcing is a toy "weekday/weekend" switch (placeholder).
      If you later add a real school calendar, replace term_time_indicator().
    """
    period: float = 365.0    # Seasonal period in days (default: 1 year).
    a_cos: float = 0.0       # Cosine amplitude for seasonal forcing.
    b_sin: float = 0.0       # Sine amplitude for seasonal forcing.
    term_amp: float = 0.0    # Weekday/weekend term-time amplitude (toy).


def term_time_indicator(t: float) -> float:
    """
    Very simple 'term-time' proxy:
    - 1 on weekdays, 0 on weekends

    This is NOT a real school calendar. It's a placeholder to show the *code path*.
    """
    day = int(t) % 7  # Weekly cycle: 0-6.
    return 1.0 if day in [0, 1, 2, 3, 4] else 0.0  # Weekdays=1, weekends=0.


def beta_t(t: float, p: SEIRParams, f: ForcingParams) -> float:
    """
    Time-varying beta(t).

    seasonal: 1 + a*cos(2πt/T) + b*sin(2πt/T)
    term:     1 + term_amp*(term_indicator - 0.5)

    Multiply them to get the final beta(t).
    """
    # Seasonal modulation via Fourier basis (cos/sin) with flexible phase.
    seasonal = 1.0 + f.a_cos * np.cos(2 * np.pi * t / f.period) + f.b_sin * np.sin(2 * np.pi * t / f.period)
    # Term-time modulation: weekday boost, weekend reduction (mean near 1.0).
    term = 1.0 + f.term_amp * (term_time_indicator(t) - 0.5)
    return p.beta0 * seasonal * term  # Final beta(t) = baseline * seasonal * term.


def simulate_seir(
    t_end: float,
    dt: float,
    x0: np.ndarray,
    p: SEIRParams,
    f: ForcingParams,
):
    """
    Simulate SEIR as an ODE.

    Args:
        t_end: total simulation time (days)
        dt: time step in days for saving output
        x0: initial state [S0, E0, I0, R0] in absolute counts
        p: SEIRParams
        f: ForcingParams

    Returns:
        t: array of times
        X: array shape (len(t), 4) containing [S,E,I,R] at each time
    """
    def rhs(t, y):
        # RHS of SEIR ODEs (mass-action, normalized by N).
        S, E, I, R = y
        bt = beta_t(t, p, f)
        dS = -bt * S * I / p.N
        dE = bt * S * I / p.N - p.sigma * E
        dI = p.sigma * E - p.gamma * I
        dR = p.gamma * I
        return [dS, dE, dI, dR]

    # Output grid for storing trajectories.
    t_eval = np.arange(0, t_end + dt, dt)
    # Adaptive ODE solve; returns results only at t_eval.
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y.T  # Rows are [S,E,I,R] at each time.


def simulate_sir(
    t_end: float,
    dt: float,
    x0: np.ndarray,
    p: SIRParams,
):
    """
    Simulate SIR as an ODE.

    Args:
        t_end: total simulation time (days)
        dt: time step in days for saving output
        x0: initial state [S0, I0, R0] in absolute counts
        p: SIRParams

    Returns:
        t: array of times
        X: array shape (len(t), 3) containing [S,I,R] at each time
    """
    def rhs(t, y):
        S, I, R = y
        dS = -p.beta0 * S * I / p.N
        dI = p.beta0 * S * I / p.N - p.gamma * I
        dR = p.gamma * I
        return [dS, dI, dR]

    t_eval = np.arange(0, t_end + dt, dt)
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y.T  # Rows are [S,I,R] at each time.
