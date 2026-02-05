"""
Phase 2 (PySINDy): Slightly different mechanisms via a "mechanism library".

Key trick:
- Many epidemic models are **non-autonomous** (beta depends on time: seasonality, term time).
- Standard SINDy assumes autonomous dynamics: xdot = f(x).
- To handle forcing, we provide forcing basis functions u(t) and learn xdot = f(x, u).

Implementation approach:
- Build U(t) = [sin(2πt/T), cos(2πt/T), term(t), ...]
- Augment inputs: [X, U]
- Fit SINDy on the augmented variables, then interpret selected terms.

This is a minimal demo that lets SINDy "choose" whether forcing matters.


"""
