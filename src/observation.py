"""
Observation model: map latent SEIR states -> observed reported cases.

Why do we need this?
- Real epidemic data rarely gives you S/E/I/R directly.
- It gives you "reported cases", often noisy, delayed, under-reported, and aggregated.

This file implements:
1) incidence ~ sigma * E * dt  (flow from E -> I)
2) reporting rate rho
3) reporting delay (shift)
4) negative binomial noise (over-dispersion)
5) aggregation to weekly/monthly counts
"""

import numpy as np  

