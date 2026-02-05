"""
Phase 1: Symbolic Regression (SR) recovery for SIR with reduced coordinates.

We learn only S and I equations from (S, I), then derive R by conservation:
    R = 1 - S - I
    dR/dt = -(dS/dt + dI/dt)

This module is standalone and does NOT change existing SINDy code.
"""
