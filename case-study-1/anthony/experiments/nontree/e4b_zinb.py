"""Exp 4b: missing-aware zero-inflated log-normal NB (custom), small var-smoothing grid."""
from ntcommon import ZInfNB, run
for vs in [0.0, 0.1, 0.5]:
    run(ZInfNB(var_smooth=vs), f"ZeroInfl-LogNormal NB vs{vs} (marginalised)", "NB")
