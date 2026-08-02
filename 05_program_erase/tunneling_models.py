# ============================================================
# Tunneling Models
#
# Future models:
#
# Fowler-Nordheim tunneling
# Direct tunneling
# Trap-assisted tunneling
# ============================================================


import math
import tunneling_parameters as tp


# ============================================================
# Fowler-Nordheim tunneling
# ============================================================

def fowler_nordheim_current(Eox):

    if Eox <= 0:
        return 0.0

    J = (
        tp.FN_A
        * Eox**2
        * math.exp(
            -tp.FN_B / Eox
        )
    )

    return J


# ============================================================
# Direct tunneling
# ============================================================

def direct_tunneling_current(Eox):

    # Placeholder

    return 0.0


# ============================================================
# Trap-assisted tunneling
# ============================================================

def tat_current(Eox):

    # Placeholder

    return 0.0


# ============================================================
# Total injection current
# ============================================================

def total_tunneling_current(Eox):

    return (
        fowler_nordheim_current(Eox)
        + direct_tunneling_current(Eox)
        + tat_current(Eox)
    )