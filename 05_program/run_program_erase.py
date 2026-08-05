# ============================================================
# Program / Erase Simulation
# ============================================================


PROGRAM_VOLTAGE = 10.0
ERASE_VOLTAGE = -10.0

TIME_STEP = 1.0e-8

PROGRAM_TIME = 1.0e-6
ERASE_TIME = 1.0e-6


# Initial charge

Qtrap = 0.0


# ============================================================
# Programming
# ============================================================

print()
print("========================================")
print("PROGRAMMING")
print("========================================")


time = 0.0


while time <= PROGRAM_TIME:

    # Future:
    #
    # 1. apply Vprogram
    #
    # 2. solve electrostatics
    #
    # 3. obtain tunnel oxide electric field
    #
    # 4. calculate tunneling current
    #
    # 5. update trapped charge
    #
    # dQtrap/dt = Jin - Jout

    Jin = 0.0
    Jout = 0.0

    dQdt = Jin - Jout

    Qtrap += dQdt * TIME_STEP

    time += TIME_STEP


print("Programmed Qtrap =", Qtrap)


# ============================================================
# Erase
# ============================================================

print()
print("========================================")
print("ERASE")
print("========================================")


time = 0.0


while time <= ERASE_TIME:

    Jin = 0.0
    Jout = 0.0

    dQdt = Jin - Jout

    Qtrap += dQdt * TIME_STEP

    time += TIME_STEP


print("Erased Qtrap =", Qtrap)