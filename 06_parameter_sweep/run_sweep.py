# ============================================================
# Parameter Sweep
# MoS2 GAA Charge-Trap Memory
# ============================================================


# ============================================================
# Sweep parameters
# ============================================================


mos2_thickness_list = (
    0.7,
    1.4,
    2.0,
    3.0,
)
# nm


tunnel_oxide_list = (
    2.0,
    3.0,
    4.0,
    5.0,
)
# nm


trap_density_list = (
    1e17,
    5e17,
    1e18,
    5e18,
)
# cm^-3


program_voltage_list = (
    4.0,
    6.0,
    8.0,
    10.0,
)
# V


# ============================================================
# Results
# ============================================================

results = []


# ============================================================
# Sweep
# ============================================================

for mos2_thickness in mos2_thickness_list:

    for tox in tunnel_oxide_list:

        for trap_density in trap_density_list:

            for Vpgm in program_voltage_list:

                print()
                print("--------------------------------")
                print(
                    "MoS2 =",
                    mos2_thickness,
                    "nm",
                )

                print(
                    "TOX =",
                    tox,
                    "nm",
                )

                print(
                    "Nt =",
                    trap_density,
                )

                print(
                    "Vpgm =",
                    Vpgm,
                    "V",
                )


                # ============================================
                # Future simulation
                # ============================================
                #
                # 1. regenerate geometry
                #
                # 2. solve transistor
                #
                # 3. program device
                #
                # 4. calculate trapped charge
                #
                # 5. run ID-VG
                #
                # 6. extract Vth
                #
                # 7. calculate Memory Window
                #
                # ============================================


                memory_window = None

                results.append(
                    {
                        "mos2_thickness":
                            mos2_thickness,

                        "tox":
                            tox,

                        "trap_density":
                            trap_density,

                        "program_voltage":
                            Vpgm,

                        "memory_window":
                            memory_window,
                    }
                )


print()
print("========================================")
print("PARAMETER SWEEP FINISHED")
print("========================================")

print(
    "Number of simulations =",
    len(results),
)