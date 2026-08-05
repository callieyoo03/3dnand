# ============================================================
# Fowler-Nordheim tunneling-model test
# ============================================================

import math

import tunneling_parameters as tp

from tunneling_models import (
    evaluate_tunneling_current_density,
    print_tunneling_result,
)


TEST_FIELDS_V_CM = (
    0.0,
    1.0e6,
    3.0e6,
    5.0e6,
    7.0e6,
)


def main():
    tp.print_tunneling_parameter_summary()

    previous_current_density = -1.0

    for electric_field_V_cm in TEST_FIELDS_V_CM:
        result = (
            evaluate_tunneling_current_density(
                electric_field_V_cm
            )
        )

        print_tunneling_result(
            result
        )

        current_density = result[
            "fowler_nordheim_current_density_A_cm2"
        ]

        if not math.isfinite(
            current_density
        ):
            raise RuntimeError(
                "Non-finite Fowler-Nordheim current density."
            )

        if current_density < 0.0:
            raise RuntimeError(
                "Negative current-density magnitude."
            )

        if (
            previous_current_density >= 0.0
            and current_density
            < previous_current_density
        ):
            raise RuntimeError(
                "FN current density did not increase "
                "with electric-field magnitude."
            )

        previous_current_density = (
            current_density
        )

    print()
    print("=" * 70)
    print("TUNNELING MODEL TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()