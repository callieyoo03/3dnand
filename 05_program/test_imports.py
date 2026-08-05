# ============================================================
# Import test
# 05_program_erase module dependency check
# ============================================================

import sys
import traceback


MODULES = (
    "device_structure",
    "material_parameters",
    "physics_models",
    "trap_parameters",
    "trap_models",
    "tunneling_parameters",
    "tunneling_models",
)


def test_import(module_name):
    """
    Import one module and print the result.

    Returns
    -------
    bool
        True when the import succeeds.
        False when the import fails.
    """

    try:
        __import__(module_name)

        print(
            f"[PASS] {module_name}"
        )

        return True

    except Exception:
        print(
            f"[FAIL] {module_name}"
        )

        traceback.print_exc()

        return False


def main():

    print()
    print("========================================")
    print("05_program_erase import test")
    print("========================================")
    print()

    passed_count = 0
    failed_modules = []

    for module_name in MODULES:

        success = test_import(
            module_name=module_name,
        )

        if success:
            passed_count += 1
        else:
            failed_modules.append(
                module_name
            )

    print()
    print("========================================")
    print("Import test summary")
    print("========================================")
    print(
        f"Passed: {passed_count}/{len(MODULES)}"
    )

    if failed_modules:
        print(
            "Failed modules:"
        )

        for module_name in failed_modules:
            print(
                f"  - {module_name}"
            )

        return 1

    print(
        "All modules imported successfully."
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )