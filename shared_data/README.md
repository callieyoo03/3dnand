# MoS2 GAA compact-model handoff data

This directory contains the static DC/read handoff for the final nominal
2-D axisymmetric MoS2 GAA charge-trap device.

- Geometry: core radius 10 nm, MoS2 2 nm, tunnel Al2O3 3 nm, charge-trap
  HfO2 5 nm, blocking Al2O3 16 nm, gate metal 2 nm, air 2 nm, active length
  100 nm.
- Dielectrics: Al2O3 relative permittivity 8.9; HfO2 19.65; MoS2 6.7;
  core oxide 3.9.
- Five static trapped-electron states span 0 to 2e18 cm^-3. No erased state,
  ERASE simulation, retention dynamics, or time evolution is included.
- `idvg_by_state.csv`, `idvd_by_state.csv`, `metrics_by_state.csv`, and
  `memory_state_map.csv` are the standard electrical interfaces.
- `baseline_comparison_legacy_vs_final.csv` compares against the committed
  legacy 4/5/8 nm, relative-permittivity 9/20 dataset.
- The parameter and topology CSVs state both provenance and runtime support.
- `mesh/` contains the final DEVSIM restart mesh and empty/programmed off/on
  VTK multiblock references.
- Terminal charge/capacitance support and limitations are defined in
  `terminal_charge_capacitance_README.md`.
- `reproducibility_report.csv` is a cell-by-cell repeat-run comparison of all
  five DC and all three Q/C CSVs. It requires exact schemas, row order, text,
  and NaN placement. Numeric cells use `|a-b| <= atol +
  1e-4*max(|a|,|b|)`, with `atol=1e-18` for DC (covering currents below the
  SS-window floor) and `atol=1e-28 C/F` for Q/C. Raw file hashes can differ
  because of solver-level floating-point roundoff. Only the derived,
  near-zero diagnostics use field-specific absolute repeat tolerances:
  `1e-8` for relative-sensitivity values and `1e-23 F` for the cancelling
  `Cgg+Cgd+Cgs` row sum. Primary charge and capacitance values do not receive
  those overrides and remain subject to `1e-28 C/F` plus relative tolerance.
- `handoff_manifest.csv` hashes every delivered artifact except itself; a
  stable self-hash inside the manifest is impossible by construction.

The output current is a full cylindrical contact integral in amperes, not a
per-width current. Local current fields in VTK are A/cm^2. Coordinates are in
centimeters, electric field is V/cm, volume charge is C/cm^3, terminal charge
is C, and capacitance is F.
