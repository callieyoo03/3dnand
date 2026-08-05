# Terminal charge and quasi-static capacitance

These files are a static, fixed-trap extension of the electron-only DEVSIM
read model. They use the same 2-D axisymmetric `r-z` device as the IV data.
All reported charges are full `2*pi*r` cylindrical integrals in coulombs; no
planar area, per-unit-depth multiplier, fitting scale, or current-times-time
conversion is used.

## Supported quantities

- `Qg_C` is the instantaneous electrostatic gate-boundary charge returned by
  DEVSIM's `get_contact_charge(..., equation="PotentialEquation")`. The gate
  is the insulating outer boundary of `BlockingOxide`.
- `Qtrap_C` is `sum(TrappedChargeDensity*CylindricalNodeVolume)` in HfO2.
- `Qmobile_C` is
  `q*sum((EquilibriumHoles-Electrons)*CylindricalNodeVolume)` in MoS2.
  Electrons are solved; holes are a fixed equilibrium reference, not a solved
  hole-continuity variable.
- `Qfixed_C` is
  `q*sum(NetDoping*CylindricalNodeVolume)` in MoS2.
- `Cgg`, `Cgd`, and `Cgs` use the direct derivative convention
  `Cij=dQi/dVj`, so the off-diagonal values are normally negative. Ntrap is
  held fixed during every perturbation.

## Intentionally unsupported quantities

`Qd_C` and `Qs_C` are `NaN`. The present source/drain contacts are ideal
conducting end faces of the MoS2 channel, without explicit source/drain
regions or a Ward-Dutton channel-charge partition. The DEVSIM Poisson
boundary values are therefore retained only as
`Qd_poisson_boundary_flux_C` and `Qs_poisson_boundary_flux_C` for a discrete
Gauss-law diagnostic. They are not labeled as compact-model Qd/Qs.

Rows whose measured terminal is drain or source in
`capacitance_matrix_by_state.csv` are likewise explicit unsupported `NaN`
entries. Only the measured-terminal `gate` rows are supported.

## Conservation and finite differences

The diagnostic sum is

```text
Qg + Qd_poisson_boundary_flux + Qs_poisson_boundary_flux
   + Qtrap + Qmobile + Qfixed
```

`charge_conservation_error_C` is the absolute value of that signed sum. The
calculation uses exactly the mesh's `CylindricalNodeVolume` and contact-flux
weights.

Capacitance is obtained by converged central DC re-solves at `+deltaV` and
`-deltaV`, followed by a converged restoration of the reference point.
Sensitivity is reported for 0.5, 1.0, and 2.0 mV; 1.0 mV is the nominal value.
A failed endpoint or restore produces `NaN` and an error rather than a
one-sided derivative.

The generator fails if the maximum relative delta-voltage sensitivity exceeds
`1e-2`, or if `abs(Cgg+Cgd+Cgs)` exceeds `1e-22 F`. Repeat-run comparison keeps
the primary Q/C tolerance at `1e-28 C/F + 1e-4*scale`; only these near-zero
derived diagnostics use the explicitly documented absolute overrides in the
top-level README and reproducibility report.

## Scope and limitations

- This is quasi-static DC capacitance, not an AC/RF frequency response.
- Trap occupancy does not respond to the perturbation; there is no trap
  capacitance or retention dynamics.
- The gate boundary currently uses zero work-function offset. Absolute Qg
  depends on that uncalibrated boundary assumption.
- Quantum confinement and quantum capacitance are absent.
- The mesh-volume quadrature is discrete; conservation uses the same weights
  and is therefore the relevant numerical check.
- Ioff and ON/OFF in the companion IV data are sensitive to the numerical
  leakage floor and are not primary fitting targets.
