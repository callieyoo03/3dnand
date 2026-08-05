# Final DEVSIM mesh and VTK references

This directory contains the nominal MoS2 cylindrical GAA charge-trap device
used for the final compact-model handoff. It contains static read states only;
no erased-state or retention result is represented here.

## Geometry and coordinates

The DEVSIM mesh is two-dimensional axisymmetric `r-z`. DEVSIM uses `x=r` and
`y=z`; coordinates stored in MSH/VTK are centimeters. Cylindrical integration
uses `CylindricalNodeVolume` and `CylindricalEdgeCouple`, including the physical
`2*pi*r` rotational weighting. No planar-area multiplier is applied.

The active axial length is 100 nm. Nominal radial boundaries are:

| Region | Inner radius (nm) | Outer radius (nm) | Thickness (nm) |
|---|---:|---:|---:|
| CoreOxide | 0 | 10 | 10 |
| MoS2 | 10 | 12 | 2 |
| TunnelOxide (Al2O3) | 12 | 15 | 3 |
| ChargeTrap (HfO2) | 15 | 20 | 5 |
| BlockingOxide (Al2O3) | 20 | 36 | 16 |
| GateMetal | 36 | 38 | 2 |
| Air | 38 | 40 | 2 |

The channel inner/outer diameters are 20/24 nm. `GateMetal` and `Air` are
mesh-only regions in the present solver. The electrostatic gate contact lies on
the outer boundary of `BlockingOxide` at `r=36 nm`.

## Files and reference biases

- `mos2_gaa_final_3_5_16nm.msh`: DEVSIM restart/mesh written at the converged
  empty-trap, zero-bias state.
- `empty_off_reference/`: `Ntrap=0 cm^-3`, `VGS=-1 V`, `VDS=+0.05 V`.
- `empty_on_reference/`: `Ntrap=0 cm^-3`, `VGS=+3 V`, `VDS=+0.05 V`.
- `programmed_off_reference/`: `Ntrap=2e18 cm^-3`, `VGS=-1 V`,
  `VDS=+0.05 V`.
- `programmed_on_reference/`: `Ntrap=2e18 cm^-3`, `VGS=+3 V`,
  `VDS=+0.05 V`.

Each reference directory contains one `.vtm` multiblock file, one `.visit`
index, and one `.vtu` file for each of the seven regions. Open the `.vtm` file
in ParaView to load the complete device. Both index files use relative VTU
basenames, so a reference directory can be moved without retaining the export
machine's absolute path.

DEVSIM emits the blocks in the following deterministic order. The exporter
does not infer these names from the numeric filename: before writing each VTM
`name`, it decodes that VTU's coordinates and verifies the exact radial bounds.

| Block index | Region | Verified radial range (nm) |
|---:|---|---:|
| 0 | Air | 38..40 |
| 1 | BlockingOxide | 20..36 |
| 2 | ChargeTrap | 15..20 |
| 3 | CoreOxide | 0..10 |
| 4 | GateMetal | 36..38 |
| 5 | MoS2 | 10..12 |
| 6 | TunnelOxide | 12..15 |

The final validator also checks the restart mesh declarations as exact sets:
seven regions, three contacts (`source`, `drain`, `gate`), and four dielectric/
channel interfaces. It rejects an unnamed/mislabeled VTM block, an absolute or
nested VTU reference, a VTM/VISIT ordering mismatch, or a per-block radial
range that differs from the nominal stack.

## Exported physical fields

Only existing solved/model quantities and exact component/magnitude transforms
are exported:

- `Potential`: V.
- `ElectricField_x`, `ElectricField_y`, `ElectricFieldMagnitude`: V/cm,
  reconstructed at element centers from the existing edge field.
- `Electrons`: cm^-3, MoS2 only.
- `NetDoping`: cm^-3, MoS2 only.
- `ElectronCurrent_x`, `ElectronCurrent_y`, `ElectronCurrentMagnitude`: local
  A/cm^2 in MoS2. These are not the cylindrically integrated drain current.
- `TrappedElectronDensity`: cm^-3, ChargeTrap only.
- `TrappedChargeDensity`: C/cm^3, ChargeTrap only; trapped electrons are
  negative charge.
- `CylindricalNodeVolume`: cm^3 and `CylindricalSurfaceArea`: cm^2 where the
  corresponding cylindrical models exist.
- `AtContactNode`: generic mesh marker for nodes on a contact.

`Holes` is intentionally omitted: the current electron-only device does not
solve a hole continuity equation and retains only a fixed equilibrium-hole
expression. `Permittivity` is a region parameter, not an existing node or
element model, so it is recorded in `shared_data/material_parameters.csv`
rather than fabricated as a VTK field. Region identity is supplied by the VTM
blocks/VTU filenames; exact region, interface, and contact definitions are in
`shared_data/region_contact_map.csv`. There is no contact-specific VTK ID field.

The CSV `ID_A` values elsewhere in the handoff are full cylindrical contact
integrals in amperes. Local VTK current density, node-volume-integrated charge,
and terminal contact charge must not be interchanged.
