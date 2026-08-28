# Solvers

CriSTool provides three solvers for the population balance equation:

- `MoM()` (Method of Moments): fast, returns moments only.
- `FiniteVol(meshsize=..., lmax=...)`: full PSD, moderate cost.
- `WENO(meshsize=..., lmax=...)`: higher-order finite volume, more accurate for sharp fronts.

## Outputs by solver

- **MoM**: `concentration`, `d10`, `d32`, `d43`, `mu2`.
- **FiniteVol/WENO**: `concentration`, `numberdensity`, `voldensity`,
  volume-density quantiles `d10q`, `d50q`, `d90q`, **and** the
  moment-derived sizes `d10`, `d32`, `d43`, `mu2` (computed once at
  solve time so `sol.d43` works the same way as on a MoM solution).

## Mesh storage

For `FiniteVol` and `WENO`, the mesh data (`cell_face`, `cell_centre`,
`cell_dL`, `meshsize`, `lmin`, `lmax`) lives as fields on the solver
struct alongside the algorithm config. See
[`docs/adr/0001-mesh-stays-on-discretised-solver.md`](../../docs/adr/0001-mesh-stays-on-discretised-solver.md)
for the rationale (no separate `Mesh` type until a second mesh shape
lands). The single rate function that needed mesh access from outside
the solver, `growth_dissolution_length`, takes an explicit
`mesh::AbstractVector` argument rather than reaching into the solver.

## Example: choosing solvers

```julia
using CriSTool

params = [38.0, 0.7, 1.0, 3.0]
nucl, gr = nucl_CNT(), growth_empirical()
agg, br  = noaggregation(), nobreakage()

# MoM
_, sol_mom = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)

# Finite volume
_, sol_fv = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = FiniteVol(meshsize = 100, lmax = 50e-6),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)

# WENO
_, sol_weno = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = WENO(meshsize = 100, lmax = 50e-6),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)
```

## Overriding the time-stepper

You can override the ODE time-stepping algorithm via the
`timestepping_solver` keyword in `runsimulation` (or by setting the
solver field). This is used in the benchmarking scripts.

```julia
using CriSTool
import OrdinaryDiffEqSSPRK

params = [38.0, 0.7, 1.0, 3.0]

_, sol = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = FiniteVol(meshsize = 200, lmax = 50e-6),
    timestepping_solver = OrdinaryDiffEqSSPRK.SSPRK43(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)
```
