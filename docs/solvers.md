# Solvers

CriSTool provides four solvers for the population balance equation:

- `MoM()` (Method of Moments): fast, returns moments only.
- `QMOM(nquadrature=...)`: evolves raw moments and reconstructs a Gaussian
  quadrature for moment-source closures.
- `FiniteVol(meshsize=..., lmax=...)`: full PSD, moderate cost.
- `WENO(meshsize=..., lmax=...)`: higher-order finite volume, more accurate for sharp fronts.

The full runnable examples are [Tutorial 6](<../examples/Tutorial 6 Dissolution.jl>)
for signed dissolution across the three solver families and
[Tutorial 7](<../examples/Tutorial 7 QMOM.jl>) for moment inversion and
quadrature output.

## Outputs by solver

- **MoM**: `concentration`, `d10`, `d32`, `d43`, `moment2`.
- **QMOM**: `concentration`, raw `moments`, reconstructed
  `quadrature_nodes`/`quadrature_weights`, and the moment-derived
  `d10`, `d32`, `d43`, `moment2` metrics. A three-node QMOM evolves `M₀:M₅`.
- **FiniteVol/WENO**: `concentration`, `numberdensity`, `voldensity`,
  volume-density quantiles `d10q`, `d50q`, `d90q`, **and** the
  moment-derived sizes `d10`, `d32`, `d43`, `moment2` (computed once at
  solve time so `sol.d43` works the same way as on a MoM solution).

## Mesh storage

For `FiniteVol` and `WENO`, the mesh data (`cell_face`, `cell_centre`,
`cell_dL`, `meshsize`, `lmin`, and `lmax`) is stored on the solver. The
length-dependent dissolution rate accepts an explicit mesh argument when it
is evaluated outside the solver; use the solver-owned scratch buffer with
`growthrate!` in custom discretised code.

## QMOM and signed rates

QMOM stores raw physical moments in the state and uses `coordinate_scale` only
while inverting those moments. The public quadrature values are crystal
lengths in metres and particle-number weights. Use `quadrature(solution, i)`
to reconstruct the active rule at saved time index `i`.

QMOM currently accepts scalar growth or dissolution kinetics and the supported
volume-additive aggregation and uniform-in-volume breakage closures. The
length-dependent `growth_dissolution_length` model is intentionally a
FiniteVol/WENO model and is rejected by QMOM with an `ArgumentError`.

## Example: QMOM

```julia
using CriSTool

params = [38.0, 0.0007, 1e-9 / 60, 3.0]
problem, solution = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = QMOM(nquadrature = 3),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

@show solution.moments[:, end]
@show quadrature(solution, length(solution.time)).nodes
@show solution.d43[end]
```

## Example: choosing solvers

```julia
using CriSTool

params = [38.0, 0.0007, 1e-9 / 60, 3.0]
nucl, gr = nucl_CNT(), growth_empirical()
agg, br  = noaggregation(), nobreakage()

# MoM
_, sol_mom = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

# Finite volume
_, sol_fv = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = FiniteVol(meshsize = 100, lmax = 50e-6),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

# WENO
_, sol_weno = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = WENO(meshsize = 100, lmax = 50e-6),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)
```

## Overriding the time-stepper

You can override the ODE time-stepping algorithm via the
`timestepping_solver` keyword in `runsimulation` (or by setting the
solver field). This is used in the benchmarking scripts.

```julia
using CriSTool
import OrdinaryDiffEqSSPRK

params = [38.0, 0.0007, 1e-9 / 60, 3.0]

_, sol = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = FiniteVol(meshsize = 200, lmax = 50e-6),
    timestepping_solver = OrdinaryDiffEqSSPRK.SSPRK43(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)
```
