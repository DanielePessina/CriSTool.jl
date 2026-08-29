# Running simulations

CriSTool's main entry point is `runsimulation`, which builds a
`CrystallisationProblem` and returns a `(problem, solution)` tuple.

## Parameter ordering

The parameter vector is always concatenated as:

```
[p_nucleation; p_growth; p_aggregation; p_breakage]
```

The lengths of each block come from the `nparams` field on the kinetic
structs you pass in. `runsimulation` validates this length.

`runsimulation` has two entry points sharing the same kwargs:

```julia
# 1. ComponentArray-native core. Reads parameters.nucl/.gr/.agg/.br directly.
runsimulation(parameters::ComponentArray; nucl, gr, agg, br, solver,
              initial_concentration, save_idx, ...)

# 2. Backward-compatible flat-vector wrapper. Validates length, builds a
#    ComponentArray view via `paramaxis(nucl, gr, agg, br)`, and forwards.
runsimulation(parameters::AbstractVector; nucl, gr, agg, br, ...)
```

Both forms accept AD types (`Vector{Dual}` from ForwardDiff, etc.) and pass
them through unchanged, so optimisers and PE routines that hand in flat
vectors keep working without modification.

For coupled solvent variables, provide `initial_solvent_state` and a
`solvent_dynamics` callable. Solver states keep the population variables first
and the named solvent variables last; `solution.solvent_state` exposes the
resulting trajectories.

To build a structured parameter vector explicitly — useful when you want to
read or set a single field by name — use the composite axis:

```julia
using CriSTool, ComponentArrays

nucl, gr = nucl_CNT(), growth_empirical()
agg, br  = noaggregation(), nobreakage()

flat = [38.0, 0.7, 1.0, 3.0]
p    = ComponentArray(flat, paramaxis(nucl, gr, agg, br))

p.nucl.Aj   # 38.0
p.gr.g      # 3.0
p.gr.g = 2.5
runsimulation(p; nucl, gr, agg, br, solver = MoM(),
              initial_concentration = 18.0, save_idx = 0.0:60.0:480.0)
```

For the kinetic-side per-family axes (e.g. `Axis(Aj=1, γ=2)` for
`nucl_CNT`), see [Kinetics](kinetics.md) and Tutorial 4.

## Basic MoM simulation (keyword interface)

```julia
using CriSTool

params = [38.0, 0.7, 1.0, 3.0]

problem, solution = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)

@show solution.success
@show solution.concentration[end]
@show solution.d43[end]
```

MoM returns moment-based outputs (`d10`, `d32`, `d43`, `mu2`) and does not
provide the full particle size distribution.

## Finite-volume simulation with mesh and time-stepper

```julia
using CriSTool
import OrdinaryDiffEqTsit5

params = [38.0, 0.7, 1.0, 3.0]

problem, solution = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = FiniteVol(meshsize = 200, lmax = 50e-6),
    timestepping_solver = OrdinaryDiffEqTsit5.Tsit5(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)

@show solution.d10q[end]
@show solution.d50q[end]
@show solution.d90q[end]
@show solution.d43[end]   # moment-based volume-mean diameter
@show solution.d32[end]   # moment-based Sauter diameter
```

Finite volume and WENO solvers return a full number density over size,
plus volume-density quantiles (`d10q`, `d50q`, `d90q`) **and** the
moment-derived sizes (`d10`, `d32`, `d43`, `mu2`). The moment-derived
sizes are computed once at solution-construction time, so `sol.d43`
behaves identically on `CrystallisationFVSolution` and
`CrystallisationMoMSolution`. The legacy
`getmomentsizes(prob, sol)` helper is now a thin field-access wrapper
kept for backward compatibility — new code can read `sol.d43` /
`sol.d32` directly.

## Optional initial state

For discretised solvers (`FiniteVol`, `WENO`), you can pass an initial
number density via `initial_state` to match the solver mesh size.

```julia
using CriSTool

solver = FiniteVol(meshsize = 100, lmax = 50e-6)
initial_state = zeros(solver.meshsize)

problem, solution = runsimulation(
    [38.0, 0.7, 1.0, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = solver,
    initial_state = initial_state,
    initial_concentration = 18.0,
    save_idx = 0.0:120.0:480.0,
)
```

## Passing extra problem settings

`runsimulation` forwards extra keyword arguments to
`CrystallisationProblem`, which means you can pass things like
`temp_profile` or `loading` directly:

```julia
using CriSTool

problem, solution = runsimulation(
    [38.0, 0.7, 1.0, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    solver = MoM(),
    initial_concentration = 18.0,
    temp_profile = ConstantTemperature(293.15),
    loading = 0.0,
    save_idx = 0.0:60.0:480.0,
)
```
