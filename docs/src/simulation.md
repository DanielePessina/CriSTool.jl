# Running simulations

CriSTool's main entry point is `runsimulation`, which builds a
`CrystallisationProblem` and returns a `(problem, solution)` tuple.

For a complete comparison of temperature profiles, start with
[Tutorial 1](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl). The tutorial
uses the same kinetics and initial condition for three different temperature
profiles, which makes it a useful template for a first simulation script.

## Choose a solver

Choose the solver from the output you need:

| Need | Solver | Main solution fields |
| --- | --- | --- |
| moment-derived size metrics with a compact state | `MoM()` | `concentration`, `d10`, `d32`, `d43`, `moment2` |
| moments plus a reconstructed Gaussian rule | `QMOM(nquadrature = N)` | the MoM fields, `moments`, `quadrature_nodes`, `quadrature_weights` |
| seeded direct quadrature variables | `DQMOM(nquadrature = N)` | the MoM fields, `moments`, `nodes`, `weights`, and projection diagnostics |
| a resolved particle-size distribution | `FiniteVol(meshsize = ..., lmax = ...)` | `numberdensity`, `voldensity`, `d10q`, `d50q`, `d90q`, and moment-derived sizes |
| a higher-order finite-volume discretisation | `WENO(meshsize = ..., lmax = ...)` | the finite-volume fields and quantiles |

`MoM`, `QMOM`, and `DQMOM` do not store a full size-distribution mesh. Use
`FiniteVol` or `WENO` when you need distribution quantiles. DQMOM can evaluate
length-dependent growth at its nodes, but it currently requires seeded
crystals and does not support length-dependent dissolution.
Use `QMOM` when a compact moment state and a small set of reconstructed nodes
are sufficient. See [Solvers](solvers.md) for the solver-specific details.

## Parameter ordering

The parameter vector is concatenated as:

```
[p_nucleation; p_growth; p_dissolution; p_aggregation; p_breakage]
```

The dissolution block is empty for the default `nodissolution()`. The lengths
of each block come from the `nparams` field on the kinetic structs you pass in.
`runsimulation` validates this length.

For a flat vector, omitting `diss` selects the compatibility four-block layout
`[nucleation; growth; aggregation; breakage]`; the empty dissolution block is
not represented by a placeholder value. Pass an independent dissolution model
with `diss = ...` to select the five-block layout shown above.

`runsimulation` has two entry points sharing the same kwargs:

```julia
# 1. ComponentArray-native core. Reads parameters.nucl/.gr/.diss/.agg/.br directly.
runsimulation(parameters::ComponentArray; nucl, gr, diss, agg, br, solver,
              initial_concentration, save_idx, ...)

# 2. Flat-vector wrapper. Validates length, builds a ComponentArray view via
#    `paramaxis(nucl, gr, diss, agg, br)`, and forwards.
runsimulation(parameters::AbstractVector; nucl, gr, diss, agg, br, ...)
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

flat = [38.0, 0.0007, 1e-9 / 60, 3.0]
p    = ComponentArray(flat, paramaxis(nucl, gr, agg, br))

p.nucl.ln_nucleation_prefactor   # 38.0
p.gr.growth_order                 # 3.0
p.gr.growth_order = 2.5
runsimulation(p; nucl, gr, agg, br, solver = MoM(),
              initial_concentration = 18.0, save_idx = 0.0:3600.0:28800.0)
```

For the kinetic-side per-family axes (for example
`Axis(ln_nucleation_prefactor=1, surface_energy=2)` for
`nucl_CNT`), see [Kinetics](kinetics.md) and Tutorial 4.

## Basic MoM simulation (keyword interface)

```julia
using CriSTool

params = [38.0, 0.0007, 1e-9 / 60, 3.0]

problem, solution = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

@show solution.success
@show solution.concentration[end]
@show solution.d43[end]
```

MoM returns moment-based outputs (`d10`, `d32`, `d43`, `moment2`) and does not
provide the full particle size distribution. Check `solution.success` before
treating a trajectory as a successful simulation.

## QMOM simulation

QMOM evolves `2N` raw moments and reconstructs an `N`-node Gaussian rule at
the saved times. The default `QMOM(nquadrature = 3)` tracks `M₀:M₅`, which
provides the same d32 and d43 observables used by the moment-based losses.

```julia
using CriSTool

problem, solution = runsimulation(
    [38.0, 0.0007, 1e-9 / 60, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = QMOM(nquadrature = 3),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

@show solution.d43[end]
@show solution.moments[:, end]
@show quadrature(solution, length(solution.time)).nodes
```

QMOM accepts scalar signed growth/dissolution kinetics and the validated
volume-additive aggregation and uniform-in-volume breakage closures. Use
`FiniteVol` or `WENO` for `growth_dissolution_length`; QMOM rejects arbitrary
length-dependent kinetics until a corresponding moment closure is defined.

## DQMOM simulation

DQMOM is a seeded direct-quadrature solver. It evolves `N` weights and `N`
physical crystal lengths, then reconstructs the first `2N` raw moments for
observables. Supply either `initial_crystals` or a complete direct initial
state; an empty unseeded population is rejected in v1.

```julia
using CriSTool

initial_crystals = LogNormalInitialCrystals(
    mass_concentration = 0.25,
    d43 = 12e-6,
    geometric_std = 1.25)

problem, solution = runsimulation(
    [3e-9, 1.2];
    nucl = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                nucleation_order = 1.0),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = DQMOM(nquadrature = 3),
    initial_crystals = initial_crystals,
    initial_concentration = 20.0,
    saturation_model = ConstantSolubility(10.0),
    save_idx = 0.0:3600.0:14400.0)

@show solution.nodes[:, end]
@show solution.weights[:, end]
@show solution.d43[end]
```

The public DQMOM population state is `[weights; nodes; solvent_state...]`.
Weights are particle numbers per volume and nodes are metres. Nodes must remain
distinct and above `solver.minimum_size`. Scalar dissolution is supported only
until a node reaches that lower boundary; DQMOM then fails explicitly rather
than deleting a node or changing the active set. See [Solvers](solvers.md) for
the projection equation and the supported-kinetics matrix.

## Finite-volume simulation with mesh and time-stepper

```julia
using CriSTool

params = [38.0, 0.0007, 1e-9 / 60, 3.0]

problem, solution = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = FiniteVol(meshsize = 200, lmax = 50e-6,
                       timestepping_algorithm = :tsit5),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

@show solution.d10q[end]
@show solution.d50q[end]
@show solution.d90q[end]
@show solution.d43[end]   # moment-based volume-mean diameter
@show solution.d32[end]   # moment-based Sauter diameter
```

Finite volume and WENO solvers return a full number density over size,
plus volume-density quantiles (`d10q`, `d50q`, `d90q`) **and** the
moment-derived sizes (`d10`, `d32`, `d43`, `moment2`). The moment-derived
sizes are computed once at solution-construction time, so `sol.d43`
behaves identically on `CrystallisationFVSolution` and
`CrystallisationMoMSolution`. `getmomentsizes(prob, sol)` remains a thin
field-access wrapper. All stored times and sizes are SI seconds and metres.

## Optional initial state

For seeded batches, describe the initial crystal population once and let
CriSTool construct the solver-specific state. The public characteristics use
kg/m³ for crystal mass concentration and metres for `d43`.

```julia
using CriSTool

initial_crystals = LogNormalInitialCrystals(; mass_concentration = 0.25,
                                            d43 = 12e-6,
                                            geometric_std = 1.25)

solver = FiniteVol(meshsize = 100, lmax = 50e-6)

problem, solution = runsimulation(
    [38.0, 0.0007, 1e-9 / 60, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = solver,
    initial_crystals = initial_crystals,
    initial_concentration = 18.0,
    save_idx = 0.0:7200.0:28800.0,
)
```

Use `GaussianInitialCrystals` with `standard_deviation` in metres for a
truncated positive Gaussian profile. Set `d43 = 0.0` and
`mass_concentration = 0.0` for an empty
initial population. The lower-level `initial_state` keyword remains available
when a solver-specific state is already known.

The same state builder is available directly:

```julia
state = initial_state_from_characteristics(
    CrystallisationProblem(; solver = MoM()),
    initial_crystals,
)
```

## Passing extra problem settings

`runsimulation` forwards extra keyword arguments to
`CrystallisationProblem`, which means you can pass things like
`temp_profile` directly:

```julia
using CriSTool

problem, solution = runsimulation(
    [38.0, 0.0007, 1e-9 / 60, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    solver = MoM(),
    initial_concentration = 18.0,
    temp_profile = ConstantTemperature(293.15),
    save_idx = 0.0:3600.0:28800.0,
)
```

`runsimulation` forwards additional keywords to `CrystallisationProblem`. This
is the route for process settings such as `temp_profile`, `saturation_model`,
`initial_solvent_state`, and a custom `solvent_dynamics` callable. See
[Temperature profiles](temperature-profiles.md), [Saturation
models](saturation-models.md), and [Bringing your own system](bring-your-own-system.md)
for those extensions.
