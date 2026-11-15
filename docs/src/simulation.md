# Running simulations

```julia
using CriSTool

crystal_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
    growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))

batch_run = RunConditions(temperature = 293.15, concentration = 18.0)
problem = CrystallisationProblem(crystal_model, batch_run, (0.0, 28_800.0))
trajectory = solve(problem, MoM(); saveat = 3600.0)
trajectory.t                      # saved times (s)
trajectory.retcode                # ReturnCode.Success
successful_retcode(trajectory)
trajectory[:concentration][end]   # same as trajectory.concentration[end]
trajectory[:d43][end]

mesh_trajectory = solve(problem, FiniteVol(meshsize = 100, lmax = 100e-6);
    saveat = 0.0:600.0:28_800.0)
mesh_trajectory[:d50q][end]
cooling_run = merge(batch_run; temperature = LinearTemperature(298.15, -1e-4))
quadrature_trajectory = solve(remake(problem; conditions = cooling_run), QMOM(nquadrature = 3))
```

A model contains the material system and the bound kinetic laws. A law called
with all of its parameter names returns a bound `KineticModel`; `CNTNucleation()`
alone is the unbound law, and a partial set of names is an error. Keywords that
name a law's configuration fields (for example `Lref` on
`EmpiricalLengthGrowth`) configure the law instead of binding a parameter. Read named
values through `crystal_model.growth.parameters`.

`RunConditions(; temperature, concentration, solvent=(;), crystals=nothing,
operation=BatchOperation())` holds everything else about one run. A numeric
temperature is Kelvin and becomes a `ConstantTemperature`; any
`AbstractTemperature` is accepted. `merge(conditions; concentration = 20.0)`
returns a copy with the named settings replaced.

`CrystallisationProblem(model, conditions, tspan)` is one run over
`tspan = (t0, tf)` seconds. Like an `ODEProblem` it holds no numerical method:
`solve(problem, solver; saveat)` picks the solver, so one problem can be solved
with several. `saveat` is a save step (t0, every step and tf are saved) or a
vector of times inside `tspan`; without it every accepted step is saved. There is
no dense interpolation `sol(t)`; choose `saveat` instead.
`remake(problem; model, conditions, tspan, p)` returns a validated copy.
Always check `successful_retcode(trajectory)` (or `trajectory.retcode`) before
interpreting outputs. `trajectory[:name]` returns any named trajectory, the same
as `observable_values(trajectory, :name)`: `:concentration`, `:d43`, `:d50q`
(mesh solvers), `:moments`, or a configured solvent name. Fields such as
`trajectory.concentration` remain available.

For another material, supply a `CrystallisationSystem(saturation_model=...,
crystal_density=..., volume_shape_factor=...)`. Molecular volume may be `nothing`
when unused; CNT requires it and fails explicitly if it is missing.

## Inspecting objects

Conditions, seeds, solvers, operations, models, solutions, fits, inference
results and predictions print a short summary at the REPL. Containers and
`repr` use a one-line form. Sizes are stored in metres and displayed in µm;
`FromObservation` placeholders are shown as such, and failed solves say so.

```text
julia> batch_run
RunConditions
  temperature:   ConstantTemperature(293.15 K)
  concentration: 18.0 kg/m³
  crystals:      none
  operation:     BatchOperation()

julia> trajectory
CrystallisationSolution{MomentPopulation} — moments µ0 … µ4, retcode = Success
  t:             0.0 … 28800.0 s (9 saves)
  concentration: 18.0 → 7.8 kg/m³
  d43:           ...
  ODE:           ... RHS evaluations, ... accepted / ... rejected steps
```

A `FitResult` lists the fitted parameter paths and values with the loss and, for
Optimization.jl solvers, whether the optimiser reported convergence.

## Seeds and distributions

```julia
using Distributions
seed_population = DistributionInitialCrystals(
    LogNormal(log(12e-6), 0.2); mass_concentration = 0.25)
seeded_run = RunConditions(temperature = 293.15, concentration = 18.0,
    crystals = seed_population)
seeded_trajectory = solve(CrystallisationProblem(crystal_model, seeded_run, (0.0, 3600.0)),
    FiniteVol(meshsize = 400, lmax = 100e-6); saveat = [0.0, 3600.0])
```

Lengths are metres; solid mass concentration is kg/m³. Number weighting
(`NumberWeighted()`) is the default. A volume-weighted source uses
`weighting = VolumeWeighted()`; `number_weighted(seed)`
provides explicit conversion. Conversion divides density by length cubed and
fails when its number normalisation diverges. Explicit truncation belongs to
the supplied distribution. Negative support is rejected.

Measured characteristics remain supported: `LogNormalInitialCrystals` requires
mass, d43 and geometric standard deviation; `GaussianInitialCrystals` requires
mass, d43 and standard deviation. In the model workflow they enter the same
checked distribution builder. Mesh seeds report omitted number/mass fractions
through `seed_domain_diagnostics(solver, seed)`, and fail at `solve` above `mass_loss_tolerance` (default
1e-4). Cell populations are integrated without rescaling lost mass. Mesh moment
accuracy remains a numerical resolution question.

## Solver and operation selection

Use MoM for compact scalar-rate moments, QMOM for quadrature reconstruction,
seeded DQMOM for node-dependent growth, or FiniteVol/WENO for a resolved PSD.
Unsupported combinations fail when `solve` is called; solvers are never changed
automatically.
See [Solvers](solvers.md) for details and [Operations](operations.md) for batch,
MSMPR and fed-batch contracts. Supply `operation` to `RunConditions`.

## Advanced numerical control

`solve(problem, solver; saveat, alg, solve_options, callback_factory)` shares
the validated numerical solve policy with prepared fitting. `alg` replaces the
solver's default OrdinaryDiffEq algorithm (for example `alg = Rodas5P()` from
its subpackage). Custom callbacks are constructed once per solve and compose
with package callbacks. `crystallisation_odeproblem(problem, solver)` and
`crystallisation_solution(problem, solver, ode_solution)` expose the generated
SciML problem and physical reconstruction. Internal DQMOM coordinates differ
from public weights and nodes; preserve that contract when solving directly.

The kinetic values are the problem's `p`, a `ComponentArray` on
`paramaxis(problem)` that defaults to the model's bound values
(`kinetic_parameter_vector(problem)`). `solve(remake(problem; p = values), solver; saveat)`
solves for other values, given as a flat or `ComponentArray` vector; their
element type may be a ForwardDiff dual, so this is the interface for automatic
differentiation and sensitivity work.

A configured problem stores each value once: `problem.model` (material system
and bound kinetic laws), `problem.conditions` (temperature, initial solvent
values — read the concentration with `initial_concentration(problem)` — seed
and operation), `problem.tspan`, `problem.p`, `problem.solvent_dynamics` and
`problem.solid_mass_concentration_threshold`. The seed becomes a solver
population only at `solve` time; `initial_state_from_characteristics(problem,
solver, seed)` returns that state. Material properties are read
from the system, e.g. `problem.model.system.crystal_density`. The gas and
Boltzmann constants are the module constants `CriSTool.CRISTOOL_GAS_CONSTANT`
and `CriSTool.CRISTOOL_BOLTZMANN_CONSTANT`.
