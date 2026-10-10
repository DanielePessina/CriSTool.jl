# Running simulations

```julia
using CriSTool

crystal_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
    growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))

batch_run = RunConditions(temperature = 293.15, concentration = 18.0)
trajectory = simulate(crystal_model, batch_run; solver = MoM(), saveat = 0.0:3600.0:28800.0)
trajectory.success
trajectory.concentration[end]
trajectory.d43[end]
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
returns a copy with the named settings replaced. `simulate` returns the solution
without the input problem. Always inspect `trajectory.success` before
interpreting outputs.

For another material, supply a `CrystallisationSystem(saturation_model=...,
crystal_density=..., volume_shape_factor=...)`. Molecular volume may be `nothing`
when unused; CNT requires it and fails explicitly if it is missing.

## Seeds and distributions

```julia
using Distributions
seed_population = DistributionInitialCrystals(
    LogNormal(log(12e-6), 0.2); mass_concentration = 0.25)
seeded_run = RunConditions(temperature = 293.15, concentration = 18.0,
    crystals = seed_population)
seeded_trajectory = simulate(crystal_model, seeded_run;
    solver = FiniteVol(meshsize = 400, lmax = 100e-6), saveat = [0.0, 3600.0])
```

Lengths are metres; solid mass concentration is kg/m³. Number weighting is the
default. A volume-weighted source uses `weighting=:volume`; `number_weighted(seed)`
provides explicit conversion. Conversion divides density by length cubed and
fails when its number normalisation diverges. Explicit truncation belongs to
the supplied distribution. Negative support is rejected.

Measured characteristics remain supported: `LogNormalInitialCrystals` requires
mass, d43 and geometric standard deviation; `GaussianInitialCrystals` requires
mass, d43 and standard deviation. In the model workflow they enter the same
checked distribution builder. Mesh seeds report omitted number/mass fractions
through `seed_domain_diagnostics`, and fail above `mass_loss_tolerance` (default
1e-4). Cell populations are integrated without rescaling lost mass. Mesh moment
accuracy remains a numerical resolution question.

## Solver and operation selection

Use MoM for compact scalar-rate moments, QMOM for quadrature reconstruction,
seeded DQMOM for node-dependent growth, or FiniteVol/WENO for a resolved PSD.
Unsupported combinations fail; solvers are never changed automatically.
See [Solvers](solvers.md) for details and [Operations](operations.md) for batch,
MSMPR and fed-batch contracts. Supply `operation` to `RunConditions`.

## Advanced numerical control

`simulate(model, conditions; solver, saveat, algorithm, solve_options,
callback_factory)` shares the validated numerical solve policy with prepared
fitting. Custom callbacks are constructed once per solve and compose with
package callbacks. `CrystallisationProblem(model, conditions; solver)` builds the
configured problem explicitly; `simulate(problem; saveat)` solves it.
`crystallisation_odeproblem` and `crystallisation_solution` expose the generated
SciML problem and physical reconstruction. Internal DQMOM coordinates differ
from public weights and nodes; preserve that contract when solving directly.

`runsimulation` is the low-level interface for automatic differentiation and
sensitivity work: it takes a flat or `ComponentArray` kinetic parameter vector
and returns `(problem, solution)`. Use `simulate` for ordinary simulations.
