# Running simulations

```julia
using CriSTool

crystal_model = CrystallisationModel(
    system = LysozymeSystem(),
    nucleation = KineticModel(nucl_CNT(); parameters = (
        ln_nucleation_prefactor = 38.0, surface_energy = 0.0006)),
    growth = KineticModel(growth_empirical(); parameters = (
        growth_coefficient = 1e-9 / 60, growth_order = 3.0)))

batch_run = CrystallisationProblem(crystal_model;
    initial_conditions = (concentration = 18.0,),
    temperature = ConstantTemperature(293.15), solver = MoM())
trajectory = simulate(batch_run; saveat = 0.0:3600.0:28800.0)
trajectory.success
trajectory.concentration[end]
trajectory.d43[end]

```

Read or replace named kinetic values through `model.growth.parameters` and
construct another `KineticModel` with the same law. A model contains physical
laws; a problem contains the conditions and numerical representation for one run.
`simulate` returns the solution without redundantly returning the input problem.
Always inspect `trajectory.success` before interpreting outputs.

For another material, supply a `CrystallisationSystem(saturation_model=...,
crystal_density=..., volume_shape_factor=...)`. Molecular volume may be `nothing`
when unused; CNT requires it and fails explicitly if it is missing.

## Seeds and distributions

```julia
using Distributions
seed_population = DistributionInitialCrystals(
    LogNormal(log(12e-6), 0.2); mass_concentration = 0.25)
seeded_run = CrystallisationProblem(crystal_model;
    initial_conditions = (concentration = 18.0,),
    temperature = ConstantTemperature(293.15), initial_crystals = seed_population,
    solver = FiniteVol(meshsize = 400, lmax = 100e-6))
seeded_trajectory = simulate(seeded_run; saveat = [0.0, 3600.0])
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
MSMPR and fed-batch contracts. Supply `operation` to the configured problem.

## Advanced numerical control

`simulate(run; saveat, algorithm, solve_options, callback_factory)` shares the
validated numerical solve policy with prepared fitting. Custom callbacks are
constructed once per solve and compose with package callbacks.
`crystallisation_odeproblem` and `crystallisation_solution` expose the generated
SciML problem and physical reconstruction. Internal DQMOM coordinates differ
from public weights and nodes; preserve that contract when solving directly.
