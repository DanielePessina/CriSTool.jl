# CriSTool

Build a material system and kinetic model once, then reuse it for simulation,
parameter estimation and uncertainty propagation. CriSTool supports batch,
MSMPR and fed-batch population balances with five solver families.

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

Start with [Model workflow](model-workflow.md) and [Running simulations](simulation.md).
Then follow [Measurements](measurements.md), [Parameter estimation](parameter-estimation.md)
and [Uncertainty](uq-ensembles.md). [Migration](migration.md) explains the breaking
changes. [Solvers](solvers.md) lists capabilities and restrictions;
[Reactor operations](operations.md) covers flow and steady-state contracts.

[Custom kinetics](kinetics.md), [Bringing your own system](bring-your-own-system.md),
[Temperature profiles](temperature-profiles.md), [Solubility](saturation-models.md)
and [Sensitivity](sensitivity.md) cover advanced extensions.

Instantiate from a checkout with `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
Runnable [tutorials](tutorials.md) use the `examples` environment.
See the [API reference](api.md) for exported interfaces.
