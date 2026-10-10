# CriSTool

Build a material system and kinetic model once, then reuse it for simulation,
parameter estimation and uncertainty propagation. CriSTool supports batch,
MSMPR and fed-batch population balances with five solver families.

```julia
using CriSTool

crystal_model = CrystallisationModel(LysozymeSystem();
    nucleation = nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
    growth = growth_empirical(growth_coefficient = 1e-9 / 60, growth_order = 3.0))

batch_run = RunConditions(temperature = 293.15, concentration = 18.0)
trajectory = simulate(crystal_model, batch_run; solver = MoM(), saveat = 0.0:3600.0:28800.0)
trajectory.success
trajectory.concentration[end]
trajectory.d43[end]
```

Calling a law with all of its parameter names binds the values; `nucl_CNT()`
alone is the unbound law. `RunConditions` hold everything else about a run:
temperature, initial concentration, an optional seed and the reactor operation.
`runsimulation` remains the low-level interface for automatic differentiation
and sensitivity work.

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
