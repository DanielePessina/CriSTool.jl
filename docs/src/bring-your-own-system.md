# Bringing your own system

Material properties have no implicit lysozyme fallback in a generic system:

```julia
material = CrystallisationSystem(
    saturation_model = ConstantSolubility(10.0),
    crystal_density = 1000.0, volume_shape_factor = 0.5)
```

`molecular_volume` defaults to `nothing`; supply it when a chosen law requires it.
Advanced laws can declare `required_properties(law)`. Validation fails before
solving if required properties are missing. The gas and Boltzmann constants are
fixed SI module constants (`CriSTool.CRISTOOL_GAS_CONSTANT`,
`CriSTool.CRISTOOL_BOLTZMANN_CONSTANT`), not problem settings.

A callable growth law takes named parameters and a physical context:

```julia
custom_growth = CallableGrowth(
    (parameters, context) -> parameters.coefficient *
        max(supersaturation(context) - 1, 0)^parameters.order;
    parameters = (coefficient = 1e-9, order = 2.0))
crystal_model = CrystallisationModel(material;
    nucleation = EmpiricalNucleation(log10_nucleation_prefactor = 8.0, nucleation_order = 2.0),
    growth = custom_growth)
```

Only `growth` is required: nucleation defaults to `NoNucleation()` and the
dissolution, aggregation and breakage slots default to their `No*` laws.

```julia
```

Callable laws carry their values and are bound automatically;
`KineticModel(custom_growth)` binds one explicitly.

`temperature(context)` and `solvent_state(context)` expose physical values;
`system_properties(context)` returns the material system's additional `properties`.
Advanced callers may inspect `context.problem`, `context.state` and `context.time`.
`CallableNucleation` has the same two-argument pattern. `CallableLengthGrowth`
takes an additional length in metres and fills solver-owned buffers internally.
Custom types and `paramaxis`/rate dispatch remain available for specialised laws.

Additional solvent variables use `RunConditions(solvent = (pH = 7.0,))` and a
custom `solvent_dynamics` callable, passed as a `CrystallisationProblem` keyword or to
`prepare_fit` through `problem_options = (solvent_dynamics = ...,)`. Experiments
read those initial values from time-zero data with `solvent = (pH =
FromObservation(:pH),)`. Custom observables use local `observable_projections`
during preparation. Only kinetic parameters are selectable for fitting.

Additional material values can be supplied as `properties=(offset=..., viscosity=...)`
and read as `system_properties(context).offset`. Declare needed names with a callable law's
`required=(...)` tuple. Missing properties and nonfinite scalar values fail;
custom laws own the domain rules for their extra properties, including valid
zero/signed values or supplied callable models. Density, shape factor and the
molecular volume required by CNT retain their physical positivity requirements.
