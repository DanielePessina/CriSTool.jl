# Bringing your own system

Material properties have no implicit lysozyme fallback in a generic system:

```julia
material = CrystallisationSystem(
    saturation_model = ConstantSolubility(10.0),
    crystal_density = 1000.0, volume_shape_factor = 0.5)
```

`molecular_volume` defaults to `nothing`; supply it when a chosen law requires it.
Advanced laws can declare `required_properties(law)`. Validation fails before
solving if required properties are missing. Universal constants retain their SI
defaults on the numerical problem.

A callable growth law takes named parameters and a physical context:

```julia
custom_growth = CallableGrowth(
    (parameters, context) -> parameters.coefficient *
        max(supersaturation(context) - 1, 0)^parameters.order;
    parameters = (coefficient = 1e-9, order = 2.0))
bound_growth = KineticModel(custom_growth)
```

`temperature(context)` and `solvent_state(context)` expose physical values.
Advanced callers may inspect `context.problem`, `context.state` and `context.time`.
`CallableNucleation` has the same two-argument pattern. `CallableLengthGrowth`
takes an additional length in metres and fills solver-owned buffers internally.
Custom types and `paramaxis`/rate dispatch remain available for specialised laws.

Additional solvent variables use named initial conditions and custom
`solvent_dynamics`. Experiments map those initial quantities explicitly through
`initial_from`. Custom observables use local `observable_projections` during
preparation. Only kinetic parameters are selectable for fitting.

Additional material values can be supplied as `properties=(offset=..., viscosity=...)`
and read through `context.system`. Declare needed names with a callable law's
`required=(...)` tuple. Missing properties and nonfinite scalar values fail;
custom laws own the domain rules for their extra properties, including valid
zero/signed values or supplied callable models. Density, shape factor and the
molecular volume required by CNT retain their physical positivity requirements.
