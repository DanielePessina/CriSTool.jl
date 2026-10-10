"""
    CrystallisationSystem(; saturation_model, crystal_density, volume_shape_factor,
                          molecular_volume=nothing)

Material properties in SI units. Missing optional properties are `nothing`;
selected kinetic laws declare their requirements through `required_properties`.
"""
Base.@kwdef struct CrystallisationSystem{Solubility, MolecularVolume, Properties <: NamedTuple}
    saturation_model::Solubility
    crystal_density::Float64
    volume_shape_factor::Float64
    molecular_volume::MolecularVolume = nothing
    properties::Properties = (;)
end

function Base.getproperty(material_system::CrystallisationSystem, field_name::Symbol)
    field_name in fieldnames(typeof(material_system)) && return getfield(material_system, field_name)
    return getproperty(getfield(material_system, :properties), field_name)
end
Base.propertynames(material_system::CrystallisationSystem) =
    (fieldnames(typeof(material_system))..., keys(getfield(material_system, :properties))...)


"""`LysozymeSystem(; kwargs...)` returns an ordinary material system with lysozyme defaults."""
LysozymeSystem(; kwargs...) = CrystallisationSystem(;
    merge((; saturation_model = lysozyme_solubility(), crystal_density = 1370.0,
        volume_shape_factor = 0.81, molecular_volume = 2.97e-26), (; kwargs...))...)

"""`required_properties(law)` declares additional material properties used by a law."""
required_properties(kinetic_law) = ()
required_properties(::Union{CNTNucleation, CNTNucleationNoSupersaturation}) = (:molecular_volume,)
required_properties(combined_law::CombinedNucleation) =
    Tuple(union(required_properties(combined_law.primary), required_properties(combined_law.secondary)))

function _validate_required_material_property(kinetic_law, required_name, required_value)
    required_value !== nothing && (!(required_value isa Real) || isfinite(required_value)) ||
        throw(ArgumentError("$(typeof(kinetic_law)) requires supplied finite $required_name."))
    if required_name === :molecular_volume
        required_value isa Real && required_value > 0 ||
            throw(ArgumentError("$(typeof(kinetic_law)) requires positive molecular_volume."))
    end
    return nothing
end

"""
    KineticModel(law; parameters=(;))

Bind a law to named values. Names and nesting must match `paramaxis(law)`;
the same schema serves simulation, selection of fit parameters, and predictions.
Built-in laws bind more briefly through their keyword call, e.g.
`EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 2.0)`.
"""
struct KineticModel{Law, Parameters <: NamedTuple}
    law::Law
    parameters::Parameters
    function KineticModel(kinetic_law; parameters::NamedTuple = (;))
        declared_values = ComponentArray(zeros(_parameter_count(kinetic_law)), paramaxis(kinetic_law))
        canonical_parameters = _canonical_kinetic_parameters(declared_values, parameters)
        supplied_values = ComponentArray(; canonical_parameters...)
        all(parameter_value -> parameter_value isa Real && isfinite(parameter_value), supplied_values) ||
            throw(ArgumentError("Kinetic parameter values must be finite real numbers."))
        new{typeof(kinetic_law), typeof(canonical_parameters)}(kinetic_law, canonical_parameters)
    end
end

function _canonical_kinetic_parameters(declared_values, supplied_tree::NamedTuple)
    declared_names = propertynames(declared_values)
    missing_names = setdiff(declared_names, keys(supplied_tree))
    unknown_names = setdiff(keys(supplied_tree), declared_names)
    isempty(missing_names) && isempty(unknown_names) || throw(ArgumentError(
        "Kinetic parameters must match the law's paramaxis $declared_names" *
        (isempty(missing_names) ? "" : "; missing $(Tuple(missing_names))") *
        (isempty(unknown_names) ? "" : "; unknown $(Tuple(unknown_names))") * "."))
    return NamedTuple{Tuple(declared_names)}(map(Tuple(declared_names)) do declared_name
        declared_value = getproperty(declared_values, declared_name)
        supplied_value = getproperty(supplied_tree, declared_name)
        if declared_value isa ComponentArray
            supplied_value isa NamedTuple || throw(ArgumentError("Nested kinetic parameters require a NamedTuple."))
            return _canonical_kinetic_parameters(declared_value, supplied_value)
        end
        supplied_value isa Real || throw(ArgumentError("Kinetic values must be real scalars."))
        supplied_value
    end)
end

paramaxis(bound_kinetic::KineticModel) = paramaxis(bound_kinetic.law)

CombinedNucleation(primary::KineticModel, secondary::KineticModel) =
    KineticModel(CombinedNucleation(primary.law, secondary.law);
        parameters = (primary = primary.parameters, secondary = secondary.parameters))

"""
    CrystallisationModel(system; growth, nucleation=NoNucleation(), dissolution=NoDissolution(),
                         aggregation=NoAggregation(), breakage=NoBreakage())

A material system and five bound kinetic slots, independent of run conditions
and solver. Each slot takes a bound `KineticModel`, for example
`CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)`; laws without
parameters and callable laws are bound automatically.
"""
struct CrystallisationModel{System, Nucleation, Growth, Dissolution, Aggregation, Breakage}
    system::System
    nucleation::Nucleation
    growth::Growth
    dissolution::Dissolution
    aggregation::Aggregation
    breakage::Breakage
end

CrystallisationModel(material_system; growth, nucleation = NoNucleation(), dissolution = NoDissolution(),
        aggregation = NoAggregation(), breakage = NoBreakage()) =
    CrystallisationModel(material_system, _bound_kinetic(nucleation), _bound_kinetic(growth),
        _bound_kinetic(dissolution), _bound_kinetic(aggregation), _bound_kinetic(breakage))

_bound_kinetic(bound_kinetic::KineticModel) = bound_kinetic
function _bound_kinetic(kinetic_law)
    _parameter_count(kinetic_law) == 0 || throw(ArgumentError("$(nameof(typeof(kinetic_law))) needs parameter values; " *
        "bind them with its keyword call, e.g. $(nameof(typeof(kinetic_law)))(name = value, ...)."))
    return KineticModel(kinetic_law)
end

const _MODEL_KINETIC_SLOTS = (:nucleation, :growth, :dissolution, :aggregation, :breakage)

function Base.show(display_io::IO, ::MIME"text/plain", crystal_model::CrystallisationModel)
    println(display_io, "CrystallisationModel")
    println(display_io, "  density: ", crystal_model.system.crystal_density, " kg/m³; shape factor: ",
        crystal_model.system.volume_shape_factor)
    println(display_io, "  solubility: ", typeof(crystal_model.system.saturation_model))
    for slot_name in _MODEL_KINETIC_SLOTS
        bound_kinetic = getproperty(crystal_model, slot_name)
        println(display_io, "  ", slot_name, ": ", nameof(typeof(bound_kinetic.law)), " ", bound_kinetic.parameters)
    end
end

function _validate_material_model(crystal_model::CrystallisationModel)
    material_system = crystal_model.system
    if material_system isa CrystallisationSystem
        isempty(intersect(keys(material_system.properties), fieldnames(typeof(material_system)))) ||
            throw(ArgumentError("Additional material properties cannot shadow named system fields."))
    end
    material_system.saturation_model isa AbstractSolubilityModel ||
        throw(ArgumentError("saturation_model must implement AbstractSolubilityModel."))
    for required_name in (:crystal_density, :volume_shape_factor)
        required_value = getproperty(material_system, required_name)
        isfinite(required_value) && required_value > 0 ||
            throw(ArgumentError("$required_name must be finite and positive."))
    end
    for (slot_name, expected_family) in zip(_MODEL_KINETIC_SLOTS,
            (AbstractNucleationFunction, AbstractGrowthFunction, AbstractDissolutionFunction,
                AbstractAggregationFunction, AbstractBreakageFunction))
        bound_kinetic = getproperty(crystal_model, slot_name)
        bound_kinetic isa KineticModel && bound_kinetic.law isa expected_family ||
            throw(ArgumentError("$slot_name requires a KineticModel of $expected_family."))
        for required_name in required_properties(bound_kinetic.law)
            hasproperty(material_system, required_name) ||
                throw(ArgumentError("Material system lacks required property $required_name."))
            required_value = getproperty(material_system, required_name)
            _validate_required_material_property(bound_kinetic.law, required_name, required_value)
        end
    end
    return crystal_model
end

_bound_parameter_vector(bound_kinetic::KineticModel) =
    isempty(bound_kinetic.parameters) ? Float64[] : ComponentArray(; bound_kinetic.parameters...)


"""
    CrystallisationProblem(model, conditions; solver=MoM(), kwargs...)

Configure a model under numeric `RunConditions` for simulation or steady
solving. A missing seed means an empty population. Remaining keywords are
numerical settings; material, kinetic and condition fields cannot be overridden.
"""
function CrystallisationProblem(crystal_model::CrystallisationModel, run_conditions::RunConditions;
        solver::AbstractSolver = MoM(), kwargs...)
    _validate_material_model(crystal_model)
    unresolved_sources = observation_sources(run_conditions)
    isempty(unresolved_sources) || throw(ArgumentError(
        "Conditions read $(Tuple(unresolved_sources)) from observations; supply numbers or use an experiment."))
    initial_solvent = run_conditions.solvent
    model_settings = (;
        crystal_density = crystal_model.system.crystal_density,
        volume_shape_factor = crystal_model.system.volume_shape_factor,
        molecular_volume = crystal_model.system.molecular_volume,
        saturation_model = crystal_model.system.saturation_model,
        material_system = crystal_model.system,
        kinetics_nucleationfunction = crystal_model.nucleation.law,
        parameterset_nucleation = _bound_parameter_vector(crystal_model.nucleation),
        kinetics_growthfunction = crystal_model.growth.law,
        parameterset_growth = _bound_parameter_vector(crystal_model.growth),
        kinetics_dissolutionfunction = crystal_model.dissolution.law,
        parameterset_dissolution = _bound_parameter_vector(crystal_model.dissolution),
        kinetics_aggregationfunction = crystal_model.aggregation.law,
        parameterset_aggregation = _bound_parameter_vector(crystal_model.aggregation),
        kinetics_breakagefunction = crystal_model.breakage.law,
        parameterset_breakage = _bound_parameter_vector(crystal_model.breakage),
        initial_concentration = initial_solvent.concentration,
        initial_solvent_state = initial_solvent, temp_profile = run_conditions.temperature, solver,
        operation = run_conditions.operation)
    reserved_settings = intersect((keys(model_settings)..., :initial_state), keys(kwargs))
    isempty(reserved_settings) || throw(ArgumentError(
        "Material, kinetic and run conditions are set by the model and conditions: $(Tuple(reserved_settings))."))
    configured_run = CrystallisationProblem(; model_settings..., kwargs...)
    _validate_crystallisation_problem(configured_run)
    if run_conditions.crystals !== nothing
        configured_run = _problem_with_initial_state(configured_run,
            initial_state_from_characteristics(configured_run, run_conditions.crystals))
        _validate_crystallisation_problem(configured_run)
    end
    return configured_run
end

"""
    simulate(model, conditions; solver=MoM(), saveat, kwargs...)
    simulate(problem; saveat, kwargs...)

Return the physical trajectory at the requested times in seconds.
"""
simulate(crystal_model::CrystallisationModel, run_conditions::RunConditions;
        solver::AbstractSolver = MoM(), saveat, kwargs...) =
    simulate(CrystallisationProblem(crystal_model, run_conditions; solver); saveat, kwargs...)
simulate(configured_run::CrystallisationProblem; saveat, kwargs...) =
    last(runsimulation(configured_run; save_idx = saveat, kwargs...))

"""Context for callable kinetics, with physical values and advanced problem/state access."""
struct KineticContext{Problem, State, Time, System}
    problem::Problem
    state::State
    time::Time
    system::System
end
KineticContext(configured_run::CrystallisationProblem, numerical_state, clock_time) =
    KineticContext(configured_run, numerical_state, clock_time, configured_run.material_system)
supersaturation(rate_context::KineticContext) =
    supersaturation(rate_context.problem, rate_context.state, rate_context.time)
temperature(rate_context::KineticContext) = temperature(rate_context.problem.temp_profile, rate_context.time)
solvent_state(rate_context::KineticContext) = solvent_state(rate_context.problem, rate_context.state)

"""`CallableGrowth(callable; parameters, required=())`: scalar law `(named_parameters, context) -> rate`."""
struct CallableGrowth{Callable, Schema, Requirements} <: AbstractScalarGrowthFunction
    callable::Callable
    schema::Schema
    required::Requirements
end
CallableGrowth(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableGrowth(kinetic_callable, parameters, required)

"""`CallableNucleation(callable; parameters, required=())`: law `(named_parameters, context) -> rate`."""
struct CallableNucleation{Callable, Schema, Requirements} <: AbstractNucleationFunction
    callable::Callable
    schema::Schema
    required::Requirements
end
CallableNucleation(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableNucleation(kinetic_callable, parameters, required)

"""Length law `(named_parameters, context, length_metres) -> rate`, including DQMOM node evaluation."""
struct CallableLengthGrowth{Callable, Schema, Requirements} <: AbstractLengthGrowthFunction
    callable::Callable
    schema::Schema
    required::Requirements
end
CallableLengthGrowth(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableLengthGrowth(kinetic_callable, parameters, required)

const _CallableKinetic = Union{CallableGrowth, CallableNucleation, CallableLengthGrowth}
paramaxis(callable_law::_CallableKinetic) = first(getaxes(ComponentArray(; callable_law.schema...)))
required_properties(callable_law::_CallableKinetic) = callable_law.required
KineticModel(callable_law::_CallableKinetic) = KineticModel(callable_law; parameters = callable_law.schema)
_bound_kinetic(callable_law::_CallableKinetic) = KineticModel(callable_law)

growthrate(callable_law::CallableGrowth, candidate_values, configured_run::CrystallisationProblem, numerical_state, clock_time) =
    callable_law.callable(_named_params(callable_law, candidate_values), KineticContext(configured_run, numerical_state, clock_time))
nucleationrate(callable_law::CallableNucleation, candidate_values, configured_run::CrystallisationProblem, numerical_state, clock_time) =
    callable_law.callable(_named_params(callable_law, candidate_values), KineticContext(configured_run, numerical_state, clock_time))
growthrate_at_length(callable_law::CallableLengthGrowth, candidate_values, configured_run::CrystallisationProblem, numerical_state, clock_time, crystal_length) =
    callable_law.callable(_named_params(callable_law, candidate_values), KineticContext(configured_run, numerical_state, clock_time), crystal_length)
function growthrate!(destination::AbstractVector, callable_law::CallableLengthGrowth, candidate_values,
        configured_run::CrystallisationProblem, numerical_state, clock_time, length_mesh::AbstractVector)
    length(destination) == length(length_mesh) || throw(DimensionMismatch("Rate buffer must match mesh."))
    named_values = _named_params(callable_law, candidate_values)
    rate_context = KineticContext(configured_run, numerical_state, clock_time)
    for mesh_index in eachindex(destination, length_mesh)
        destination[mesh_index] = callable_law.callable(named_values, rate_context, length_mesh[mesh_index])
    end
    return destination
end
