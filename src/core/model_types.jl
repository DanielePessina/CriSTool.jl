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
required_properties(::Union{nucl_CNT, nucl_CNTnoS, nucl_CNT_fixed, nucl_CNT_plus_second}) =
    (:molecular_volume,)

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
"""
struct KineticModel{Law, Parameters <: NamedTuple}
    law::Law
    parameters::Parameters
    function KineticModel(kinetic_law; parameters::NamedTuple = (;))
        declared_values = ComponentArray(collect(Float64, zeros(kinetic_law.nparams)), paramaxis(kinetic_law))
        canonical_parameters = _canonical_kinetic_parameters(declared_values, parameters)
        supplied_values = ComponentArray(; canonical_parameters...)
        all(parameter_value -> parameter_value isa Real && isfinite(parameter_value), supplied_values) ||
            throw(ArgumentError("Kinetic parameter values must be finite real numbers."))
        new{typeof(kinetic_law), typeof(canonical_parameters)}(kinetic_law, canonical_parameters)
    end
end

function _canonical_kinetic_parameters(declared_values, supplied_tree::NamedTuple)
    declared_names = propertynames(declared_values)
    Set(declared_names) == Set(keys(supplied_tree)) ||
        throw(ArgumentError("Parameter names/nesting must match the law's paramaxis: $declared_names."))
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

"""A reusable physical system and five kinetic slots, independent of experiment and solver."""
Base.@kwdef struct CrystallisationModel{System, Nucleation, Growth, Dissolution, Aggregation, Breakage}
    system::System = LysozymeSystem()
    nucleation::Nucleation
    growth::Growth
    dissolution::Dissolution = KineticModel(nodissolution())
    aggregation::Aggregation = KineticModel(noaggregation())
    breakage::Breakage = KineticModel(nobreakage())
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

_checked_operation(operation_settings::BatchOperation) = operation_settings
function _checked_operation(operation_settings::Union{MSMPROperation, FedBatchOperation})
    original_feed = operation_settings.feed
    checked_feed = CrystallisationFeed(
        concentration = original_feed.solvent_state.concentration,
        solvent_state = Base.structdiff(original_feed.solvent_state, (; concentration = nothing)),
        crystals = original_feed.crystals === nothing ? nothing : _checked_seed_description(original_feed.crystals),
        transport = original_feed.transport)
    return operation_settings isa MSMPROperation ?
        MSMPROperation(volume = operation_settings.volume, inflow = operation_settings.inflow, feed = checked_feed) :
        FedBatchOperation(initial_volume = operation_settings.initial_volume, inflow = operation_settings.inflow, feed = checked_feed)
end
_checked_operation(operation_settings::AbstractCrystallisationOperation) = operation_settings

"""
    CrystallisationProblem(model; initial_conditions, temperature, solver=MoM(),
                           initial_crystals=nothing, operation=BatchOperation(), kwargs...)

Configure a reusable model for simulation. Initial conditions are explicit named
solvent values; a missing seed means an empty population. Material and kinetic
properties cannot be overridden through numerical/experiment keywords.
"""
function CrystallisationProblem(crystal_model::CrystallisationModel;
        initial_conditions::NamedTuple, temperature::AbstractTemperature,
        solver::AbstractSolver = MoM(), initial_crystals = nothing,
        operation::AbstractCrystallisationOperation = BatchOperation(), kwargs...)
    _validate_material_model(crystal_model)
    hasproperty(initial_conditions, :concentration) ||
        throw(ArgumentError("initial_conditions must supply concentration."))
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
        initial_concentration = initial_conditions.concentration,
        initial_solvent_state = initial_conditions, temp_profile = temperature, solver,
        operation = _checked_operation(operation))
    forbidden_settings = intersect(keys(model_settings), keys(kwargs))
    isempty(forbidden_settings) || throw(ArgumentError("Model/condition overrides are reserved: $forbidden_settings."))
    configured_run = CrystallisationProblem(; model_settings..., kwargs...)
    _validate_crystallisation_problem(configured_run)
    if configured_run.initial_state !== nothing
        explicit_solvent = solvent_state(configured_run, configured_run.initial_state)
        all(isapprox(getproperty(explicit_solvent, solvent_name), getproperty(initial_conditions, solvent_name);
            rtol = 1e-12, atol = 0.0) for solvent_name in keys(initial_conditions)) ||
            throw(ArgumentError("initial_state conflicts with declared initial_conditions."))
    end
    if initial_crystals !== nothing
        haskey(kwargs, :initial_state) && throw(ArgumentError("Supply seeds or initial_state, not both."))
        configured_run = _problem_with_initial_state(configured_run,
            initial_state_from_characteristics(configured_run, _checked_seed_description(initial_crystals)))
    end
    _validate_crystallisation_problem(configured_run)
    return configured_run
end

"""`simulate(problem; saveat, kwargs...)` returns a physical trajectory."""
simulate(configured_run::CrystallisationProblem; saveat = 0.0:300.0:28800.0, kwargs...) =
    last(runsimulation(configured_run; save_idx = saveat, kwargs...))

"""Context for callable kinetics, with physical values and advanced problem/state access."""
struct KineticContext{Problem, State, Time}
    problem::Problem
    state::State
    time::Time
end
function Base.getproperty(rate_context::KineticContext, field_name::Symbol)
    field_name === :system && return getfield(rate_context, :problem).material_system
    return getfield(rate_context, field_name)
end
Base.propertynames(::KineticContext) = (:problem, :state, :time, :system)
supersaturation(rate_context::KineticContext) =
    supersaturation(rate_context.problem, rate_context.state, rate_context.time)
temperature(rate_context::KineticContext) = temperature(rate_context.problem.temp_profile, rate_context.time)
solvent_state(rate_context::KineticContext) = solvent_state(rate_context.problem, rate_context.state)

"""`CallableGrowth(callable; parameters, required=())`: scalar law `(named_parameters, context) -> rate`."""
struct CallableGrowth{Callable, Schema, Requirements} <: AbstractFPScalarGrowthFunction
    callable::Callable
    schema::Schema
    required::Requirements
end
CallableGrowth(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableGrowth(kinetic_callable, parameters, required)

"""`CallableNucleation(callable; parameters, required=())`: law `(named_parameters, context) -> rate`."""
struct CallableNucleation{Callable, Schema, Requirements} <: AbstractFPNucleationFunction
    callable::Callable
    schema::Schema
    required::Requirements
end
CallableNucleation(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableNucleation(kinetic_callable, parameters, required)

"""Length law `(named_parameters, context, length_metres) -> rate`, including DQMOM node evaluation."""
struct CallableLengthGrowth{Callable, Schema, Requirements} <: AbstractFPLengthGrowthFunction
    callable::Callable
    schema::Schema
    required::Requirements
end
CallableLengthGrowth(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableLengthGrowth(kinetic_callable, parameters, required)

const _CallableKinetic = Union{CallableGrowth, CallableNucleation, CallableLengthGrowth}
Base.propertynames(callable_law::_CallableKinetic) = (:callable, :schema, :required, :nparams, :string, :symbols)
function Base.getproperty(callable_law::_CallableKinetic, field_name::Symbol)
    field_name === :nparams && return length(ComponentArray(; getfield(callable_law, :schema)...))
    field_name === :string && return string(nameof(typeof(callable_law)))
    field_name === :symbols && return Symbol.(ComponentArrays.labels(ComponentArray(; getfield(callable_law, :schema)...)))
    return getfield(callable_law, field_name)
end
paramaxis(callable_law::_CallableKinetic) = first(getaxes(ComponentArray(; callable_law.schema...)))
required_properties(callable_law::_CallableKinetic) = callable_law.required
KineticModel(callable_law::_CallableKinetic) = KineticModel(callable_law; parameters = callable_law.schema)

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
