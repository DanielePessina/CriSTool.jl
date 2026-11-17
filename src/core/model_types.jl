"""
    CrystallisationSystem(; saturation_model, crystal_density, volume_shape_factor,
                          molecular_volume=nothing, properties=(;))

Material properties in SI units. Missing optional properties are `nothing`;
selected kinetic laws declare their requirements through `required_properties`.
Additional named material values live in `properties` and are read as
`system.properties.name`.
"""
Base.@kwdef struct CrystallisationSystem{Solubility, MolecularVolume, Properties <: NamedTuple}
    saturation_model::Solubility
    crystal_density::Float64
    volume_shape_factor::Float64
    molecular_volume::MolecularVolume = nothing
    properties::Properties = (;)
end

"""
    _material_property(system, property_name) -> value or nothing

Look up a property a kinetic law requires: a named system field first, then
the system's additional `properties`. Missing properties are `nothing`.
"""
function _material_property(material_system::CrystallisationSystem, property_name::Symbol)
    hasfield(typeof(material_system), property_name) && return getfield(material_system, property_name)
    return get(material_system.properties, property_name, nothing)
end
_material_property(material_system, property_name::Symbol) =
    hasproperty(material_system, property_name) ? getproperty(material_system, property_name) : nothing


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
    CrystallisationModel(system; growth, nucleation=NoNucleation(), dissolution=NoDissolution(),
                         aggregation=NoAggregation(), breakage=NoBreakage())

A material system and five kinetic laws, independent of run conditions and
solver. Each law holds its parameter values, for example
`CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)`; the
values seed the problem's kinetic vector `p`.
"""
struct CrystallisationModel{System, Nucleation <: AbstractNucleationFunction, Growth <: AbstractGrowthFunction,
                            Dissolution <: AbstractDissolutionFunction, Aggregation <: AbstractAggregationFunction,
                            Breakage <: AbstractBreakageFunction}
    system::System
    nucleation::Nucleation
    growth::Growth
    dissolution::Dissolution
    aggregation::Aggregation
    breakage::Breakage
    function CrystallisationModel(material_system, nucleation::AbstractNucleationFunction,
            growth::AbstractGrowthFunction, dissolution::AbstractDissolutionFunction,
            aggregation::AbstractAggregationFunction, breakage::AbstractBreakageFunction)
        for kinetic_law in (nucleation, growth, dissolution, aggregation, breakage)
            _validate_law_values(kinetic_law, _law_values(kinetic_law))
        end
        return new{typeof(material_system), typeof(nucleation), typeof(growth), typeof(dissolution),
            typeof(aggregation), typeof(breakage)}(material_system, nucleation, growth, dissolution,
            aggregation, breakage)
    end
end

CrystallisationModel(material_system; growth, nucleation = NoNucleation(), dissolution = NoDissolution(),
        aggregation = NoAggregation(), breakage = NoBreakage()) =
    CrystallisationModel(material_system, nucleation, growth, dissolution, aggregation, breakage)

# Every value a law holds must be a finite real scalar or real vector, at any nesting depth.
_validate_law_values(kinetic_law, law_values::NamedTuple) =
    foreach(law_value -> _validate_law_values(kinetic_law, law_value), values(law_values))
_validate_law_values(kinetic_law, law_value::Real) = isfinite(law_value) ||
    throw(ArgumentError("$(nameof(typeof(kinetic_law))) parameter values must be finite; got $law_value."))
_validate_law_values(kinetic_law, law_values::AbstractVector{<:Real}) =
    foreach(law_value -> _validate_law_values(kinetic_law, law_value), law_values)
_validate_law_values(kinetic_law, law_value) = throw(ArgumentError(
    "$(nameof(typeof(kinetic_law))) parameter values must be real numbers; got $(typeof(law_value))."))

const _MODEL_KINETIC_SLOTS = (:nucleation, :growth, :dissolution, :aggregation, :breakage)

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
    # The constructor's slot type bounds already enforce each law's family.
    for slot_name in _MODEL_KINETIC_SLOTS
        kinetic_law = getproperty(crystal_model, slot_name)
        for required_name in required_properties(kinetic_law)
            _validate_required_material_property(kinetic_law, required_name,
                _material_property(material_system, required_name))
        end
    end
    return crystal_model
end

"""
    CrystallisationProblem(model, conditions, tspan, p = <model values>;
                           solvent_dynamics=default_solvent_dynamics,
                           solid_mass_concentration_threshold=1e-12)

Configure a model under numeric `RunConditions` over `tspan = (t0, tf)`
seconds, with kinetic values `p` (any vector in `paramaxis(model)` order,
defaulting to the values the model's laws hold). A missing seed means an empty
population. The keywords are the only other settings; the numerical solver is
chosen in `solve(problem, solver)`.
"""
function CrystallisationProblem(crystal_model::CrystallisationModel, run_conditions::RunConditions, tspan,
        kinetic_values::AbstractVector = _model_kinetic_values(crystal_model);
        solvent_dynamics = default_solvent_dynamics,
        solid_mass_concentration_threshold::Real = 1e-12, unsupported_settings...)
    isempty(unsupported_settings) || throw(ArgumentError(
        "Unsupported problem settings $(Tuple(keys(unsupported_settings))); material, kinetic and run " *
        "conditions are set by the model and conditions, the solver by solve(problem, solver)."))
    unresolved_sources = observation_sources(run_conditions)
    isempty(unresolved_sources) || throw(ArgumentError(
        "Conditions read $(Tuple(unresolved_sources)) from observations; supply numbers or use an experiment."))
    parameter_axis = paramaxis(crystal_model.nucleation, crystal_model.growth,
        crystal_model.dissolution, crystal_model.aggregation, crystal_model.breakage)
    configured_run = CrystallisationProblem(crystal_model, run_conditions, _checked_tspan(tspan),
        _kinetic_values_on_axis(parameter_axis, kinetic_values), solvent_dynamics,
        Float64(solid_mass_concentration_threshold))
    return _validate_crystallisation_problem(configured_run)
end

"""The values the model's laws hold, as a `Float64` vector in `paramaxis` order."""
_model_kinetic_values(crystal_model::CrystallisationModel) =
    Vector{Float64}(ComponentArray(; _model_named_values(crystal_model)...))

"""
    remake(problem::CrystallisationProblem; model, conditions, tspan, p,
           solvent_dynamics, solid_mass_concentration_threshold)

Copy `problem` with the given fields replaced and validated again. Without `p`,
a replaced `model` brings the kinetic values its laws hold; `remake(problem; p)`
is the differentiable way to solve one run for other kinetic values.
"""
function SciMLBase.remake(configured_run::CrystallisationProblem; model = missing, conditions = missing,
        tspan = missing, p = missing, solvent_dynamics = missing,
        solid_mass_concentration_threshold = missing)
    remade_model = coalesce(model, configured_run.model)
    return CrystallisationProblem(remade_model, coalesce(conditions, configured_run.conditions),
        coalesce(tspan, configured_run.tspan), _remade_kinetic_values(p, model, configured_run);
        solvent_dynamics = coalesce(solvent_dynamics, configured_run.solvent_dynamics),
        solid_mass_concentration_threshold = coalesce(solid_mass_concentration_threshold,
            configured_run.solid_mass_concentration_threshold))
end

# Explicit values win; a new model without values brings the values its laws hold.
_remade_kinetic_values(kinetic_values, model, configured_run) = kinetic_values
_remade_kinetic_values(::Missing, ::Missing, configured_run) = configured_run.p
_remade_kinetic_values(::Missing, crystal_model::CrystallisationModel, configured_run) =
    _model_kinetic_values(crystal_model)

"""
    DiscretisedProblem(problem::CrystallisationProblem, solver::AbstractSolver)

Resolve `problem` into the state layout of `solver`, validating the
solver-specific contract and turning the seed into the initial population block.
"""
function DiscretisedProblem(configured_run::CrystallisationProblem, solver::AbstractSolver)
    run_conditions = configured_run.conditions
    discretised_run = DiscretisedProblem(configured_run.model, run_conditions.temperature,
        run_conditions.solvent, nothing, run_conditions.operation, solver,
        configured_run.solvent_dynamics, configured_run.solid_mass_concentration_threshold)
    _validate_discretised_problem(discretised_run)
    return _with_initial_crystals(discretised_run, run_conditions.crystals)
end

_with_initial_crystals(discretised_run::DiscretisedProblem, ::Nothing) = discretised_run
_with_initial_crystals(discretised_run::DiscretisedProblem, initial_crystals::AbstractInitialCrystals) =
    _problem_with_initial_values(discretised_run, _initial_population(discretised_run, initial_crystals),
        discretised_run.initial_solvent_state)

"""Context for callable kinetics, with physical values and advanced problem/state access."""
struct KineticContext{Problem, State, Time}
    problem::Problem
    state::State
    time::Time
end
"""
    system_properties(context::KineticContext)

The additional named material values (`properties`) of the model's material
system, e.g. `system_properties(context).offset`.
"""
system_properties(rate_context::KineticContext) = rate_context.problem.model.system.properties
supersaturation(rate_context::KineticContext) =
    supersaturation(rate_context.problem, rate_context.state, rate_context.time)
temperature(rate_context::KineticContext) = temperature(rate_context.problem.temp_profile, rate_context.time)
solvent_state(rate_context::KineticContext) = solvent_state(rate_context.problem, rate_context.state)

"""`CallableGrowth(callable; parameters, required=())`: scalar law `(named_parameters, context) -> rate`."""
struct CallableGrowth{Callable, Parameters <: NamedTuple, Requirements} <: AbstractScalarGrowthFunction
    callable::Callable
    parameters::Parameters
    required::Requirements
end
CallableGrowth(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableGrowth(kinetic_callable, parameters, required)

"""`CallableNucleation(callable; parameters, required=())`: law `(named_parameters, context) -> rate`."""
struct CallableNucleation{Callable, Parameters <: NamedTuple, Requirements} <: AbstractNucleationFunction
    callable::Callable
    parameters::Parameters
    required::Requirements
end
CallableNucleation(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableNucleation(kinetic_callable, parameters, required)

"""Length law `(named_parameters, context, length_metres) -> rate`, including DQMOM node evaluation."""
struct CallableLengthGrowth{Callable, Parameters <: NamedTuple, Requirements} <: AbstractLengthGrowthFunction
    callable::Callable
    parameters::Parameters
    required::Requirements
end
CallableLengthGrowth(kinetic_callable; parameters::NamedTuple, required::Tuple = ()) =
    CallableLengthGrowth(kinetic_callable, parameters, required)

const _CallableKinetic = Union{CallableGrowth, CallableNucleation, CallableLengthGrowth}
paramaxis(callable_law::_CallableKinetic) = first(getaxes(ComponentArray(; callable_law.parameters...)))
required_properties(callable_law::_CallableKinetic) = callable_law.required
_law_values(callable_law::_CallableKinetic) = callable_law.parameters
_law_with_values(callable_law::_CallableKinetic, law_values::NamedTuple) =
    ConstructionBase.constructorof(typeof(callable_law))(callable_law.callable, law_values, callable_law.required)

growthrate(callable_law::CallableGrowth, candidate_values, configured_run::DiscretisedProblem, numerical_state, clock_time) =
    callable_law.callable(_named_params(callable_law, candidate_values), KineticContext(configured_run, numerical_state, clock_time))
nucleationrate(callable_law::CallableNucleation, candidate_values, configured_run::DiscretisedProblem, numerical_state, clock_time) =
    callable_law.callable(_named_params(callable_law, candidate_values), KineticContext(configured_run, numerical_state, clock_time))
growthrate_at_length(callable_law::CallableLengthGrowth, candidate_values, configured_run::DiscretisedProblem, numerical_state, clock_time, crystal_length) =
    callable_law.callable(_named_params(callable_law, candidate_values), KineticContext(configured_run, numerical_state, clock_time), crystal_length)
function growthrate!(destination::AbstractVector, callable_law::CallableLengthGrowth, candidate_values,
        configured_run::DiscretisedProblem, numerical_state, clock_time, length_mesh::AbstractVector)
    length(destination) == length(length_mesh) || throw(DimensionMismatch("Rate buffer must match mesh."))
    named_values = _named_params(callable_law, candidate_values)
    rate_context = KineticContext(configured_run, numerical_state, clock_time)
    for mesh_index in eachindex(destination, length_mesh)
        destination[mesh_index] = callable_law.callable(named_values, rate_context, length_mesh[mesh_index])
    end
    return destination
end
