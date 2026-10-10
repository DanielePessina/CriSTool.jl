"""
    RunConditions(; temperature, concentration=FromObservation(:concentration),
                  solvent=(;), crystals=nothing, operation=BatchOperation())

Everything that defines one run apart from the model and solver: temperature
(Kelvin or an `AbstractTemperature`), initial concentration (kg/m³), extra
initial solvent values, the initial seed, and the reactor operation.

Initial values may be `FromObservation` placeholders, resolved per experiment
from time-zero observations. Steady fits relax from explicit numeric values, so
their conditions must not contain placeholders.
"""
struct RunConditions{Temperature <: AbstractTemperature, Solvent <: NamedTuple, Crystals,
                     Operation <: AbstractCrystallisationOperation}
    temperature::Temperature
    solvent::Solvent
    crystals::Crystals
    operation::Operation
end

function RunConditions(; temperature, concentration = FromObservation(:concentration),
        solvent::NamedTuple = (;), crystals = nothing,
        operation::AbstractCrystallisationOperation = BatchOperation())
    haskey(solvent, :concentration) &&
        throw(ArgumentError("Supply the initial concentration once, using concentration=."))
    solvent_values = merge((; concentration), solvent)
    all(solvent_value -> solvent_value isa Union{Real, FromObservation}, solvent_values) ||
        throw(ArgumentError("Initial solvent values must be numbers or FromObservation placeholders."))
    crystals === nothing || crystals isa AbstractInitialCrystals ||
        throw(ArgumentError("crystals must be an initial crystal description or nothing."))
    temperature_profile = temperature isa Real ? ConstantTemperature(Float64(temperature)) : temperature
    temperature_profile isa AbstractTemperature ||
        throw(ArgumentError("temperature must be a number in Kelvin or an AbstractTemperature."))
    return RunConditions(temperature_profile, solvent_values, crystals, operation)
end

"""`merge(conditions; changes...)` returns conditions with the named settings replaced."""
function Base.merge(run_conditions::RunConditions; condition_changes...)
    current_settings = (; temperature = run_conditions.temperature,
        concentration = run_conditions.solvent.concentration,
        solvent = Base.structdiff(run_conditions.solvent, (; concentration = nothing)),
        crystals = run_conditions.crystals, operation = run_conditions.operation)
    return RunConditions(; merge(current_settings, values(condition_changes))...)
end

# Placeholder discovery and resolution walk solvent values and seed fields.
_observation_placeholders(condition_value) = FromObservation[]
_observation_placeholders(condition_value::FromObservation) = [condition_value]
_observation_placeholders(condition_values::NamedTuple) =
    reduce(vcat, map(_observation_placeholders, values(condition_values)); init = FromObservation[])
_observation_placeholders(seed_description::AbstractInitialCrystals) =
    reduce(vcat, (_observation_placeholders(getfield(seed_description, field_name))
        for field_name in fieldnames(typeof(seed_description))); init = FromObservation[])
_observation_placeholders(run_conditions::RunConditions) = vcat(
    _observation_placeholders(run_conditions.solvent), _observation_placeholders(run_conditions.crystals))

"""Names of the observables whose time-zero values define these conditions."""
observation_sources(run_conditions::RunConditions) =
    unique(getfield.(_observation_placeholders(run_conditions), :observable_name))

_resolved_condition_value(condition_value, observation_lookup) = condition_value
_resolved_condition_value(condition_value::FromObservation, observation_lookup) =
    observation_lookup(condition_value.observable_name)
_resolved_condition_value(condition_values::NamedTuple, observation_lookup) =
    map(condition_value -> _resolved_condition_value(condition_value, observation_lookup), condition_values)
_resolved_condition_value(seed_description::LogNormalInitialCrystals, observation_lookup) =
    LogNormalInitialCrystals(; _resolved_seed_fields(seed_description, observation_lookup)...)
_resolved_condition_value(seed_description::GaussianInitialCrystals, observation_lookup) =
    GaussianInitialCrystals(; _resolved_seed_fields(seed_description, observation_lookup)...)
_resolved_seed_fields(seed_description, observation_lookup) =
    NamedTuple{fieldnames(typeof(seed_description))}(map(fieldnames(typeof(seed_description))) do field_name
        _resolved_condition_value(getfield(seed_description, field_name), observation_lookup)
    end)

"""Replace every placeholder with `observation_lookup(observable_name)`."""
function _resolved_conditions(run_conditions::RunConditions, observation_lookup)
    isempty(_observation_placeholders(run_conditions)) && return run_conditions
    return RunConditions(run_conditions.temperature,
        _resolved_condition_value(run_conditions.solvent, observation_lookup),
        _resolved_condition_value(run_conditions.crystals, observation_lookup),
        run_conditions.operation)
end
