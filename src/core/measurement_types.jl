## Measurements

"""
    ObservableColumns(; time=:Time, mean, variance=nothing)

Describe the source columns used to build one measured observable. Each
observable may use a different time column; the normalized result is always an
`Observable` time series.
"""
struct ObservableColumns
    time::Symbol
    mean::Symbol
    variance::Union{Nothing, Symbol}
end

function ObservableColumns(; time = :Time, mean, variance = nothing)
    source_time = time isa Symbol ? time : Symbol(time)
    source_mean = mean isa Symbol ? mean : Symbol(mean)
    source_variance = variance === nothing ? nothing :
                      (variance isa Symbol ? variance : Symbol(variance))
    return ObservableColumns(source_time, source_mean, source_variance)
end

"""
    Observable(; time, mean, variance=nothing) -> Observable

A measured observable (concentration, d43, pH, ...) represented uniformly as a
time series. A single final-state measurement is a one-element time series,
not a different data shape.

`variance = nothing` represents an unreplicated measurement. A scalar variance
is accepted as a convenience and expanded across all measured points.

The constructor enforces the measurement contract used by loaders, losses,
bootstrap, and plotting:

- `time` and `mean` are non-empty vectors of finite real values;
- times are strictly increasing;
- a supplied variance has one value per measurement and is finite and
  nonnegative.
"""
struct Observable{Tt <: AbstractVector{<:Real}, Tm <: AbstractVector{<:Real}, Tv}
    time::Tt
    mean::Tm
    variance::Tv
end

function Observable(; time, mean, variance = nothing)
    time_values = collect(time)
    mean_values = collect(mean)

    isempty(time_values) && throw(ArgumentError("Observable time cannot be empty."))
    length(time_values) == length(mean_values) ||
        throw(ArgumentError("Observable time and mean must have the same length."))
    all(value -> value isa Real && isfinite(value), time_values) ||
        throw(ArgumentError("Observable time must contain only finite real values."))
    all(value -> value isa Real && isfinite(value), mean_values) ||
        throw(ArgumentError("Observable mean must contain only finite real values."))
    all(diff(time_values) .> 0) ||
        throw(ArgumentError("Observable time must be strictly increasing."))

    variance_values = if variance === nothing
        nothing
    elseif variance isa Real
        fill(variance, length(mean_values))
    else
        collected_variance = collect(variance)
        length(collected_variance) == length(mean_values) ||
            throw(ArgumentError("Observable variance must have one value per mean."))
        collected_variance
    end

    if variance_values !== nothing
        all(value -> value isa Real && isfinite(value) && value >= 0,
            variance_values) ||
            throw(ArgumentError("Observable variance must contain finite nonnegative values."))
    end

    return Observable{typeof(time_values), typeof(mean_values), typeof(variance_values)}(
        time_values, mean_values, variance_values)
end

"""
    CrystallisationExperiment(; observables, exp_id, conditions, metadata=(;))

A single crystallisation experiment: a typed `NamedTuple` of observables
(e.g. `(concentration = Observable(...), d43 = Observable(...))`), an integer
identifier, the `RunConditions` it was run under, and additional metadata.
Conditions may refer to the experiment's own time-zero observations through
`FromObservation` placeholders.

The `NamedTuple` shape keeps the container type-stable and Tables.jl
compatible; additional observables (pH, mass, PSD, ...) are added as new
fields, not new container types.
"""
Base.@kwdef struct CrystallisationExperiment{O <: NamedTuple, C <: RunConditions, M <: NamedTuple}
    observables::O
    exp_id::Int
    conditions::C
    metadata::M = NamedTuple()
end

"""
    _experiment_time_span(expt::CrystallisationExperiment) -> (tmin, tmax)

Return the time span covered by every observable in an experiment.
"""
function _experiment_time_span(expt::CrystallisationExperiment)
    times = vcat((observable.time for observable in values(expt.observables))...)
    return extrema(times)
end

function _initial_observation(experiment_run::CrystallisationExperiment, observable_name::Symbol)
    hasproperty(experiment_run.observables, observable_name) ||
        throw(ArgumentError("Experiment $(experiment_run.exp_id) lacks initial observable :$observable_name."))
    initial_series = getproperty(experiment_run.observables, observable_name)
    initial_index = findfirst(iszero, initial_series.time)
    initial_index === nothing && throw(ArgumentError(
        "Experiment $(experiment_run.exp_id): :$observable_name must have a time-zero initial observation."))
    return initial_series.mean[initial_index]
end

"""Run conditions with placeholders replaced by the experiment's time-zero observations."""
_experiment_conditions(experiment_run::CrystallisationExperiment) =
    _resolved_conditions(experiment_run.conditions,
        observable_name -> _initial_observation(experiment_run, observable_name))

"""Run temperature in Kelvin at time zero, for labels and summaries."""
_experiment_start_temperature(experiment_run::CrystallisationExperiment) =
    temperature(experiment_run.conditions.temperature, 0.0)

"""
    initial_concentration(expt::CrystallisationExperiment) -> Real

First measured concentration of an experiment.
"""
initial_concentration(expt::CrystallisationExperiment) =
    expt.observables.concentration.mean[1]
