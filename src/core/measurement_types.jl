## Measurements

"""
    AbstractObservable

Abstract supertype for observable containers (see `Observable`).
"""
abstract type AbstractObservable end

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
struct Observable{Tt <: AbstractVector{<:Real}, Tm <: AbstractVector{<:Real}, Tv} <:
       AbstractObservable
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
    AbstractExperiment

Abstract supertype for a single experimental run (see `CrystallisationExperiment`).
"""
abstract type AbstractExperiment end

"""
    CrystallisationExperiment{O<:NamedTuple,I,M<:NamedTuple} <: AbstractExperiment

A single crystallisation experiment: a typed `NamedTuple` of observables
plus the run conditions.

Fields:
- `observables::O`: e.g. `(concentration = Observable(...), d43 = Observable(...))`.
  The `concentration` observable is mandatory for loss evaluation.
- `temperature::Float64`: run temperature in Kelvin
- `exp_id::Int`: experiment identifier
- `initial_crystals::Union{Nothing,NamedTuple}`: optional initial seed
  characteristics used to construct the solver state
- `metadata::M`: additional typed run metadata

The `NamedTuple` shape keeps the container type-stable and Tables.jl
compatible; additional observables (pH, mass, PSD, ...) are added as new
fields, not new container types.
"""
Base.@kwdef @concrete struct CrystallisationExperiment{O <: NamedTuple,
                                                       I <: Union{Nothing, NamedTuple},
                                                       M <: NamedTuple} <:
                 AbstractExperiment
    observables::O
    temperature::Float64
    exp_id::Int
    initial_crystals::I = nothing
    metadata::M = NamedTuple()
end

"""
    initial_concentration(expt::CrystallisationExperiment) -> Real

Initial solute concentration of an experiment, read from the
`concentration` series observable (first timepoint).
"""
initial_concentration(expt::CrystallisationExperiment) =
    expt.observables.concentration.mean[1]

"""
    _experiment_time_span(expt::CrystallisationExperiment) -> (tmin, tmax)

Return the time span covered by every observable in an experiment.
"""
function _experiment_time_span(expt::CrystallisationExperiment)
    times = vcat((observable.time for observable in values(expt.observables))...)
    return extrema(times)
end
