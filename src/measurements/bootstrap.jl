"""One measured point of an observable series, as the bootstrap resamples it."""
struct _BootstrapEntry
    exp_id::Int
    time::Float64
    mean::Float64
    variance::Union{Nothing, Float64}
end

_observable_variance_value(observable, index::Int) = _variance_value(observable.variance, index)
_variance_value(::Nothing, index::Int) = nothing
_variance_value(variances::AbstractVector, index::Int) = Float64(variances[index])

_bootstrap_entry(exp_id::Integer, observable, index::Int) = _BootstrapEntry(exp_id,
    Float64(observable.time[index]), Float64(observable.mean[index]), _observable_variance_value(observable, index))

function _bootstrap_observable_pool(experiments, observable_name::Symbol)
    pool = _BootstrapEntry[]
    for expt in experiments
        observable = getproperty(expt.observables, observable_name)
        for index in 2:length(observable.mean)
            push!(pool, _bootstrap_entry(expt.exp_id, observable, index))
        end
    end
    return pool
end

function _bootstrap_observable(observable::Observable, exp_id::Integer, entries::Vector{_BootstrapEntry})
    all_entries = [_bootstrap_entry(exp_id, observable, 1); entries]
    sort!(all_entries; by = entry -> entry.time)

    # Sampling with replacement can select the same time point more than once.
    # An Observable has one value per time, so retain the first sampled value at
    # each time rather than constructing an ambiguous duplicate time grid.
    unique_entries = _BootstrapEntry[]
    seen_times = Set{Float64}()
    for entry in all_entries
        entry.time in seen_times && continue
        push!(unique_entries, entry)
        push!(seen_times, entry.time)
    end

    times = [entry.time for entry in unique_entries]
    means = [entry.mean for entry in unique_entries]
    all(entry -> entry.variance === nothing, unique_entries) && return Observable(; time = times, mean = means)
    variances = [something(entry.variance, 0.0) for entry in unique_entries]
    return Observable(; time = times, mean = means, variance = variances)
end

"""
    bootstrap_measurements(experiments, n_bootstrap=1; observable_names=...,
                           n_samples=nothing, seed=nothing)
        -> Vector{Vector{CrystallisationExperiment}}

Bootstrap arbitrary observable entries independently. The first point of each
series is retained as its experiment anchor; later points are sampled with
replacement. Metadata, initial mappings, conditions and unselected observables
are preserved, including initial measurements needed by the fit workflow.
"""
function bootstrap_measurements(experiments::Vector{<:CrystallisationExperiment},
                                n_bootstrap::Integer = 1;
                                observable_names = isempty(experiments) ?
                                                    () : propertynames(first(experiments).observables),
                                n_samples::Union{Nothing, Integer} = nothing,
                                seed::Union{Nothing, Integer} = nothing)
    n_bootstrap >= 0 || throw(ArgumentError("n_bootstrap must be non-negative."))
    isempty(experiments) && return [CrystallisationExperiment[] for _ in 1:n_bootstrap]
    all(name -> all(hasproperty(expt.observables, name) for expt in experiments),
        observable_names) || throw(ArgumentError("Every experiment must contain every requested observable."))

    pools = Dict(name => _bootstrap_observable_pool(experiments, name)
                 for name in observable_names)
    base_seed = isnothing(seed) ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    output = Vector{Vector{CrystallisationExperiment}}(undef, n_bootstrap)

    for bootstrap_index in 1:n_bootstrap
        rng = Random.Xoshiro(base_seed + UInt64(bootstrap_index - 1))
        sampled = Dict{Symbol, Dict{Int, Vector{_BootstrapEntry}}}()
        for observable_name in observable_names
            pool = pools[observable_name]
            draw_count = isnothing(n_samples) ? length(pool) : n_samples
            draw_count >= 0 || throw(ArgumentError("n_samples must be non-negative."))
            by_id = Dict{Int, Vector{_BootstrapEntry}}()
            if !isempty(pool) && draw_count > 0
                for pool_index in rand(rng, 1:length(pool), draw_count)
                    entry = pool[pool_index]
                    push!(get!(() -> _BootstrapEntry[], by_id, entry.exp_id), entry)
                end
            end
            sampled[observable_name] = by_id
        end

        replicate = CrystallisationExperiment[]
        for expt in experiments
            rebuilt = map(observable_names) do observable_name
                original = getproperty(expt.observables, observable_name)
                entries = get(() -> _BootstrapEntry[], sampled[observable_name], expt.exp_id)
                _bootstrap_observable(original, expt.exp_id, entries)
            end
            observables = merge(expt.observables, NamedTuple{Tuple(observable_names)}(Tuple(rebuilt)))
            push!(replicate, CrystallisationExperiment(;
                observables = observables,
                conditions = expt.conditions,
                exp_id = expt.exp_id,
                metadata = expt.metadata))
        end
        output[bootstrap_index] = replicate
    end
    return output
end


function _bootstrap_default_samples(experiments::Vector{<:CrystallisationExperiment},
                                    observable_names)
    return sum(max(length(getproperty(experiment.observables, name).mean) - 1, 0)
               for experiment in experiments for name in observable_names)
end

"""
    bootstrap_repeatmeasurements(experiments, n_bootstrap, include_ps; kwargs...)

Generate bootstrap datasets for the concentration series and, when
`include_ps` is true, the `d43` series. All selected observables use the same
generic time-series bootstrap implementation as `bootstrap_measurements`.
"""
function bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                      n_bootstrap::Integer, include_ps::Bool;
                                      n_samples::Union{Nothing, Integer} = nothing,
                                      seed::Union{Nothing, Integer} = nothing)
    if n_bootstrap < 0
        throw(ArgumentError("n_bootstrap must be non-negative."))
    end
    observable_names = include_ps ? (:concentration, :d43) : (:concentration,)
    all(name -> all(hasproperty(expt.observables, name) for expt in experiments),
        observable_names) ||
        throw(ArgumentError("Every experiment must contain the selected bootstrap observables."))
    ns = isnothing(n_samples) ? _bootstrap_default_samples(experiments, observable_names) :
         n_samples
    return bootstrap_measurements(experiments, n_bootstrap;
                                  observable_names = observable_names,
                                  n_samples = ns, seed = seed)
end

"""
    bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                 include_ps::Bool; n_samples::Union{Nothing, Integer} = nothing,
                                 seed::Union{Nothing, Integer} = nothing)
        -> Vector{CrystallisationExperiment}

Generate a single bootstrap dataset by resampling the concentration series and,
when `include_ps` is true, the `d43` series. Experiments retain the first point
of each selected observable as their anchor.
"""
function bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                      include_ps::Bool;
                                      n_samples::Union{Nothing, Integer} = nothing,
                                      seed::Union{Nothing, Integer} = nothing)
    observable_names = include_ps ? (:concentration, :d43) : (:concentration,)
    all(name -> all(hasproperty(expt.observables, name) for expt in experiments),
        observable_names) ||
        throw(ArgumentError("Every experiment must contain the selected bootstrap observables."))
    ns = isnothing(n_samples) ? _bootstrap_default_samples(experiments, observable_names) :
         n_samples
    return first(bootstrap_measurements(experiments, 1;
                                        observable_names = observable_names,
                                        n_samples = ns, seed = seed))
end
