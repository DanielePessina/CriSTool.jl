function _observable_variance_value(observable, index::Int)
    observable.variance === nothing && return nothing
    return observable.variance[index]
end

function _bootstrap_observable_pool(experiments, observable_name::Symbol)
    pool = Tuple{Int, Float64, Float64, Any}[]
    for expt in experiments
        observable = getproperty(expt.observables, observable_name)
        for index in 2:length(observable.mean)
            push!(pool, (expt.exp_id, Float64(observable.time[index]),
                         Float64(observable.mean[index]),
                         _observable_variance_value(observable, index)))
        end
    end
    return pool
end

function _bootstrap_observable(observable::Observable, entries::Vector{<:Tuple})
    anchor_time = Float64(observable.time[1])
    anchor_mean = Float64(observable.mean[1])
    anchor_variance = _observable_variance_value(observable, 1)
    all_entries = [(anchor_time, anchor_mean, anchor_variance)]
    append!(all_entries, [(entry[2], entry[3], entry[4]) for entry in entries])
    sort!(all_entries, by = first)

    # Sampling with replacement can select the same time point more than once.
    # An Observable has one value per time, so retain the first sampled value at
    # each time rather than constructing an ambiguous duplicate time grid.
    unique_entries = eltype(all_entries)[]
    seen_times = Set{Float64}()
    for entry in all_entries
        entry[1] in seen_times && continue
        push!(unique_entries, entry)
        push!(seen_times, entry[1])
    end

    times = Float64[entry[1] for entry in unique_entries]
    means = Float64[entry[2] for entry in unique_entries]
    if all(entry -> entry[3] === nothing, unique_entries)
        return Observable(; time = times, mean = means)
    end
    variances = Float64[entry[3] === nothing ? 0.0 : Float64(entry[3])
                        for entry in unique_entries]
    return Observable(; time = times, mean = means, variance = variances)
end

"""
    bootstrap_measurements(experiments, n_bootstrap=1; observable_names=...,
                           n_samples=nothing, seed=nothing)
        -> Vector{Vector{CrystallisationExperiment}}

Bootstrap arbitrary observable entries independently. The first point of each
series is retained as its experiment anchor; later points are sampled with
replacement. Original experiment metadata and unselected observable fields
are not included in the returned bootstrap experiment.
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
        sampled = Dict{Symbol, Dict{Int, Vector{Tuple{Int, Float64, Float64, Any}}}}()
        for observable_name in observable_names
            pool = pools[observable_name]
            draw_count = isnothing(n_samples) ? length(pool) : n_samples
            draw_count >= 0 || throw(ArgumentError("n_samples must be non-negative."))
            by_id = Dict{Int, Vector{Tuple{Int, Float64, Float64, Any}}}()
            if !isempty(pool) && draw_count > 0
                for pool_index in rand(rng, 1:length(pool), draw_count)
                    entry = pool[pool_index]
                    push!(get!(by_id, entry[1],
                               Tuple{Int, Float64, Float64, Any}[]), entry)
                end
            end
            sampled[observable_name] = by_id
        end

        replicate = CrystallisationExperiment[]
        for expt in experiments
            rebuilt = map(observable_names) do observable_name
                original = getproperty(expt.observables, observable_name)
                entries = get(sampled[observable_name], expt.exp_id,
                              Tuple{Int, Float64, Float64, Any}[])
                _bootstrap_observable(original, entries)
            end
            observables = NamedTuple{Tuple(observable_names)}(Tuple(rebuilt))
            push!(replicate, CrystallisationExperiment(;
                observables = observables,
                temperature = expt.temperature,
                initial_crystals = expt.initial_crystals,
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
