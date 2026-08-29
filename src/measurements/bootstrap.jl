function _observable_variance_value(observable, index::Int)
    observable.variance === nothing && return nothing
    return observable.variance isa AbstractArray ? observable.variance[index] : observable.variance
end

function _bootstrap_observable_pool(experiments, observable_name::Symbol)
    pool = Tuple{Int, Float64, Float64, Any}[]
    for expt in experiments
        observable = getproperty(expt.observables, observable_name)
        if observable.mean isa AbstractArray
            for index in 2:length(observable.mean)
                push!(pool, (expt.exp_id, Float64(observable.time[index]),
                             Float64(observable.mean[index]),
                             _observable_variance_value(observable, index)))
            end
        else
            push!(pool, (expt.exp_id, Float64(observable.time), Float64(observable.mean),
                         _observable_variance_value(observable, 1)))
        end
    end
    return pool
end

function _bootstrap_observable(observable::Observable{T, Tt, Tv},
                               entries::Vector{<:Tuple}) where {T <: AbstractArray, Tt, Tv}
    anchor_time = Float64(observable.time[1])
    anchor_mean = Float64(observable.mean[1])
    anchor_variance = _observable_variance_value(observable, 1)
    all_entries = [(anchor_time, anchor_mean, anchor_variance)]
    append!(all_entries, [(entry[2], entry[3], entry[4]) for entry in entries])
    sort!(all_entries, by = first)

    times = Float64[entry[1] for entry in all_entries]
    means = Float64[entry[2] for entry in all_entries]
    if all(entry -> entry[3] === nothing, all_entries)
        return Observable(; time = times, mean = means)
    end
    variances = Float64[entry[3] === nothing ? 0.0 : Float64(entry[3])
                        for entry in all_entries]
    return Observable(; time = times, mean = means, variance = variances)
end

function _bootstrap_observable(observable::Observable{T, Tt, Tv},
                               entries::Vector{<:Tuple}) where {T <: Real, Tt, Tv}
    isempty(entries) && return observable
    entry = entries[end]
    variance = entry[4]
    return variance === nothing ?
           Observable(; time = entry[2], mean = entry[3]) :
           Observable(; time = entry[2], mean = entry[3], variance = Float64(variance))
end

"""
    bootstrap_measurements(experiments, n_bootstrap=1; observable_names=...,
                           n_samples=nothing, seed=nothing)
        -> Vector{Vector{CrystallisationExperiment}}

Bootstrap arbitrary observable entries independently. The first point of each
series is retained as its experiment anchor; later series points and scalar
observables are sampled with replacement. Original experiment metadata and
unselected observable fields are preserved.
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
                loading = expt.loading,
                exp_id = expt.exp_id,
                metadata = expt.metadata))
        end
        output[bootstrap_index] = replicate
    end
    return output
end

"""
    _bootstrap_default_samples(experiments::Vector{CrystallisationExperiment}, include_ps::Bool) -> Int

Calculate the default number of bootstrap samples based on pooled data size
(concentration points excluding the initial timepoint, plus one PS entry per
experiment when `include_ps`).
"""
function _bootstrap_default_samples(experiments::Vector{CrystallisationExperiment},
                                    include_ps::Bool)
    pool_len = sum(max(length(expt.observables.concentration.time) - 1, 0) for expt in experiments)
    return pool_len + (include_ps ? length(experiments) : 0)
end

"""
    _bootstrap_repeatmeasurements_rng(experiments::Vector{CrystallisationExperiment},
                                      rng, n_samples::Integer; include_ps::Bool) -> Vector{CrystallisationExperiment}

Generate a single bootstrap dataset using the provided RNG: sample pooled
(non-initial) concentration tuples and optional final-PS entries with
replacement, then rebuild experiments. Experiments without sampled tuples are
omitted.
"""
function _bootstrap_repeatmeasurements_rng(experiments::Vector{CrystallisationExperiment},
                                           rng, n_samples::Integer; include_ps::Bool)

    if n_samples < 0
        throw(ArgumentError("n_samples must be non-negative."))
    end
    if isempty(experiments) || n_samples == 0
        return CrystallisationExperiment[]
    end

    pool = Vector{Tuple{Float64, Float64, Float64, Int, Symbol}}()
    for expt in experiments
        conc = expt.observables.concentration
        n = length(conc.time)
        for j in 2:n
            push!(pool, (conc.time[j], conc.mean[j], conc.variance[j],
                         expt.exp_id, :conc))
        end
        if include_ps
            push!(pool, (conc.time[end], expt.observables.d43.mean,
                         expt.observables.d43.variance, expt.exp_id, :ps))
        end
    end

    if isempty(pool)
        return CrystallisationExperiment[]
    end

    idx = rand(rng, 1:length(pool), n_samples)
    conc_by_id = Dict{Int, Vector{Tuple{Float64, Float64, Float64}}}()
    ps_by_id = Dict{Int, Vector{Tuple{Float64, Float64, Float64}}}()

    for k in idx
        time, value, var, exp_id, kind = pool[k]
        if kind == :conc
            push!(get!(conc_by_id, exp_id, Vector{Tuple{Float64, Float64, Float64}}()),
                  (time, value, var))
        else
            push!(get!(ps_by_id, exp_id, Vector{Tuple{Float64, Float64, Float64}}()),
                  (time, value, var))
        end
    end

    exp_ids = include_ps ? union(keys(conc_by_id), keys(ps_by_id)) : keys(conc_by_id)

    out = CrystallisationExperiment[]
    for expt in experiments
        exp_id = expt.exp_id
        exp_id in exp_ids || continue
        conc_entries = get(conc_by_id, exp_id, Tuple{Float64, Float64, Float64}[])
        n_conc = length(conc_entries)

        time = Vector{Float64}(undef, n_conc + 1)
        conc = Vector{Float64}(undef, n_conc + 1)
        conc_var = Vector{Float64}(undef, n_conc + 1)

        time[1] = expt.observables.concentration.time[1]
        conc[1] = expt.observables.concentration.mean[1]
        conc_var[1] = expt.observables.concentration.variance[1]

        for (j, entry) in enumerate(conc_entries)
            t, c, v = entry
            time[j + 1] = t
            conc[j + 1] = c
            conc_var[j + 1] = v
        end

        order = sortperm(time)
        time = time[order]
        conc = conc[order]
        conc_var = conc_var[order]

        if include_ps
            ps_entries = get(ps_by_id, exp_id, Tuple{Float64, Float64, Float64}[])
            if isempty(ps_entries)
                ps_mean = CRISTOOL_MISSING_SIZE_SENTINEL
                ps_var = CRISTOOL_MISSING_SIZE_SENTINEL
            else
                ps_time = [entry[1] for entry in ps_entries]
                ps_idx = argmax(ps_time)
                ps_mean = ps_entries[ps_idx][2]
                ps_var = ps_entries[ps_idx][3]
            end
        else
            ps_mean = expt.observables.d43.mean
            ps_var = expt.observables.d43.variance
        end

        push!(out,
              CrystallisationExperiment(;
                  observables = (;
                      concentration = Observable(; time = time, mean = conc,
                                                       variance = conc_var),
                      d43 = Observable(; time = time[end], mean = ps_mean, variance = ps_var),
                      d50q = Observable(; time = time[end], mean = ps_mean, variance = ps_var),
                  ),
                  temperature = expt.temperature,
                  loading = expt.loading,
                  exp_id = exp_id,
                  metadata = expt.metadata))
    end

    return out
end

"""
    bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                 n_bootstrap::Integer, include_ps::Bool;
                                 n_samples::Union{Nothing, Integer} = nothing,
                                 seed::Union{Nothing, Integer} = nothing)
        -> Vector{Vector{CrystallisationExperiment}}

Generate `n_bootstrap` bootstrap datasets, each sampled with replacement from
the pooled non-initial tuples. A fresh `Xoshiro` RNG is created for each
bootstrap using a base seed (`seed` if provided, otherwise drawn from
`Random.default_rng()`) and incremented by `i - 1`. If `n_samples` is omitted,
the original pooled data size is used for every bootstrap. Experiments without
sampled tuples are omitted.
"""
function bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                      n_bootstrap::Integer, include_ps::Bool;
                                      n_samples::Union{Nothing, Integer} = nothing,
                                      seed::Union{Nothing, Integer} = nothing)
    if n_bootstrap < 0
        throw(ArgumentError("n_bootstrap must be non-negative."))
    end
    ns = isnothing(n_samples) ? _bootstrap_default_samples(experiments, include_ps) :
         n_samples

    base_seed = isnothing(seed) ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    out = Vector{Vector{CrystallisationExperiment}}(undef, n_bootstrap)
    for i in 1:n_bootstrap
        rng = Random.Xoshiro(base_seed + UInt64(i - 1))
        out[i] = _bootstrap_repeatmeasurements_rng(experiments, rng, ns;
                                                   include_ps = include_ps)
    end

    return out
end

"""
    bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                 include_ps::Bool; n_samples::Union{Nothing, Integer} = nothing,
                                 seed::Union{Nothing, Integer} = nothing)
        -> Vector{CrystallisationExperiment}

Generate a single bootstrap dataset by resampling pooled non-initial tuples
with replacement. A `Xoshiro` RNG is created from `seed` (if provided) or from
a base seed drawn from `Random.default_rng()`. If `n_samples` is omitted, the
original pooled data size is used. Experiments without sampled tuples are
omitted.
"""
function bootstrap_repeatmeasurements(experiments::Vector{CrystallisationExperiment},
                                      include_ps::Bool;
                                      n_samples::Union{Nothing, Integer} = nothing,
                                      seed::Union{Nothing, Integer} = nothing)
    ns = isnothing(n_samples) ? _bootstrap_default_samples(experiments, include_ps) :
         n_samples
    base_seed = isnothing(seed) ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    rng = Random.Xoshiro(base_seed)
    return _bootstrap_repeatmeasurements_rng(experiments, rng, ns; include_ps = include_ps)
end
