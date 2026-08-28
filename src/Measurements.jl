"""
Utilities for loading and representing experimental measurement data for
crystallisation runs. Experiments are stored in the typed
`CrystallisationExperiment` container: a `NamedTuple` of per-observable
`SeriesObservable`/`ScalarObservable` entries plus run conditions.
"""

using Random

"""
    load_experiments(filepath::AbstractString, sheet_name::AbstractString,
                     loading::Real, temperature_range::Tuple=(nothing, nothing))
        -> Vector{CrystallisationExperiment}

Load experiments from a single Excel sheet formatted like the Python importer
(`data_import.py`), filtering by a specific `loading` and grouping by `Exp_ID`.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `sheet_name::AbstractString`: Name of the sheet to load (e.g., a system name)
- `loading::Real`: Loading value to filter experiments by
- `temperature_range::Tuple=(nothing, nothing)`: Optional `(Tmin, Tmax)` bounds to
  filter rows by temperature; pass `nothing` for no bound on that side.

# Returns
- `Vector{CrystallisationExperiment}`: One experiment per unique `Exp_ID`

The sheet is expected to include the following columns:
- `Exp_ID` (Int), `System` (String), `Temperature` (Real), `Loading` (Real),
  `Time` (Real), `Concentration` (Real), optional `Concentration_var` (Real),
  optional `PS` (Real), optional `PS_var` (Real).

Notes:
- The `concentration` observable is a `SeriesObservable(time, mean, variance)`.
- Particle size (PS) is taken at the last timepoint only and stored as both a
  `d43` and a `d50q` `ScalarObservable` (the legacy loader stored the same
  value in both slots). If `PS` at the last timepoint is `-1` or `missing`, a
  dummy value of `10.0` is used and variance is set to `100.0`, matching the
  legacy behaviour.
"""
function load_experiments(filepath::AbstractString, sheet_name::AbstractString,
                          loading::Real,
                          temperature_range::Tuple = (nothing, nothing))

    # Load the sheet as a DataFrame
    df = DataFrame(XLSX.readtable(filepath, sheet_name))

    # Normalize expected numeric columns to Float64 where present
    for col in
        (:Time, :Concentration, :Concentration_var, :Temperature, :Loading, :PS, :PS_var)
        if col in names(df)
            df[!, col] = Float64.(coalesce.(df[!, col], NaN))
        end
    end

    # Ensure required columns exist
    req_cols = [:Exp_ID, :Time, :Concentration]
    for col in req_cols
        if !(string(col) in names(df))
            println("Available columns: ", names(df))
            throw(ArgumentError("Expected column '$(col)' not found in sheet '$(sheet_name)'"))
        end
    end

    # Provide default concentration variance if missing
    if "Concentration_var" ∉ names(df)
        df[!, :Concentration_var] = zeros(size(df, 1))
    end

    # Optional columns defaults if missing
    if "Temperature" ∉ names(df)
        df[!, :Temperature] = fill(NaN, size(df, 1))
    end
    if "Loading" ∉ names(df)
        df[!, :Loading] = fill(NaN, size(df, 1))
    end

    # Filter by the requested loading, and optionally by temperature bounds
    df_filtered = df[df.Loading .== Float64(loading), :]

    # Apply temperature bounds only if at least one bound is provided
    tmin, tmax = temperature_range
    if !(tmin === nothing && tmax === nothing)
        lower = tmin === nothing ? -Inf : Float64(tmin)
        upper = tmax === nothing ? Inf : Float64(tmax)
        mask = (df_filtered.Temperature .>= lower) .& (df_filtered.Temperature .<= upper)
        df_filtered = df_filtered[mask, :]
    end

    # If nothing matches, return empty vector
    if nrow(df_filtered) == 0
        @info "No experiments found for loading=$(loading) in sheet '$(sheet_name)' of $(filepath)"
        return CrystallisationExperiment[]
    end

    sort!(df_filtered, [:Exp_ID, :Time])

    # Group by experiment id and build an experiment per id
    gdf = groupby(df_filtered, :Exp_ID, sort = true)
    out = Vector{CrystallisationExperiment}(undef, length(gdf))

    for (i, sdf) in enumerate(gdf)
        # Extract experiment ID from the grouped data
        exp_id = Int(first(sdf.Exp_ID))

        # Extract vectors
        time = collect(skipmissing(sdf.Time)) |> Vector{Float64}
        conc = collect(skipmissing(sdf.Concentration)) |> Vector{Float64}
        conc_var_raw = coalesce.(sdf.Concentration_var, 0.0)
        conc_var = Vector{Float64}(conc_var_raw)

        # Basic sanity: align lengths if needed
        L = min(length(time), length(conc), length(conc_var))
        time = time[1:L]
        conc = conc[1:L]
        conc_var = conc_var[1:L]

        # Temperature assumed constant within experiment
        Texp = begin
            Tvals = skipmissing(sdf.Temperature)
            isempty(Tvals) ? NaN : Float64(first(Tvals))
        end

        # Particle size: last element only (check for -1 sentinel)
        ps_last = begin
            if "PS" in names(sdf)
                v = sdf.PS[end]
                v === missing ? -1.0 : Float64(v)
            else
                -1.0
            end
        end
        ps_var_last = begin
            if "PS_var" in names(sdf)
                v = sdf.PS_var[end]
                v === missing ? -1.0 : Float64(v)
            else
                -1.0
            end
        end

        if ps_last == -1.0 || !isfinite(ps_last)
            ps_mean = 10.0
            ps_var = 100.0
        else
            ps_mean = ps_last
            # If PS_var at last is missing or sentinel, set to 100.0 as instructed
            ps_var = (ps_var_last == -1.0 || !isfinite(ps_var_last)) ? 100.0 : ps_var_last
        end

        out[i] = CrystallisationExperiment(;
            observables = (;
                concentration = SeriesObservable(; time = time, mean = conc,
                                                 variance = conc_var),
                d43 = ScalarObservable(; value = ps_mean, variance = ps_var,
                                       time = time[end]),
                d50q = ScalarObservable(; value = ps_mean, variance = ps_var,
                                        time = time[end]),
            ),
            temperature = round(Texp + 273.15, digits = 2),
            loading = Float64(loading),
            exp_id = exp_id)
    end

    @info("Loaded $(filepath) - $(length(gdf)) experiments - Initial Concentration: "*join([" $(round(out[i].observables.concentration.mean[1], sigdigits = 3) )  [mg/ml] "
                                                                                            for i in
                                                                                                 eachindex(out)]))

    return out

end

"""
    load_experiments_legacy(filepath::AbstractString, n_sheets::Int64=2) -> Vector{CrystallisationExperiment}

Load repeated crystallization measurements from legacy Excel files containing
sheets named "c i", "q i", and "d i" for each measurement i (concentration,
quantile, and diameter data respectively).

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `n_sheets::Int64=2`: Number of measurement sheets to load (default: 2)

The `q i` sheet's first quantile becomes the `d50q` scalar observable; the
`d i` sheet's last quantile row becomes the `d43` scalar observable (the
legacy layout stored d10/d32/d43 in successive rows). Temperatures and
loadings are unknown in this format and set to `NaN`/`0.0`.
"""
function load_experiments_legacy(filepath::AbstractString, n_sheets::Int64 = 2)
    return load_experiments_legacy(filepath, collect(1:n_sheets))
end

"""
    load_experiments_legacy(filepath::AbstractString, sheet_ids::Vector{Int64}) -> Vector{CrystallisationExperiment}

Load repeated crystallization measurements from specific legacy sheets
("c i"/"q i"/"d i" format, see `load_experiments_legacy`).
"""
function load_experiments_legacy(filepath::AbstractString, sheet_ids::Vector{Int64})

    measurements = Vector{CrystallisationExperiment}(undef, length(sheet_ids))

    for (i, id) in enumerate(sheet_ids)
        df_c = DataFrame(XLSX.readtable(filepath, "c $id"))
        df_c[!, :] = convert.(Float64, df_c[!, :])

        df_q = dropmissing(DataFrame(XLSX.readtable(filepath, "q $id")))
        df_q[!, :] = convert.(Float64, df_q[!, :])

        df_d = dropmissing(DataFrame(XLSX.readtable(filepath, "d $id")))
        df_d[!, :] = convert.(Float64, df_d[!, :])

        measurements[i] = CrystallisationExperiment(;
            observables = (;
                concentration = SeriesObservable(; time = df_c.time,
                                                 mean = df_c.concentrationmean,
                                                 variance = df_c.concentrationvariance),
                d50q = ScalarObservable(; value = df_q.qmean[1],
                                        variance = df_q.qvariance[1]),
                d43 = ScalarObservable(; value = df_d.qmean[end],
                                       variance = df_d.qvariance[end]),
            ),
            temperature = NaN,
            loading = 0.0,
            exp_id = id)
    end

    @info("Loaded $(filepath) - $(length(measurements)) measurements - Initial Concentration: "*join([" $(round(measurements[i].observables.concentration.mean[1], sigdigits = 3) )  [mg/ml] "
                                                                                                      for i in
                                                                                                          1:length(sheet_ids)]))

    return measurements

end

"""
    load_experiments_legacy_single(filepath::AbstractString, n_sheets::Int64=2) -> Vector{CrystallisationExperiment}

Load single (unreplicated) crystallization measurements from legacy Excel
files ("c i"/"q i"/"d i" format, the `d i` sheet carrying `d10`/`d32`/`d43`
columns). No variance information exists in this format; observables carry
`variance = nothing`.
"""
function load_experiments_legacy_single(filepath::AbstractString, n_sheets::Int64 = 2)

    measurements = Vector{CrystallisationExperiment}(undef, n_sheets)

    for i in 1:n_sheets
        df_c = DataFrame(XLSX.readtable(filepath, "c $i"))
        df_c[!, :] = convert.(Float64, df_c[!, :])

        df_q = dropmissing(DataFrame(XLSX.readtable(filepath, "q $i")))
        df_q[!, :] = convert.(Float64, df_q[!, :])

        df_d = dropmissing(DataFrame(XLSX.readtable(filepath, "d $i")))
        df_d[!, :] = convert.(Float64, df_d[!, :])

        measurements[i] = CrystallisationExperiment(;
            observables = (;
                concentration = SeriesObservable(; time = df_c.time,
                                                 mean = df_c.concentrationmean),
                d50q = ScalarObservable(; value = df_q.qmean[1],
                                        variance = df_q.qvariance[1]),
                d10 = ScalarObservable(; value = df_d.d10[1]),
                d32 = ScalarObservable(; value = df_d.d32[1]),
                d43 = ScalarObservable(; value = df_d.d43[1]),
            ),
            temperature = NaN,
            loading = 0.0,
            exp_id = i)
    end

    @info("Loaded $(filepath) - $(n_sheets) measurements - Initial Concentration: "*join([" $(round(measurements[i].observables.concentration.mean[1], sigdigits = 3) )  [mg/ml] "
                                                                                          for i in
                                                                                              1:n_sheets]))

    return measurements

end

"""
    repeatmeasurementbalancer(experiments::Vector{CrystallisationExperiment}, minconcstd::Real=10) -> Vector{CrystallisationExperiment}

Balance concentration measurement variances by enforcing a minimum relative
variance (at least `minconcstd`% of the mean value at each timepoint).
"""
function repeatmeasurementbalancer(experiments::Vector{CrystallisationExperiment},
                                   minconcstd::Real = 10)

    newmeasurements = Vector{CrystallisationExperiment}(undef, length(experiments))

    for (i, expt) in enumerate(experiments)
        conc = expt.observables.concentration
        balanced_conc = SeriesObservable(;
            time = conc.time,
            mean = conc.mean,
            variance = max.(conc.variance,
                            (minconcstd * 1e-2 * conc.mean) .^ 2))
        newmeasurements[i] = CrystallisationExperiment(;
            observables = merge(expt.observables,
                                (; concentration = balanced_conc)),
            temperature = expt.temperature,
            loading = expt.loading,
            exp_id = expt.exp_id)
    end

    return newmeasurements
end

"""
    psd_measurementbalancer(experiments::Vector{CrystallisationExperiment}, psd_std_pc::Real=10) -> Vector{CrystallisationExperiment}

Balance particle size measurement variances by enforcing a minimum relative
variance (at least `psd_std_pc`% of the value for the `d43`/`d50q` scalar
observables).
"""
function psd_measurementbalancer(experiments::Vector{CrystallisationExperiment},
                                 psd_std_pc::Real = 10)

    newmeasurements = Vector{CrystallisationExperiment}(undef, length(experiments))

    for (i, expt) in enumerate(experiments)
        obs = expt.observables
        balanced_obs = map(obs) do observable
            observable isa ScalarObservable || return observable
            var_floor = (observable.value * psd_std_pc * 1e-2)^2
            return ScalarObservable(; value = observable.value,
                                    variance = max(observable.variance, var_floor),
                                    time = observable.time)
        end
        newmeasurements[i] = CrystallisationExperiment(;
            observables = balanced_obs,
            temperature = expt.temperature,
            loading = expt.loading,
            exp_id = expt.exp_id)
    end

    return newmeasurements
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
            push!(pool, (conc.time[end], expt.observables.d43.value,
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
                ps_mean = -1.0
                ps_var = -1.0
            else
                ps_time = [entry[1] for entry in ps_entries]
                ps_idx = argmax(ps_time)
                ps_mean = ps_entries[ps_idx][2]
                ps_var = ps_entries[ps_idx][3]
            end
        else
            ps_mean = expt.observables.d43.value
            ps_var = expt.observables.d43.variance
        end

        push!(out,
              CrystallisationExperiment(;
                  observables = (;
                      concentration = SeriesObservable(; time = time, mean = conc,
                                                       variance = conc_var),
                      d43 = ScalarObservable(; value = ps_mean, variance = ps_var,
                                             time = time[end]),
                      d50q = ScalarObservable(; value = ps_mean, variance = ps_var,
                                              time = time[end]),
                  ),
                  temperature = expt.temperature,
                  loading = expt.loading,
                  exp_id = exp_id))
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