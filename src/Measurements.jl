"""
Utilities for loading and representing experimental measurement data for
crystallisation runs. Experiments are stored in the typed
`CrystallisationExperiment` container: a `NamedTuple` of per-observable
`Observable` entries plus run conditions.
"""

function _measurement_source_column(column)
    return column isa Symbol ? column : Symbol(column)
end

function _measurement_column_spec(spec)
    if spec isa Tuple
        1 <= length(spec) <= 2 ||
            throw(ArgumentError("Observable column specifications need one mean column and optionally one variance column."))
        return spec[1], length(spec) == 2 ? spec[2] : nothing
    end
    return spec, nothing
end

function _first_measurement_value(table, column, default)
    column === nothing && return default
    values = table[!, _measurement_source_column(column)]
    for value in values
        value === missing && continue
        return value
    end
    return default
end

function _measurement_series(table, time_col, mean_col, variance_col)
    times = table[!, _measurement_source_column(time_col)]
    means = table[!, _measurement_source_column(mean_col)]
    indices = [i for i in eachindex(times, means)
               if times[i] !== missing && means[i] !== missing]
    isempty(indices) && throw(ArgumentError("Observable column :$mean_col has no usable measurements."))

    observed_times = Float64[Float64(times[i]) for i in indices]
    observed_means = Float64[Float64(means[i]) for i in indices]
    if variance_col === nothing
        observed_variance = nothing
    else
        variances = table[!, _measurement_source_column(variance_col)]
        observed_variance = Float64[variances[i] === missing ? 0.0 : Float64(variances[i])
                                    for i in indices]
    end
    return Observable(; time = observed_times, mean = observed_means,
                      variance = observed_variance)
end

function _measurement_scalar(table, time_col, mean_col, variance_col)
    times = table[!, _measurement_source_column(time_col)]
    means = table[!, _measurement_source_column(mean_col)]
    index = findlast(i -> times[i] !== missing && means[i] !== missing,
                     eachindex(times, means))
    index === nothing && throw(ArgumentError("Observable column :$mean_col has no usable scalar measurement."))

    variance = if variance_col === nothing
        nothing
    else
        value = table[!, _measurement_source_column(variance_col)][index]
        value === missing ? nothing : Float64(value)
    end
    return Observable(; time = Float64(times[index]), mean = Float64(means[index]),
                      variance = variance)
end

"""
    load_measurements(filepath, sheet_name; time_col=:Time, id_col=:Exp_ID,
                      observables=(concentration = (:Concentration,
                      :Concentration_var),), scalar_observables=(),
                      metadata_cols=(temperature=:Temperature, loading=:Loading),
                      temperature_transform=identity, filters=NamedTuple())
        -> Vector{CrystallisationExperiment}

Load a table-driven measurement sheet. Each `observables` entry maps an
observable name to either a mean column or `(mean_column, variance_column)`.
Names listed in `scalar_observables` use the final available row of each
experiment; all others retain their own time series. `metadata_cols` maps
typed metadata names to source columns and is retained on each experiment.
The fixed `temperature`, `loading`, and `exp_id` fields are also populated
from metadata columns when those names are supplied.
"""
function load_measurements(filepath::AbstractString, sheet_name::AbstractString;
                            time_col = :Time,
                            id_col = :Exp_ID,
                            observables::NamedTuple =
                                (; concentration = (:Concentration, :Concentration_var)),
                            scalar_observables = (),
                            metadata_cols::NamedTuple =
                                (; temperature = :Temperature, loading = :Loading),
                            temperature_transform = identity,
                            filters::NamedTuple = NamedTuple())
    table = DataFrame(XLSX.readtable(filepath, sheet_name))
    id_source = _measurement_source_column(id_col)
    time_source = _measurement_source_column(time_col)
    for source in (id_source, time_source)
        string(source) in names(table) ||
            throw(ArgumentError("Expected column '$source' not found in sheet '$sheet_name'."))
    end

    for (column, expected) in pairs(filters)
        source = _measurement_source_column(column)
        string(source) in names(table) ||
            throw(ArgumentError("Filter column '$source' not found in sheet '$sheet_name'."))
        table = table[table[!, source] .== expected, :]
    end
    isempty(table) && return CrystallisationExperiment[]

    observable_names = keys(observables)
    for observable_name in observable_names
        mean_col, variance_col = _measurement_column_spec(getproperty(observables,
                                                                        observable_name))
        for source in (mean_col, variance_col)
            source === nothing && continue
            string(_measurement_source_column(source)) in names(table) ||
                throw(ArgumentError("Expected column '$source' for observable :$observable_name."))
        end
    end

    groups = groupby(table, id_source, sort = true)
    experiments = Vector{CrystallisationExperiment}(undef, length(groups))
    for (index, group) in enumerate(groups)
        observable_values = map(observable_names) do observable_name
            mean_col, variance_col = _measurement_column_spec(getproperty(observables,
                                                                            observable_name))
            if observable_name in scalar_observables
                _measurement_scalar(group, time_source, mean_col, variance_col)
            else
                _measurement_series(group, time_source, mean_col, variance_col)
            end
        end
        named_observables = NamedTuple{observable_names}(Tuple(observable_values))

        metadata_names = keys(metadata_cols)
        metadata_values = map(metadata_names) do metadata_name
            _first_measurement_value(group, getproperty(metadata_cols, metadata_name), missing)
        end
        metadata = NamedTuple{metadata_names}(Tuple(metadata_values))

        temperature_source = hasproperty(metadata_cols, :temperature) ?
                             getproperty(metadata_cols, :temperature) : nothing
        loading_source = hasproperty(metadata_cols, :loading) ?
                         getproperty(metadata_cols, :loading) : nothing
        raw_temperature = _first_measurement_value(group, temperature_source, NaN)
        raw_loading = _first_measurement_value(group, loading_source, 0.0)
        temperature_value = raw_temperature === missing ? NaN :
                            Float64(temperature_transform(Float64(raw_temperature)))
        loading_value = raw_loading === missing ? 0.0 : Float64(raw_loading)
        experiment_id = Int(_first_measurement_value(group, id_source, index))

        experiments[index] = CrystallisationExperiment(;
            observables = named_observables,
            temperature = temperature_value,
            loading = loading_value,
            exp_id = experiment_id,
            metadata = metadata)
    end
    return experiments
end

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
- The `concentration` observable is a `Observable(time, mean, variance)`.
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
        if string(col) in names(df)
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
                v === missing ? CRISTOOL_MISSING_SIZE_SENTINEL : Float64(v)
            else
                CRISTOOL_MISSING_SIZE_SENTINEL
            end
        end
        ps_var_last = begin
            if "PS_var" in names(sdf)
                v = sdf.PS_var[end]
                v === missing ? CRISTOOL_MISSING_SIZE_SENTINEL : Float64(v)
            else
                CRISTOOL_MISSING_SIZE_SENTINEL
            end
        end

        if ps_last == CRISTOOL_MISSING_SIZE_SENTINEL || !isfinite(ps_last)
            ps_mean = CRISTOOL_MISSING_SIZE_VALUE
            ps_var = CRISTOOL_MISSING_SIZE_VARIANCE
        else
            ps_mean = ps_last
            # If PS_var at last is missing or sentinel, set to 100.0 as instructed
            ps_var = (ps_var_last == CRISTOOL_MISSING_SIZE_SENTINEL || !isfinite(ps_var_last)) ?
                     CRISTOOL_MISSING_SIZE_VARIANCE : ps_var_last
        end

        out[i] = CrystallisationExperiment(;
            observables = (;
                concentration = Observable(; time = time, mean = conc,
                                                 variance = conc_var),
                d43 = Observable(; time = time[end], mean = ps_mean, variance = ps_var),
                d50q = Observable(; time = time[end], mean = ps_mean, variance = ps_var),
            ),
            temperature = round(Texp + 273.15, digits = 2),
            loading = Float64(loading),
            exp_id = exp_id,
            metadata = (;))
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
                concentration = Observable(; time = df_c.time,
                                                 mean = df_c.concentrationmean,
                                                 variance = df_c.concentrationvariance),
                d50q = Observable(; mean = df_q.qmean[1], variance = df_q.qvariance[1]),
                d43 = Observable(; mean = df_d.qmean[end], variance = df_d.qvariance[end]),
            ),
            temperature = NaN,
            loading = 0.0,
            exp_id = id,
            metadata = (;))
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
                concentration = Observable(; time = df_c.time,
                                                 mean = df_c.concentrationmean),
                d50q = Observable(; mean = df_q.qmean[1], variance = df_q.qvariance[1]),
                d10 = Observable(; mean = df_d.d10[1]),
                d32 = Observable(; mean = df_d.d32[1]),
                d43 = Observable(; mean = df_d.d43[1]),
            ),
            temperature = NaN,
            loading = 0.0,
            exp_id = i,
            metadata = (;))
    end

    @info("Loaded $(filepath) - $(n_sheets) measurements - Initial Concentration: "*join([" $(round(measurements[i].observables.concentration.mean[1], sigdigits = 3) )  [mg/ml] "
                                                                                          for i in
                                                                                              1:n_sheets]))

    return measurements

end


"""
    _variance_floor_observable(observable, min_rel_std_pc) -> Observable

Raise an observable's variance to at least `(min_rel_std_pc% of mean)^2`.
The methods dispatch on scalar versus series shape and create a variance
vector when a single-replicate observable has no variance.
"""
function _variance_floor_observable(observable::Observable{T, Tt, Nothing},
                                    min_rel_std_pc) where {T <: AbstractArray, Tt}
    floor = (min_rel_std_pc * 1e-2 .* abs.(observable.mean)) .^ 2
    return Observable(; time = observable.time, mean = observable.mean,
                      variance = floor)
end

function _variance_floor_observable(observable::Observable{T, Tt, Tv},
                                    min_rel_std_pc) where {T <: AbstractArray, Tt, Tv}
    floor = (min_rel_std_pc * 1e-2 .* abs.(observable.mean)) .^ 2
    variance = max.(observable.variance, floor)
    return Observable(; time = observable.time, mean = observable.mean, variance = variance)
end

function _variance_floor_observable(observable::Observable{T, Tt, Nothing},
                                    min_rel_std_pc) where {T <: Real, Tt}
    floor = (min_rel_std_pc * 1e-2 * abs(observable.mean))^2
    return Observable(; time = observable.time, mean = observable.mean, variance = floor)
end

function _variance_floor_observable(observable::Observable{T, Tt, Tv},
                                    min_rel_std_pc) where {T <: Real, Tt, Tv}
    floor = (min_rel_std_pc * 1e-2 * abs(observable.mean))^2
    variance = max(observable.variance, floor)
    return Observable(; time = observable.time, mean = observable.mean, variance = variance)
end

"""
    balance_variances(experiments; obs, min_rel_std_pc=10)
        -> Vector{CrystallisationExperiment}

Apply a relative standard-deviation floor to one named observable across all
experiments. The experiment metadata and all other observables are preserved.
"""
function balance_variances(experiments::Vector{<:CrystallisationExperiment};
                           obs::Symbol,
                           min_rel_std_pc::Real = 10)
    newmeasurements = Vector{CrystallisationExperiment}(undef, length(experiments))
    for (index, expt) in enumerate(experiments)
        hasproperty(expt.observables, obs) ||
            throw(ArgumentError("Experiment $(expt.exp_id) has no observable :$obs."))
        measured = getproperty(expt.observables, obs)
        balanced = _variance_floor_observable(measured, min_rel_std_pc)
        newmeasurements[index] = CrystallisationExperiment(;
            observables = merge(expt.observables, NamedTuple{(obs,)}((balanced,))),
            temperature = expt.temperature,
            loading = expt.loading,
            exp_id = expt.exp_id,
            metadata = expt.metadata)
    end
    return newmeasurements
end

"""
    repeatmeasurementbalancer(experiments::Vector{CrystallisationExperiment}, minconcstd::Real=10) -> Vector{CrystallisationExperiment}

Balance concentration measurement variances by enforcing a minimum relative
variance (at least `minconcstd`% of the mean value at each timepoint).
"""
function repeatmeasurementbalancer(experiments::Vector{CrystallisationExperiment},
                                   minconcstd::Real = 10)
    return balance_variances(experiments; obs = :concentration,
                             min_rel_std_pc = minconcstd)
end

"""
    psd_measurementbalancer(experiments::Vector{CrystallisationExperiment}, psd_std_pc::Real=10) -> Vector{CrystallisationExperiment}

Balance particle size measurement variances by enforcing a minimum relative
variance (at least `psd_std_pc`% of the value for the `d43`/`d50q` scalar
observables).
"""
function psd_measurementbalancer(experiments::Vector{CrystallisationExperiment},
                                 psd_std_pc::Real = 10)
    balanced = experiments
    for observable_name in (:d43, :d50q)
        if !isempty(balanced) && hasproperty(first(balanced).observables, observable_name)
            balanced = balance_variances(balanced; obs = observable_name,
                                         min_rel_std_pc = psd_std_pc)
        end
    end
    return balanced
end

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
