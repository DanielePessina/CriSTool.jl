"""
Utilities for loading and representing experimental measurement data for crystallisation runs. Provides helper constructors for repeated and single measurements from Excel files.
"""

using Random

"""
    makerepeatmeasurements(filepath::AbstractString, n_sheets::Int64=2) -> Vector{CrystallisationRepeatMeasurements}

Load repeated crystallization measurements from an Excel file.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `n_sheets::Int64=2`: Number of measurement sheets to load (default: 2)

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects

The Excel file should contain sheets named "c i", "q i", and "d i" for each measurement i,
containing concentration, quantile, and diameter data respectively.
"""
function makerepeatmeasurements(filepath::AbstractString, n_sheets::Int64 = 2)
    # # Open results

    # Create CrystallisationMeasurements objects
    measurements = Vector{CrystallisationRepeatMeasurements}(undef, n_sheets)

    for i in 1:n_sheets
        df_c = DataFrame(XLSX.readtable(filepath, "c $i"))
        df_c[!, :] = convert.(Float64, df_c[!, :])

        df_q = dropmissing(DataFrame(XLSX.readtable(filepath, "q $i")))

        df_q[!, :] = convert.(Float64, df_q[!, :])

        df_d = dropmissing(DataFrame(XLSX.readtable(filepath, "d $i")))

        df_d[!, :] = convert.(Float64, df_d[!, :])

        measurements[i] = CrystallisationRepeatMeasurements(df_c.time,
                                                            df_c.concentrationmean,
                                                            df_c.concentrationvariance,
                                                            df_q.quantilepercent[1],
                                                            df_q.qmean[1],
                                                            df_q.qvariance[1],
                                                            df_d.qmean[1],
                                                            df_d.qvariance[1],
                                                            df_d.qmean[3],
                                                            df_d.qvariance[3],
                                                            df_d.qmean[4],
                                                            df_d.qvariance[4],
                                                            NaN,
                                                            NaN,
                                                            i)

    end

    # @info("$(n_sheets) measurements loaded - Initial Concentration:" * join([" $(round(measurements[i].concentrationmean[1], sigdigits = 3) )  [mg/ml] " for i in 1:n_sheets]))
    @info("Loaded $(filepath) - $(n_sheets) measurements - Initial Concentration: "*join([" $(round(measurements[i].concentrationmean[1], sigdigits = 3) )  [mg/ml] "
                                                                                          for i in
                                                                                              1:n_sheets]))


    return measurements

end

"""
    makesinglemeasurements(filepath::AbstractString, n_sheets::Int64=2) -> Vector{CrystallisationSingleMeasurement}

Load single crystallization measurements from an Excel file.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `n_sheets::Int64=2`: Number of measurement sheets to load (default: 2)

# Returns
- `Vector{CrystallisationSingleMeasurement}`: Vector of measurement objects

The Excel file should contain sheets named "c i", "q i", and "d i" for each measurement i,
containing concentration, quantile, and diameter data respectively.
"""
function makesinglemeasurements(filepath::AbstractString, n_sheets::Int64 = 2)

    measurements = Vector{CrystallisationSingleMeasurement}(undef, n_sheets)

    for i in 1:n_sheets
        df_c = DataFrame(XLSX.readtable(filepath, "c $i"))
        df_c[!, :] = convert.(Float64, df_c[!, :])

        df_q = dropmissing(DataFrame(XLSX.readtable(filepath, "q $i")))

        df_q[!, :] = convert.(Float64, df_q[!, :])

        df_d = dropmissing(DataFrame(XLSX.readtable(filepath, "d $i")))

        df_d[!, :] = convert.(Float64, df_d[!, :])

        measurements[i] = CrystallisationSingleMeasurement(df_c.time,
                                                           df_c.concentrationmean,
                                                           df_q.quantilepercent, df_q.qmean,
                                                           df_d.d10, df_d.d32, df_d.d43)

    end


    # @info("$(n_sheets) measurements loaded - Initial Concentration: " * join([" $(round(measurements[i].concentrationmean[1], sigdigits = 3) )  [mg/ml] " for i in 1:n_sheets]))
    @info("Loaded $(filepath) - $(n_sheets) measurements - Initial Concentration: "*join([" $(round(measurements[i].concentrationmean[1], sigdigits = 3) )  [mg/ml] "
                                                                                          for i in
                                                                                              1:n_sheets]))

    return measurements

end

"""
    makerepeatmeasurements(filepath::AbstractString, sheet_ids::Vector{Int64}) -> Vector{CrystallisationRepeatMeasurements}

Load repeated crystallization measurements from specific sheets in an Excel file.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `sheet_ids::Vector{Int64}`: Vector of sheet indices to load

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects

The Excel file should contain sheets named "c i", "q i", and "d i" for each measurement i,
containing concentration, quantile, and diameter data respectively.
"""
function makerepeatmeasurements(filepath::AbstractString, sheet_ids::Vector{Int64})
    # # Open results

    # Create CrystallisationMeasurements objects
    measurements = Vector{CrystallisationRepeatMeasurements}(undef, length(sheet_ids))

    for (i, id) in enumerate(sheet_ids)
        try
            df_c = DataFrame(XLSX.readtable(filepath, "c $id"))
            df_c[!, :] = convert.(Float64, df_c[!, :])

            df_q = dropmissing(DataFrame(XLSX.readtable(filepath, "q $id")))

            df_q[!, :] = convert.(Float64, df_q[!, :])

            df_d = dropmissing(DataFrame(XLSX.readtable(filepath, "d $id")))

            df_d[!, :] = convert.(Float64, df_d[!, :])

            measurements[i] = CrystallisationRepeatMeasurements(df_c.time,
                                                                df_c.concentrationmean,
                                                                df_c.concentrationvariance,
                                                                df_q.qmean[1],
                                                                df_q.qvariance[1],
                                                                df_d.qmean[1],
                                                                df_d.qvariance[1],
                                                                df_d.qmean[3],
                                                                df_d.qvariance[3],
                                                                df_c.temperature[1],
                                                                NaN,
                                                                id)
        catch err
            if err isa ArgumentError
                @error("Failed to load measurements from filepath $filepath sheet $id\n $err")
            else
                @error("Unexpected error loading measurements from filepath $filepath sheet $id: $err")
            end

        end
    end

    # @info("Loaded $(filepath) - measurements - Initial Concentration: " * join([" $(round(measurements[i].concentrationmean[1], sigdigits = 3) )  [mg/ml] " for i in 1:length(sheet_ids)]))

    return measurements

end

"""
    makerepeatmeasurements(filepath::AbstractString, sheet_name::AbstractString,
                           loading::Real, temperature_range::Tuple=(nothing, nothing))
        -> Vector{CrystallisationRepeatMeasurements}

Load repeated crystallization measurements from a single Excel sheet formatted like the Python importer
(`data_import.py`), filtering by a specific `loading` and grouping by `Exp_ID`.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `sheet_name::AbstractString`: Name of the sheet to load (e.g., a system name)
- `loading::Real`: Loading value to filter experiments by
- `temperature_range::Tuple=(nothing, nothing)`: Optional `(Tmin, Tmax)` bounds to
  filter rows by temperature; pass `nothing` for no bound on that side.

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects, one per unique `Exp_ID`

The sheet is expected to include the following columns:
- `Exp_ID` (Int), `System` (String), `Temperature` (Real), `Loading` (Real),
  `Time` (Real), `Concentration` (Real), optional `Concentration_var` (Real),
  optional `PS` (Real), optional `PS_var` (Real).

Notes:
- Particle size (PS) is taken at the last timepoint only. If `PS` at the last
  timepoint is `-1` or `missing`, a dummy value of `10.0` is used and variance
  is set to `100.0`. The same values are assigned to both `quantile*` and `d43*`
  fields of `CrystallisationRepeatMeasurements` for compatibility with both
  discretised and MoM solvers.
"""
function makerepeatmeasurements(filepath::AbstractString, sheet_name::AbstractString,
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
    # Use ≈ comparison tolerance for floats if needed; here we use exact as data is clean.
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
        return Vector{CrystallisationRepeatMeasurements}()
    end

    sort!(df_filtered, [:Exp_ID, :Time])

    # Group by experiment id and build a measurement per experiment
    gdf = groupby(df_filtered, :Exp_ID, sort = true)
    out = Vector{CrystallisationRepeatMeasurements}(undef, length(gdf))

    for (i, sdf) in enumerate(gdf)
        # Extract experiment ID from the grouped data
        exp_id = first(sdf.Exp_ID)

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
            qmean = 10.0
            qvar = 100.0
            d43 = 10.0
            d43var = 100.0
        else
            qmean = ps_last
            # If PS_var at last is missing or sentinel, set to 100.0 as instructed
            qvar = (ps_var_last == -1.0 || !isfinite(ps_var_last)) ? 100.0 : ps_var_last
            d43 = qmean
            d43var = qvar
        end

        out[i] = CrystallisationRepeatMeasurements(time, conc, conc_var,
                                                   qmean, qvar,
                                                   d43, d43var,
                                                   round(Texp+273.15, digits = 2),
                                                   Float64(loading),
                                                   exp_id)
    end

    return out
end
"""
    makerepeatmeasurements(filepath::AbstractString, sheet_name::AbstractString,
                           loading::AbstractArray{<:Real},
                           temperature_range::Tuple=(nothing, nothing))
        -> Vector{CrystallisationRepeatMeasurements}

Load repeated crystallization measurements for multiple loading values.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `sheet_name::AbstractString`: Name of the sheet to load
- `loading::AbstractArray{<:Real}`: Array of loading values to filter experiments by
- `temperature_range::Tuple=(nothing, nothing)`: Optional `(Tmin, Tmax)` bounds

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Combined measurement objects for all loadings
"""
function makerepeatmeasurements(filepath::AbstractString, sheet_name::AbstractString,
                                loading::AbstractArray{<:Real},
                                temperature_range::Tuple = (nothing, nothing))
    all_measurements = Vector{CrystallisationRepeatMeasurements}()
    for l in loading
        ms = makerepeatmeasurements(filepath, sheet_name, l, temperature_range)
        append!(all_measurements, ms)
    end
    return all_measurements
end


"""
    makesinglemeasurements(filepath::AbstractString, sheet_ids::Vector{Int64}) -> Vector{CrystallisationSingleMeasurement}

Load single crystallization measurements from specific sheets in an Excel file.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `sheet_ids::Vector{Int64}`: Vector of sheet indices to load

# Returns
- `Vector{CrystallisationSingleMeasurement}`: Vector of measurement objects

The Excel file should contain sheets named "c i", "q i", and "d i" for each measurement i,
containing concentration, quantile, and diameter data respectively.
"""
function makesinglemeasurements(filepath::AbstractString, sheet_ids::Vector{Int64})

    measurements = Vector{CrystallisationSingleMeasurement}(undef, length(sheet_ids))

    for (i, id) in enumerate(sheet_ids)
        df_c = DataFrame(XLSX.readtable(filepath, "c $id"))
        df_c[!, :] = convert.(Float64, df_c[!, :])

        df_q = dropmissing(DataFrame(XLSX.readtable(filepath, "q $id")))

        df_q[!, :] = convert.(Float64, df_q[!, :])

        df_d = dropmissing(DataFrame(XLSX.readtable(filepath, "d $id")))

        df_d[!, :] = convert.(Float64, df_d[!, :])

        measurements[i] = CrystallisationSingleMeasurement(df_c.time,
                                                           df_c.concentrationmean,
                                                           df_q.quantilepercent, df_q.qmean,
                                                           df_d.d10, df_d.d32, df_d.d43)

    end

    # @info("Loaded $(filepath) - measurements - Initial Concentration: " * join([" $(round(measurements[i].concentrationmean[1], sigdigits = 3) )  [mg/ml] " for i in 1:n_sheets]))

    return measurements

end

"""
    repeatmeasurementbalancer(measurements::Vector{CrystallisationRepeatMeasurements}, minconcvariance::Real=10) -> Vector{CrystallisationRepeatMeasurements}

Balance measurement variances by enforcing a minimum relative variance.

# Arguments
- `measurements::Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects
- `minconcvariance::Real=10`: Minimum relative variance in percent (default: 10)

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Vector of balanced measurement objects

Ensures measurement variances are at least minconcvariance% of the mean value.
"""

"""
    psd_measurementbalancer(measurements::Vector{CrystallisationRepeatMeasurements}, psd_std_pc::Real=10) -> Vector{CrystallisationRepeatMeasurements}

Balance particle size measurement variances by enforcing a minimum relative variance.

# Arguments
- `measurements::Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects
- `psd_std_pc::Real=10`: Minimum relative standard deviation in percent (default: 10%)

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Vector of balanced measurement objects

Ensures particle size variances (quantile and d43) are at least psd_std_pc% of their mean values.
"""
function psd_measurementbalancer(measurements::Vector{CrystallisationRepeatMeasurements},
                                 psd_std_pc::Real = 10)

    newmeasurements = Vector{CrystallisationRepeatMeasurements}(undef, length(measurements))

    for (i, singlemeasurement) in enumerate(measurements)
        newmeasurements[i] = CrystallisationRepeatMeasurements(singlemeasurement.time,
                                                               singlemeasurement.concentrationmean,
                                                               singlemeasurement.concentrationvariance,
                                                               singlemeasurement.quantilemean,
                                                               max(singlemeasurement.quantilevariance,
                                                                   (singlemeasurement.quantilemean .*
                                                                    psd_std_pc * 1e-2) .^ 2),
                                                               singlemeasurement.d43,
                                                               max(singlemeasurement.d43var,
                                                                   (singlemeasurement.d43 .*
                                                                    psd_std_pc * 1e-2) .^ 2),
                                                               singlemeasurement.temperature,
                                                               singlemeasurement.loading,
                                                               singlemeasurement.exp_id)
    end

    return newmeasurements
end

"""
    repeatmeasurementbalancer(measurements::Vector{CrystallisationRepeatMeasurements}, minconcstd::Real=10) -> Vector{CrystallisationRepeatMeasurements}

Balance concentration measurement variances by enforcing a minimum relative variance.

# Arguments
- `measurements::Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects
- `minconcstd::Real=10`: Minimum relative standard deviation in percent (default: 10%)

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Vector of balanced measurement objects

Ensures concentration variances are at least minconcstd% of their mean values.
"""
function repeatmeasurementbalancer(measurements::Vector{CrystallisationRepeatMeasurements},
                                   minconcstd::Real = 10)

    newmeasurements = Vector{CrystallisationRepeatMeasurements}(undef, length(measurements))

    for (i, singlemeasurement) in enumerate(measurements)
        newmeasurements[i] = CrystallisationRepeatMeasurements(singlemeasurement.time,
                                                               singlemeasurement.concentrationmean,
                                                               max.(singlemeasurement.concentrationvariance,
                                                                    (minconcstd * 1e-2 *
                                                                     singlemeasurement.concentrationmean) .^
                                                                    2),
                                                               singlemeasurement.quantilemean,
                                                               singlemeasurement.quantilevariance,
                                                               singlemeasurement.d43,
                                                               singlemeasurement.d43var,
                                                               singlemeasurement.temperature,
                                                               singlemeasurement.loading,
                                                               singlemeasurement.exp_id)
    end

    return newmeasurements
end

"""
    _bootstrap_default_samples(measurements::Vector{CrystallisationRepeatMeasurements}, include_ps::Bool) -> Int

Calculate the default number of bootstrap samples based on pooled data size.

# Arguments
- `measurements::Vector{CrystallisationRepeatMeasurements}`: Vector of measurement objects
- `include_ps::Bool`: Whether to include particle size measurements in the pool

# Returns
- `Int`: Total number of non-initial data points (plus PS entries if include_ps is true)
"""
function _bootstrap_default_samples(measurements::Vector{CrystallisationRepeatMeasurements},
                                    include_ps::Bool)
    pool_len = sum(max(length(m.time) - 1, 0) for m in measurements)
    return pool_len + (include_ps ? length(measurements) : 0)
end

"""
    _bootstrap_repeatmeasurements_rng(measurements::Vector{CrystallisationRepeatMeasurements},
                                      rng, n_samples::Integer; include_ps::Bool) -> Vector{CrystallisationRepeatMeasurements}

Generate a single bootstrap dataset using the provided RNG.

# Arguments
- `measurements::Vector{CrystallisationRepeatMeasurements}`: Original measurement data
- `rng`: Random number generator instance
- `n_samples::Integer`: Number of tuples to sample with replacement
- `include_ps::Bool`: Whether to include particle size measurements in the pool

# Returns
- `Vector{CrystallisationRepeatMeasurements}`: Bootstrapped measurement data

Samples tuples (excluding initial time points) with replacement and reconstructs
measurement objects. Experiments without sampled tuples are omitted.
"""
function _bootstrap_repeatmeasurements_rng(measurements::Vector{CrystallisationRepeatMeasurements},
                                           rng, n_samples::Integer; include_ps::Bool)

    if n_samples < 0
        throw(ArgumentError("n_samples must be non-negative."))
    end
    if isempty(measurements) || n_samples == 0
        return CrystallisationRepeatMeasurements[]
    end

    pool = Vector{Tuple{Float64, Float64, Float64, Int, Symbol}}()
    for m in measurements
        n = length(m.time)
        for j in 2:n
            push!(pool,
                  (m.time[j], m.concentrationmean[j], m.concentrationvariance[j],
                   m.exp_id, :conc))
        end
        if include_ps
            push!(pool, (m.time[end], m.d43, m.d43var, m.exp_id, :ps))
        end
    end

    if isempty(pool)
        return CrystallisationRepeatMeasurements[]
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

    out = CrystallisationRepeatMeasurements[]
    for m in measurements
        exp_id = m.exp_id
        exp_id in exp_ids || continue
        conc_entries = get(conc_by_id, exp_id, Tuple{Float64, Float64, Float64}[])
        n_conc = length(conc_entries)

        time = Vector{Float64}(undef, n_conc + 1)
        conc = Vector{Float64}(undef, n_conc + 1)
        conc_var = Vector{Float64}(undef, n_conc + 1)

        time[1] = m.time[1]
        conc[1] = m.concentrationmean[1]
        conc_var[1] = m.concentrationvariance[1]

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
                qmean = -1.0
                qvar = -1.0
                d43 = -1.0
                d43var = -1.0
            else
                ps_time = [entry[1] for entry in ps_entries]
                ps_idx = argmax(ps_time)
                ps_val = ps_entries[ps_idx][2]
                ps_var = ps_entries[ps_idx][3]
                qmean = ps_val
                qvar = ps_var
                d43 = ps_val
                d43var = ps_var
            end
        else
            qmean = m.quantilemean
            qvar = m.quantilevariance
            d43 = m.d43
            d43var = m.d43var
        end

        push!(out,
              CrystallisationRepeatMeasurements(time, conc, conc_var,
                                                qmean, qvar, d43, d43var,
                                                m.temperature, m.loading, m.exp_id))
    end

    return out
end

"""
    bootstrap_repeatmeasurements(measurements::Vector{CrystallisationRepeatMeasurements},
                                 n_bootstrap::Integer, include_ps::Bool;
                                 n_samples::Union{Nothing, Integer} = nothing,
                                 seed::Union{Nothing, Integer} = nothing)
        -> Vector{Vector{CrystallisationRepeatMeasurements}}

Generate `n_bootstrap` bootstrap datasets, each sampled with replacement from the pooled
non-initial tuples. A fresh `Xoshiro` RNG is created for each bootstrap using a base seed
(`seed` if provided, otherwise drawn from `Random.default_rng()`) and incremented by `i - 1`.
If `n_samples` is omitted, the original pooled data size is used for every bootstrap.
Experiments without sampled tuples are omitted.
"""
function bootstrap_repeatmeasurements(measurements::Vector{CrystallisationRepeatMeasurements},
                                      n_bootstrap::Integer, include_ps::Bool;
                                      n_samples::Union{Nothing, Integer} = nothing,
                                      seed::Union{Nothing, Integer} = nothing)
    if n_bootstrap < 0
        throw(ArgumentError("n_bootstrap must be non-negative."))
    end
    ns = isnothing(n_samples) ? _bootstrap_default_samples(measurements, include_ps) :
         n_samples

    base_seed = isnothing(seed) ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    out = Vector{Vector{CrystallisationRepeatMeasurements}}(undef, n_bootstrap)
    for i in 1:n_bootstrap
        rng = Random.Xoshiro(base_seed + UInt64(i - 1))
        out[i] = _bootstrap_repeatmeasurements_rng(measurements, rng, ns;
                                                   include_ps = include_ps)
    end

    return out
end

"""
    bootstrap_repeatmeasurements(measurements::Vector{CrystallisationRepeatMeasurements},
                                 include_ps::Bool; n_samples::Union{Nothing, Integer} = nothing,
                                 seed::Union{Nothing, Integer} = nothing)
        -> Vector{CrystallisationRepeatMeasurements}

Generate a single bootstrap dataset by resampling pooled non-initial tuples with replacement.
A `Xoshiro` RNG is created from `seed` (if provided) or from a base seed drawn from
`Random.default_rng()`. If `n_samples` is omitted, the original pooled data size is used.
Experiments without sampled tuples are omitted.
"""
function bootstrap_repeatmeasurements(measurements::Vector{CrystallisationRepeatMeasurements},
                                      include_ps::Bool;
                                      n_samples::Union{Nothing, Integer} = nothing,
                                      seed::Union{Nothing, Integer} = nothing)
    ns = isnothing(n_samples) ? _bootstrap_default_samples(measurements, include_ps) :
         n_samples
    base_seed = isnothing(seed) ? rand(Random.default_rng(), UInt64) : UInt64(seed)
    rng = Random.Xoshiro(base_seed)
    return _bootstrap_repeatmeasurements_rng(measurements, rng, ns; include_ps = include_ps)
end

"""
    CrystallisationAugmentedRepeatMeasurements <: AbstractMeasurements

Structure for storing augmented crystallization measurements with particle size information.

# Fields
- `time::Vector{Float64}`: Time points
- `concentrationmean::Vector{Float64}`: Mean concentration measurements
- `concentrationvariance::Vector{Float64}`: Variance of concentration measurements
- `time_ps::Vector{Float64}`: Time points for particle size measurements
- `ps_mean::Vector{Float64}`: Mean particle size measurements
- `ps_var::Vector{Float64}`: Variance of particle size measurements
"""
@concrete struct CrystallisationAugmentedRepeatMeasurements <: AbstractMeasurements

    time::Vector{Float64}
    concentrationmean::Vector{Float64}
    concentrationvariance::Vector{Float64}

    time_ps::Vector{Float64}

    ps_mean::Vector{Float64}
    ps_var::Vector{Float64}

end

"""
    augmentsingleexperiment(dataset::CrystallisationRepeatMeasurements, θ::AbstractVector{Float64},
                           nucl_f::NuF, growth_f::GrF, aggregation_f::AgF, breakage_f::BrF;
                           solver::AbstractSolver = MoM(), pseudo_psvariance::Real = 0.25) -> CrystallisationAugmentedRepeatMeasurements where {NuF<:AbstractFPNucleationFunction, GrF<:AbstractGrowthFunction, AgF<:AbstractAggregationFunction, BrF<:AbstractBreakageFunction}

Augment a single experiment's measurements with simulated particle size data.

# Arguments
- `dataset::CrystallisationRepeatMeasurements`: Original measurement data
- `θ::AbstractVector{Float64}`: Parameter vector for kinetics
- `nucl_f::NuF`: Nucleation function
- `growth_f::GrF`: Growth function
- `aggregation_f::AgF`: Aggregation function
- `breakage_f::BrF`: Breakage function
- `solver::AbstractSolver=MoM()`: Numerical solver (default: Method of Moments)
- `pseudo_psvariance::Real=0.25`: Relative variance for particle size (default: 0.25)

# Returns
- `CrystallisationAugmentedRepeatMeasurements`: Augmented measurement data
"""
function augmentsingleexperiment(dataset::CrystallisationRepeatMeasurements,
                                 θ::AbstractVector{Float64}, nucl_f::NuF, growth_f::GrF,
                                 aggregation_f::AgF, breakage_f::BrF;
                                 solver::AbstractSolver = MoM(),
                                 pseudo_psvariance::Real = 0.25) where {NuF <:
                                                                        AbstractFPNucleationFunction,
                                                                        GrF <:
                                                                        AbstractGrowthFunction,
                                                                        AgF <:
                                                                        AbstractAggregationFunction,
                                                                        BrF <:
                                                                        AbstractBreakageFunction}

    ### Simulate the experiment

    problem,
    simulation = runsimulation(θ, nucl_f, growth_f, aggregation_f, breakage_f,
                               dataset.concentrationmean[1], solver = solver,
                               save_idx = dataset.time)

    ps = solver isa MoM ? simulation.d43 : simulation.d50q
    psvariance = ps .* pseudo_psvariance

    return CrystallisationAugmentedRepeatMeasurements(dataset.time,
                                                      dataset.concentrationmean,
                                                      dataset.concentrationvariance,
                                                      simulation.time, ps, psvariance)

end
"""
    augmentsingleexperiment(dataset::CrystallisationRepeatMeasurements, θ::AbstractVector{Float64},
                           nucl_f::NuF, growth_f::GrF, aggregation_f::AgF, breakage_f::BrF,
                           insilico_conc::Bool; solver::AbstractSolver=MoM(),
                           pseudo_psvariance::Real=0.25) -> CrystallisationAugmentedRepeatMeasurements

Augment a single experiment's measurements with simulated data, optionally replacing concentration.

# Arguments
- `dataset::CrystallisationRepeatMeasurements`: Original measurement data
- `θ::AbstractVector{Float64}`: Parameter vector for kinetics
- `nucl_f::NuF`: Nucleation function
- `growth_f::GrF`: Growth function
- `aggregation_f::AgF`: Aggregation function
- `breakage_f::BrF`: Breakage function
- `insilico_conc::Bool`: If true, use simulated concentration; if false, use measured
- `solver::AbstractSolver=MoM()`: Numerical solver
- `pseudo_psvariance::Real=0.25`: Relative variance for particle size

# Returns
- `CrystallisationAugmentedRepeatMeasurements`: Augmented measurement data
"""
function augmentsingleexperiment(dataset::CrystallisationRepeatMeasurements,
                                 θ::AbstractVector{Float64}, nucl_f::NuF, growth_f::GrF,
                                 aggregation_f::AgF, breakage_f::BrF, insilico_conc::Bool;
                                 solver::AbstractSolver = MoM(),
                                 pseudo_psvariance::Real = 0.25,) where {NuF <:
                                                                         AbstractFPNucleationFunction,
                                                                         GrF <:
                                                                         AbstractGrowthFunction,
                                                                         AgF <:
                                                                         AbstractAggregationFunction,
                                                                         BrF <:
                                                                         AbstractBreakageFunction}

    ### Simulate the experiment

    problem,
    simulation = runsimulation(θ, nucl_f, growth_f, aggregation_f, breakage_f,
                               dataset.concentrationmean[1], solver = solver,
                               save_idx = dataset.time)

    concentration = insilico_conc ? simulation.concentration : dataset.concentrationmean
    ps = solver isa MoM ? simulation.d43 : simulation.d50q
    psvariance = ps .* pseudo_psvariance

    return CrystallisationAugmentedRepeatMeasurements(dataset.time, concentration,
                                                      dataset.concentrationvariance,
                                                      simulation.time, ps, psvariance)

end

"""
    augmentdataset(dataset::Vector{CrystallisationRepeatMeasurements}, θ::AbstractMatrix{Float64},
                  nucl_f::NuF, growth_f::GrF, aggregation_f::AgF, breakage_f::BrF;
                  solver::AbstractSolver = MoM(), pseudo_psvariance::Real = 0.25) -> Vector{CrystallisationAugmentedRepeatMeasurements} where {NuF<:AbstractFPNucleationFunction, GrF<:AbstractGrowthFunction, AgF<:AbstractAggregationFunction, BrF<:AbstractBreakageFunction}

Augment multiple experiments' measurements with simulated particle size data.

# Arguments
- `dataset::Vector{CrystallisationRepeatMeasurements}`: Vector of measurement data
- `θ::AbstractMatrix{Float64}`: Parameter matrix for kinetics (columns correspond to experiments)
- `nucl_f::NuF`: Nucleation function
- `growth_f::GrF`: Growth function
- `aggregation_f::AgF`: Aggregation function
- `breakage_f::BrF`: Breakage function
- `solver::AbstractSolver=MoM()`: Numerical solver (default: Method of Moments)
- `pseudo_psvariance::Real=0.25`: Relative variance for particle size (default: 0.25)

# Returns
- `Vector{CrystallisationAugmentedRepeatMeasurements}`: Vector of augmented measurement data
"""
function augmentdataset(dataset::Vector{CrystallisationRepeatMeasurements},
                        θ::AbstractMatrix{Float64}, nucl_f::NuF, growth_f::GrF,
                        aggregation_f::AgF, breakage_f::BrF; solver::AbstractSolver = MoM(),
                        pseudo_psvariance::Real = 0.25) where {NuF <:
                                                               AbstractFPNucleationFunction,
                                                               GrF <:
                                                               AbstractGrowthFunction,
                                                               AgF <:
                                                               AbstractAggregationFunction,
                                                               BrF <:
                                                               AbstractBreakageFunction}

    augmenteddataset = Vector{CrystallisationAugmentedRepeatMeasurements}(undef,
                                                                          length(dataset))
    for (i, singledataset) in enumerate(dataset)
        augmenteddataset[i] = augmentsingleexperiment(singledataset, θ[:, i], nucl_f,
                                                      growth_f, aggregation_f, breakage_f,
                                                      solver = solver,
                                                      pseudo_psvariance = pseudo_psvariance)
    end

    return augmenteddataset

end

"""
    augmentdataset(dataset::Vector{CriSTool.CrystallisationRepeatMeasurements}, θ::AbstractVector{Float64},
                  nucl_f::NuF, growth_f::GrF, aggregation_f::AgF, breakage_f::BrF;
                  solver::AbstractSolver = MoM(), pseudo_psvariance::Real = 0.25) -> Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements} where {NuF<:AbstractFPNucleationFunction, GrF<:AbstractGrowthFunction, AgF<:AbstractAggregationFunction, BrF<:AbstractBreakageFunction}

Augment multiple experiments' measurements with simulated particle size data using a single parameter set.

# Arguments
- `dataset::Vector{CriSTool.CrystallisationRepeatMeasurements}`: Vector of measurement data
- `θ::AbstractVector{Float64}`: Parameter vector for kinetics (same for all experiments)
- `nucl_f::NuF`: Nucleation function
- `growth_f::GrF`: Growth function
- `aggregation_f::AgF`: Aggregation function
- `breakage_f::BrF`: Breakage function
- `solver::AbstractSolver=MoM()`: Numerical solver (default: Method of Moments)
- `pseudo_psvariance::Real=0.25`: Relative variance for particle size (default: 0.25)

# Returns
- `Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}`: Vector of augmented measurement data
"""
function augmentdataset(dataset::Vector{CriSTool.CrystallisationRepeatMeasurements},
                        θ::AbstractVector{Float64}, nucl_f::NuF, growth_f::GrF,
                        aggregation_f::AgF, breakage_f::BrF; solver::AbstractSolver = MoM(),
                        pseudo_psvariance::Real = 0.25) where {NuF <:
                                                               AbstractFPNucleationFunction,
                                                               GrF <:
                                                               AbstractGrowthFunction,
                                                               AgF <:
                                                               AbstractAggregationFunction,
                                                               BrF <:
                                                               AbstractBreakageFunction}

    augmenteddataset = Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}(undef,
                                                                                   length(dataset))
    for (i, singledataset) in enumerate(dataset)
        augmenteddataset[i] = augmentsingleexperiment(singledataset, θ, nucl_f, growth_f,
                                                      aggregation_f, breakage_f,
                                                      solver = solver,
                                                      pseudo_psvariance = pseudo_psvariance)
    end

    return augmenteddataset

end

"""
    augmentdataset(dataset::Vector{CriSTool.CrystallisationRepeatMeasurements}, θ::AbstractVector{Float64},
                  nucl_f::NuF, growth_f::GrF, aggregation_f::AgF, breakage_f::BrF, insilico_conc::Bool;
                  solver::AbstractSolver=MoM(), pseudo_psvariance::Real=0.25) -> Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}

Augment multiple experiments' measurements, optionally replacing concentration with simulated data.

# Arguments
- `dataset::Vector{CriSTool.CrystallisationRepeatMeasurements}`: Vector of measurement data
- `θ::AbstractVector{Float64}`: Parameter vector for kinetics (same for all experiments)
- `nucl_f::NuF`: Nucleation function
- `growth_f::GrF`: Growth function
- `aggregation_f::AgF`: Aggregation function
- `breakage_f::BrF`: Breakage function
- `insilico_conc::Bool`: If true, use simulated concentration; if false, use measured
- `solver::AbstractSolver=MoM()`: Numerical solver
- `pseudo_psvariance::Real=0.25`: Relative variance for particle size

# Returns
- `Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}`: Vector of augmented measurement data
"""
function augmentdataset(dataset::Vector{CriSTool.CrystallisationRepeatMeasurements},
                        θ::AbstractVector{Float64}, nucl_f::NuF, growth_f::GrF,
                        aggregation_f::AgF, breakage_f::BrF, insilico_conc::Bool;
                        solver::AbstractSolver = MoM(),
                        pseudo_psvariance::Real = 0.25,) where {NuF <:
                                                                AbstractFPNucleationFunction,
                                                                GrF <:
                                                                AbstractGrowthFunction,
                                                                AgF <:
                                                                AbstractAggregationFunction,
                                                                BrF <:
                                                                AbstractBreakageFunction}

    augmenteddataset = Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}(undef,
                                                                                   length(dataset))
    for (i, singledataset) in enumerate(dataset)
        augmenteddataset[i] = augmentsingleexperiment(singledataset, θ, nucl_f, growth_f,
                                                      aggregation_f, breakage_f,
                                                      insilico_conc, solver = solver,
                                                      pseudo_psvariance = pseudo_psvariance)
    end

    return augmenteddataset

end

"""
    truncate_dataset(dataset::Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}, t_truncation::Real) -> Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}

Truncate augmented measurement data to a specified time point.

# Arguments
- `dataset::Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}`: Vector of augmented measurement data
- `t_truncation::Real`: Time point at which to truncate the data

# Returns
- `Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}`: Vector of truncated measurement data

All measurements after t_truncation are removed from the dataset.
"""
function truncate_dataset(dataset::Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements},
                          t_truncation::Real)

    newdataset = Vector{CriSTool.CrystallisationAugmentedRepeatMeasurements}(undef,
                                                                             length(dataset))

    for (i, singledataset) in enumerate(dataset)
        newtime = singledataset.time[singledataset.time .<= t_truncation]
        newconcentrationmean = singledataset.concentrationmean[singledataset.time .<= t_truncation]
        newconcentrationvariance = singledataset.concentrationvariance[singledataset.time .<= t_truncation]
        newtime_ps = singledataset.time_ps[singledataset.time .<= t_truncation]
        newps_mean = singledataset.ps_mean[singledataset.time .<= t_truncation]
        newps_var = singledataset.ps_var[singledataset.time .<= t_truncation]
        newdataset[i] = CriSTool.CrystallisationAugmentedRepeatMeasurements(newtime,
                                                                            newconcentrationmean,
                                                                            newconcentrationvariance,
                                                                            newtime_ps,
                                                                            newps_mean,
                                                                            newps_var)
    end

    return newdataset

end



# ## Type piracy? Extend the getproperty function for the CrystallisationRepeatMeasurements type
# function Base.getproperty(measurement::Vector{CrystallisationRepeatMeasurements}, f::Symbol)
#     return [getproperty(singlemeasurement, f) for singlemeasurement in measurement]
# end
