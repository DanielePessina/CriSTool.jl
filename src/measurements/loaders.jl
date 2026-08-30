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

function _initial_crystals_from_table(table, initial_crystals_cols)
    initial_crystals_cols === nothing && return nothing
    required = (:mass_concentration, :d43, :distribution, :spread)
    all(name -> hasproperty(initial_crystals_cols, name), required) ||
        throw(ArgumentError("initial_crystals_cols must map mass_concentration, d43, " *
                            "distribution, and spread columns."))

    values = map(required) do name
        _first_measurement_value(table,
                                 getproperty(initial_crystals_cols, name),
                                 missing)
    end
    all(value -> value === missing, values) && return nothing
    any(value -> value === missing, values) &&
        throw(ArgumentError("Initial crystal characteristic columns must be populated " *
                            "for every experiment or omitted entirely."))

    return (; mass_concentration = Float64(values[1]),
            d43 = Float64(values[2]),
            distribution = values[3] isa Symbol ? values[3] : Symbol(lowercase(String(values[3]))),
            spread = Float64(values[4]))
end

"""
    load_measurements(filepath, sheet_name; time_col=:Time, id_col=:Exp_ID,
                      observables=(concentration = (:Concentration,
                      :Concentration_var),), scalar_observables=(),
                      metadata_cols=(temperature=:Temperature,),
                      initial_crystals_cols=nothing, temperature_transform=identity,
                      filters=NamedTuple())
        -> Vector{CrystallisationExperiment}

Load a table-driven measurement sheet. Each `observables` entry maps an
observable name to either a mean column or `(mean_column, variance_column)`.
Names listed in `scalar_observables` use the final available row of each
experiment; all others retain their own time series. `metadata_cols` maps
typed metadata names to source columns and is retained on each experiment.
The fixed `temperature` and `exp_id` fields are populated from metadata
columns when `temperature` is supplied. `initial_crystals_cols` optionally
maps initial seed characteristics into the experiment's `initial_crystals`
field.
"""
function load_measurements(filepath::AbstractString, sheet_name::AbstractString;
                            time_col = :Time,
                            id_col = :Exp_ID,
                            observables::NamedTuple =
                                (; concentration = (:Concentration, :Concentration_var)),
                            scalar_observables = (),
                            metadata_cols::NamedTuple =
                                (; temperature = :Temperature),
                            initial_crystals_cols = nothing,
                            temperature_transform = identity,
                            filters::NamedTuple = NamedTuple())
    hasproperty(metadata_cols, :loading) &&
        throw(ArgumentError("The loading metadata field was removed; use initial_crystals_cols."))
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

    if initial_crystals_cols !== nothing
        required = (:mass_concentration, :d43, :distribution, :spread)
        all(name -> hasproperty(initial_crystals_cols, name), required) ||
            throw(ArgumentError("initial_crystals_cols must map mass_concentration, d43, " *
                                "distribution, and spread columns."))
        for name in required
            source = getproperty(initial_crystals_cols, name)
            string(_measurement_source_column(source)) in names(table) ||
                throw(ArgumentError("Expected initial crystal column '$source' for :$name."))
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
        raw_temperature = _first_measurement_value(group, temperature_source, NaN)
        temperature_value = raw_temperature === missing ? NaN :
                            Float64(temperature_transform(Float64(raw_temperature)))
        experiment_id = Int(_first_measurement_value(group, id_source, index))
        initial_crystals = _initial_crystals_from_table(group, initial_crystals_cols)

        experiments[index] = CrystallisationExperiment(;
            observables = named_observables,
            temperature = temperature_value,
            initial_crystals = initial_crystals,
            exp_id = experiment_id,
            metadata = metadata)
    end
    return experiments
end

"""
    load_experiments(filepath::AbstractString, sheet_name::AbstractString;
                     temperature_range::Tuple=(nothing, nothing),
                     filters::NamedTuple=NamedTuple(),
                     initial_crystals_cols=nothing)
        -> Vector{CrystallisationExperiment}

Load experiments from a single Excel sheet formatted like the Python importer
(`data_import.py`), applying optional generic row filters and grouping by
`Exp_ID`.

# Arguments
- `filepath::AbstractString`: Path to the Excel file containing measurements
- `sheet_name::AbstractString`: Name of the sheet to load (e.g., a system name)
- `temperature_range::Tuple=(nothing, nothing)`: Optional `(Tmin, Tmax)` bounds to
  filter rows by temperature; pass `nothing` for no bound on that side.
- `filters::NamedTuple`: Optional source-column filters applied before grouping.
- `initial_crystals_cols`: Optional mapping of initial seed characteristic
  names to source columns.

# Returns
- `Vector{CrystallisationExperiment}`: One experiment per unique `Exp_ID`

The sheet is expected to include the following columns:
- `Exp_ID` (Int), `System` (String), `Temperature` (Real), `Time` (Real),
  `Concentration` (Real), optional `Concentration_var` (Real),
  optional `PS` (Real), optional `PS_var` (Real).

Notes:
- The `concentration` observable is a `Observable(time, mean, variance)`.
- Particle size (PS) is taken at the last timepoint only and stored as both a
  `d43` and a `d50q` `ScalarObservable` (the legacy loader stored the same
  value in both slots). If `PS` at the last timepoint is `-1` or `missing`, a
  dummy value of `10.0` is used and variance is set to `100.0`, matching the
  legacy behaviour.
"""
function load_experiments(filepath::AbstractString, sheet_name::AbstractString;
                          temperature_range::Tuple = (nothing, nothing),
                          filters::NamedTuple = NamedTuple(),
                          initial_crystals_cols = nothing)

    # Load the sheet as a DataFrame
    df = DataFrame(XLSX.readtable(filepath, sheet_name))

    # Normalize expected numeric columns to Float64 where present
    for col in
        (:Time, :Concentration, :Concentration_var, :Temperature, :PS, :PS_var)
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
    for (column, expected) in pairs(filters)
        source = _measurement_source_column(column)
        string(source) in names(df) ||
            throw(ArgumentError("Filter column '$source' not found in sheet '$(sheet_name)'."))
        df = df[df[!, source] .== expected, :]
    end

    # Apply optional temperature bounds.
    df_filtered = df

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
        @info "No experiments found in sheet '$(sheet_name)' of $(filepath) after applying filters."
        return CrystallisationExperiment[]
    end

    if initial_crystals_cols !== nothing
        required = (:mass_concentration, :d43, :distribution, :spread)
        all(name -> hasproperty(initial_crystals_cols, name), required) ||
            throw(ArgumentError("initial_crystals_cols must map mass_concentration, d43, " *
                                "distribution, and spread columns."))
        for name in required
            source = getproperty(initial_crystals_cols, name)
            string(_measurement_source_column(source)) in names(df_filtered) ||
                throw(ArgumentError("Expected initial crystal column '$source' for :$name."))
        end
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
        initial_crystals = _initial_crystals_from_table(sdf, initial_crystals_cols)

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
            initial_crystals = initial_crystals,
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
legacy layout stored d10/d32/d43 in successive rows). Temperature is unknown
in this format and set to `NaN`.
"""
