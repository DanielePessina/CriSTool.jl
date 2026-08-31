"""
Measurement ingestion and normalization.

`experiments_from_table` is the semantic loader: it receives a table and an
explicit per-observable source schema, then returns typed
`CrystallisationExperiment` values. `load_measurements` is the thin file
adapter for CSV and JSON records.
"""

function _measurement_source_column(column)
    return column isa Symbol ? column : Symbol(column)
end

function _table_has_column(table, column)
    return string(_measurement_source_column(column)) in names(table)
end

function _require_measurement_column(table, column, description)
    _table_has_column(table, column) ||
        throw(ArgumentError("Expected column '$column' for $description."))
    return nothing
end

function _observable_columns_spec(spec)
    spec isa ObservableColumns ||
        throw(ArgumentError("Observable schemas must use ObservableColumns(time, mean, variance)."))
    return spec
end

function _consistent_measurement_value(table, column, default, description)
    column === nothing && return default
    values = [value for value in table[!, _measurement_source_column(column)]
              if value !== missing]
    isempty(values) && return default
    reference = first(values)
    all(value -> value == reference, values) ||
        throw(ArgumentError("Metadata column '$column' is inconsistent within $description."))
    return reference
end

function _measurement_series(table, columns::ObservableColumns, observable_name::Symbol)
    times = table[!, columns.time]
    means = table[!, columns.mean]
    indices = [index for index in eachindex(times, means)
               if times[index] !== missing && means[index] !== missing]
    isempty(indices) &&
        throw(ArgumentError("Observable :$observable_name has no usable measurements."))

    observed_times = Float64[Float64(times[index]) for index in indices]
    observed_means = Float64[Float64(means[index]) for index in indices]
    order = sortperm(observed_times)
    observed_times = observed_times[order]
    observed_means = observed_means[order]

    observed_variance = if columns.variance === nothing
        nothing
    else
        variances = table[!, columns.variance]
        sorted_indices = indices[order]
        Float64[variances[index] === missing ? 0.0 : Float64(variances[index])
                for index in sorted_indices]
    end

    return Observable(; time = observed_times, mean = observed_means,
                      variance = observed_variance)
end

function _initial_crystals_from_table(table, initial_crystals_cols)
    initial_crystals_cols === nothing && return nothing
    required = (:mass_concentration, :d43, :distribution, :spread)
    all(name -> hasproperty(initial_crystals_cols, name), required) ||
        throw(ArgumentError("initial_crystals_cols must map mass_concentration, d43, " *
                            "distribution, and spread columns."))

    values = map(required) do name
        column = getproperty(initial_crystals_cols, name)
        _consistent_measurement_value(table, column, missing, "initial crystal characteristics")
    end
    all(value -> value === missing, values) && return nothing
    any(value -> value === missing, values) &&
        throw(ArgumentError("Initial crystal characteristic columns must be populated " *
                            "for every experiment or omitted entirely."))

    return NamedTuple{required}(Tuple(values))
end

"""
    experiments_from_table(table; id_col=:Exp_ID, observables, metadata_cols,
                           temperature_transform=identity, filters=NamedTuple(),
                           initial_crystals_cols=nothing)
        -> Vector{CrystallisationExperiment}

Normalize a tabular measurement source into typed experiments. Each observable
is described by an `ObservableColumns` value, allowing a different time grid
for every observable. Rows with missing mean or time values are omitted from
that observable only. Source rows are grouped by `id_col`; each observable is
sorted independently by its declared time column.
"""
function experiments_from_table(table;
                                id_col = :Exp_ID,
                                observables::NamedTuple =
                                    (; concentration = ObservableColumns(
                                        mean = :Concentration)),
                                metadata_cols::NamedTuple =
                                    NamedTuple(),
                                temperature_transform = identity,
                                filters::NamedTuple = NamedTuple(),
                                initial_crystals_cols = nothing)
    table = DataFrame(table)
    id_source = _measurement_source_column(id_col)
    _require_measurement_column(table, id_source, "experiment identifiers")
    isempty(table) && return CrystallisationExperiment[]

    for (column, expected) in pairs(filters)
        source = _measurement_source_column(column)
        _require_measurement_column(table, source, "filter :$column")
        mask = [value !== missing && value == expected for value in table[!, source]]
        table = table[mask, :]
    end
    isempty(table) && return CrystallisationExperiment[]

    isempty(observables) &&
        throw(ArgumentError("At least one observable must be specified."))
    for observable_name in keys(observables)
        columns = _observable_columns_spec(getproperty(observables, observable_name))
        _require_measurement_column(table, columns.time, "observable :$observable_name time")
        _require_measurement_column(table, columns.mean, "observable :$observable_name mean")
        columns.variance === nothing ||
            _require_measurement_column(table, columns.variance,
                                        "observable :$observable_name variance")
    end

    if initial_crystals_cols !== nothing
        required = (:mass_concentration, :d43, :distribution, :spread)
        all(name -> hasproperty(initial_crystals_cols, name), required) ||
            throw(ArgumentError("initial_crystals_cols must map mass_concentration, d43, " *
                                "distribution, and spread columns."))
        for name in required
            _require_measurement_column(table, getproperty(initial_crystals_cols, name),
                                        "initial crystal :$name")
        end
    end

    metadata_names = keys(metadata_cols)
    for metadata_name in metadata_names
        _require_measurement_column(table, getproperty(metadata_cols, metadata_name),
                                    "metadata :$metadata_name")
    end

    groups = groupby(table, id_source, sort = true)
    experiments = Vector{CrystallisationExperiment}(undef, length(groups))
    for (experiment_index, group) in enumerate(groups)
        observable_values = map(keys(observables)) do observable_name
            columns = _observable_columns_spec(getproperty(observables, observable_name))
            _measurement_series(group, columns, observable_name)
        end
        named_observables = NamedTuple{keys(observables)}(Tuple(observable_values))

        metadata_values = map(metadata_names) do metadata_name
            column = getproperty(metadata_cols, metadata_name)
            _consistent_measurement_value(group, column, missing,
                                          "experiment $(experiment_index) metadata")
        end
        metadata = NamedTuple{metadata_names}(Tuple(metadata_values))

        temperature_source = hasproperty(metadata_cols, :temperature) ?
                             getproperty(metadata_cols, :temperature) : nothing
        raw_temperature = _consistent_measurement_value(group, temperature_source, NaN,
                                                        "experiment $(experiment_index) temperature")
        temperature_value = raw_temperature === missing ? NaN :
                            Float64(temperature_transform(Float64(raw_temperature)))
        experiment_id = Int(_consistent_measurement_value(group, id_source,
                                                          experiment_index,
                                                          "experiment identifier"))
        initial_crystals = _initial_crystals_from_table(group, initial_crystals_cols)

        experiments[experiment_index] = CrystallisationExperiment(;
            observables = named_observables,
            temperature = temperature_value,
            initial_crystals = initial_crystals,
            exp_id = experiment_id,
            metadata = metadata)
    end
    return experiments
end

function _json_records_table(filepath::AbstractString)
    payload = JSON.parsefile(filepath)
    payload isa AbstractVector ||
        throw(ArgumentError("JSON measurement files must contain an array of records."))
    records = NamedTuple[]
    for (record_index, record) in enumerate(payload)
        record isa AbstractDict ||
            throw(ArgumentError("JSON record $record_index must be an object."))
        record_keys = collect(keys(record))
        record_names = Tuple(Symbol(String(key)) for key in record_keys)
        record_values = Tuple(record[key] === nothing ? missing : record[key]
                              for key in record_keys)
        push!(records, NamedTuple{record_names}(record_values))
    end
    return DataFrame(records)
end

function _measurement_file_table(filepath::AbstractString, format::Symbol)
    extension = splitext(filepath)[2]
    normalized_format = if format === :auto
        isempty(extension) &&
            throw(ArgumentError("Cannot infer measurement format from '$filepath'. " *
                                "Pass format = :csv or :json."))
        Symbol(lowercase(extension[2:end]))
    else
        format
    end
    if normalized_format === :csv
        return DataFrame(CSV.read(filepath, DataFrame))
    elseif normalized_format === :json
        return _json_records_table(filepath)
    end
    throw(ArgumentError("Unsupported measurement format :$normalized_format. Use :csv or :json."))
end

function _default_measurement_observables(table)
    _require_measurement_column(table, :Concentration, "the default concentration observable")
    concentration_variance = _table_has_column(table, :Concentration_var) ?
                             :Concentration_var : nothing
    observables = (; concentration = ObservableColumns(
        mean = :Concentration, variance = concentration_variance))
    if _table_has_column(table, :PS)
        particle_variance = _table_has_column(table, :PS_var) ? :PS_var : nothing
        observables = merge(observables, (; d43 = ObservableColumns(
            mean = :PS, variance = particle_variance)))
    end
    return observables
end

function _default_measurement_metadata(table)
    metadata = NamedTuple()
    _table_has_column(table, :Temperature) &&
        (metadata = merge(metadata, (; temperature = :Temperature)))
    _table_has_column(table, :System) &&
        (metadata = merge(metadata, (; system = :System)))
    return metadata
end

"""
    load_measurements(filepath; format=:auto, observables=nothing, ...)
        -> Vector{CrystallisationExperiment}

Read a CSV or JSON array-of-records file and normalize it through
`experiments_from_table`. When `observables` is omitted, the conventional
columns `Concentration`, optional `Concentration_var`, optional `PS`, and
optional `PS_var` are discovered. Temperatures are interpreted as Celsius and
converted to Kelvin by default; pass `temperature_transform = identity` for
data already stored in Kelvin.
"""
function load_measurements(filepath::AbstractString;
                           format::Symbol = :auto,
                           id_col = :Exp_ID,
                           observables = nothing,
                           metadata_cols = nothing,
                           temperature_transform = value -> round(value + 273.15, digits = 2),
                           temperature_range::Tuple = (nothing, nothing),
                           filters::NamedTuple = NamedTuple(),
                           initial_crystals_cols = nothing)
    table = _measurement_file_table(filepath, format)
    if temperature_range != (nothing, nothing)
        _require_measurement_column(table, :Temperature, "temperature filtering")
        lower_bound, upper_bound = temperature_range
        lower = lower_bound === nothing ? -Inf : Float64(lower_bound)
        upper = upper_bound === nothing ? Inf : Float64(upper_bound)
        temperature_mask = [value !== missing && lower <= value <= upper
                            for value in table.Temperature]
        table = table[temperature_mask, :]
    end
    isempty(table) && return CrystallisationExperiment[]

    normalized_observables = observables === nothing ?
                              _default_measurement_observables(table) : observables
    normalized_metadata = metadata_cols === nothing ?
                          _default_measurement_metadata(table) : metadata_cols
    return experiments_from_table(table;
                                  id_col = id_col,
                                  observables = normalized_observables,
                                  metadata_cols = normalized_metadata,
                                  temperature_transform = temperature_transform,
                                  filters = filters,
                                  initial_crystals_cols = initial_crystals_cols)
end
