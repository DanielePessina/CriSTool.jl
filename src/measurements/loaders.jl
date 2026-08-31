"""
Utilities for loading and representing experimental measurement data for
crystallisation runs. Experiments are stored in the typed
`CrystallisationExperiment` container: a `NamedTuple` of per-observable
`Observable` entries plus run conditions.

The single table-driven entry point is `experiments_from_table`, which turns
any tabular source (a `DataFrame`, a `CSV.File`, ...) into experiments.
`load_measurements` and `load_experiments` are thin wrappers that read a CSV
file and apply a conventional layout.
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

    # Values are kept raw: interpretation and validation are deferred to
    # `initial_state_from_characteristics` (see `_normalise_initial_crystals`).
    return NamedTuple{required}(Tuple(values))
end

"""
    experiments_from_table(table; time_col=:Time, id_col=:Exp_ID,
                           observables=(concentration = (:Concentration,
                           :Concentration_var),),
                           metadata_cols=(temperature=:Temperature,),
                           temperature_transform=identity,
                           filters=NamedTuple(), initial_crystals_cols=nothing)
        -> Vector{CrystallisationExperiment}

Build experiments from any tabular source (a `DataFrame`, a `CSV.File`, ...).
Each `observables` entry maps an observable name to either a mean column or
`(mean_column, variance_column)`. Every observable retains all usable rows on
its own time grid. `metadata_cols` maps typed metadata names to source columns
and is retained on each experiment. The fixed `temperature` and `exp_id` fields
are populated from the `temperature` metadata column and `id_col` respectively.
`initial_crystals_cols` optionally maps the conventional initial seed
characteristics (`mass_concentration`, `d43`, `distribution`, `spread`) to
source columns; values are stored raw and validated only when the solver state
is constructed. Rows are sorted by `(id_col, time_col)` before grouping.
"""
function experiments_from_table(table;
                                time_col = :Time,
                                id_col = :Exp_ID,
                                observables::NamedTuple =
                                    (; concentration = (:Concentration,
                                                       :Concentration_var)),
                                metadata_cols::NamedTuple =
                                    (; temperature = :Temperature),
                                temperature_transform = identity,
                                filters::NamedTuple = NamedTuple(),
                                initial_crystals_cols = nothing)
    table = DataFrame(table)
    id_source = _measurement_source_column(id_col)
    time_source = _measurement_source_column(time_col)
    for source in (id_source, time_source)
        string(source) in names(table) ||
            throw(ArgumentError("Expected column '$source' not found in the measurement table."))
    end

    for (column, expected) in pairs(filters)
        source = _measurement_source_column(column)
        string(source) in names(table) ||
            throw(ArgumentError("Filter column '$source' not found in the measurement table."))
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

    sort!(table, [id_source, time_source])
    groups = groupby(table, id_source, sort = true)
    experiments = Vector{CrystallisationExperiment}(undef, length(groups))
    for (index, group) in enumerate(groups)
        observable_values = map(observable_names) do observable_name
            mean_col, variance_col = _measurement_column_spec(getproperty(observables,
                                                                            observable_name))
            _measurement_series(group, time_source, mean_col, variance_col)
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
    load_measurements(filepath; time_col=:Time, id_col=:Exp_ID,
                      observables=(concentration = (:Concentration,
                      :Concentration_var),),
                      metadata_cols=(temperature=:Temperature,),
                      temperature_transform=identity,
                      filters=NamedTuple(), initial_crystals_cols=nothing)
        -> Vector{CrystallisationExperiment}

Read a CSV file with `CSV.read` and build experiments via
`experiments_from_table`; see that function for the keyword semantics.
"""
function load_measurements(filepath::AbstractString; kwargs...)
    return experiments_from_table(CSV.read(filepath, DataFrame); kwargs...)
end

"""
    load_experiments(filepath; temperature_range=(nothing, nothing),
                     filters=NamedTuple(), initial_crystals_cols=nothing)
        -> Vector{CrystallisationExperiment}

Load experiments from a CSV file in the long-format convention (one row per
experiment, timepoint, and observable sample):

| Column | Meaning |
| --- | --- |
| `Exp_ID` | experiment identifier used for grouping |
| `Time` | observation time |
| `Concentration` | measured concentration |
| `Concentration_var` | concentration variance, when replicate measurements exist |
| `PS`, `PS_var` | optional particle-size measurement and variance at each sampled time |
| `Temperature` | experiment temperature in degrees Celsius |
| `System` | optional system label, retained in `metadata` |

Each unique `Exp_ID` becomes one `CrystallisationExperiment`. The
`concentration` observable is a time series. When a `PS` column is present,
every usable `PS` row is stored as a `d43` time series; rows without a
particle-size value are ignored. `Temperature` is converted from Celsius to
Kelvin. `temperature_range` optionally bounds the rows by temperature before
grouping; `filters` applies arbitrary source-column equality filters.
"""
function load_experiments(filepath::AbstractString;
                          temperature_range::Tuple = (nothing, nothing),
                          filters::NamedTuple = NamedTuple(),
                          initial_crystals_cols = nothing)
    table = CSV.read(filepath, DataFrame)

    if "Temperature" in names(table)
        tmin, tmax = temperature_range
        if !(tmin === nothing && tmax === nothing)
            lower = tmin === nothing ? -Inf : Float64(tmin)
            upper = tmax === nothing ? Inf : Float64(tmax)
            table = table[(table.Temperature .>= lower) .& (table.Temperature .<= upper), :]
        end
    end
    isempty(table) && return CrystallisationExperiment[]

    has_ps = "PS" in names(table)
    ps_spec = "PS_var" in names(table) ? (:PS, :PS_var) : :PS
    observables = has_ps ?
                  (; concentration = (:Concentration, :Concentration_var),
                     d43 = ps_spec) :
                  (; concentration = (:Concentration, :Concentration_var))

    metadata_cols = NamedTuple()
    "Temperature" in names(table) && (metadata_cols = merge(metadata_cols,
                                                            (; temperature = :Temperature)))
    "System" in names(table) && (metadata_cols = merge(metadata_cols,
                                                       (; system = :System)))

    return experiments_from_table(table;
                                  observables = observables,
                                  metadata_cols = metadata_cols,
                                  temperature_transform = value -> round(value + 273.15, digits = 2),
                                  filters = filters,
                                  initial_crystals_cols = initial_crystals_cols)
end
