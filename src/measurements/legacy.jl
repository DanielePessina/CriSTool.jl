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
