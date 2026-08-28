@testset "Plotting Thesis Table Formatting" begin
    params = [38.0, 0.7, 1.0, 3.0]
    save_idx = collect(0.0:120.0:480.0)
    temperature = 293.15
    loading = 12.5

    _,
    solution = runsimulation(params;
                             nucl = nucl_CNT(),
                             gr = growth_empirical(),
                             agg = noaggregation(),
                             br = nobreakage(),
                             solver = MoM(),
                             initial_concentration = 18.0,
                             save_idx = save_idx,
                             temp_profile = CriSTool.ConstantTemperature(temperature),
                             loading = loading)

    measurement = CrystallisationExperiment(;
                                                    observables = (;
                                                    concentration = SeriesObservable(; time = save_idx, mean = solution.concentration,
                                                    variance = fill(0.01, length(save_idx))),
                                                    d43 = ScalarObservable(; value = solution.d43[end], variance = 4.0),
                                                    d50q = ScalarObservable(; value = solution.d43[end], variance = 4.0)),
                                                    temperature = temperature,
                                                    loading = loading,
                                                    exp_id = 42)

    thesis_default = CriSTool.build_simulation_thesis_table_data([measurement], [solution])
    @test thesis_default.size_label == "d43"
    @test thesis_default.rows[1][1] == "42"
    @test thesis_default.rows[1][2] == string(round(loading, sigdigits = 3))
    @test thesis_default.rows[1][4] ==
          string(round(CriSTool.get_characteristic_size(solution), sigdigits = 3))
    @test occursin("±", thesis_default.rows[1][5])

    thesis_deterministic = CriSTool.build_simulation_thesis_table_data([measurement],
                                                                       [solution];
                                                                       show_measurement_uncertainty = false)
    @test thesis_deterministic.size_label == "d43"
    @test thesis_deterministic.rows[1][5] == string(round(measurement.observables.d43.value, sigdigits = 3))
    @test !occursin("±", thesis_deterministic.rows[1][5])

    param_symbols = vcat(nucl_CNT().symbols, growth_empirical().symbols)
    param_table = CriSTool.build_parameter_value_table(params, param_symbols)
    @test occursin("Parameter", param_table)
    @test occursin(string(param_symbols[1]), param_table)
    @test !occursin("±", param_table)
end

@testset "Plotting Savedir Overrides" begin
    using Random

    params = [38.0, 0.7, 1.0, 3.0]
    save_idx = collect(0.0:120.0:480.0)
    temperature = 293.15
    loading = 12.5

    _,
    solution = runsimulation(params;
                             nucl = nucl_CNT(),
                             gr = growth_empirical(),
                             agg = noaggregation(),
                             br = nobreakage(),
                             solver = MoM(),
                             initial_concentration = 18.0,
                             save_idx = save_idx,
                             temp_profile = CriSTool.ConstantTemperature(temperature),
                             loading = loading)

    measurement = CrystallisationExperiment(;
                                                    observables = (;
                                                    concentration = SeriesObservable(; time = save_idx, mean = solution.concentration,
                                                    variance = fill(0.01, length(save_idx))),
                                                    d43 = ScalarObservable(; value = solution.d43[end], variance = 4.0),
                                                    d50q = ScalarObservable(; value = solution.d43[end], variance = 4.0)),
                                                    temperature = temperature,
                                                    loading = loading,
                                                    exp_id = 42)

    mktempdir() do dir
        fig = CriSTool.plot_measurements_vs_simulation([measurement],
                                                       params,
                                                       nucl_CNT(),
                                                       growth_empirical(),
                                                       noaggregation(),
                                                       nobreakage(),
                                                       MoM();
                                                       savename = "sim_plot",
                                                       savedir = dir,
                                                       showplot = false,
                                                       showtext = false)
        @test fig !== nothing
        @test isfile(joinpath(dir, "sim_plot.png"))
    end

    mktempdir() do dir
        samples = rand(length(params), 64)
        fig = CriSTool.ChainPairPlots(samples,
                                      params;
                                      savestring = "pairplot",
                                      savedir = dir,
                                      showplot = false)
        @test fig !== nothing
        @test isfile(joinpath(dir, "pairplot Pairplot.png"))
    end

    mktempdir() do dir
        Random.seed!(1234)
        prior = CriSTool.Factored(TriangularDist(30.0, 45.0, params[1]),
                                  TriangularDist(0.2, 1.2, params[2]),
                                  TriangularDist(0.5, 2.0, params[3]),
                                  TriangularDist(2.0, 4.0, params[4]))
        res, _ = CriSTool.ABCDE_Turner_Routine(
            CriSTool.logMLE(weighting = (1.0, 1.0)),
            [measurement],
            params,
            prior,
            nucl_CNT(),
            growth_empirical(),
            noaggregation(),
            nobreakage();
            solver = MoM(),
            nparticles = 16,
            generations = 2,
            saveplot = false,
            confidenceinterval = 0.90,
            HPC = true,
            verbosity = 0,
            earlystop = false,
            test = :wilks,
            K = 1,
            savedir = dir,
        )
        @test hasproperty(res, :P)
        @test hasproperty(res, :C)
        objects_dir = joinpath(dir, "ABCDE Objects")
        @test isdir(objects_dir)
        object_files = readdir(objects_dir)
        @test !isempty(object_files)
        @test any(endswith(".jld2"), object_files)
    end
end
