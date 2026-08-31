@testset "Plotting Thesis Table Formatting" begin
    params = [38.0, 0.7, 1.0, 3.0]
    save_idx = collect(0.0:120.0:480.0)
    temperature = 293.15

    _,
    solution = runsimulation(params;
                             nucl = nucl_CNT(),
                             gr = growth_empirical(),
                             agg = noaggregation(),
                             br = nobreakage(),
                             solver = MoM(),
                             initial_concentration = 18.0,
                             save_idx = save_idx,
                             temp_profile = CriSTool.ConstantTemperature(temperature))

    measurement = CrystallisationExperiment(;
                                                    observables = (;
                                                    concentration = Observable(; time = save_idx, mean = solution.concentration,
                                                    variance = fill(0.01, length(save_idx))),
                                                    d43 = Observable(; time = save_idx, mean = solution.d43,
                                                                      variance = fill(4.0, length(save_idx)))),
                                                    temperature = temperature,
                                                    exp_id = 42)

    ensemble_samples = repeat(reshape(params, :, 1), 1, 2)
    ensemble = run_ensemble_fixed(ensemble_samples, nucl_CNT(), growth_empirical(),
                                  noaggregation(), nobreakage(), MoM();
                                  time_idx = save_idx,
                                  temp_profile = CriSTool.ConstantTemperature(temperature),
                                  initial_concentration = 18.0,
                                  verbosity = 0, HPC = true)
    optimal_solutions = [(nothing, solution)]

    mktempdir() do dir
        concentration_figure = CriSTool.plot_measurements_vs_ensemble(
            [measurement], [ensemble], optimal_solutions;
            savename = "ensemble_concentration", savedir = dir,
            showplot = false, showtext = false)
        particle_size_figure = CriSTool.plot_ps_measurements_vs_ensemble(
            [measurement], [ensemble], optimal_solutions;
            savename = "ensemble_particle_size", savedir = dir,
            showplot = false, showtext = false)
        @test concentration_figure !== nothing
        @test particle_size_figure !== nothing
        @test isfile(joinpath(dir, "ensemble_concentration.png"))
        @test isfile(joinpath(dir, "ensemble_particle_size.png"))
    end

    thesis_default = CriSTool.build_simulation_thesis_table_data([measurement], [solution])
    @test thesis_default.size_label == "d43"
    @test thesis_default.rows[1][1] == "42"
    @test thesis_default.rows[1][2] == string(round(temperature - 273, digits = 2))
    @test thesis_default.rows[1][3] ==
          string(round(CriSTool.get_characteristic_size(solution), sigdigits = 3))
    @test occursin("±", thesis_default.rows[1][4])

    thesis_deterministic = CriSTool.build_simulation_thesis_table_data([measurement],
                                                                       [solution];
                                                                       show_measurement_uncertainty = false)
    @test thesis_deterministic.size_label == "d43"
    @test thesis_deterministic.rows[1][4] ==
          string(round(measurement.observables.d43.mean[end], sigdigits = 3))
    @test !occursin("±", thesis_deterministic.rows[1][4])

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

    _,
    solution = runsimulation(params;
                             nucl = nucl_CNT(),
                             gr = growth_empirical(),
                             agg = noaggregation(),
                             br = nobreakage(),
                             solver = MoM(),
                             initial_concentration = 18.0,
                             save_idx = save_idx,
                             temp_profile = CriSTool.ConstantTemperature(temperature))

    measurement = CrystallisationExperiment(;
                                                    observables = (;
                                                    concentration = Observable(; time = save_idx, mean = solution.concentration,
                                                    variance = fill(0.01, length(save_idx))),
                                                    d43 = Observable(; time = save_idx, mean = solution.d43,
                                                                      variance = fill(4.0, length(save_idx)))),
                                                    temperature = temperature,
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
        prior = Distributions.product_distribution([TriangularDist(30.0, 45.0, params[1]),
                                                    TriangularDist(0.2, 1.2, params[2]),
                                                    TriangularDist(0.5, 2.0, params[3]),
                                                    TriangularDist(2.0, 4.0, params[4])])
        res, _ = CriSTool.ABCDE_Turner_Routine(
            CriSTool.logMLE(weighting = [1.0, 1.0]),
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
            outputdir = dir,
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
