using Test
using CriSTool
using Distributions
using XLSX

@testset "Initial crystal state construction" begin
    lognormal_initial_crystals = (; mass_concentration = 0.25,
                                  d43 = 12.0,
                                  distribution = :lognormal,
                                  spread = 1.25)
    gaussian_initial_crystals = (; mass_concentration = 0.25,
                                 d43 = 12.0,
                                 distribution = :gaussian,
                                 spread = 2.0)

    function assert_moment_state(problem, initial_crystals)
        state = initial_state_from_characteristics(problem, initial_crystals)
        moment_values = state[1:moment_count(problem.solver)]
        @test problem.ρ * problem.kv * moment_values[4] ≈
              initial_crystals.mass_concentration rtol = 1e-12
        @test 1e6 * moment_values[5] / moment_values[4] ≈ initial_crystals.d43 rtol = 1e-12
        @test state[end] == problem.initial_concentration
        return state
    end

    @testset "MoM and QMOM moments" begin
        for solver in (MoM(), QMOM(nquadrature = 3))
            problem = CrystallisationProblem(; solver = solver,
                                               initial_concentration = 18.0)
            assert_moment_state(problem, lognormal_initial_crystals)
            assert_moment_state(problem, gaussian_initial_crystals)
        end
    end

    @testset "Finite-volume mesh profiles" begin
        for solver in (FiniteVol(meshsize = 120, lmax = 80e-6),
                       WENO(meshsize = 120, lmax = 80e-6))
            problem = CrystallisationProblem(; solver = solver,
                                               initial_concentration = 18.0)
            state = initial_state_from_characteristics(problem, gaussian_initial_crystals)
            numberdensity = state[1:solver.meshsize]
            mu3 = solver.cell_dL * sum(numberdensity[i] * solver.cell_centre[i]^3
                                       for i in eachindex(numberdensity))
            mu4 = solver.cell_dL * sum(numberdensity[i] * solver.cell_centre[i]^4
                                       for i in eachindex(numberdensity))
            @test all(>=(0.0), numberdensity)
            @test problem.ρ * problem.kv * mu3 ≈ gaussian_initial_crystals.mass_concentration rtol = 1e-12
            @test 1e6 * mu4 / mu3 ≈ gaussian_initial_crystals.d43 rtol = 0.05
        end
    end

    @testset "Empty and invalid populations" begin
        problem = CrystallisationProblem(; solver = MoM(), initial_concentration = 18.0)
        empty_state = initial_state_from_characteristics(
            problem,
            (; mass_concentration = 0.0, d43 = 0.0,
               distribution = :lognormal, spread = 1.25))
        @test all(==(0.0), empty_state[1:moment_count(problem.solver)])
        @test empty_state[end] == 18.0

        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            (; mass_concentration = 1.0, d43 = 0.0,
               distribution = :lognormal, spread = 1.25))
        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            (; mass_concentration = 0.25, d43 = 12.0,
               distribution = :uniform, spread = 1.25))
        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            (; mass_concentration = 0.25, d43 = 12.0,
               distribution = :lognormal, spread = 1.0))
        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            (; mass_concentration = 0.25, d43 = 12.0,
               distribution = :gaussian, spread = 0.0))
    end

    @testset "Runner and experiment integration" begin
        params = [38.0, 0.7, 1.0, 3.0]
        problem, solution = runsimulation(params;
                                           nucl = nucl_CNT(),
                                           gr = growth_empirical(),
                                           agg = noaggregation(),
                                           br = nobreakage(),
                                           solver = MoM(),
                                           initial_concentration = 18.0,
                                           initial_crystals = lognormal_initial_crystals,
                                           save_idx = [0.0, 0.1])
        @test problem.initial_state !== nothing
        @test 1e6 * problem.initial_state[5] / problem.initial_state[4] ≈
              lognormal_initial_crystals.d43 rtol = 1e-12
        @test solution.d43[1] ≈ lognormal_initial_crystals.d43 rtol = 0.01
        @test_throws ArgumentError runsimulation(params;
                                                 nucl = nucl_CNT(),
                                                 gr = growth_empirical(),
                                                 agg = noaggregation(),
                                                 br = nobreakage(),
                                                 solver = MoM(),
                                                 initial_concentration = 18.0,
                                                 initial_state = problem.initial_state,
                                                 initial_crystals = lognormal_initial_crystals,
                                                 save_idx = [0.0, 0.1])

        experiment = CrystallisationExperiment(;
            observables = (;
                concentration = Observable(; time = [0.0, 0.1],
                                           mean = [18.0, solution.concentration[end]],
                                           variance = [1.0, 1.0]),
                d43 = Observable(; time = 0.1, mean = solution.d43[end], variance = 1.0),
                d50q = Observable(; time = 0.1, mean = solution.d43[end], variance = 1.0)),
            temperature = 293.15,
            initial_crystals = lognormal_initial_crystals,
            exp_id = 1)
        prepared_problem = CriSTool._experiment_problem(problem, experiment)
        @test prepared_problem.initial_state !== nothing
        @test prepared_problem.initial_state[4] ≈ problem.initial_state[4] rtol = 1e-12
        @test !hasproperty(problem, :loading)
        @test !hasproperty(experiment, :loading)

        ensemble = run_ensemble_fixed(reshape(params, :, 1),
                                      nucl_CNT(), growth_empirical(),
                                      noaggregation(), nobreakage(), MoM();
                                      time_idx = [0.0, 0.1],
                                      temp_profile = CriSTool.ConstantTemperature(293.15),
                                      initial_concentration = 18.0,
                                      initial_crystals = lognormal_initial_crystals,
                                      verbosity = 0,
                                      HPC = true)
        @test ensemble.d43[1, 1] ≈ lognormal_initial_crystals.d43 rtol = 0.01
    end

    @testset "Table loader initial-crystal mapping" begin
        mktempdir() do directory
            filepath = joinpath(directory, "seeded.xlsx")
            columns = [:Exp_ID, :Time, :Concentration, :Temperature,
                       :SeedMass, :SeedD43, :SeedDistribution, :SeedSpread]
            rows = [
                (1, 0.0, 18.0, 20.0, 0.25, 12.0, "gaussian", 2.0),
                (1, 30.0, 16.0, 20.0, 0.25, 12.0, "gaussian", 2.0),
            ]
            XLSX.openxlsx(filepath, mode = "w") do workbook
                sheet = workbook[1]
                XLSX.rename!(sheet, "Seeded")
                for (column_index, column) in enumerate(columns)
                    sheet[1, column_index] = String(column)
                end
                for (row_index, row) in enumerate(rows)
                    for (column_index, value) in enumerate(row)
                        sheet[row_index + 1, column_index] = value
                    end
                end
            end

            experiments = load_measurements(
                filepath,
                "Seeded";
                observables = (; concentration = :Concentration),
                metadata_cols = (; temperature = :Temperature),
                initial_crystals_cols = (; mass_concentration = :SeedMass,
                                         d43 = :SeedD43,
                                         distribution = :SeedDistribution,
                                         spread = :SeedSpread),
                temperature_transform = value -> value + 273.15)
            @test length(experiments) == 1
            @test experiments[1].initial_crystals ==
                  (; mass_concentration = 0.25, d43 = 12.0,
                     distribution = :gaussian, spread = 2.0)
            @test experiments[1].temperature == 293.15
        end
    end
end
