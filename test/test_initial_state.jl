using Test
using CriSTool
using Distributions
using DataFrames

@testset "Initial crystal state construction" begin
    lognormal_initial_crystals = LogNormalInitialCrystals(;
        mass_concentration = 0.25, d43 = 12e-6, geometric_std = 1.25)
    gaussian_initial_crystals = GaussianInitialCrystals(;
        mass_concentration = 0.25, d43 = 12e-6, standard_deviation = 2e-6)

    function assert_moment_state(problem, initial_crystals)
        state = initial_state_from_characteristics(problem, initial_crystals)
        moment_values = state[1:moment_count(problem.solver)]
        @test problem.crystal_density * problem.volume_shape_factor * moment_values[4] ≈
              initial_crystals.mass_concentration rtol = 1e-12
        @test moment_values[5] / moment_values[4] ≈ initial_crystals.d43 rtol = 1e-12
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
            @test problem.crystal_density * problem.volume_shape_factor * mu3 ≈ gaussian_initial_crystals.mass_concentration rtol = 1e-12
            @test mu4 / mu3 ≈ gaussian_initial_crystals.d43 rtol = 0.05
        end
    end

    @testset "Empty and invalid populations" begin
        problem = CrystallisationProblem(; solver = MoM(), initial_concentration = 18.0)
        empty_state = initial_state_from_characteristics(
            problem,
            LogNormalInitialCrystals(; mass_concentration = 0.0, d43 = 0.0,
                                     geometric_std = 1.25))
        @test all(==(0.0), empty_state[1:moment_count(problem.solver)])
        @test empty_state[end] == 18.0

        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            LogNormalInitialCrystals(; mass_concentration = 1.0, d43 = 0.0,
                                     geometric_std = 1.25))
        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            LogNormalInitialCrystals(; mass_concentration = 0.25, d43 = 12e-6,
                                     geometric_std = 1.0))
        @test_throws ArgumentError initial_state_from_characteristics(
            problem,
            GaussianInitialCrystals(; mass_concentration = 0.25, d43 = 12e-6,
                                    standard_deviation = 0.0))
    end

    @testset "Runner and experiment integration" begin
        params = [38.0, 0.0007, 1e-9 / 60, 3.0]
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
        @test problem.initial_state[5] / problem.initial_state[4] ≈
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
                d43 = Observable(; time = [0.1], mean = [solution.d43[end]],
                                 variance = [1e-12])),
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
        _, reference = runsimulation(params;
                                     nucl = nucl_CNT(), gr = growth_empirical(),
                                     agg = noaggregation(), br = nobreakage(),
                                     solver = MoM(), initial_concentration = 18.0,
                                     initial_crystals = lognormal_initial_crystals,
                                     temp_profile = CriSTool.ConstantTemperature(293.15),
                                     save_idx = [0.0, 0.1])
        @test ensemble.time == [0.0, 0.1]
        @test ensemble.concentration[1, :] ≈ reference.concentration
        @test ensemble.d43[1, :] ≈ reference.d43
        @test ensemble.d43_mean ≈ reference.d43
        @test ensemble.d43[1, 1] ≈ lognormal_initial_crystals.d43 rtol = 0.01

        prior = Distributions.product_distribution([
            Distributions.Dirac(value) for value in params
        ])
        measurement_ensemble = run_ensemble(
            prior, [experiment], nucl_CNT(), growth_empirical(),
            noaggregation(), nobreakage(), MoM(); n_samples = 1,
            time_idx = [0.0, 0.1], verbosity = 0, HPC = true)
        @test measurement_ensemble[1].time == [0.0, 0.1]
        @test measurement_ensemble[1].concentration[1, :] ≈ reference.concentration
        @test measurement_ensemble[1].d43[1, :] ≈ reference.d43
    end

    @testset "Table loader initial-crystal mapping" begin
        rows = DataFrame(;
            Exp_ID = [1, 1],
            Time = [0.0, 1800.0],
            Concentration = [18.0, 16.0],
            Temperature = [293.15, 293.15],
            SeedMass = [0.25, 0.25],
            SeedD43 = [12e-6, 12e-6],
            SeedStandardDeviation = [2e-6, 2e-6])

        experiments = experiments_from_table(
            rows;
            observables = (; concentration = ObservableColumns(mean = :Concentration)),
            metadata_cols = (; temperature = :Temperature),
            initial_crystals_cols = (; mass_concentration = :SeedMass,
                                     d43 = :SeedD43,
                                     standard_deviation = :SeedStandardDeviation))
        @test length(experiments) == 1
        @test experiments[1].temperature == 293.15
        initial_crystals = experiments[1].initial_crystals

        @test initial_crystals == GaussianInitialCrystals(;
            mass_concentration = 0.25, d43 = 12e-6, standard_deviation = 2e-6)
        problem = CrystallisationProblem(; solver = MoM(),
                                         initial_concentration = 18.0)
        state = initial_state_from_characteristics(problem, initial_crystals)
        moment_values = state[1:moment_count(problem.solver)]
        @test problem.crystal_density * problem.volume_shape_factor * moment_values[4] ≈ 0.25 rtol = 1e-12
        @test moment_values[5] / moment_values[4] ≈ 12e-6 rtol = 1e-12
    end
end
