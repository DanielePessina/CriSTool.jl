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
                                           save_idx = [0.0, 6.0])
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
                                                 save_idx = [0.0, 6.0])

experiment = CrystallisationExperiment(;
            observables = (;
                concentration = Observable(; time = [0.0, 6.0],
                                           mean = [18.0, solution.concentration[end]],
                                           variance = [1.0, 1.0]),
                d43 = Observable(; time = [6.0], mean = [solution.d43[end]],
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
                                      time_idx = [0.0, 6.0],
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
                                     save_idx = [0.0, 6.0])
        @test ensemble.time == [0.0, 6.0]
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
            time_idx = [0.0, 6.0], verbosity = 0, HPC = true)
        @test measurement_ensemble[1].time == [0.0, 6.0]
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

@testset "Population and solvent layout regression" begin
    coupled_problem = CrystallisationProblem(; solver = MoM(nmoments = 3),
        initial_solvent_state = (concentration = 20.0, pH = 7.0))
    coupled_derivative = CriSTool._mom_rhs(coupled_problem, 2.0, 11.0,
        (-13.0, -17.0), StaticArrays.SVector(1.0, 2.0, 3.0, 4.0, 20.0, 7.0))
    @test coupled_derivative == [11.0, 2.0, 8.0, 18.0, -13.0, -17.0]

    # Same total length, different physical layout: five moments, one solvent.
    ordinary_problem = CrystallisationProblem(; solver = MoM())
    ordinary_derivative = CriSTool._mom_rhs(ordinary_problem, 2.0, 11.0,
        (-13.0,), StaticArrays.SVector(1.0, 2.0, 3.0, 4.0, 5.0, 20.0))
    @test ordinary_derivative == [11.0, 2.0, 8.0, 18.0, 32.0, -13.0]

    _, coupled_trajectory = runsimulation(Float64[];
        nucl = nucl_empirical_fixed([0.0, 1.0]),
        gr = growth_empirical_fixed([0.0, 1.0]),
        solver = MoM(nmoments = 3), initial_concentration = 20.0,
        saturation_model = ConstantSolubility(20.0),
        initial_solvent_state = (concentration = 20.0, pH = 7.0),
        solvent_dynamics = (problem, state, time, growth) ->
            (concentration = -13.0, pH = -17.0), save_idx = [0.0, 1.0])
    @test coupled_trajectory.concentration[end] ≈ 7.0 atol = 1e-10
    @test coupled_trajectory.solvent_state.pH[end] ≈ -10.0 atol = 1e-10
end

@testset "Experiment seed preservation and authority" begin
    seed_moments = [1e12, 1e6, 1.0, 1e-6, 1e-12]
    seeded_template = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed([0.0, 1.0]),
        kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]),
        parameterset_nucleation = Float64[], parameterset_growth = Float64[],
        initial_concentration = 10.0, saturation_model = ConstantSolubility(10.0),
        initial_solvent_state = (pH = 7.0, concentration = 10.0),
        initial_state = [seed_moments; 7.5; 12.0], solver = MoM())
    measured_concentration = Observable(; time = [0.0, 1.0], mean = [18.0, 18.0],
                                        variance = 1.0)
    seed_experiment = CrystallisationExperiment(;
        observables = (concentration = measured_concentration,
            d43 = Observable(; time = [1.0], mean = [1e-6], variance = 1e-12)),
        temperature = 300.0, exp_id = 1)
    prepared_seed = prepare_loss(seeded_template, [seed_experiment])
    @test collect(prepared_seed.prepared[1].odeproblem.u0) == [seed_moments; 7.5; 18.0]
    @test loss(mae(), prepared_seed, Float64[]) ≈ 0.0 atol = 1e-14
    @test seeded_template.initial_state == [seed_moments; 7.5; 12.0]
    @test initial_concentration(seeded_template) == 12.0
    @test initial_concentration(prepared_seed.prepared[1].problem) == 18.0
    @test CriSTool.temperature(prepared_seed.prepared[1].problem.temp_profile, 0.0) == 300.0

    override_experiment = CrystallisationExperiment(;
        observables = (concentration = measured_concentration,),
        temperature = 300.0, exp_id = 2,
        initial_crystals = LogNormalInitialCrystals(; mass_concentration = 0.0,
            d43 = 0.0, geometric_std = 1.25))
    overridden_seed = CriSTool._experiment_problem(seeded_template, override_experiment)
    @test overridden_seed.initial_state == [zeros(5); 7.5; 18.0]
    @test seeded_template.initial_state == [seed_moments; 7.5; 12.0]

    # Public DQMOM states contain weights and physical nodes, not raw moments.
    direct_template = CriSTool._copy_crystallisation_problem(seeded_template;
        solver = DQMOM(nquadrature = 2),
        initial_state = [4e12, 6e12, 1e-6, 2e-6, 7.0, 12.0])
    direct_experiment = CriSTool._experiment_problem(direct_template, seed_experiment)
    @test direct_experiment.initial_state == [4e12, 6e12, 1e-6, 2e-6, 7.0, 18.0]
    @test direct_template.initial_state == [4e12, 6e12, 1e-6, 2e-6, 7.0, 12.0]
end
