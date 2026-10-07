using Random
using Metaheuristics
import OptimizationOptimJL

@testset "Named model workflow" begin
    interface_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.0),
        crystal_density = 1000.0, volume_shape_factor = 1.0)
    interface_nucleation = KineticModel(CallableNucleation((rate_values, rate_context) -> 0.0;
        parameters = (;)))
    # Constant physical growth gives an independent translation oracle for the
    # seed moments. A second, unused coefficient exercises subset selection.
    interface_growth = KineticModel(CallableGrowth((rate_values, rate_context) -> rate_values.speed * 1e-6;
        parameters = (speed = 0.01, held_order = 2.0)))
    interface_model = CrystallisationModel(system = interface_system,
        nucleation = interface_nucleation, growth = interface_growth)
    interface_seed = DistributionInitialCrystals(Uniform(10e-6, 20e-6); mass_concentration = 0.25)
    interface_run = CrystallisationProblem(interface_model;
        initial_conditions = (concentration = 20.0,), temperature = ConstantTemperature(293.15),
        initial_crystals = interface_seed)
    @test interface_model.system.molecular_volume === nothing
    @test LysozymeSystem() isa CrystallisationSystem
    offset_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.),
        crystal_density = 1000., volume_shape_factor = 1., properties = (signed_offset = -1e-9,))
    offset_growth = KineticModel(CallableGrowth((rate_values, rate_context) -> rate_context.system.signed_offset;
        parameters = (;), required = (:signed_offset,)))
    offset_model = CrystallisationModel(system = offset_system, nucleation = interface_nucleation, growth = offset_growth)
    offset_run = CrystallisationProblem(offset_model;
        initial_conditions = (concentration = 20.,), temperature = ConstantTemperature(293.15))
    @test CriSTool.growthrate(offset_growth.law, offset_run.parameterset_growth,
        offset_run, CriSTool._get_initial_state(offset_run), 0.) == -1e-9
    @test_throws ArgumentError KineticModel(growth_empirical(); parameters = (wrong = 1.0, growth_order = 2.0))
    @test_throws ArgumentError CrystallisationProblem(CrystallisationModel(system = interface_system,
        nucleation = KineticModel(nucl_CNT(); parameters = (ln_nucleation_prefactor = 38., surface_energy = 0.0006)),
        growth = interface_growth); initial_conditions = (concentration = 20.,), temperature = ConstantTemperature(293.15))

    interface_trajectory = simulate(interface_run; saveat = [0., 100.])
    # Uniform lengths in [a,b]: E[L^k] = (b^(k+1)-a^(k+1))/((k+1)(b-a)).
    # Frozen analytic mass after translation by 1 micron gives final C.
    @test interface_trajectory.concentration[end] ≈ 20.0 - 0.25 * (4.496e-15 / 3.75e-15 - 1) atol = 1e-8
    @test interface_trajectory.d43[1] ≈ (6.2e-20 / 3.75e-15) rtol = 1e-8

    initial_observables = (concentration = Observable(time = [0., 100.],
        mean = [20., interface_trajectory.concentration[end]], variance = [0., 0.01]),
        seed_mass = Observable(time = [0.], mean = [0.25]))
    interface_experiment = CrystallisationExperiment(observables = initial_observables,
        initial_from = (concentration = :concentration, crystals = (mass_concentration = :seed_mass,)),
        seed_shape = (distribution = Uniform(10e-6, 20e-6),), temperature = 293.15, exp_id = 31)
    interface_spec = OptimisationSpec(bounds = (growth = (speed = (0.001, 0.03),),))
    interface_prepared = prepare_fit(interface_model, [interface_experiment], interface_spec)
    @test loss(interface_prepared, [0.01]) ≈ 0.5 * log(2π * 0.01) atol = 1e-10
    @test interface_prepared.selection.names == ["growth.speed"]
    @test interface_prepared.setup.included_observations[1][:seed_mass] == Int[]
    @test interface_prepared.setup.included_observations[1][:concentration] == [2]
    selected_candidate = CriSTool._selected_model(interface_model, interface_prepared.selection, [0.02])
    @test selected_candidate.growth.parameters == (speed = 0.02, held_order = 2.0)
    @test interface_model.growth.parameters == (speed = 0.01, held_order = 2.0)
    @test CriSTool._expanded_parameters(interface_prepared.selection, [0.02]) == [0.02, 2.0]
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment],
        OptimisationSpec(bounds = (system = (crystal_density = (900., 1100.),),)))
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment],
        OptimisationSpec(bounds = (growth = (speed = (0.03, 0.001),),)))

    interface_gradient_ad = DI.gradient(candidate_values -> loss(interface_prepared, candidate_values),
        DI.AutoForwardDiff(), [0.015])
    interface_gradient_fd = DI.gradient(candidate_values -> loss(interface_prepared, candidate_values),
        DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1)), [0.015])
    @test interface_gradient_ad ≈ interface_gradient_fd rtol = 1e-4 atol = 1e-7

    optimized_fit = fit(interface_prepared; algorithm = OptimizationOptimJL.LBFGS(),
        searchoptions = (maxiters = 50,))
    @test optimized_fit.model.growth.parameters.speed ≈ 0.01 atol = 1e-5
    @test optimized_fit.parameters.growth.speed == optimized_fit.model.growth.parameters.speed
    @test optimized_fit.model.growth.parameters.held_order == 2.0
    @test interface_model.growth.parameters.speed == 0.01

    pareto_experiment = CrystallisationExperiment(observables = merge(initial_observables,
        (d43 = Observable(time = [100.], mean = [18e-6]),)),
        initial_from = interface_experiment.initial_from, seed_shape = interface_experiment.seed_shape,
        temperature = 293.15, exp_id = 31)
    pareto_prepared = prepare_fit(interface_model, [pareto_experiment],
        OptimisationSpec(bounds = interface_spec.bounds, loss = mae()))
    pareto_results = fit(pareto_prepared; algorithm = Metaheuristics.NSGA2(N = 12,
        options = Metaheuristics.Options(iterations = 3, seed = 42, parallel_evaluation = true)))
    native_front = Metaheuristics.get_non_dominated_solutions(first(pareto_results).backend_result.population)
    @test [pareto_result.parameters.growth.speed for pareto_result in pareto_results] ==
        vec(Metaheuristics.positions(native_front))
    @test permutedims(hcat([collect(values(pareto_result.objectives)) for pareto_result in pareto_results]...)) ≈
        Metaheuristics.pareto_front(first(pareto_results).backend_result)
    @test first(pareto_results).model.growth.parameters.held_order == 2.0

    interface_draws = ParameterSamples(growth = (speed = [0.01, 0.02],))
    interface_predictions = predict(interface_model, [interface_experiment, interface_experiment], interface_draws;
        saveat = [0., 100.], observables = (:concentration, :d43))
    @test interface_predictions.samples.sample_ids == [1, 2]
    @test interface_predictions.ensembles[1].concentration == interface_predictions.ensembles[2].concentration
    @test interface_predictions.ensembles[1].concentration[1, end] ≈ interface_trajectory.concentration[end] atol = 1e-8
    @test prediction_summary(interface_predictions, :concentration)[1].sample_count == 2
    physical_predictions_before = copy(interface_predictions.ensembles[1].concentration)
    noisy_predictions = measurement_samples(interface_predictions, :concentration;
        noise = Uniform(1., 2.), rng = MersenneTwister(43))
    reference_rng = MersenneTwister(43)
    @test noisy_predictions[1][1, 1] == physical_predictions_before[1, 1] + rand(reference_rng, Uniform(1., 2.))
    @test interface_predictions.ensembles[1].concentration == physical_predictions_before
    @test_throws ArgumentError ParameterSamples(["growth.speed", "growth.speed"], ones(2, 3))
    @test_throws ArgumentError predict(interface_model, [interface_experiment], ParameterSamples(["concentration"], ones(1, 2)))
    incompatible_run = CriSTool._copy_crystallisation_problem(interface_run; crystal_density = 1100.)
    @test_throws ArgumentError predict(interface_model, [incompatible_run], interface_draws)
    @test predict(interface_model, [interface_run], interface_draws; saveat = [0., 100.],
        observables = (:concentration,)).ensembles[1].concentration[1, end] ≈
        interface_trajectory.concentration[end] atol = 1e-8

    late_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [10., 100.], mean = [20., 19.9], variance = [0.01, 0.01]),), temperature = 293.15, exp_id = 32)
    @test_throws ArgumentError prepare_fit(interface_model, [late_experiment], interface_spec)
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment], interface_spec; initial_time = 10.)
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment], interface_spec;
        problem_options = (initial_state = interface_run.initial_state,))
    incomplete_seed_experiment = CrystallisationExperiment(observables = initial_observables,
        initial_from = (concentration = :concentration, crystals = (mass_concentration = :seed_mass,)),
        seed_shape = (family = :lognormal, geometric_std = 1.2), temperature = 293.15, exp_id = 33)
    @test_throws ArgumentError prepare_fit(interface_model, [incomplete_seed_experiment], interface_spec)
    conflicting_shape_experiment = CrystallisationExperiment(observables = initial_observables,
        initial_from = interface_experiment.initial_from,
        seed_shape = (distribution = Uniform(10e-6, 20e-6), family = :lognormal), temperature = 293.15, exp_id = 33)
    @test_throws ArgumentError prepare_fit(interface_model, [conflicting_shape_experiment], interface_spec)
    conflicting_seed_experiment = CrystallisationExperiment(observables = initial_observables,
        initial_from = interface_experiment.initial_from, seed_shape = interface_experiment.seed_shape,
        initial_crystals = LogNormalInitialCrystals(mass_concentration = 0., d43 = 0., geometric_std = 1.2),
        temperature = 293.15, exp_id = 34)
    @test_throws ArgumentError prepare_fit(interface_model, [conflicting_seed_experiment], interface_spec)

    no_variance_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [0., 100.], mean = [20., 19.9]),), temperature = 293.15, exp_id = 35)
    @test_throws ArgumentError prepare_fit(interface_model, [no_variance_experiment], interface_spec)
    @test prepare_fit(interface_model, [no_variance_experiment],
        OptimisationSpec(bounds = interface_spec.bounds, loss = mae())) isa PreparedFit
    ignored_noise_experiment = CrystallisationExperiment(observables = (
        concentration = no_variance_experiment.observables.concentration,
        d43 = Observable(time = [100.], mean = [17e-6], variance = [1e-12])),
        temperature = 293.15, exp_id = 35)
    @test prepare_fit(interface_model, [ignored_noise_experiment], OptimisationSpec(bounds = interface_spec.bounds,
        loss = logMLE(weighting = (concentration = 0., d43 = 1.)))) isa PreparedFit

    bayesian_prepared = prepare_fit(interface_model, [interface_experiment],
        BayesianSpec(priors = (growth = (speed = Uniform(0.001, 0.03),),)))
    prior_candidate = [0.015]
    bayesian_logjoint = CriSTool.Turing.DynamicPPL.logjoint(nuts_model(bayesian_prepared),
        (θ = prior_candidate,))
    @test bayesian_logjoint ≈ logpdf(Uniform(0.001, 0.03), 0.015) - loss(bayesian_prepared, prior_candidate)
    tiny_posterior = fit(bayesian_prepared; sampler = CriSTool.Turing.NUTS(5, 0.65),
        n_samples = 5, n_chains = 2, rng = MersenneTwister(72), progress = false)
    @test parameter_samples(tiny_posterior).names == ["growth.speed"]
    @test size(parameter_samples(tiny_posterior).values) == (1, 10)
    @test parameter_samples(tiny_posterior).values == chains_to_matrix(tiny_posterior.backend_result;
        params = [Symbol("growth.speed")])
    @test tiny_posterior.model.growth.parameters.held_order == 2.0

    abc_prepared = prepare_fit(interface_model, [interface_experiment], ABCSpec(
        priors = (growth = (speed = Uniform(.005, .02),),), loss = mae(), target = 1.0))
    tiny_abc = fit(abc_prepared; nparticles = 8, generations = 1, HPC = true)
    @test vec(parameter_samples(tiny_abc).values) == collect(tiny_abc.backend_result.P.particles)
    @test parameter_samples(tiny_abc).names == ["growth.speed"]
    @test parameter_samples([optimized_fit, optimized_fit]).values == fill(optimized_fit.model.growth.parameters.speed, 1, 2)

    failed_predictions = predict(interface_model, [interface_experiment], interface_draws;
        saveat = [0., 100.], solve_options = (maxiters = 1,), observables = (:concentration,))
    @test failed_predictions.ensembles[1].success == [false, false]
    @test_throws ArgumentError prediction_summary(failed_predictions, :concentration)
    @test_throws ArgumentError prediction_summary(failed_predictions, :concentration; skip_failed = true)

    reordered_kinetic = KineticModel(growth_empirical();
        parameters = (growth_order = 2., growth_coefficient = 1e-9))
    @test reordered_kinetic.parameters == (growth_coefficient = 1e-9, growth_order = 2.)

    length_kinetic = KineticModel(CallableLengthGrowth(
        (rate_values, rate_context, crystal_length) -> rate_values.speed * 1e-6;
        parameters = (speed = .01,)))
    length_model = CrystallisationModel(system = interface_system,
        nucleation = interface_nucleation, growth = length_kinetic)
    @test_throws ArgumentError CrystallisationProblem(length_model;
        initial_conditions = (concentration = 20.,), temperature = ConstantTemperature(293.15))
    length_run = CrystallisationProblem(length_model;
        initial_conditions = (concentration = 20.,), temperature = ConstantTemperature(293.15),
        initial_crystals = interface_seed, solver = DQMOM(nquadrature = 3))
    @test simulate(length_run; saveat = [0., 100.]).concentration[end] ≈ interface_trajectory.concentration[end] atol = 1e-7

    msmpr_experiment = CrystallisationExperiment(observables = (
        concentration = Observable(time = [0.], mean = [2.], variance = [.01]),),
        temperature = 293.15, exp_id = 40, relaxation_initial = (concentration = 8.,),
        operation = MSMPROperation(volume = 2., inflow = .5,
            feed = CrystallisationFeed(concentration = 2.)))
    @test_throws ArgumentError prepare_fit(interface_model, [msmpr_experiment], interface_spec; mode = :steady)
    steady_prepared = prepare_fit(interface_model, [msmpr_experiment], interface_spec;
        mode = :steady, steady_options = (autonomous = true,))
    @test loss(steady_prepared, [.01]) ≈ .5 * log(2π * .01) atol = 1e-8
    @test steady_prepared.setup.included_observations[1][:concentration] == [1]
    steady_predictions = predict(interface_model, [msmpr_experiment], interface_draws;
        mode = :steady, observables = (:concentration,), steady_options = (autonomous = true,))
    @test steady_predictions.ensembles[1].concentration ≈ fill(2., 2, 1) atol = 1e-8
    @test_throws ArgumentError predict(interface_model, [interface_experiment], interface_draws; initial_time = 10.)

    noncontiguous_model = CrystallisationModel(system = interface_system,
        nucleation = KineticModel(CallableNucleation((rate_values, rate_context) -> rate_values.amplitude * 1e9;
            parameters = (amplitude = .1,))),
        growth = KineticModel(CallableGrowth((rate_values, rate_context) ->
            rate_values.speed * 1e-6 * supersaturation(rate_context)^rate_values.order;
            parameters = (speed = .001, order = 2.))))
    noncontiguous_fit = prepare_fit(noncontiguous_model, [interface_experiment],
        OptimisationSpec(bounds = (nucleation = (amplitude = (.01, .5),), growth = (order = (1., 3.),))))
    @test CriSTool._expanded_parameters(noncontiguous_fit.selection, [.2, 2.5]) == [.2, .001, 2.5]
    noncontiguous_ad = DI.gradient(candidate_values -> loss(noncontiguous_fit, candidate_values),
        DI.AutoForwardDiff(), [.2, 2.5])
    noncontiguous_fd = DI.gradient(candidate_values -> loss(noncontiguous_fit, candidate_values),
        DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1)), [.2, 2.5])
    @test noncontiguous_ad ≈ noncontiguous_fd rtol = 1e-4 atol = 1e-8

    bootstrapped_runs = bootstrap_measurements([interface_experiment], 1;
        observable_names = (:concentration,), seed = 17)
    @test only(only(bootstrapped_runs)).initial_from == interface_experiment.initial_from
    @test only(only(bootstrapped_runs)).observables.seed_mass.mean == [.25]
    @test loss(prepare_fit(interface_model, only(bootstrapped_runs), interface_spec), [.01]) ≈
        loss(interface_prepared, [.01]) atol = 1e-10
    table_runs = experiments_from_table(DataFrame(Exp_ID = [31, 31], Time = [0., 100.],
        Concentration = [20., interface_trajectory.concentration[end]], Concentration_var = [.01, .01],
        SeedMass = [.25, missing], Temperature = [293.15, 293.15]);
        observables = (concentration = ObservableColumns(mean = :Concentration, variance = :Concentration_var),
            seed_mass = ObservableColumns(mean = :SeedMass)), metadata_cols = (temperature = :Temperature,),
        initial_from = interface_experiment.initial_from, seed_shape = interface_experiment.seed_shape)
    @test loss(prepare_fit(interface_model, table_runs, interface_spec), [.01]) ≈
        loss(interface_prepared, [.01]) atol = 1e-10
end

@testset "Distribution seed physical identities" begin
    seed_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.),
        crystal_density = 1000., volume_shape_factor = 1.)
    seed_model = CrystallisationModel(system = seed_system,
        nucleation = KineticModel(CallableNucleation((rate_values, rate_context) -> 0.; parameters = (;))),
        growth = KineticModel(CallableGrowth((rate_values, rate_context) -> 0.; parameters = (;))))
    number_seed = DistributionInitialCrystals(LogNormal(log(12e-6), 0.2); mass_concentration = 0.25)
    volume_seed = DistributionInitialCrystals(LogNormal(log(12e-6) + 3 * 0.2^2, 0.2);
        mass_concentration = 0.25, weighting = :volume)
    @test number_weighted(volume_seed).distribution == number_seed.distribution
    @test volume_seed.d43 ≈ number_seed.d43 rtol = 1e-12
    uniform_volume_seed = DistributionInitialCrystals(Uniform(10e-6, 20e-6);
        mass_concentration = .25, weighting = :volume)
    @test uniform_volume_seed.d43 ≈ 15e-6 rtol = 1e-9
    @test number_weighted(uniform_volume_seed).d43 ≈ 15e-6 rtol = 1e-8
    number_run = CrystallisationProblem(seed_model; initial_conditions = (concentration = 20.,),
        temperature = ConstantTemperature(293.15), initial_crystals = number_seed)
    volume_run = CrystallisationProblem(seed_model; initial_conditions = (concentration = 20.,),
        temperature = ConstantTemperature(293.15), initial_crystals = volume_seed)
    @test number_run.initial_state ≈ volume_run.initial_state rtol = 1e-12
    @test number_run.initial_state[4] * 1000.0 ≈ 0.25 rtol = 1e-12
    @test_throws ArgumentError DistributionInitialCrystals(Normal(12e-6, 1e-6); mass_concentration = .25)
    @test_throws ArgumentError DistributionInitialCrystals(Uniform(0., 20e-6); mass_concentration = .25, weighting = :volume)
    @test_throws ArgumentError DistributionInitialCrystals(Gamma(2., 1e-6); mass_concentration = .25, weighting = :volume)
    @test_throws ArgumentError CrystallisationProblem(seed_model;
        initial_conditions = (concentration = 20.,), temperature = ConstantTemperature(293.15),
        solver = FiniteVol(meshsize = 100, lmax = 13e-6), initial_crystals = number_seed)
    mesh_seed_run = CrystallisationProblem(seed_model;
        initial_conditions = (concentration = 20.,), temperature = ConstantTemperature(293.15),
        solver = FiniteVol(meshsize = 400, lmax = 60e-6), initial_crystals = number_seed)
    @test seed_domain_diagnostics(mesh_seed_run, number_seed).omitted_mass_fraction < 1e-8
    @test mesh_seed_run.solver.cell_dL * sum(mesh_seed_run.initial_state[1:400] .* mesh_seed_run.solver.cell_centre.^3) * 1000 ≈ .25 rtol = 1e-4
end
