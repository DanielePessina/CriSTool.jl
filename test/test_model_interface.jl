using Random
using Metaheuristics
import OptimizationOptimJL

@testset "Named model workflow" begin
    interface_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.0),
        crystal_density = 1000.0, volume_shape_factor = 1.0)
    interface_nucleation = CallableNucleation((rate_values, rate_context) -> 0.0; parameters = (;))
    # Constant physical growth gives an independent translation oracle for the
    # seed moments. A second, unused coefficient exercises subset selection.
    interface_growth = CallableGrowth((rate_values, rate_context) -> rate_values.speed * 1e-6;
        parameters = (speed = 0.01, held_order = 2.0))
    interface_model = CrystallisationModel(interface_system;
        nucleation = interface_nucleation, growth = interface_growth)
    interface_seed = DistributionInitialCrystals(Uniform(10e-6, 20e-6); mass_concentration = 0.25)
    interface_conditions = RunConditions(temperature = 293.15, concentration = 20.0, crystals = interface_seed)
    @test interface_model.system.molecular_volume === nothing
    @test interface_model.growth.parameters == (speed = 0.01, held_order = 2.0)
    @test LysozymeSystem() isa CrystallisationSystem
    offset_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.),
        crystal_density = 1000., volume_shape_factor = 1., properties = (signed_offset = -1e-9,))
    offset_growth = CallableGrowth((rate_values, rate_context) -> rate_context.system.signed_offset;
        parameters = (;), required = (:signed_offset,))
    offset_model = CrystallisationModel(offset_system; nucleation = interface_nucleation, growth = offset_growth)
    offset_run = CrystallisationProblem(offset_model, RunConditions(temperature = 293.15, concentration = 20.))
    @test CriSTool.growthrate(offset_growth, offset_run.parameterset_growth,
        offset_run, CriSTool._get_initial_state(offset_run), 0.) == -1e-9

    # Keyword law calls bind values; partial or unknown names fail with the expected names.
    bound_growth = growth_empirical(growth_coefficient = 1e-9, growth_order = 2.0)
    @test bound_growth isa KineticModel
    @test bound_growth.parameters == (growth_coefficient = 1e-9, growth_order = 2.0)
    @test growth_empirical() isa growth_empirical
    @test growth_energy(activation_energy = 50e3).activation_energy == 50e3
    @test_throws ArgumentError growth_empirical(growth_order = 2.0)
    @test_throws ArgumentError growth_empirical(wrong = 1.0, growth_order = 2.0)
    @test_throws ArgumentError CrystallisationModel(interface_system; nucleation = nucl_CNT(), growth = interface_growth)
    @test_throws ArgumentError CrystallisationProblem(CrystallisationModel(interface_system;
        nucleation = nucl_CNT(ln_nucleation_prefactor = 38., surface_energy = 0.0006),
        growth = interface_growth), RunConditions(temperature = 293.15, concentration = 20.))
    @test_throws ArgumentError CrystallisationProblem(interface_model, RunConditions(temperature = 293.15))
    @test_throws ArgumentError CrystallisationProblem(interface_model, interface_conditions;
        initial_state = zeros(5))

    interface_trajectory = simulate(interface_model, interface_conditions; saveat = [0., 100.])
    # Uniform lengths in [a,b]: E[L^k] = (b^(k+1)-a^(k+1))/((k+1)(b-a)).
    # Frozen analytic mass after translation by 1 micron gives final C.
    @test interface_trajectory.concentration[end] ≈ 20.0 - 0.25 * (4.496e-15 / 3.75e-15 - 1) atol = 1e-8
    @test interface_trajectory.d43[1] ≈ (6.2e-20 / 3.75e-15) rtol = 1e-8

    initial_observables = (concentration = Observable(time = [0., 100.],
        mean = [20., interface_trajectory.concentration[end]], variance = [0., 0.01]),
        seed_mass = Observable(time = [0.], mean = [0.25]))
    observed_conditions = RunConditions(temperature = 293.15,
        crystals = DistributionInitialCrystals(Uniform(10e-6, 20e-6); mass_concentration = FromObservation(:seed_mass)))
    @test CriSTool.observation_sources(observed_conditions) == [:concentration, :seed_mass]
    interface_experiment = CrystallisationExperiment(observables = initial_observables,
        conditions = observed_conditions, exp_id = 31)
    interface_bounds = (growth = (speed = (0.001, 0.03),),)
    interface_spec = OptimisationSpec(bounds = interface_bounds, algorithm = OptimizationOptimJL.LBFGS(),
        searchoptions = (maxiters = 50,))
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
        OptimisationSpec(bounds = (system = (crystal_density = (900., 1100.),),), algorithm = DE()))
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment],
        OptimisationSpec(bounds = (growth = (speed = (0.03, 0.001),),), algorithm = DE()))

    interface_gradient_ad = DI.gradient(candidate_values -> loss(interface_prepared, candidate_values),
        DI.AutoForwardDiff(), [0.015])
    interface_gradient_fd = DI.gradient(candidate_values -> loss(interface_prepared, candidate_values),
        DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1)), [0.015])
    @test interface_gradient_ad ≈ interface_gradient_fd rtol = 1e-4 atol = 1e-7

    optimized_fit = fit(interface_prepared)
    @test optimized_fit.model.growth.parameters.speed ≈ 0.01 atol = 1e-5
    @test optimized_fit.parameters.growth.speed == optimized_fit.model.growth.parameters.speed
    @test optimized_fit.model.growth.parameters.held_order == 2.0
    @test interface_model.growth.parameters.speed == 0.01
    # A fit predicts its own fitted draw on the fitted experiments with the fitted solver.
    fitted_prediction = predict(optimized_fit; observables = (:concentration,))
    @test fitted_prediction.experiments == [31]
    @test fitted_prediction.ensembles[1].concentration[1, end] ≈ interface_trajectory.concentration[end] atol = 1e-6

    pareto_experiment = CrystallisationExperiment(observables = merge(initial_observables,
        (d43 = Observable(time = [100.], mean = [18e-6]),)), conditions = observed_conditions, exp_id = 31)
    pareto_prepared = prepare_fit(interface_model, [pareto_experiment], OptimisationSpec(bounds = interface_bounds,
        algorithm = Metaheuristics.NSGA2(N = 12, options = Metaheuristics.Options(iterations = 3, seed = 42)),
        loss = mae()))
    pareto_results = fit(pareto_prepared; threaded = true)
    native_front = Metaheuristics.get_non_dominated_solutions(first(pareto_results).backend_result.population)
    @test [pareto_result.parameters.growth.speed for pareto_result in pareto_results] ==
        vec(Metaheuristics.positions(native_front))
    @test permutedims(hcat([collect(values(pareto_result.objectives)) for pareto_result in pareto_results]...)) ≈
        Metaheuristics.pareto_front(first(pareto_results).backend_result)
    @test first(pareto_results).model.growth.parameters.held_order == 2.0

    interface_draws = ParameterSamples(growth = (speed = [0.01, 0.02],))
    interface_predictions = predict(interface_model, [interface_experiment, interface_experiment], interface_draws;
        solver = MoM(), saveat = [0., 100.], observables = (:concentration, :d43))
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
    @test_throws ArgumentError predict(interface_model, [interface_experiment],
        ParameterSamples(["concentration"], ones(1, 2)); solver = MoM())
    @test_throws UndefKeywordError predict(interface_model, [interface_experiment], interface_draws)
    @test_throws ArgumentError predict(interface_model, [interface_conditions], interface_draws; solver = MoM())
    @test predict(interface_model, [interface_conditions], interface_draws; solver = MoM(), saveat = [0., 100.],
        observables = (:concentration,)).ensembles[1].concentration[1, end] ≈
        interface_trajectory.concentration[end] atol = 1e-8

    late_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [10., 100.], mean = [20., 19.9], variance = [0.01, 0.01]),),
        conditions = RunConditions(temperature = 293.15), exp_id = 32)
    @test_throws ArgumentError prepare_fit(interface_model, [late_experiment], interface_spec)
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment], interface_spec; initial_time = 10.)
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment], interface_spec;
        problem_options = (initial_state = zeros(5),))
    unobserved_seed_experiment = CrystallisationExperiment(observables = initial_observables,
        conditions = RunConditions(temperature = 293.15, crystals = LogNormalInitialCrystals(
            mass_concentration = FromObservation(:seed_mass), d43 = FromObservation(:d43), geometric_std = 1.2)),
        exp_id = 33)
    @test_throws ArgumentError prepare_fit(interface_model, [unobserved_seed_experiment], interface_spec)

    no_variance_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [0., 100.], mean = [20., 19.9]),), conditions = RunConditions(temperature = 293.15), exp_id = 35)
    @test_throws ArgumentError prepare_fit(interface_model, [no_variance_experiment], interface_spec)
    @test prepare_fit(interface_model, [no_variance_experiment],
        OptimisationSpec(bounds = interface_bounds, algorithm = DE(), loss = mae())) isa PreparedFit
    ignored_noise_experiment = CrystallisationExperiment(observables = (
        concentration = no_variance_experiment.observables.concentration,
        d43 = Observable(time = [100.], mean = [17e-6], variance = [1e-12])),
        conditions = RunConditions(temperature = 293.15), exp_id = 35)
    @test prepare_fit(interface_model, [ignored_noise_experiment], OptimisationSpec(bounds = interface_bounds,
        algorithm = DE(), loss = logMLE(weighting = (concentration = 0., d43 = 1.)))) isa PreparedFit

    bayesian_prepared = prepare_fit(interface_model, [interface_experiment], BayesianSpec(
        priors = (growth = (speed = Uniform(0.001, 0.03),),), sampler = CriSTool.Turing.NUTS(5, 0.65),
        n_samples = 5, n_chains = 2))
    prior_candidate = [0.015]
    bayesian_logjoint = CriSTool.Turing.DynamicPPL.logjoint(CriSTool.nuts_model(bayesian_prepared),
        (θ = prior_candidate,))
    @test bayesian_logjoint ≈ logpdf(Uniform(0.001, 0.03), 0.015) - loss(bayesian_prepared, prior_candidate)
    tiny_posterior = fit(bayesian_prepared; rng = MersenneTwister(72), threaded = true, progress = false)
    @test tiny_posterior.samples.names == ["growth.speed"]
    @test size(tiny_posterior.samples.values) == (1, 10)
    @test tiny_posterior.samples.values == CriSTool.chains_to_matrix(tiny_posterior.backend_result;
        params = [Symbol("growth.speed")])
    @test tiny_posterior.model.growth.parameters.held_order == 2.0
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment],
        BayesianSpec(priors = (growth = (speed = Uniform(0.001, 0.03),),), loss = mae()))

    abc_prepared = prepare_fit(interface_model, [interface_experiment], ABCSpec(
        priors = (growth = (speed = Uniform(.005, .02),),), loss = mae(), target = 1.0,
        n_particles = 8, n_generations = 1))
    tiny_abc = fit(abc_prepared; threaded = true)
    @test vec(tiny_abc.samples.values) == collect(tiny_abc.backend_result.P.particles)
    @test tiny_abc.samples.names == ["growth.speed"]
    @test_throws ArgumentError prepare_fit(interface_model, [interface_experiment], ABCSpec(
        priors = (growth = (speed = Uniform(.005, .02),),), target = Inf))
    @test parameter_samples([optimized_fit, optimized_fit]).values == fill(optimized_fit.model.growth.parameters.speed, 1, 2)

    failed_predictions = predict(interface_model, [interface_experiment], interface_draws; solver = MoM(),
        saveat = [0., 100.], solve_options = (maxiters = 1,), observables = (:concentration,))
    @test failed_predictions.ensembles[1].success == [false, false]
    @test_throws ArgumentError prediction_summary(failed_predictions, :concentration)
    @test_throws ArgumentError prediction_summary(failed_predictions, :concentration; skip_failed = true)

    reordered_kinetic = KineticModel(growth_empirical();
        parameters = (growth_order = 2., growth_coefficient = 1e-9))
    @test reordered_kinetic.parameters == (growth_coefficient = 1e-9, growth_order = 2.)

    length_growth = CallableLengthGrowth(
        (rate_values, rate_context, crystal_length) -> rate_values.speed * 1e-6; parameters = (speed = .01,))
    length_model = CrystallisationModel(interface_system; nucleation = interface_nucleation, growth = length_growth)
    @test_throws ArgumentError CrystallisationProblem(length_model, RunConditions(temperature = 293.15, concentration = 20.))
    @test simulate(length_model, interface_conditions; solver = DQMOM(nquadrature = 3),
        saveat = [0., 100.]).concentration[end] ≈ interface_trajectory.concentration[end] atol = 1e-7
    # A posterior predicts with the solver it was fitted with: this length law
    # cannot run under the MoM default, so a silent solver change would throw.
    length_prepared = prepare_fit(length_model, [interface_experiment], BayesianSpec(
        priors = (growth = (speed = Uniform(0.001, 0.03),),)); solver = DQMOM(nquadrature = 3))
    length_posterior = CriSTool.InferenceResult(length_model, ParameterSamples(growth = (speed = [0.01],)),
        nothing, length_prepared)
    @test predict(length_posterior; observables = (:concentration,)).ensembles[1].concentration[1, end] ≈
        interface_trajectory.concentration[end] atol = 1e-7

    msmpr_operation = MSMPROperation(volume = 2., inflow = .5, feed = CrystallisationFeed(concentration = 2.))
    msmpr_experiment = CrystallisationExperiment(observables = (
        concentration = Observable(time = [0.], mean = [2.], variance = [.01]),),
        conditions = RunConditions(temperature = 293.15, concentration = 8., operation = msmpr_operation), exp_id = 40)
    @test_throws ArgumentError prepare_fit(interface_model, [msmpr_experiment], interface_spec; mode = :steady)
    observed_msmpr_experiment = CrystallisationExperiment(observables = msmpr_experiment.observables,
        conditions = RunConditions(temperature = 293.15, operation = msmpr_operation), exp_id = 41)
    @test_throws ArgumentError prepare_fit(interface_model, [observed_msmpr_experiment], interface_spec;
        mode = :steady, steady_options = (autonomous = true,))
    steady_prepared = prepare_fit(interface_model, [msmpr_experiment], interface_spec;
        mode = :steady, steady_options = (autonomous = true,))
    @test loss(steady_prepared, [.01]) ≈ .5 * log(2π * .01) atol = 1e-8
    @test steady_prepared.setup.included_observations[1][:concentration] == [1]
    steady_predictions = predict(interface_model, [msmpr_experiment], interface_draws; solver = MoM(),
        mode = :steady, observables = (:concentration,), steady_options = (autonomous = true,))
    @test steady_predictions.ensembles[1].concentration ≈ fill(2., 2, 1) atol = 1e-8

    noncontiguous_model = CrystallisationModel(interface_system;
        nucleation = CallableNucleation((rate_values, rate_context) -> rate_values.amplitude * 1e9;
            parameters = (amplitude = .1,)),
        growth = CallableGrowth((rate_values, rate_context) ->
            rate_values.speed * 1e-6 * supersaturation(rate_context)^rate_values.order;
            parameters = (speed = .001, order = 2.)))
    noncontiguous_fit = prepare_fit(noncontiguous_model, [interface_experiment],
        OptimisationSpec(bounds = (nucleation = (amplitude = (.01, .5),), growth = (order = (1., 3.),)), algorithm = DE()))
    @test CriSTool._expanded_parameters(noncontiguous_fit.selection, [.2, 2.5]) == [.2, .001, 2.5]
    noncontiguous_ad = DI.gradient(candidate_values -> loss(noncontiguous_fit, candidate_values),
        DI.AutoForwardDiff(), [.2, 2.5])
    noncontiguous_fd = DI.gradient(candidate_values -> loss(noncontiguous_fit, candidate_values),
        DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1)), [.2, 2.5])
    @test noncontiguous_ad ≈ noncontiguous_fd rtol = 1e-4 atol = 1e-8

    bootstrapped_runs = bootstrap_measurements([interface_experiment], 1;
        observable_names = (:concentration,), seed = 17)
    @test only(only(bootstrapped_runs)).conditions === interface_experiment.conditions
    @test only(only(bootstrapped_runs)).observables.seed_mass.mean == [.25]
    @test loss(prepare_fit(interface_model, only(bootstrapped_runs), interface_spec), [.01]) ≈
        loss(interface_prepared, [.01]) atol = 1e-10
    table_runs = experiments_from_table(DataFrame(Exp_ID = [31, 31], Time = [0., 100.],
        Concentration = [20., interface_trajectory.concentration[end]], Concentration_var = [.01, .01],
        SeedMass = [.25, missing], Temperature = [293.15, 293.15]);
        observables = (concentration = ObservableColumns(mean = :Concentration, variance = :Concentration_var),
            seed_mass = ObservableColumns(mean = :SeedMass)), metadata_cols = (temperature = :Temperature,),
        conditions = (crystals = observed_conditions.crystals,))
    @test loss(prepare_fit(interface_model, table_runs, interface_spec), [.01]) ≈
        loss(interface_prepared, [.01]) atol = 1e-10
    @test_throws ArgumentError experiments_from_table(DataFrame(Exp_ID = [1], Time = [0.], Concentration = [20.]))
end

@testset "Distribution seed physical identities" begin
    seed_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.),
        crystal_density = 1000., volume_shape_factor = 1.)
    seed_model = CrystallisationModel(seed_system;
        nucleation = CallableNucleation((rate_values, rate_context) -> 0.; parameters = (;)),
        growth = CallableGrowth((rate_values, rate_context) -> 0.; parameters = (;)))
    number_seed = DistributionInitialCrystals(LogNormal(log(12e-6), 0.2); mass_concentration = 0.25)
    volume_seed = DistributionInitialCrystals(LogNormal(log(12e-6) + 3 * 0.2^2, 0.2);
        mass_concentration = 0.25, weighting = :volume)
    @test number_weighted(volume_seed).distribution == number_seed.distribution
    @test volume_seed.d43 ≈ number_seed.d43 rtol = 1e-12
    uniform_volume_seed = DistributionInitialCrystals(Uniform(10e-6, 20e-6);
        mass_concentration = .25, weighting = :volume)
    @test uniform_volume_seed.d43 ≈ 15e-6 rtol = 1e-9
    @test number_weighted(uniform_volume_seed).d43 ≈ 15e-6 rtol = 1e-8
    seed_conditions(seed_description) = RunConditions(temperature = 293.15, concentration = 20., crystals = seed_description)
    number_run = CrystallisationProblem(seed_model, seed_conditions(number_seed))
    volume_run = CrystallisationProblem(seed_model, seed_conditions(volume_seed))
    @test number_run.initial_state ≈ volume_run.initial_state rtol = 1e-12
    @test number_run.initial_state[4] * 1000.0 ≈ 0.25 rtol = 1e-12
    @test_throws ArgumentError DistributionInitialCrystals(Normal(12e-6, 1e-6); mass_concentration = .25)
    @test_throws ArgumentError DistributionInitialCrystals(Uniform(0., 20e-6); mass_concentration = .25, weighting = :volume)
    @test_throws ArgumentError DistributionInitialCrystals(Gamma(2., 1e-6); mass_concentration = .25, weighting = :volume)
    @test_throws ArgumentError CrystallisationProblem(seed_model, seed_conditions(number_seed);
        solver = FiniteVol(meshsize = 100, lmax = 13e-6))
    mesh_seed_run = CrystallisationProblem(seed_model, seed_conditions(number_seed);
        solver = FiniteVol(meshsize = 400, lmax = 60e-6))
    @test seed_domain_diagnostics(mesh_seed_run, number_seed).omitted_mass_fraction < 1e-8
    @test mesh_seed_run.solver.cell_dL * sum(mesh_seed_run.initial_state[1:400] .* mesh_seed_run.solver.cell_centre.^3) * 1000 ≈ .25 rtol = 1e-4
end
