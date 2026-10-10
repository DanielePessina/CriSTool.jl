# End-to-end model -> prepare_fit -> fit -> predict on the translation fixture.
# Synthetic data come from the closed-form trajectory at the true speed 0.01, so
# fits are checked by recovery of that speed and predictions by the closed form.
using Random
using Metaheuristics
import OptimizationOptimJL
using MCMCChains: Chains

const fitpredict_true_speed = 0.01
const fitpredict_times = [0.0, 300.0, 600.0, 1000.0]
const fitpredict_variances = (concentration = 0.01, d43 = 1e-14)
const fitpredict_trajectory = (time = fitpredict_times,
    concentration = fixture_translation_concentration.(fitpredict_times),
    d43 = fixture_translation_d43.(fitpredict_times))
# Initial concentration is read from the t = 0 observation, which is then not scored.
const fitpredict_conditions = RunConditions(temperature = 293.15, crystals = fixture_uniform_seed())
const fitpredict_experiment = fixture_experiment(fitpredict_trajectory, fitpredict_conditions; exp_id = 31)
# A second, unused growth coefficient checks that fits only touch selected values.
const fitpredict_model = CrystallisationModel(fixture_system; nucleation = fixture_no_nucleation,
    growth = CallableGrowth((rate_values, rate_context) -> rate_values.speed * 1e-6;
        parameters = (speed = fitpredict_true_speed, held_order = 2.0)))
const fitpredict_bounds = (growth = (speed = (0.001, 0.03),),)
const fitpredict_gaussian_nll = (residual, variance) -> -logpdf(Normal(0.0, sqrt(variance)), residual)

# Records the target and reference loss handed to an ABC sampler.
struct FitPredictTargetSampler <: CriSTool.AbstractABCSampler
    received_target::Base.RefValue{Float64}
    received_reference_loss::Base.RefValue{Float64}
end
function CriSTool._runsampler(target_sampler::FitPredictTargetSampler, prior, candidate_loss,
        target, reference_loss; kwargs...)
    target_sampler.received_target[] = target
    target_sampler.received_reference_loss[] = reference_loss
    return (P = [0.4 0.6],), true
end

# Draws a known sequence of unit uniforms for inverse-CDF sampling checks.
mutable struct FitPredictSequenceRNG <: Random.AbstractRNG
    unit_draws::Vector{Float64}
    draw_index::Int
end
Random.rand(sequence_rng::FitPredictSequenceRNG) =
    (sequence_rng.draw_index += 1; sequence_rng.unit_draws[sequence_rng.draw_index - 1])

@testset "Prepared loss: hand-computed likelihood and preparation rules" begin
    fitpredict_prepared = prepare_fit(fitpredict_model, [fitpredict_experiment],
        OptimisationSpec(bounds = fitpredict_bounds, algorithm = OptimizationOptimJL.LBFGS()))
    @test fitpredict_prepared.selection.names == ["growth.speed"]
    # At the truth every residual vanishes: 3 scored concentrations (t = 0 is the
    # initial condition) and 4 scored d43 values contribute only their normalisers.
    truth_terms = loss_terms(fitpredict_prepared, [0.01])
    @test keys(truth_terms) == (:concentration, :d43)
    @test collect(values(truth_terms)) ≈
          [3 * fitpredict_gaussian_nll(0.0, 0.01), 4 * fitpredict_gaussian_nll(0.0, 1e-14)] rtol = 1e-9
    # Away from the truth the residuals are the closed-form trajectory differences.
    trial_residuals = fixture_translation_concentration.(fitpredict_times[2:end]; speed = 0.02) .-
                      fitpredict_trajectory.concentration[2:end]
    trial_d43_residuals = fixture_translation_d43.(fitpredict_times; speed = 0.02) .- fitpredict_trajectory.d43
    @test loss(fitpredict_prepared, [0.02]) ≈ sum(fitpredict_gaussian_nll.(trial_residuals, 0.01)) +
          sum(fitpredict_gaussian_nll.(trial_d43_residuals, 1e-14)) rtol = 1e-6

    @test_throws ArgumentError prepare_fit(fitpredict_model, [fitpredict_experiment],
        OptimisationSpec(bounds = (system = (crystal_density = (900.0, 1100.0),),), algorithm = DE()))
    @test_throws ArgumentError prepare_fit(fitpredict_model, [fitpredict_experiment],
        OptimisationSpec(bounds = (growth = (speed = (0.03, 0.001),),), algorithm = DE()))
    @test_throws ArgumentError prepare_fit(fitpredict_model, [fitpredict_experiment],
        OptimisationSpec(bounds = (growth = (unknown_rate = (0.0, 1.0),),), algorithm = DE()))
    # The time-zero start is not a preparation option.
    @test_throws MethodError prepare_fit(fitpredict_model, [fitpredict_experiment],
        fitpredict_prepared.specification; initial_time = 10.0)
    @test_throws ArgumentError prepare_fit(fitpredict_model, [fitpredict_experiment],
        fitpredict_prepared.specification; problem_options = (initial_state = zeros(5),))
    # Initial concentration from observations requires a t = 0 observation.
    late_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [10.0, 100.0], mean = [20.0, 19.9], variance = [0.01, 0.01]),),
        conditions = RunConditions(temperature = 293.15), exp_id = 32)
    @test_throws ArgumentError prepare_fit(fitpredict_model, [late_experiment], fitpredict_prepared.specification)
    # A likelihood needs supplied variances; a discrepancy does not.
    no_variance_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [0.0, 100.0], mean = [20.0, 19.9]),), conditions = RunConditions(temperature = 293.15), exp_id = 35)
    @test_throws ArgumentError prepare_fit(fitpredict_model, [no_variance_experiment], fitpredict_prepared.specification)
    @test loss(prepare_fit(fitpredict_model, [no_variance_experiment],
        OptimisationSpec(bounds = fitpredict_bounds, algorithm = DE(), loss = MeanAbsoluteError())), [0.0]) ≈ 0.1 atol = 1e-12
    @test_throws ArgumentError prepare_fit(fitpredict_model, [fitpredict_experiment],
        BayesianSpec(priors = (growth = (speed = Uniform(0.001, 0.03),),), loss = MeanAbsoluteError()))
    @test_throws ArgumentError prepare_fit(fitpredict_model, [fitpredict_experiment],
        ABCSpec(priors = (growth = (speed = Uniform(0.005, 0.02),),), target = Inf))
end

@testset "Pooled observable objectives (hand-computed MAE and NLL)" begin
    # Nothing grows and nothing nucleates, so C stays at its observed C0 = 10 and
    # pH stays at its supplied value 7.
    pooled_model = fixture_translation_model(speed = 0.0)
    pooled_conditions = RunConditions(temperature = 293.15, solvent = (pH = 7.0,))
    pooled_run(exp_id, observed_series) = CrystallisationExperiment(; exp_id, observables = observed_series,
        conditions = pooled_conditions)
    concentration_series = Observable(time = [0.0, 1.0], mean = [10.0, 11.0])
    ph_series = Observable(time = [0.0, 1.0], mean = [8.0, 8.0])
    pooled_spec(pooled_loss) = OptimisationSpec(bounds = (growth = (speed = (0.0, 1.0),),),
        algorithm = DE(), loss = pooled_loss)
    pooled_loss_value(pooled_loss, pooled_runs; kwargs...) =
        loss(prepare_fit(pooled_model, pooled_runs, pooled_spec(pooled_loss); kwargs...), [0.0])
    # Concentration: one scored unit error per run; pH: two unit errors.
    for pooled_runs in ([pooled_run(1, (concentration = concentration_series,)),
                         pooled_run(2, (concentration = concentration_series, pH = ph_series))],
                        [pooled_run(2, (pH = ph_series, concentration = concentration_series)),
                         pooled_run(1, (concentration = concentration_series,))])
        @test pooled_loss_value(MeanAbsoluteError(), pooled_runs) ≈ 2.0
        @test pooled_loss_value(MeanAbsoluteError(weighting = (concentration = 2.0, pH = 3.0)), pooled_runs) ≈ 5.0
    end
    # Two pH errors of 1 and four of 4 pool to a mean absolute error of 3.
    uneven_runs = [pooled_run(1, (concentration = concentration_series, pH = ph_series)),
        pooled_run(2, (concentration = concentration_series,
            pH = Observable(time = [0.0, 0.25, 0.75, 1.0], mean = fill(11.0, 4))))]
    uneven_prepared = prepare_fit(pooled_model, uneven_runs, pooled_spec(MeanAbsoluteError()))
    @test loss_terms(MeanAbsoluteError(), uneven_prepared.setup, [0.0]) == (concentration = 1.0, pH = 3.0)
    ordered_prepared = prepare_fit(pooled_model, uneven_runs, pooled_spec(MeanAbsoluteError(weighting = [3.0, 2.0]));
        observable_order = [:pH, :concentration])
    @test loss_terms(MeanAbsoluteError(weighting = [3.0, 2.0]), ordered_prepared.setup, [0.0]) == (pH = 9.0, concentration = 2.0)

    likelihood_runs = [pooled_run(1, (concentration = Observable(time = [0.0, 1.0], mean = [10.0, 11.0], variance = 1.0),
            pH = Observable(time = [0.0, 1.0], mean = [8.0, 8.0], variance = 4.0))),
        pooled_run(2, (concentration = Observable(time = [0.0, 1.0], mean = [10.0, 11.0], variance = 1.0),))]
    likelihood_prepared = prepare_fit(pooled_model, likelihood_runs, pooled_spec(GaussianNLL()))
    likelihood_terms = loss_terms(GaussianNLL(), likelihood_prepared.setup, [0.0])
    @test keys(likelihood_terms) == (:concentration, :pH)
    @test collect(values(likelihood_terms)) ≈ [2 * fitpredict_gaussian_nll(1.0, 1.0), 2 * fitpredict_gaussian_nll(1.0, 4.0)]

    # A local projection scores a derived signal: 2C matches, 3C misses by 20.
    signal_run = pooled_run(3, (concentration = concentration_series,
        optical_signal = Observable(time = [0.0, 1.0], mean = [20.0, 20.0])))
    signal_terms(signal_scale) = loss_terms(MeanAbsoluteError(), prepare_fit(pooled_model, [signal_run], pooled_spec(MeanAbsoluteError());
        observable_projections = (optical_signal = physical_solution -> signal_scale .* physical_solution.concentration,)).setup, [0.0])
    @test signal_terms(2.0).optical_signal == 0.0
    @test signal_terms(3.0).optical_signal ≈ 10.0
    @test_throws ArgumentError prepare_fit(pooled_model, [signal_run], pooled_spec(MeanAbsoluteError());
        observable_projections = (concentration = physical_solution -> physical_solution.concentration,))
end

@testset "Gradient fit (LBFGS) recovers the true speed and predicts the closed form" begin
    lbfgs_prepared = prepare_fit(fitpredict_model, [fitpredict_experiment], OptimisationSpec(
        bounds = fitpredict_bounds, algorithm = OptimizationOptimJL.LBFGS(), searchoptions = (maxiters = 50,)))
    lbfgs_fit = fit(lbfgs_prepared)
    fitted_speed = lbfgs_fit.parameters.growth.speed
    @test fitted_speed ≈ fitpredict_true_speed rtol = 1e-4
    @test lbfgs_fit.model.growth.parameters == (speed = fitted_speed, held_order = 2.0)
    @test fitpredict_model.growth.parameters.speed == fitpredict_true_speed
    @test sum(values(lbfgs_fit.objectives)) ≈ loss(lbfgs_prepared, [fitted_speed]) rtol = 1e-12
    # predict(fit) uses the fitted draw, experiments, solver and observation times.
    fitted_prediction = predict(lbfgs_fit)
    @test fitted_prediction.experiments == [31]
    @test only(fitted_prediction.ensembles).time == fitpredict_times
    @test vec(observable_values(only(fitted_prediction.ensembles), :concentration)) ≈
          fixture_translation_concentration.(fitpredict_times; speed = fitted_speed) atol = 1e-8
    @test vec(observable_values(only(fitted_prediction.ensembles), :d43)) ≈ fixture_translation_d43.(fitpredict_times; speed = fitted_speed) rtol = 1e-7
    # New run conditions at new times extrapolate the fitted model.
    extrapolated = predict(lbfgs_fit, [fixture_seeded_conditions(concentration = 25.0)]; saveat = [0.0, 5000.0],
        observables = (:concentration,))
    @test observable_values(only(extrapolated.ensembles), :concentration)[1, end] ≈
          fixture_translation_concentration(5000.0; speed = fitted_speed, concentration = 25.0) atol = 1e-8
end

@testset "Metaheuristics DE: fit equals a direct optimiser run; configuration untouched" begin
    supplied_de = Metaheuristics.DE(; N = 8, strategy = :rand2, F = 0.43, CR = 0.81,
        options = Metaheuristics.Options(; iterations = 3, seed = 921, rng = MersenneTwister(921),
            parallel_evaluation = false, f_calls_limit = 500, store_convergence = true, time_limit = 60.0))
    supplied_de.status.f_calls = 123
    supplied_snapshot = deepcopy(supplied_de)
    de_prepared = prepare_fit(fitpredict_model, [fitpredict_experiment],
        OptimisationSpec(bounds = fitpredict_bounds, algorithm = supplied_de))
    direct_de = deepcopy(supplied_de)
    direct_de.status = Metaheuristics.State(nothing, [])
    direct_result = Metaheuristics.optimize(trial -> loss(de_prepared, trial),
        Metaheuristics.boxconstraints([0.001], [0.03]), direct_de)
    de_fit = fit(de_prepared)
    @test Metaheuristics.minimizer(de_fit.backend_result) == Metaheuristics.minimizer(direct_result)
    @test de_fit.backend_result.f_calls == direct_result.f_calls
    @test [de_fit.parameters.growth.speed] == Metaheuristics.minimizer(direct_result)
    @test sum(values(de_fit.objectives)) ≈ Metaheuristics.minimum(direct_result) rtol = 1e-12
    @test 0.001 <= de_fit.parameters.growth.speed <= 0.03
    # The supplied algorithm, its options, status and RNG stream are not consumed.
    @test all(getfield(supplied_de.parameters, field_name) == getfield(supplied_snapshot.parameters, field_name)
              for field_name in fieldnames(typeof(supplied_de.parameters)))
    @test (supplied_de.options.iterations, supplied_de.options.f_calls_limit, supplied_de.status.f_calls) == (3, 500.0, 123)
    @test rand(deepcopy(supplied_de.options.rng), 4) == rand(supplied_snapshot.options.rng, 4)
    # Explicit overrides change only what they name.
    overridden_de = CriSTool._configured_mh_algorithm(supplied_de; nparticles = 16, generations = 7,
        parallel_evaluation = true)
    @test (overridden_de.parameters.N, overridden_de.options.iterations, overridden_de.options.parallel_evaluation,
           overridden_de.parameters.strategy, overridden_de.parameters.F, overridden_de.status.f_calls) ==
          (16, 7, true, :rand2, 0.43, 0)
    @test_throws ArgumentError CriSTool._configured_mh_algorithm(supplied_de; nparticles = 0)
end

@testset "NSGA2 (threaded): non-dominated Pareto set within bounds" begin
    pareto_prepared = prepare_fit(fitpredict_model, [fitpredict_experiment], OptimisationSpec(bounds = fitpredict_bounds,
        algorithm = Metaheuristics.NSGA2(N = 12, options = Metaheuristics.Options(iterations = 3, seed = 42)), loss = MeanAbsoluteError()))
    pareto_results = fit(pareto_prepared; threaded = true)
    pareto_objectives = [collect(values(pareto_result.objectives)) for pareto_result in pareto_results]
    @test all(pareto_result -> 0.001 <= pareto_result.parameters.growth.speed <= 0.03, pareto_results)
    @test all(pareto_result -> pareto_result.model.growth.parameters.held_order == 2.0, pareto_results)
    dominates(first_objectives, second_objectives) =
        all(first_objectives .<= second_objectives) && any(first_objectives .< second_objectives)
    @test !any(dominates(first_objectives, second_objectives)
               for first_objectives in pareto_objectives, second_objectives in pareto_objectives)
    # Each candidate's named objectives are the serial loss at its parameters.
    @test all(pareto_result -> sum(values(pareto_result.objectives)) ≈
                               loss(pareto_prepared, [pareto_result.parameters.growth.speed]), pareto_results)
    @test sort(first.(pareto_objectives)) ≈ sort(Metaheuristics.pareto_front(first(pareto_results).backend_result)[:, 1])
end

@testset "Bootstrap fits on noise-free data all recover the truth" begin
    bootstrap_spec = OptimisationSpec(bounds = fitpredict_bounds, algorithm = OptimizationOptimJL.LBFGS(),
        searchoptions = (maxiters = 50,))
    bootstrap_fits = [fit(prepare_fit(fitpredict_model, replicate_runs, bootstrap_spec))
        for replicate_runs in bootstrap_measurements([fitpredict_experiment], 2; observable_names = (:concentration,), seed = 17)]
    bootstrap_draws = parameter_samples(bootstrap_fits)
    @test bootstrap_draws.names == ["growth.speed"]
    @test bootstrap_draws.source == :bootstrap
    @test bootstrap_draws.values ≈ fill(fitpredict_true_speed, 1, 2) rtol = 1e-4
end

@testset "Table-loaded experiments give the same loss as hand-built ones" begin
    # Seed mass comes from a t = 0 observation through FromObservation.
    observed_seed_conditions = RunConditions(temperature = 293.15,
        crystals = DistributionInitialCrystals(Uniform(fixture_seed_bounds...); mass_concentration = FromObservation(:seed_mass)))
    hand_built_run = CrystallisationExperiment(exp_id = 31, conditions = observed_seed_conditions,
        observables = (concentration = fitpredict_experiment.observables.concentration,
            seed_mass = Observable(time = [0.0], mean = [fixture_seed_mass])))
    table_runs = experiments_from_table(DataFrame(Exp_ID = 31, Time = fitpredict_times,
            Concentration = fitpredict_trajectory.concentration, Concentration_var = 0.01,
            SeedMass = [fixture_seed_mass, missing, missing, missing]);
        observables = (concentration = ObservableColumns(mean = :Concentration, variance = :Concentration_var),
            seed_mass = ObservableColumns(mean = :SeedMass)),
        conditions = (temperature = 293.15, crystals = observed_seed_conditions.crystals))
    table_spec = OptimisationSpec(bounds = fitpredict_bounds, algorithm = DE())
    hand_built_prepared = prepare_fit(fitpredict_model, [hand_built_run], table_spec)
    @test hand_built_prepared.setup.included_observations[1][:seed_mass] == Int[]
    @test loss(hand_built_prepared, [0.02]) ≈ loss(prepare_fit(fitpredict_model, table_runs, table_spec), [0.02]) rtol = 1e-12
    @test loss(hand_built_prepared, [0.01]) ≈ 3 * fitpredict_gaussian_nll(0.0, 0.01) atol = 1e-6
end

@testset "Size objective: hand-computed d43 likelihood with offsets" begin
    # Offsets of 1 and 2 µm against variances 1e-12 and 4e-12, concentration unweighted.
    size_times = [300.0, 1000.0]
    size_offsets, size_variances = [1e-6, 2e-6], [1e-12, 4e-12]
    size_run = CrystallisationExperiment(exp_id = 5, conditions = fixture_seeded_conditions(),
        observables = (d43 = Observable(time = size_times, mean = fixture_translation_d43.(size_times) .+ size_offsets,
            variance = size_variances),))
    size_prepared = prepare_fit(fitpredict_model, [size_run], OptimisationSpec(bounds = fitpredict_bounds,
        algorithm = DE(), loss = GaussianNLL(weighting = (d43 = 1.0,))))
    @test loss(size_prepared, [0.01]) ≈ 0.5 * sum(log.(2π .* size_variances) .+ size_offsets .^ 2 ./ size_variances) rtol = 1e-6
end

@testset "NUTS (2 threaded chains): posterior density and posterior predictions" begin
    bayesian_prepared = prepare_fit(fitpredict_model, [fitpredict_experiment], BayesianSpec(
        priors = (growth = (speed = Uniform(0.001, 0.03),),), sampler = CriSTool.Turing.NUTS(5, 0.65),
        n_samples = 5, n_chains = 2))
    # log p(theta | data) = log prior - negative log likelihood.
    @test CriSTool.Turing.DynamicPPL.logjoint(CriSTool.nuts_model(bayesian_prepared), (θ = [0.015],)) ≈
          logpdf(Uniform(0.001, 0.03), 0.015) - loss(bayesian_prepared, [0.015]) rtol = 1e-10
    tiny_posterior = fit(bayesian_prepared; rng = MersenneTwister(72), threaded = true, progress = false)
    @test tiny_posterior.samples.names == ["growth.speed"]
    @test size(tiny_posterior.samples.values) == (1, 10)
    @test tiny_posterior.samples.values ==
          CriSTool.chains_to_matrix(tiny_posterior.backend_result; params = [Symbol("growth.speed")])
    @test all(0.001 .< tiny_posterior.samples.values .< 0.03)
    # Each posterior draw predicts its own closed-form trajectory.
    posterior_predictions = predict(tiny_posterior; observables = (:concentration,))
    @test observable_values(only(posterior_predictions.ensembles), :concentration) ≈
          [fixture_translation_concentration(prediction_time; speed = posterior_speed)
           for posterior_speed in vec(tiny_posterior.samples.values), prediction_time in fitpredict_times] atol = 1e-8
end

@testset "ABC: target forwarding, tiny ABCDE run and KissABC helpers" begin
    received_target, received_reference_loss = Ref(NaN), Ref(NaN)
    target_sampler = FitPredictTargetSampler(received_target, received_reference_loss)
    abc_priors = (growth = (speed = Uniform(0.005, 0.02),),)
    forwarded_result = fit(prepare_fit(fitpredict_model, [fitpredict_experiment],
        ABCSpec(priors = abc_priors, target = 1.5, sampler = target_sampler, loss = GaussianNLL())))
    # The reference loss is the likelihood at the model's current (true) speed.
    @test received_reference_loss[] ≈ 3 * fitpredict_gaussian_nll(0.0, 0.01) + 4 * fitpredict_gaussian_nll(0.0, 1e-14) rtol = 1e-9
    @test received_target[] == 1.5
    @test forwarded_result.samples.values == [0.4 0.6]
    @test forwarded_result.samples.source == :abc

    abc_prepared = prepare_fit(fitpredict_model, [fitpredict_experiment], ABCSpec(priors = abc_priors,
        target = 1.0, n_particles = 8, n_generations = 1))
    tiny_abc = fit(abc_prepared; threaded = true)
    @test vec(tiny_abc.samples.values) == collect(tiny_abc.backend_result.P.particles)
    @test all(0.005 .<= tiny_abc.samples.values .<= 0.02)
    _, unreachable_target_reached = CriSTool._runsampler(ABCDESampler(),
        product_distribution([Uniform(0.0, 1.0)]), candidate -> 100.0, 0.0, 100.0;
        nparticles = 8, generations = 1, HPC = false, earlystop = false)
    @test !unreachable_target_reached

    @test CriSTool.KissABC.logsumexp_stable([1.0, 2.0, 3.0]) ≈ log(exp(1.0) + exp(2.0) + exp(3.0))
    @test CriSTool.KissABC.logsumexp_stable([-1000.0, -1000.0]) ≈ -1000.0 + log(2.0)
    @test CriSTool.KissABC.log_kernel_weight(2.0, 0.0, 1.0; kernel = :gaussian) ≈ -2.0
    @test CriSTool.KissABC.log_kernel_weight(1.0, 0.0, 1.0; kernel = :laplace) ≈ -1.0
    @test CriSTool.KissABC.log_kernel_weight(100.0, 0.0, 1.0; kernel = :none) == 0.0
    @test CriSTool.KissABC.group_range(3, 5) == 11:15
    # Inverse-CDF draws 0.05, 0.20, 0.90 against cumulative weights 0.1, 0.4, 1.0.
    sequence_rng = FitPredictSequenceRNG([0.05, 0.20, 0.90], 1)
    @test [CriSTool.KissABC.sample_from_logweights(sequence_rng, log.([0.1, 0.3, 0.6])) for _ in 1:3] == [1, 2, 3]
end

@testset "Predictions from user draws: closed form per draw, noise and failures" begin
    user_draws = parameter_samples(["growth.speed"], Uniform(0.005, 0.02); n_samples = 3, rng = Xoshiro(714))
    @test user_draws.values == reshape(rand(Xoshiro(714), Uniform(0.005, 0.02), 3), 1, 3)
    user_predictions = predict(fitpredict_model, [fitpredict_experiment, fitpredict_experiment], user_draws;
        solver = MoM(), observables = (:concentration, :d43))
    @test user_predictions.samples.sample_ids == [1, 2, 3]
    @test observable_values(user_predictions.ensembles[1], :concentration) == observable_values(user_predictions.ensembles[2], :concentration)
    @test observable_values(user_predictions.ensembles[1], :concentration) ≈
          [fixture_translation_concentration(prediction_time; speed = draw_speed)
           for draw_speed in vec(user_draws.values), prediction_time in fitpredict_times] atol = 1e-8
    concentration_summary = prediction_summary(user_predictions, :concentration)[1]
    @test concentration_summary.sample_count == 3
    @test concentration_summary.mean ≈ vec(sum(observable_values(user_predictions.ensembles[1], :concentration); dims = 1)) ./ 3

    physical_before = copy(observable_values(user_predictions.ensembles[1], :concentration))
    noisy_values = measurement_samples(user_predictions, :concentration; noise = Uniform(1.0, 2.0), rng = MersenneTwister(43))
    @test noisy_values[1][1, 1] == physical_before[1, 1] + rand(MersenneTwister(43), Uniform(1.0, 2.0))
    @test observable_values(user_predictions.ensembles[1], :concentration) == physical_before
    conditions_prediction = predict(fitpredict_model, [fixture_seeded_conditions()],
        ParameterSamples(growth = (speed = [0.01],)); solver = MoM(), saveat = [0.0, 100.0], observables = (:concentration,))
    @test observable_values(conditions_prediction.ensembles[1], :concentration)[1, end] ≈ fixture_translation_concentration(100.0) atol = 1e-8

    @test_throws ArgumentError ParameterSamples(["growth.speed", "growth.speed"], ones(2, 3))
    @test_throws ArgumentError predict(fitpredict_model, [fitpredict_experiment],
        ParameterSamples(["concentration"], ones(1, 2)); solver = MoM())
    @test_throws UndefKeywordError predict(fitpredict_model, [fitpredict_experiment], user_draws)
    @test_throws ArgumentError predict(fitpredict_model, [fixture_seeded_conditions()], user_draws; solver = MoM())
    failed_predictions = predict(fitpredict_model, [fitpredict_experiment], user_draws; solver = MoM(),
        solve_options = (maxiters = 1,), observables = (:concentration,))
    @test failed_predictions.ensembles[1].success == [false, false, false]
    @test_throws ArgumentError prediction_summary(failed_predictions, :concentration)
    @test_throws ArgumentError prediction_summary(failed_predictions, :concentration; skip_failed = true)

    # Joint draws keep named row order and correlations.
    ordered_draws = parameter_samples(["nucleation.rate", "growth.speed"],
        product_distribution([Dirac(2.0), Dirac(-1.0)]); n_samples = 2, rng = Xoshiro(12))
    @test ordered_draws.values == [2.0 2.0; -1.0 -1.0]
    joint_distribution = MvNormal([1.0, -2.0], [1.0 0.8; 0.8 1.0])
    @test parameter_samples(["growth.speed", "nucleation.rate"], joint_distribution;
        n_samples = 5, rng = MersenneTwister(3)).values == rand(MersenneTwister(3), joint_distribution, 5)
    @test_throws DimensionMismatch parameter_samples(["growth.speed"], joint_distribution; n_samples = 3)
    @test_throws ArgumentError parameter_samples(["growth.speed"], Normal(0, 1); n_samples = 0)
end

@testset "A posterior predicts with the solver it was fitted with" begin
    # This length law cannot run under the MoM default: a silent solver change would throw.
    length_model = CrystallisationModel(fixture_system; nucleation = fixture_no_nucleation,
        growth = CallableLengthGrowth((rate_values, rate_context, crystal_length) -> rate_values.speed * 1e-6;
            parameters = (speed = 0.01,)))
    length_prepared = prepare_fit(length_model, [fitpredict_experiment], BayesianSpec(
        priors = (growth = (speed = Uniform(0.001, 0.03),),)); solver = DQMOM(nquadrature = 3))
    length_posterior = CriSTool.InferenceResult(length_model, ParameterSamples(growth = (speed = [0.01, 0.02],)),
        nothing, length_prepared)
    @test observable_values(predict(length_posterior; observables = (:concentration,)).ensembles[1], :concentration) ≈
          [fixture_translation_concentration(prediction_time; speed = draw_speed)
           for draw_speed in (0.01, 0.02), prediction_time in fitpredict_times] atol = 1e-7
end

@testset "Chain conversion keeps parameter, iteration and chain order" begin
    labelled_chain = Chains(reshape(collect(1.0:30.0), 3, 5, 2),
        [:ln_nucleation_prefactor, :surface_energy, :growth_coefficient, :growth_order, :lp],
        Dict(:internals => [:lp]))
    @test CriSTool.chains_to_matrix(labelled_chain; burnin = 0) == [1.0 2.0 3.0 16.0 17.0 18.0;
        4.0 5.0 6.0 19.0 20.0 21.0; 7.0 8.0 9.0 22.0 23.0 24.0; 10.0 11.0 12.0 25.0 26.0 27.0]
    @test CriSTool.chains_to_matrix(labelled_chain; burnin = 2, params = [:growth_order]) == [12.0 27.0]
    @test_throws ArgumentError CriSTool.chains_to_matrix(labelled_chain; burnin = -1)
    @test_throws ArgumentError CriSTool.chains_to_matrix(labelled_chain; burnin = 3)
    renamed_chain = rename_chain(Chains(reshape(collect(1.0:12.0), 3, 2, 2), [:θ1, :θ2]), [:speed, :order])
    @test names(renamed_chain, :parameters) == [:speed, :order]
    @test renamed_chain.value.data == reshape(collect(1.0:12.0), 3, 2, 2)
    @test_throws ArgumentError rename_chain(renamed_chain, [:speed])
end
