# End-to-end model -> EstimationProblem -> solve -> predict on the translation fixture.
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
const fitpredict_model = CrystallisationModel(fixture_system;
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
    target_sampler.received_reference_loss[] = reference_loss()
    return (P = [0.4 0.6],)
end

# Draws a known sequence of unit uniforms for inverse-CDF sampling checks.
mutable struct FitPredictSequenceRNG <: Random.AbstractRNG
    unit_draws::Vector{Float64}
    draw_index::Int
end
Random.rand(sequence_rng::FitPredictSequenceRNG) =
    (sequence_rng.draw_index += 1; sequence_rng.unit_draws[sequence_rng.draw_index - 1])

@testset "Estimation problem: hand-computed likelihood, construction rules and remake" begin
    fitpredict_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment]; bounds = fitpredict_bounds)
    @test fitpredict_estimation.selection.names == ["growth.speed"]
    # At the truth every residual vanishes: 3 scored concentrations (t = 0 is the
    # initial condition) and 4 scored d43 values contribute only their normalisers.
    truth_terms = loss_terms(fitpredict_estimation, [0.01])
    @test keys(truth_terms) == (:concentration, :d43)
    @test collect(values(truth_terms)) ≈
          [3 * fitpredict_gaussian_nll(0.0, 0.01), 4 * fitpredict_gaussian_nll(0.0, 1e-14)] rtol = 1e-9
    # Away from the truth the residuals are the closed-form trajectory differences.
    trial_residuals = fixture_translation_concentration.(fitpredict_times[2:end]; speed = 0.02) .-
                      fitpredict_trajectory.concentration[2:end]
    trial_d43_residuals = fixture_translation_d43.(fitpredict_times; speed = 0.02) .- fitpredict_trajectory.d43
    @test loss(fitpredict_estimation, [0.02]) ≈ sum(fitpredict_gaussian_nll.(trial_residuals, 0.01)) +
          sum(fitpredict_gaussian_nll.(trial_d43_residuals, 1e-14)) rtol = 1e-6

    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        bounds = (system = (crystal_density = (900.0, 1100.0),),))
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        bounds = (growth = (speed = (0.03, 0.001),),))
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        bounds = (growth = (unknown_rate = (0.0, 1.0),),))
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        priors = (growth = (speed = (0.001, 0.03),),))
    # A selection is required; bounds and priors together must select the same names.
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment])
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        bounds = fitpredict_bounds, priors = (growth = (held_order = Uniform(1.0, 3.0),),))
    @test EstimationProblem(fitpredict_model, [fitpredict_experiment]; bounds = fitpredict_bounds,
        priors = (growth = (speed = Uniform(0.001, 0.03),),)).selection.names == ["growth.speed"]
    # The time-zero start is not a problem option.
    @test_throws MethodError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        bounds = fitpredict_bounds, initial_time = 10.0)
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [fitpredict_experiment];
        bounds = fitpredict_bounds, problem_options = (initial_state = zeros(5),))
    # Initial concentration from observations requires a t = 0 observation.
    late_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [10.0, 100.0], mean = [20.0, 19.9], variance = [0.01, 0.01]),),
        conditions = RunConditions(temperature = 293.15), exp_id = 32)
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [late_experiment]; bounds = fitpredict_bounds)
    # A likelihood needs supplied variances; a discrepancy does not.
    no_variance_experiment = CrystallisationExperiment(observables = (concentration = Observable(
        time = [0.0, 100.0], mean = [20.0, 19.9]),), conditions = RunConditions(temperature = 293.15), exp_id = 35)
    @test_throws ArgumentError EstimationProblem(fitpredict_model, [no_variance_experiment]; bounds = fitpredict_bounds)
    @test loss(EstimationProblem(fitpredict_model, [no_variance_experiment]; bounds = fitpredict_bounds,
        loss = MeanAbsoluteError()), [0.0]) ≈ 0.1 atol = 1e-12

    # remake replaces only the named keywords: the closed-form MAE at speed 0.02
    # pools each observable's absolute residuals (3 concentrations, 4 d43 values).
    mae_estimation = remake(fitpredict_estimation; loss = MeanAbsoluteError())
    @test mae_estimation.selection.names == fitpredict_estimation.selection.names
    @test mae_estimation.setup.experiments == fitpredict_estimation.setup.experiments
    @test loss(mae_estimation, [0.02]) ≈ sum(abs, trial_residuals) / 3 + sum(abs, trial_d43_residuals) / 4 rtol = 1e-6
    @test loss(remake(mae_estimation; loss = GaussianNLL()), [0.02]) == loss(fitpredict_estimation, [0.02])
    @test_throws ArgumentError remake(fitpredict_estimation; bounds = nothing)
    # Optimisers need bounds and samplers need priors.
    prior_only_estimation = remake(fitpredict_estimation; bounds = nothing,
        priors = (growth = (speed = Uniform(0.001, 0.03),),))
    @test_throws ArgumentError solve(prior_only_estimation, DE())
    @test_throws ArgumentError solve(fitpredict_estimation, ABCDE(); target = 1.0)
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
    pooled_estimation(pooled_loss, pooled_runs; kwargs...) = EstimationProblem(pooled_model, pooled_runs;
        bounds = (growth = (speed = (0.0, 1.0),),), loss = pooled_loss, kwargs...)
    pooled_loss_value(pooled_loss, pooled_runs; kwargs...) =
        loss(pooled_estimation(pooled_loss, pooled_runs; kwargs...), [0.0])
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
    uneven_estimation = pooled_estimation(MeanAbsoluteError(), uneven_runs)
    @test loss_terms(MeanAbsoluteError(), uneven_estimation.setup, [0.0]) == (concentration = 1.0, pH = 3.0)
    ordered_estimation = pooled_estimation(MeanAbsoluteError(weighting = [3.0, 2.0]), uneven_runs;
        observable_order = [:pH, :concentration])
    @test loss_terms(MeanAbsoluteError(weighting = [3.0, 2.0]), ordered_estimation.setup, [0.0]) == (pH = 9.0, concentration = 2.0)

    likelihood_runs = [pooled_run(1, (concentration = Observable(time = [0.0, 1.0], mean = [10.0, 11.0], variance = 1.0),
            pH = Observable(time = [0.0, 1.0], mean = [8.0, 8.0], variance = 4.0))),
        pooled_run(2, (concentration = Observable(time = [0.0, 1.0], mean = [10.0, 11.0], variance = 1.0),))]
    likelihood_estimation = pooled_estimation(GaussianNLL(), likelihood_runs)
    likelihood_terms = loss_terms(GaussianNLL(), likelihood_estimation.setup, [0.0])
    @test keys(likelihood_terms) == (:concentration, :pH)
    @test collect(values(likelihood_terms)) ≈ [2 * fitpredict_gaussian_nll(1.0, 1.0), 2 * fitpredict_gaussian_nll(1.0, 4.0)]

    # A local projection scores a derived signal: 2C matches, 3C misses by 20.
    signal_run = pooled_run(3, (concentration = concentration_series,
        optical_signal = Observable(time = [0.0, 1.0], mean = [20.0, 20.0])))
    signal_terms(signal_scale) = loss_terms(MeanAbsoluteError(), pooled_estimation(MeanAbsoluteError(), [signal_run];
        observable_projections = (optical_signal = physical_solution -> signal_scale .* physical_solution.concentration,)).setup, [0.0])
    @test signal_terms(2.0).optical_signal == 0.0
    @test signal_terms(3.0).optical_signal ≈ 10.0
    @test_throws ArgumentError pooled_estimation(MeanAbsoluteError(), [signal_run];
        observable_projections = (concentration = physical_solution -> physical_solution.concentration,))
end

@testset "Gradient fit (LBFGS) recovers the true speed and predicts the closed form" begin
    lbfgs_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment]; bounds = fitpredict_bounds)
    # Keywords other than adtype and parallel reach the Optimization.jl solve.
    callback_count = Ref(0)
    lbfgs_fit = solve(lbfgs_estimation, OptimizationOptimJL.LBFGS(); maxiters = 50,
        callback = (optimisation_state, current_loss) -> (callback_count[] += 1; false))
    @test callback_count[] > 0
    @test lbfgs_fit.alg isa OptimizationOptimJL.Optim.LBFGS
    fitted_speed = lbfgs_fit.parameters.growth.speed
    @test fitted_speed ≈ fitpredict_true_speed rtol = 1e-4
    @test lbfgs_fit.model.growth.parameters == (speed = fitted_speed, held_order = 2.0)
    @test fitpredict_model.growth.parameters.speed == fitpredict_true_speed
    @test sum(values(lbfgs_fit.objectives)) ≈ loss(lbfgs_estimation, [fitted_speed]) rtol = 1e-12
    # predict(fit) uses the fitted draw, experiments, solver and observation times.
    fitted_prediction = predict(lbfgs_fit)
    @test fitted_prediction.experiments == [31]
    @test only(fitted_prediction.ensembles).t == fitpredict_times
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
    de_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment]; bounds = fitpredict_bounds)
    direct_de = deepcopy(supplied_de)
    direct_de.status = Metaheuristics.State(nothing, [])
    direct_result = Metaheuristics.optimize(trial -> loss(de_estimation, trial),
        Metaheuristics.boxconstraints([0.001], [0.03]), direct_de)
    de_fit = solve(de_estimation, supplied_de)
    @test de_fit.alg === supplied_de
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
    pareto_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment]; bounds = fitpredict_bounds,
        loss = MeanAbsoluteError())
    pareto_results = solve(pareto_estimation,
        Metaheuristics.NSGA2(N = 12, options = Metaheuristics.Options(iterations = 3, seed = 42)); parallel = EnsembleThreads())
    pareto_objectives = [collect(values(pareto_result.objectives)) for pareto_result in pareto_results]
    @test all(pareto_result -> 0.001 <= pareto_result.parameters.growth.speed <= 0.03, pareto_results)
    @test all(pareto_result -> pareto_result.model.growth.parameters.held_order == 2.0, pareto_results)
    dominates(first_objectives, second_objectives) =
        all(first_objectives .<= second_objectives) && any(first_objectives .< second_objectives)
    @test !any(dominates(first_objectives, second_objectives)
               for first_objectives in pareto_objectives, second_objectives in pareto_objectives)
    # Each candidate's named objectives are the serial loss at its parameters.
    @test all(pareto_result -> sum(values(pareto_result.objectives)) ≈
                               loss(pareto_estimation, [pareto_result.parameters.growth.speed]), pareto_results)
    @test sort(first.(pareto_objectives)) ≈ sort(Metaheuristics.pareto_front(first(pareto_results).backend_result)[:, 1])
end

@testset "Bootstrap fits on noise-free data all recover the truth" begin
    bootstrap_fits = [solve(EstimationProblem(fitpredict_model, replicate_runs; bounds = fitpredict_bounds),
            OptimizationOptimJL.LBFGS(); maxiters = 50)
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
    hand_built_estimation = EstimationProblem(fitpredict_model, [hand_built_run]; bounds = fitpredict_bounds)
    @test hand_built_estimation.setup.included_observations[1][:seed_mass] == Int[]
    @test loss(hand_built_estimation, [0.02]) ≈ loss(EstimationProblem(fitpredict_model, table_runs; bounds = fitpredict_bounds), [0.02]) rtol = 1e-12
    @test loss(hand_built_estimation, [0.01]) ≈ 3 * fitpredict_gaussian_nll(0.0, 0.01) atol = 1e-6
end

@testset "Size objective: hand-computed d43 likelihood with offsets" begin
    # Offsets of 1 and 2 µm against variances 1e-12 and 4e-12, concentration unweighted.
    size_times = [300.0, 1000.0]
    size_offsets, size_variances = [1e-6, 2e-6], [1e-12, 4e-12]
    size_run = CrystallisationExperiment(exp_id = 5, conditions = fixture_seeded_conditions(),
        observables = (d43 = Observable(time = size_times, mean = fixture_translation_d43.(size_times) .+ size_offsets,
            variance = size_variances),))
    size_estimation = EstimationProblem(fitpredict_model, [size_run]; bounds = fitpredict_bounds,
        loss = GaussianNLL(weighting = (d43 = 1.0,)))
    @test loss(size_estimation, [0.01]) ≈ 0.5 * sum(log.(2π .* size_variances) .+ size_offsets .^ 2 ./ size_variances) rtol = 1e-6
end

@testset "Turing extension: solve contract and AIS guard" begin
    prior_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment];
        priors = (growth = (speed = Uniform(0.001, 0.03),),))
    @test Base.get_extension(CriSTool, :CriSToolTuringExt) !== nothing
    # `using CriSTool, Turing` must not make `predict` ambiguous: both export StatsAPI.predict.
    @test CriSTool.predict === Turing.predict
    # Posterior sampling needs priors, a likelihood and positive counts; nothing is sampled first.
    @test_throws ArgumentError solve(remake(prior_estimation; loss = MeanAbsoluteError()), Turing.NUTS())
    @test_throws ArgumentError solve(remake(prior_estimation; priors = nothing, bounds = fitpredict_bounds),
        Turing.NUTS())
    @test_throws ArgumentError solve(prior_estimation, Turing.NUTS(); n_samples = 0)
    @test_throws ArgumentError solve(prior_estimation, Turing.NUTS(); n_chains = 0)
    # The vendored AIS step must not dispatch ambiguously into DynamicPPL.
    @test_throws ErrorException Turing.AbstractMCMC.step(Random.default_rng(),
        CriSTool.nuts_model(prior_estimation), CriSTool.KissABC.AIS(25))
end

@testset "NUTS (2 threaded chains): posterior density and posterior predictions" begin
    bayesian_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment];
        priors = (growth = (speed = Uniform(0.001, 0.03),),))
    # log p(theta | data) = log prior - negative log likelihood.
    @test Turing.DynamicPPL.logjoint(CriSTool.nuts_model(bayesian_estimation), (θ = [0.015],)) ≈
          logpdf(Uniform(0.001, 0.03), 0.015) - loss(bayesian_estimation, [0.015]) rtol = 1e-10
    nuts_sampler = Turing.NUTS(5, 0.65)
    tiny_posterior = solve(bayesian_estimation, nuts_sampler; n_samples = 5, n_chains = 2, rng = MersenneTwister(72),
        parallel = EnsembleThreads(), progress = false)
    @test tiny_posterior.alg === nuts_sampler
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
    forwarded_result = solve(EstimationProblem(fitpredict_model, [fitpredict_experiment]; priors = abc_priors),
        target_sampler; target = 1.5)
    # The reference loss is the likelihood at the model's current (true) speed.
    @test received_reference_loss[] ≈ 3 * fitpredict_gaussian_nll(0.0, 0.01) + 4 * fitpredict_gaussian_nll(0.0, 1e-14) rtol = 1e-9
    @test received_target[] == 1.5
    @test forwarded_result.samples.values == [0.4 0.6]
    @test forwarded_result.samples.source == :abc

    abc_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment]; priors = abc_priors,
        loss = MeanAbsoluteError())
    @test_throws ArgumentError solve(abc_estimation, ABCDE(); target = Inf)
    @test_throws ArgumentError solve(abc_estimation, ABCDE(); target = 1.0, n_particles = 0)
    tiny_abc = solve(abc_estimation, ABCDE(); target = 1.0, n_particles = 8, n_generations = 1,
        parallel = EnsembleThreads())
    @test tiny_abc.alg === ABCDE()
    @test vec(tiny_abc.samples.values) == collect(tiny_abc.backend_result.P.particles)
    @test all(0.005 .<= tiny_abc.samples.values .<= 0.02)
    unreachable_result = CriSTool._runsampler(ABCDE(),
        product_distribution([Uniform(0.0, 1.0)]), candidate -> 100.0, 0.0, () -> 100.0;
        nparticles = 8, generations = 1, HPC = false, stopping = AllGenerations())
    @test !unreachable_result.reached_ϵ
    # One Particles per parameter becomes the parameters × samples draw matrix.
    @test CriSTool._abc_particles_matrix([CriSTool.MonteCarloMeasurements.Particles([1.0, 2.0, 3.0]),
        CriSTool.MonteCarloMeasurements.Particles([10.0, 20.0, 30.0])]) == [1.0 2.0 3.0; 10.0 20.0 30.0]

    @test CriSTool.KissABC.logsumexp_stable([1.0, 2.0, 3.0]) ≈ log(exp(1.0) + exp(2.0) + exp(3.0))
    @test CriSTool.KissABC.logsumexp_stable([-1000.0, -1000.0]) ≈ -1000.0 + log(2.0)
    @test CriSTool.KissABC.log_kernel_weight(GaussianABCKernel(), 2.0, 0.0, 1.0) ≈ -2.0
    @test CriSTool.KissABC.log_kernel_weight(LaplaceABCKernel(), 1.0, 0.0, 1.0) ≈ -1.0
    @test CriSTool.KissABC.log_kernel_weight(FlatABCKernel(), 100.0, 0.0, 1.0) == 0.0
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
    @test all(==(ReturnCode.MaxIters), getproperty.(failed_predictions.ensembles[1].diagnostics, :retcode))
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
    length_model = CrystallisationModel(fixture_system;
        growth = CallableLengthGrowth((rate_values, rate_context, crystal_length) -> rate_values.speed * 1e-6;
            parameters = (speed = 0.01,)))
    length_estimation = EstimationProblem(length_model, [fitpredict_experiment];
        priors = (growth = (speed = Uniform(0.001, 0.03),),), solver = DQMOM(nquadrature = 3))
    length_posterior = CriSTool.InferenceResult(length_model, ParameterSamples(growth = (speed = [0.01, 0.02],)),
        nothing, nothing, length_estimation)
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

# Experiments solved on threads must reproduce the serial run exactly: scores are
# reduced in experiment order, so equality is bitwise, not approximate.
@testset "parallel = EnsembleThreads(): bitwise identical to serial (loss, solve, predict)" begin
    parallel_times = [[0.0, 300.0, 600.0, 1000.0], [0.0, 200.0, 800.0], [0.0, 100.0, 500.0, 900.0, 1000.0]]
    parallel_experiments = [fixture_experiment((time = elapsed_times,
            concentration = fixture_translation_concentration.(elapsed_times; speed = trial_speed),
            d43 = fixture_translation_d43.(elapsed_times; speed = trial_speed)),
        fitpredict_conditions; exp_id = 40 + experiment_index)
        for (experiment_index, (elapsed_times, trial_speed)) in enumerate(zip(parallel_times, (0.009, 0.011, 0.01)))]
    serial_fit = EstimationProblem(fitpredict_model, parallel_experiments; bounds = fitpredict_bounds)
    threaded_fit = CriSTool._with_parallel(serial_fit, EnsembleThreads())
    @test serial_fit.setup.parallel isa EnsembleSerial
    @test threaded_fit.setup.parallel isa EnsembleThreads
    for trial_speed in (0.004, 0.0095, 0.02)
        @test loss(threaded_fit, [trial_speed]) === loss(serial_fit, [trial_speed])
        @test collect(values(loss_terms(threaded_fit, [trial_speed]))) ==
              collect(values(loss_terms(serial_fit, [trial_speed])))
    end
    # The loss is not trivially constant: each experiment contributes to it.
    @test loss(serial_fit, [0.004]) != loss(serial_fit, [0.02])
    # solve(...; parallel) threads the experiments of each loss and yields the same optimum.
    serial_result = solve(serial_fit, OptimizationOptimJL.LBFGS())
    threaded_result = solve(serial_fit, OptimizationOptimJL.LBFGS(); parallel = EnsembleThreads())
    @test threaded_result.estimation_problem.setup.parallel isa EnsembleThreads
    @test threaded_result.parameters == serial_result.parameters
    @test threaded_result.objectives == serial_result.objectives
    @test serial_fit.setup.parallel isa EnsembleSerial
    # Predictions: run x draw pairs across threads match the serial solve.
    prediction_draws = ParameterSamples(; growth = (speed = [0.008, 0.01, 0.012],))
    serial_prediction = predict(fitpredict_model, parallel_experiments, prediction_draws; solver = MoM())
    threaded_prediction = predict(fitpredict_model, parallel_experiments, prediction_draws; solver = MoM(),
        parallel = EnsembleThreads())
    @test length(serial_prediction.ensembles) == 3
    for (serial_ensemble, threaded_ensemble) in zip(serial_prediction.ensembles, threaded_prediction.ensembles)
        @test propertynames(threaded_ensemble) == propertynames(serial_ensemble)
        @test threaded_ensemble.predictions == serial_ensemble.predictions
    end
    # A user-code exception surfaces unwrapped, with the type the serial loss raises.
    raising_model = CrystallisationModel(fixture_system;
        growth = CallableGrowth((rate_values, rate_context) ->
            rate_values.speed < 0 ? throw(DomainError(rate_values.speed, "negative speed")) : rate_values.speed * 1e-6;
            parameters = (speed = fitpredict_true_speed,)))
    for raising_parallel in (EnsembleSerial(), EnsembleThreads())
        raising_fit = CriSTool._with_parallel(
            EstimationProblem(raising_model, parallel_experiments; bounds = fitpredict_bounds), raising_parallel)
        @test_throws DomainError loss(raising_fit, [-0.01])
    end
    @test_throws ArgumentError loss(CriSTool._with_parallel(serial_fit, CriSTool.SciMLBase.EnsembleDistributed()), [0.01])
    @test_throws ArgumentError solve(serial_fit, OptimizationOptimJL.LBFGS();
        parallel = CriSTool.SciMLBase.EnsembleDistributed())
end

# One parallel setting: the algorithm picks the level that receives it, the other runs serially.
@testset "parallel level: candidates or experiments, chosen by the algorithm" begin
    seeded_de() = Metaheuristics.DE(; N = 8, options = Metaheuristics.Options(; iterations = 3, seed = 314))
    expected_levels = [
        seeded_de() => CriSTool.ParallelCandidates(),
        Metaheuristics.NSGA2(N = 12) => CriSTool.ParallelCandidates(),
        OptimizationOptimJL.LBFGS() => CriSTool.ParallelExperiments(),
        ABCDE() => CriSTool.ParallelCandidates(),
    ]
    for (level_algorithm, expected_level) in expected_levels
        @test CriSTool._parallel_level(level_algorithm) === expected_level
    end
    # MCMC threads several chains, or the experiments of a single chain.
    @test CriSTool._chain_parallel_level(1) === CriSTool.ParallelExperiments()
    @test CriSTool._chain_parallel_level(4) === CriSTool.ParallelCandidates()
    # The chosen level gets the supplied ensemble; the loss of the other level is serial.
    candidate_estimation = EstimationProblem(fitpredict_model, [fitpredict_experiment]; bounds = fitpredict_bounds)
    candidate_ensemble, candidate_split = CriSTool._split_parallel(candidate_estimation,
        CriSTool._parallel_level(seeded_de()), EnsembleThreads())
    @test (candidate_ensemble, candidate_split.setup.parallel) === (EnsembleThreads(), EnsembleSerial())
    experiment_ensemble, experiment_split = CriSTool._split_parallel(candidate_estimation,
        CriSTool._parallel_level(OptimizationOptimJL.LBFGS()), EnsembleThreads())
    @test (experiment_ensemble, experiment_split.setup.parallel) === (EnsembleSerial(), EnsembleThreads())
    # A seeded DE run evaluates the same candidates serially and on threads.
    serial_de_fit = solve(candidate_estimation, seeded_de(); parallel = EnsembleSerial())
    threaded_de_fit = solve(candidate_estimation, seeded_de(); parallel = EnsembleThreads())
    @test threaded_de_fit.parameters == serial_de_fit.parameters
    @test threaded_de_fit.objectives == serial_de_fit.objectives
    # Unsupported ensembles are rejected at the candidate level as well.
    @test_throws ArgumentError solve(candidate_estimation, seeded_de(); parallel = CriSTool.SciMLBase.EnsembleDistributed())
end
