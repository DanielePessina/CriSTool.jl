# Gradient harness. Every case compares ForwardDiff (through DifferentiationInterface)
# against an independent reference: the derivative of a closed-form trajectory, an
# exact hand value, or a five-point central finite difference of the same function.
# Each distinct differentiation path (solver, operation, steady relaxation, loss,
# forward sensitivity, rate kernels) appears once.

# One shared ForwardDiff tag: the Dual type then depends only on the chunk size,
# so solver code compiled for one objective is reused by the next.
struct GradientHarnessTag end
const gradient_forward_backend = DI.AutoForwardDiff(; tag = GradientHarnessTag())
const gradient_difference_backend = DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1))

# Lysozyme kinetics in O(1) coordinates so finite-difference steps are well scaled.
gradient_lysozyme_values(scaled_values) =
    [scaled_values[1], scaled_values[2] * 1e-3, scaled_values[3] * 1e-9 / 60, scaled_values[4]]
const gradient_lysozyme_scaled_point = [38.0, 0.7, 1.0, 3.0]

@testset "Fit loss gradient equals the closed-form chain rule" begin
    # Synthetic concentrations from the true speed 0.01; the gradient is taken
    # away from the optimum so it is not trivially zero.
    gradient_times = [0.0, 300.0, 1000.0]
    gradient_observed = fixture_translation_concentration.(gradient_times)
    gradient_experiment = fixture_experiment((time = gradient_times, concentration = gradient_observed),
        fixture_seeded_conditions(); observables = (:concentration,))
    gradient_prepared = prepare_fit(fixture_translation_model(), [gradient_experiment],
        OptimisationSpec(bounds = (growth = (speed = (0.001, 0.05),),), algorithm = nothing))
    gradient_speed = 0.015
    # d/ds sum (C(t; s) - y)^2 / (2 var), differentiating the analytic trajectory.
    closed_form_gradient = sum(gradient_times .|> elapsed_time ->
        (fixture_translation_concentration(elapsed_time; speed = gradient_speed) -
         fixture_translation_concentration(elapsed_time)) / 0.01 *
        ForwardDiff.derivative(trial_speed -> fixture_translation_concentration(elapsed_time; speed = trial_speed),
            gradient_speed))
    forward_gradient = DI.gradient(trial -> loss(gradient_prepared, trial), gradient_forward_backend, [gradient_speed])
    difference_gradient = DI.gradient(trial -> loss(gradient_prepared, trial), gradient_difference_backend, [gradient_speed])
    @test only(forward_gradient) ≈ closed_form_gradient rtol = 1e-5
    @test forward_gradient ≈ difference_gradient rtol = 1e-6
end

# Nonlinear MoM case: nucleation and growth both depend on supersaturation, and the
# selected parameters are not contiguous in the model's parameter vector.
@testset "Non-contiguous selection gradient (supersaturation-dependent kinetics)" begin
    noncontiguous_model = CrystallisationModel(fixture_system;
        nucleation = CallableNucleation((rate_values, rate_context) -> rate_values.amplitude * 1e9;
            parameters = (amplitude = 0.1,)),
        growth = CallableGrowth((rate_values, rate_context) ->
            rate_values.speed * 1e-6 * supersaturation(rate_context)^rate_values.order;
            parameters = (speed = 0.001, order = 2.0)))
    noncontiguous_experiment = fixture_experiment(
        (time = [0.0, 100.0], concentration = [20.0, 19.0]), fixture_seeded_conditions();
        observables = (:concentration,))
    noncontiguous_prepared = prepare_fit(noncontiguous_model, [noncontiguous_experiment],
        OptimisationSpec(bounds = (nucleation = (amplitude = (0.01, 0.5),), growth = (order = (1.0, 3.0),)),
            algorithm = nothing))
    noncontiguous_objective = trial -> loss(noncontiguous_prepared, trial)
    @test DI.gradient(noncontiguous_objective, gradient_forward_backend, [0.2, 2.5]) ≈
          DI.gradient(noncontiguous_objective, gradient_difference_backend, [0.2, 2.5]) rtol = 1e-4 atol = 1e-8
end

@testset "FiniteVol (30 cells) final concentration gradient" begin
    finite_volume_problem = CrystallisationProblem(fixture_lysozyme_model, fixture_lysozyme_conditions;
        solver = FiniteVol(meshsize = 30, lmax = 50e-6))
    finite_volume_objective = scaled_values -> last(runsimulation(gradient_lysozyme_values(scaled_values),
        finite_volume_problem; save_idx = [0.0, 7200.0, 14400.0])).concentration[end]
    @test DI.gradient(finite_volume_objective, gradient_forward_backend, gradient_lysozyme_scaled_point) ≈
          DI.gradient(finite_volume_objective, gradient_difference_backend, gradient_lysozyme_scaled_point) rtol = 1e-3 atol = 1e-8
end

@testset "Forward sensitivity (MoM) equals finite differences" begin
    sensitivity_problem = CrystallisationProblem(fixture_lysozyme_model,
        RunConditions(temperature = 290.15, concentration = 14.67))
    sensitivity_times = [0.0, 3600.0, 7200.0, 10800.0]
    _, sensitivity_columns = CriSTool.extract_local_sensitivities(
        CriSTool.forwardsensitivity(sensitivity_problem, sensitivity_times))
    sensitivity_values = collect(sensitivity_problem.parameterset_nucleation)
    sensitivity_values = vcat(sensitivity_values, collect(sensitivity_problem.parameterset_growth))
    final_concentration_of(kinetic_values) =
        last(runsimulation(kinetic_values, sensitivity_problem; save_idx = sensitivity_times)).concentration[end]
    # Only the O(1) parameters (ln prefactor, growth order) are differenced directly.
    for parameter_index in (1, 4)
        difference_sensitivity = FiniteDifferences.central_fdm(5, 1)(trial_value ->
            final_concentration_of(setindex!(copy(sensitivity_values), trial_value, parameter_index)),
            sensitivity_values[parameter_index])
        # State 6 is the concentration in the MoM layout (5 moments, then C).
        @test sensitivity_columns[parameter_index][6, end] ≈ difference_sensitivity rtol = 1e-4
    end
end

@testset "Fed-batch translation: d C / d speed equals the inventory closed form" begin
    # With no outflow every seed crystal stays: C V = C0 V0 + F Cf t − V0 m0 (E[(L+Gt)^3]/E[L^3] − 1).
    fed_batch_volume, fed_batch_inflow, fed_batch_time = 2.0, 0.002, 1000.0
    fed_batch_problem = CrystallisationProblem(fixture_translation_model(),
        fixture_seeded_conditions(operation = fixture_fed_batch(inflow = fed_batch_inflow));
        solver = MoM(reltol = 1e-12, abstol = 1e-20))
    fed_batch_closed_form(trial_speed) = (20.0 * fed_batch_volume + fed_batch_inflow * 2.0 * fed_batch_time -
        fed_batch_volume * fixture_seed_mass * (fixture_uniform_moment(3, trial_speed * 1e-6 * fed_batch_time) /
        fixture_uniform_moment(3) - 1)) / (fed_batch_volume + fed_batch_inflow * fed_batch_time)
    fed_batch_concentration(trial_speed) = last(runsimulation([trial_speed], fed_batch_problem;
        save_idx = [0.0, fed_batch_time])).concentration[end]
    @test fed_batch_concentration(0.01) ≈ fed_batch_closed_form(0.01) rtol = 1e-9
    @test DI.derivative(fed_batch_concentration, gradient_forward_backend, 0.01) ≈
          ForwardDiff.derivative(fed_batch_closed_form, 0.01) rtol = 1e-7
end

@testset "DQMOM under MSMPR flow: gradient equals the washout closed form" begin
    # Clear feed Cf, residence time tau: C = Cf + (C0 − Cf + m0 − m0 E[(L+Gt)^3]/E[L^3]) e^{−t/tau}.
    msmpr_time = 4.0
    msmpr_problem = CrystallisationProblem(fixture_translation_model(),
        fixture_seeded_conditions(operation = fixture_msmpr());
        solver = DQMOM(nquadrature = 3, coordinate_scale = 1e-6, weight_scale = 1e12, reltol = 1e-10, abstol = 1e-12))
    msmpr_residence = 2.0 / 0.2
    msmpr_closed_form(trial_speed) = 2.0 + (20.0 - 2.0 - fixture_seed_mass *
        (fixture_uniform_moment(3, trial_speed * 1e-6 * msmpr_time) / fixture_uniform_moment(3) - 1)) *
        exp(-msmpr_time / msmpr_residence)
    msmpr_concentration(trial_values) = last(runsimulation(trial_values, msmpr_problem;
        save_idx = [0.0, msmpr_time])).concentration[end]
    @test msmpr_concentration([1.0]) ≈ msmpr_closed_form(1.0) rtol = 1e-7
    @test only(DI.gradient(msmpr_concentration, gradient_forward_backend, [1.0])) ≈
          ForwardDiff.derivative(msmpr_closed_form, 1.0) rtol = 1e-5
end

@testset "WENO (30 cells) seeded transport gradient" begin
    weno_problem = CrystallisationProblem(fixture_translation_model(), fixture_seeded_conditions();
        solver = WENO(meshsize = 30, lmax = 40e-6, reltol = 1e-6, abstol = 1e-10))
    weno_time = 20.0
    weno_concentration(trial_values) = last(runsimulation(trial_values, weno_problem;
        save_idx = [0.0, weno_time])).concentration[end]
    weno_forward = DI.gradient(weno_concentration, gradient_forward_backend, [0.1])
    @test weno_forward ≈ DI.gradient(weno_concentration, gradient_difference_backend, [0.1]) rtol = 2e-5
    # The 30-cell discretisation only approximates the moving seed, so the
    # analytic derivative is a loose physical check.
    @test only(weno_forward) ≈ ForwardDiff.derivative(trial_speed ->
        fixture_translation_concentration(weno_time; speed = trial_speed), 0.1) rtol = 0.05
end

@testset "Steady MSMPR: d d43 / dG = 4 tau exactly" begin
    # Constant B and G give n(L) = B/G exp(−L/(G tau)), so d43 = 4 G tau and the
    # logMLE gradient on one d43 datum is (4 G tau − y)/var · 4 tau · 1e-6 per unit speed.
    steady_model = fixture_translation_model(speed = 0.2, nucleation = fixture_constant_nucleation(birth_rate = 3e5))
    steady_residence = 10.0
    steady_observed_d43, steady_variance = 7e-6, 1e-12
    steady_experiment = CrystallisationExperiment(observables = (d43 = Observable(time = [0.0],
            mean = [steady_observed_d43], variance = [steady_variance]),),
        conditions = RunConditions(temperature = 298.15, concentration = 1.0,
            operation = fixture_msmpr(feed = fixture_clear_feed(concentration = 1.0))), exp_id = 1)
    steady_prepared = prepare_fit(steady_model, [steady_experiment],
        OptimisationSpec(bounds = (growth = (speed = (0.01, 1.0),),), algorithm = nothing);
        solver = MoM(reltol = 1e-10, abstol = 1e-24), mode = SteadyMode(SteadyStateOptions(autonomous = true, relaxation_horizon = 1000.0)))
    steady_speed = 0.2
    steady_d43 = 4 * steady_speed * 1e-6 * steady_residence
    steady_gradient = DI.gradient(trial -> loss(steady_prepared, trial), gradient_forward_backend, [steady_speed])
    @test only(steady_gradient) ≈ (steady_d43 - steady_observed_d43) / steady_variance * 4e-6 * steady_residence rtol = 2e-4
    @test steady_gradient ≈ DI.gradient(trial -> loss(steady_prepared, trial),
        DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1; max_range = 1e-3)), [steady_speed]) rtol = 2e-4
end

@testset "Rate-kernel gradients (length growth, binary moment sources)" begin
    kernel_mesh = [0.5e-6, 1.0e-6, 2.0e-6]
    kernel_problem = CrystallisationProblem(kinetics_growthfunction = growth_empirical_length(Lref = 1e-6),
        saturation_model = ConstantSolubility(10.0), initial_concentration = 10.0,
        solver = FiniteVol(meshsize = length(kernel_mesh), lmax = 3e-6))
    kernel_state = [zeros(length(kernel_mesh)); 15.0]
    function length_growth_objective(growth_values)
        growth_destination = similar(kernel_mesh, eltype(growth_values))
        CriSTool.growthrate!(growth_destination, growth_empirical_length(Lref = 1e-6), growth_values,
            kernel_problem, kernel_state, 0.0, kernel_mesh)
        return sum([1.0, 2.0, 4.0] .* growth_destination) * 1e8
    end
    length_growth_point = [3e-8, 1.4, 0.45, 1.3]
    @test DI.gradient(length_growth_objective, gradient_forward_backend, length_growth_point) ≈
          DI.gradient(length_growth_objective, gradient_difference_backend, length_growth_point) rtol = 2e-4 atol = 2e-5

    kernel_rule = QMOMQuadrature([0.5e-6, 1.5e-6], [2.0, 3.0])
    aggregation_objective(kernel_values) = aggregation_moment_source(aggr_scalar(), kernel_values, kernel_rule, 3)[1]
    breakage_objective(kernel_values) = breakage_moment_source(breakage_empirical(), kernel_values, kernel_rule, 3)[1]
    @test DI.gradient(aggregation_objective, gradient_forward_backend, [0.0]) ≈
          DI.gradient(aggregation_objective, gradient_difference_backend, [0.0]) rtol = 1e-4
    @test DI.gradient(breakage_objective, gradient_forward_backend, [0.0, 1.0]) ≈
          DI.gradient(breakage_objective, gradient_difference_backend, [0.0, 1.0]) rtol = 1e-4
end
