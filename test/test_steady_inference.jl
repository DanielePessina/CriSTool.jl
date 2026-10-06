@testset "One-point steady preparation and named predictions" begin
    prepared_steady_problem = steady_state_test_problem(nucleation_rate = 3e5,
        growth_rate = 2e-7, feed_concentration = 1.0, initial_concentration_value = 8.0,
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 8.0])
    prepared_steady_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [300.0], mean = [1.2], variance = 0.01),
            d43 = Observable(time = [300.0], mean = [9e-6], variance = 1e-12)),
        temperature = 300.0, exp_id = 801)
    prepared_steady_options = (; relaxation_horizon = 1_000.0, autonomous = true)
    prepared_steady_setup = prepare_loss([prepared_steady_problem], [prepared_steady_experiment];
        mode = :steady, steady_options = prepared_steady_options)
    @test initial_concentration(prepared_steady_setup.prepared[1].problem) == 8.0
    @test prepared_steady_setup.prepared[1].saveat == [300.0]
    @test prepared_steady_setup.included_observations[1][:concentration] == [1]
    @test prepared_steady_setup.prepared[1].odeproblem.tspan == (0.0, 1_000.0)
    default_horizon_setup = prepare_loss([prepared_steady_problem], [prepared_steady_experiment];
        mode = :steady, steady_options = (; autonomous = true))
    # V=2 m³ and Q=0.2 m³/s give tau=10 s and the shared 100-tau budget.
    @test only(default_horizon_setup.prepared).odeproblem.tspan == (0.0, 1_000.0)
    @test loss(logMLE(), default_horizon_setup, [3e5, 2e-7]) ≈
        -11.780215388611698 atol = 1e-6
    @test_throws ArgumentError prepare_loss([CriSTool._copy_crystallisation_problem(
        prepared_steady_problem; operation = BatchOperation())], [prepared_steady_experiment];
        mode = :steady, steady_options = (; autonomous = true))
    # Frozen Gaussian NLL for C*=0.9999998402032, d43*=8e-6 and the targets above.
    @test loss(logMLE(), prepared_steady_setup, [3e5, 2e-7]) ≈ -11.780215388611698 atol = 1e-6
    steady_samples = [3e5 3e5; 2e-7 2e-7]
    steady_predictions = run_ensemble(steady_samples, prepared_steady_setup; verbosity = 0)
    @test steady_predictions[1].time == [300.0]
    @test observable_values(steady_predictions[1], :d43) ≈ fill(8e-6, 2, 1) rtol = 2e-6
    @test observable_values(steady_predictions[1], :volume) == fill(2.0, 2, 1)
    @test steady_predictions[1].success == [true, true]
    @test steady_predictions[1].diagnostics[1].initial_conditions.concentration == 8.0
    @test steady_predictions[1].diagnostics[1].convergence.balance_valid
    @test prediction_summary(steady_predictions[1], :concentration).mean ≈ [0.9999998402032] atol = 1e-8
    growth_objective = growth_rate_value -> loss(logMLE(), prepared_steady_setup, [3e5, growth_rate_value])
    steady_loss_ad = ForwardDiff.derivative(growth_objective, 2e-7)
    steady_loss_fd = FiniteDifferences.central_fdm(5, 1; max_range = 1e-8)(growth_objective, 2e-7)
    @test steady_loss_ad ≈ -39_999_952.0609217 rtol = 2e-4
    @test steady_loss_fd ≈ steady_loss_ad rtol = 2e-4

    template_steady_setup = prepare_loss(prepared_steady_problem, [prepared_steady_experiment];
        mode = :steady, steady_options = prepared_steady_options)
    @test initial_concentration(template_steady_setup.prepared[1].problem) == 8.0
    @test temperature(template_steady_setup.prepared[1].problem.temp_profile, 0.0) == 300.0
    exhausted_steady_setup = prepare_loss([prepared_steady_problem], [prepared_steady_experiment];
        mode = :steady, steady_options = (; relaxation_horizon = 0.01, autonomous = true))
    @test loss(mae(), exhausted_steady_setup, [3e5, 2e-7]) == 2_000_000.0
    exhausted_predictions = run_ensemble(reshape([3e5, 2e-7], 2, 1), exhausted_steady_setup; verbosity = 0)
    @test !exhausted_predictions[1].success[1]
    @test !exhausted_predictions[1].diagnostics[1].convergence.converged
    @test_throws ArgumentError prediction_summary(exhausted_predictions[1], :d43)
end

@testset "Steady reconstruction failures remain candidate failures" begin
    lower_boundary_problem = CriSTool._copy_crystallisation_problem(
        steady_state_test_problem(nucleation_rate = 0.0, growth_rate = -2e-7);
        solver = DQMOM(nquadrature = 2, coordinate_scale = 1e-6,
            weight_scale = 1e12, minimum_size = 1e-6, reltol = 1e-10, abstol = 1e-12),
        initial_state = [1e12, 1e12, 2e-6, 3e-6, 8.0])
    lower_boundary_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [6.0], mean = [2.0], variance = 0.01)),
        temperature = 298.15, exp_id = 802)
    lower_boundary_setup = prepare_loss([lower_boundary_problem], [lower_boundary_experiment];
        mode = :steady, steady_options = (; relaxation_horizon = 6.0, autonomous = true))
    lower_boundary_parameters = [0.0, -2e-7]
    # Equal constant dissolution moves the two nodes from [2,3] to [0.8,1.8]
    # micrometres in six seconds. The first is outside the declared support;
    # the ODE is finite, but a physical DQMOM result cannot be reconstructed.
    @test_throws DomainError solve_steadystate(lower_boundary_problem;
        relaxation_horizon = 6.0, autonomous = true)
    @test loss(mae(), lower_boundary_setup, lower_boundary_parameters) == 1_000_000.0
    failed_candidate = CriSTool._solve_prepared(
        only(lower_boundary_setup.prepared), lower_boundary_parameters)
    @test !failed_candidate.success
    @test failed_candidate.reason isa DomainError
    @test failed_candidate.reason.val ≈ 0.8e-6 atol = 1e-15
    failed_prediction = only(run_ensemble(reshape(lower_boundary_parameters, 2, 1),
        lower_boundary_setup; verbosity = 0))
    @test failed_prediction.success == [false]
    @test failed_prediction.diagnostics[1].reason isa DomainError

    callback_error_setup = prepare_loss([steady_state_test_problem()], [lower_boundary_experiment];
        mode = :steady, steady_options = (; relaxation_horizon = 6.0, autonomous = true),
        callback_factory = callback_problem -> throw(DomainError(0.0, "user callback failure")))
    @test_throws DomainError loss(mae(), callback_error_setup, [0.0, 0.0])
end

@testset "Steady dissolution saves the equilibrium after extinction" begin
    extinct_seed_moments = [1e12 * (1e-6)^moment_order for moment_order in 0:4]
    extinction_problem = CrystallisationProblem(; solver = MoM(reltol = 1e-10, abstol = 1e-24),
        initial_concentration = 0.1, initial_solvent_state = (; concentration = 0.1),
        initial_state = vcat(extinct_seed_moments, 0.1),
        saturation_model = ConstantSolubility(1.0),
        kinetics_growthfunction = growth_dissolution(), parameterset_growth = [1e-6, 0.0, 1.0],
        operation = MSMPROperation(volume = 2.0, inflow = 0.2,
            feed = CrystallisationFeed(concentration = 0.1)))
    extinction_equilibrium = solve_steadystate(extinction_problem; relaxation_horizon = 1000.0)
    @test extinction_equilibrium.success
    @test length(extinction_equilibrium.time) == 1
    @test only(extinction_equilibrium.concentration) ≈ 0.1 atol = 2e-11
    @test extinction_equilibrium.final_state[1:5] == zeros(5)
    @test extinction_equilibrium.product_solid_mass_flow == 0.0
    @test extinction_equilibrium.product_dissolved_solute_flow ≈ 0.02 atol = 4e-12
    extinction_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [300.0], mean = [0.1], variance = 0.01)),
        temperature = 300.0, exp_id = 821)
    extinction_setup = prepare_loss([extinction_problem], [extinction_experiment];
        mode = :steady, steady_options = (; relaxation_horizon = 1000.0))
    extinction_parameters = vcat(extinction_problem.parameterset_nucleation, [1e-6, 0.0, 1.0])
    @test loss(mae(), extinction_setup, extinction_parameters) ≈ 0.0 atol = 2e-11
    extinction_predictions = only(run_ensemble(reshape(extinction_parameters, :, 1),
        extinction_setup; verbosity = 0))
    @test extinction_predictions.success == [true]
    @test extinction_predictions.concentration ≈ fill(0.1, 1, 1) atol = 2e-11
end
