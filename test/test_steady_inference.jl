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
