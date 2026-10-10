# Steady MSMPR end-to-end through the model API. The ideal MSMPR (constant
# birth rate B, constant growth G, clear feed, residence time tau) has the
# closed-form population n(L) = B tau / (G tau) exp(-L / (G tau)), so
#   m_k = B tau k! (G tau)^k,  d43 = 4 G tau,  C* = Cf - rho kv m3
# (crystallised solute leaves as solid). Uses runtests.jl imports and fixtures.jl
# (fixture_system: rho = 1000 kg/m³, kv = 1, constant solubility).

import OptimizationOptimJL

const steady_birth_rate = 5e11      # #/m³/s
const steady_growth_speed = 0.2     # fixture units: G = 2e-7 m/s
const steady_feed_concentration = 1.0
const steady_residence_time = 10.0  # fixture_msmpr: V = 2 m³, Q = 0.2 m³/s

steady_msmpr_model(; birth_rate = steady_birth_rate, speed = steady_growth_speed) =
    CrystallisationModel(fixture_system; nucleation = fixture_constant_nucleation(; birth_rate),
        growth = fixture_constant_growth(; speed))
steady_msmpr_conditions(; concentration = steady_feed_concentration,
        feed = fixture_clear_feed(concentration = steady_feed_concentration), crystals = nothing) =
    RunConditions(; temperature = 298.15, concentration, crystals, operation = fixture_msmpr(; feed))

steady_ideal_moment(moment_order; birth_rate = steady_birth_rate, speed = steady_growth_speed) =
    birth_rate * steady_residence_time * factorial(moment_order) *
    (speed * 1e-6 * steady_residence_time)^moment_order
steady_ideal_d43(; speed = steady_growth_speed) = 4 * speed * 1e-6 * steady_residence_time
steady_ideal_concentration(; birth_rate = steady_birth_rate, speed = steady_growth_speed) =
    steady_feed_concentration - 1000.0 * steady_ideal_moment(3; birth_rate, speed)

const steady_autonomous_options = SteadyStateOptions(autonomous = true)

@testset "Steady washout contract and failure modes" begin
    # No kinetics: the tank washes out to the feed. C* = 2 and the outlet carries Q Cf.
    washout_model = fixture_translation_model(speed = 0.0)
    washout_problem = CrystallisationProblem(washout_model,
        RunConditions(temperature = 298.15, concentration = 8.0, operation = fixture_msmpr()))
    washout = solve_steadystate(washout_problem;
        options = SteadyStateOptions(minimum_relaxation = 250.0, autonomous = true))
    @test washout.success
    @test equilibrium(washout).concentration ≈ 2.0 rtol = 1e-8
    @test observable_values(washout, :concentration) ≈ [2.0] rtol = 1e-8
    @test equilibrium(washout).time >= 250.0
    @test washout.hydraulics == (; volume = 2.0, inflow = 0.2, outflow = 0.2, residence_time = 10.0)
    @test washout.product.dissolved_solute_flow ≈ 0.4 rtol = 1e-8
    @test washout.product.solid_mass_flow == 0.0

    # Callable laws are opaque: the caller must declare them time invariant.
    @test_throws ArgumentError solve_steadystate(washout_problem)
    # Only a fixed-volume MSMPR has a steady state.
    for transient_operation in (BatchOperation(), fixture_fed_batch())
        @test_throws ArgumentError solve_steadystate(CrystallisationProblem(washout_model,
            RunConditions(temperature = 298.15, concentration = 8.0, operation = transient_operation));
            options = steady_autonomous_options)
    end
    @test_throws ArgumentError SteadyStateOptions(relaxation_horizon = 10.0, minimum_relaxation = 10.1)
    @test_throws ArgumentError SteadyStateOptions(residual_reltol = 0.0, residual_abstol = 0.0)
    @test_throws ArgumentError SteadyStateOptions(balance_reltol = -1.0)
    @test_throws ArgumentError SteadyStateOptions(residual_scales = [1.0, 0.0])
    # The default horizon is 100 residence times (1000 s here).
    @test_throws ArgumentError solve_steadystate(washout_problem;
        options = SteadyStateOptions(minimum_relaxation = 1_000.1, autonomous = true))

    # Starting at the feed concentration with an empty tank, positive nucleation
    # still leaves a population residual: a concentration-only stop rule would
    # accept this state. An exhausted horizon must report non-convergence.
    ideal_problem = CrystallisationProblem(steady_msmpr_model(), steady_msmpr_conditions())
    exhausted_error = try
        solve_steadystate(ideal_problem; options = SteadyStateOptions(relaxation_horizon = 0.1,
            residual_reltol = 1e-12, residual_abstol = 1e-14, autonomous = true))
        nothing
    catch caught_error
        caught_error
    end
    @test exhausted_error isa SteadyStateConvergenceError
    @test !exhausted_error.solution.success
    @test !exhausted_error.solution.diagnostics.converged

    # Saved user-callback events must not leak extra samples into the equilibrium.
    saving_callback_factory = ode_problem -> CriSTool.DiscreteCallback(
        (numerical_state, simulation_time, integrator) -> 0.5 <= simulation_time < 0.7,
        integrator -> nothing; save_positions = (true, true))
    equilibrium_start = CrystallisationProblem(washout_model,
        RunConditions(temperature = 298.15, concentration = 2.0, operation = fixture_msmpr()))
    event_equilibrium = solve_steadystate(equilibrium_start;
        options = SteadyStateOptions(minimum_relaxation = 2.0, relaxation_horizon = 5.0, autonomous = true),
        callback_factory = saving_callback_factory, solve_options = (; dt = 0.1, adaptive = false))
    @test length(event_equilibrium.solution.time) == 1
    @test equilibrium(event_equilibrium).time >= 2.0
    @test equilibrium(event_equilibrium).concentration == 2.0
end

@testset "Ideal MSMPR closed form across solver representations" begin
    # B = 5e11, G = 2e-7, tau = 10: m = [5e12, 1e7, 40, 2.4e-4, 1.92e-9, 1.92e-14],
    # solid 0.24 kg/m³, C* = 0.76, d43 = 8 µm. Outlet flows: solid Q rho kv m3 =
    # 0.048 kg/s, dissolved Q C* = 0.152 kg/s; together they return the 0.2 kg/s fed.
    expected_moments = steady_ideal_moment.(0:5)
    expected_concentration = steady_ideal_concentration()
    # A 40-cell mesh inserts nuclei and transports mass with an O(dL) to O(dL²)
    # flow-balance defect (~3e-3 for FiniteVol, ~4e-4 for WENO here; it shrinks
    # with refinement), so meshes get a matching balance_reltol and tolerance.
    for (steady_solver, solver_rtol, balance_reltol) in (
            (MoM(reltol = 1e-10, abstol = 1e-24), 2e-6, 1e-5), (QMOM(), 2e-6, 1e-5),
            (FiniteVol(meshsize = 40, lmax = 40e-6), 0.05, 5e-3), (WENO(meshsize = 40, lmax = 40e-6), 0.05, 5e-3))
        @testset "$(nameof(typeof(steady_solver)))" begin
            steady_result = solve_steadystate(
                CrystallisationProblem(steady_msmpr_model(), steady_msmpr_conditions(); solver = steady_solver);
                options = SteadyStateOptions(autonomous = true, residual_reltol = 1e-8, residual_abstol = 1e-10,
                    balance_reltol = balance_reltol))
            @test steady_result.success
            @test equilibrium(steady_result).d43 ≈ steady_ideal_d43() rtol = solver_rtol
            @test equilibrium(steady_result).concentration ≈ expected_concentration rtol = solver_rtol
            @test steady_result.product.solid_mass_flow ≈ 0.2 * 1000.0 * expected_moments[4] rtol = solver_rtol
            @test steady_result.product.dissolved_solute_flow ≈ 0.2 * expected_concentration rtol = solver_rtol
            balance = steady_result.diagnostics.total_api_balance
            # Total solute in = dissolved + solid out + solid leaving the size domain.
            @test steady_result.product.dissolved_solute_flow + steady_result.product.solid_mass_flow +
                  balance.size_boundary_lower_solid_mass_flow + balance.size_boundary_upper_solid_mass_flow ≈
                  0.2 * steady_feed_concentration rtol = balance_reltol
            if steady_solver isa MoM
                @test steady_result.solution.final_state[1:5] ≈ expected_moments[1:5] rtol = 2e-6
            elseif steady_solver isa QMOM
                @test equilibrium(steady_result).moments ≈ expected_moments rtol = 2e-6
            end
        end
    end
end

@testset "Steady fit recovers growth from equilibrium data" begin
    # Synthetic equilibrium measurements at the closed-form truth (speed = 0.2).
    measured_concentration = steady_ideal_concentration()
    measured_d43 = steady_ideal_d43()
    concentration_variance, d43_variance = 0.01, 1e-14
    steady_experiment = CrystallisationExperiment(; exp_id = 801, conditions = steady_msmpr_conditions(),
        observables = (; concentration = Observable(time = [300.0], mean = [measured_concentration],
                           variance = [concentration_variance]),
                         d43 = Observable(time = [300.0], mean = [measured_d43], variance = [d43_variance])))
    growth_bounds = (growth = (speed = (0.1, 0.3),),)
    steady_spec = OptimisationSpec(bounds = growth_bounds, algorithm = OptimizationOptimJL.LBFGS())
    steady_prepared = prepare_fit(steady_msmpr_model(speed = 0.15), [steady_experiment], steady_spec;
        mode = :steady, steady_options = steady_autonomous_options)

    # Gaussian NLL sum_i [log(2 pi var_i) + r_i^2 / var_i] / 2 at closed-form equilibria.
    gaussian_nll(speed) = 0.5 * (log(2pi * concentration_variance) +
        (steady_ideal_concentration(; speed) - measured_concentration)^2 / concentration_variance) +
        0.5 * (log(2pi * d43_variance) + (steady_ideal_d43(; speed) - measured_d43)^2 / d43_variance)
    @test loss(steady_prepared, [0.2]) ≈ gaussian_nll(0.2) atol = 1e-5
    @test loss(steady_prepared, [0.25]) ≈ gaussian_nll(0.25) rtol = 1e-5

    steady_fit = fit(steady_prepared)
    @test steady_fit.parameters.growth.speed ≈ 0.2 rtol = 1e-4
    fitted_predictions = predict(steady_fit)
    @test fitted_predictions.ensembles[1].success == [true]
    @test vec(observable_values(fitted_predictions.ensembles[1], :d43)) ≈ [measured_d43] rtol = 1e-4

    # Explicit draws: each relaxes to its own closed-form equilibrium.
    draw_speeds = [0.15, 0.25]
    draw_predictions = predict(steady_msmpr_model(), [steady_experiment],
        ParameterSamples(growth = (speed = draw_speeds,)); solver = MoM(), mode = :steady,
        steady_options = steady_autonomous_options)
    @test vec(observable_values(draw_predictions.ensembles[1], :d43)) ≈
          [steady_ideal_d43(; speed) for speed in draw_speeds] rtol = 2e-6
    @test vec(draw_predictions.ensembles[1].concentration) ≈
          [steady_ideal_concentration(; speed) for speed in draw_speeds] rtol = 2e-6

    # Steady fits need numeric conditions, an MSMPR, and declared autonomy.
    @test_throws ArgumentError prepare_fit(steady_msmpr_model(), [CrystallisationExperiment(;
        exp_id = 802, observables = steady_experiment.observables,
        conditions = RunConditions(temperature = 298.15, operation = fixture_msmpr()))], steady_spec;
        mode = :steady, steady_options = steady_autonomous_options)
    @test_throws ArgumentError prepare_fit(steady_msmpr_model(), [CrystallisationExperiment(;
        exp_id = 803, observables = steady_experiment.observables,
        conditions = RunConditions(temperature = 298.15, concentration = 1.0))], steady_spec;
        mode = :steady, steady_options = steady_autonomous_options)
    @test_throws ArgumentError prepare_fit(steady_msmpr_model(), [steady_experiment], steady_spec; mode = :steady)

    # A candidate that cannot relax within the horizon is a failed prediction, not an error.
    exhausted_predictions = predict(steady_msmpr_model(), [steady_experiment],
        ParameterSamples(growth = (speed = [0.2],)); solver = MoM(), mode = :steady,
        steady_options = SteadyStateOptions(relaxation_horizon = 0.01, autonomous = true))
    @test exhausted_predictions.ensembles[1].success == [false]
    @test_throws ArgumentError prediction_summary(exhausted_predictions, :d43)
end
