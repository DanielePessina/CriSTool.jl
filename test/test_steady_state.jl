struct SteadyStateConstantNucleation <: CriSTool.AbstractFPNucleationFunction
    nparams::Int
end
CriSTool.paramaxis(::SteadyStateConstantNucleation) =
    CriSTool.ComponentArrays.Axis(nucleation_rate = 1)
CriSTool.nucleationrate(::SteadyStateConstantNucleation, parameters,
                        problem, state, simulation_time) = parameters[1]

struct SteadyStateConstantGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::SteadyStateConstantGrowth) =
    CriSTool.ComponentArrays.Axis(growth_rate = 1)
CriSTool.growthrate(::SteadyStateConstantGrowth, parameters,
                    problem, state, simulation_time) = parameters[1]

function steady_state_test_problem(; nucleation_rate = 0.0,
                                     growth_rate = 0.0,
                                     feed_concentration = 2.0,
                                     initial_concentration_value = 8.0,
                                     initial_state = [0.0, 0.0, 0.0, 0.0, 0.0,
                                                     initial_concentration_value],
                                     operation = MSMPROperation(
                                         volume = 2.0, inflow = 0.2,
                                         feed = CrystallisationFeed(
                                             concentration = feed_concentration)),
                                     temperature_profile = ConstantTemperature(298.15))
    return CrystallisationProblem(
        ; operation,
        solver = MoM(reltol = 1e-10, abstol = 1e-24),
        kinetics_nucleationfunction = SteadyStateConstantNucleation(1),
        parameterset_nucleation = [nucleation_rate],
        kinetics_growthfunction = SteadyStateConstantGrowth(1),
        parameterset_growth = [growth_rate],
        initial_concentration = initial_concentration_value,
        initial_solvent_state = (; concentration = initial_concentration_value),
        initial_state,
        temp_profile = temperature_profile,
        saturation_model = ConstantSolubility(1.0))
end

@testset "Steady-state MSMPR operation contract" begin
    washed_out_problem = steady_state_test_problem()
    washed_out = solve_steadystate(washed_out_problem;
                                   relaxation_horizon = 400.0,
                                   autonomous = true)

    @test washed_out isa CrystallisationSteadyStateSolution
    @test washed_out.success
    @test washed_out.converged
    @test CriSTool.OrdinaryDiffEq.SciMLBase.successful_retcode(washed_out.retcode)
    @test washed_out.time[end] > 0.0
    @test observable_values(washed_out, :concentration) ≈ [2.0] rtol = 1e-8
    @test observable_values(washed_out, :volume) == [2.0]
    @test state_vars(washed_out).concentration ≈ [2.0] rtol = 1e-8
    @test state_vars(washed_out).volume == [2.0]
    @test hasproperty(size_metrics(washed_out), :d43)
    @test washed_out.hydraulics ==
          (; volume = 2.0, inflow = 0.2, outflow = 0.2, residence_time = 10.0)
    @test washed_out.product_dissolved_solute_flow ≈ 0.4 rtol = 1e-8
    @test washed_out.product_solid_mass_flow == 0.0
    @test abs(washed_out.diagnostics.total_api_balance.total_api_residual) < 1e-9
    @test length(washed_out.residual) == length(washed_out.final_state)
    @test length(observable_values(washed_out, :d43)) == 1

    minimum_horizon_result = solve_steadystate(washed_out_problem;
        relaxation_horizon = 400.0, minimum_relaxation = 250.0,
        autonomous = true)
    @test minimum_horizon_result.success
    @test minimum_horizon_result.time[end] >= 250.0

    batch_problem = CriSTool._copy_crystallisation_problem(
        washed_out_problem; operation = BatchOperation())
    fed_batch_problem = CriSTool._copy_crystallisation_problem(
        washed_out_problem;
        operation = FedBatchOperation(initial_volume = 2.0, inflow = 0.2,
                                      feed = CrystallisationFeed(concentration = 2.0)))
    @test_throws ArgumentError solve_steadystate(batch_problem)
    @test_throws ArgumentError solve_steadystate(fed_batch_problem)

    time_profile_operation = MSMPROperation(
        volume = 2.0, inflow = simulation_time -> 0.2,
        feed = CrystallisationFeed(concentration = simulation_time -> 2.0))
    opaque_profile_problem = steady_state_test_problem(operation = time_profile_operation)
    @test_throws ArgumentError solve_steadystate(opaque_profile_problem)
    @test solve_steadystate(opaque_profile_problem;
                            relaxation_horizon = 400.0,
                            autonomous = true).success

    @test_throws ArgumentError solve_steadystate(washed_out_problem;
                                                 relaxation_horizon = 10.0,
                                                 minimum_relaxation = 10.1,
                                                 autonomous = true)

    seeded_feed_crystals = LogNormalInitialCrystals(
        mass_concentration = 0.1, d43 = 10e-6, geometric_std = 1.2)
    dqmom_problem = CrystallisationProblem(
        ; operation = MSMPROperation(volume = 2.0, inflow = 0.2,
             feed = CrystallisationFeed(concentration = 2.0,
                                        crystals = seeded_feed_crystals)),
          solver = DQMOM(),
          kinetics_nucleationfunction = SteadyStateConstantNucleation(1),
          parameterset_nucleation = [0.0],
          kinetics_growthfunction = SteadyStateConstantGrowth(1),
          parameterset_growth = [0.0],
          initial_concentration = 2.0,
          initial_solvent_state = (; concentration = 2.0),
          saturation_model = ConstantSolubility(1.0))
    seeded_state = initial_state_from_characteristics(dqmom_problem,
                                                       seeded_feed_crystals)
    dqmom_problem = CriSTool._copy_crystallisation_problem(
        dqmom_problem; initial_state = seeded_state)
    dqmom_solution = solve_steadystate(dqmom_problem;
                                       relaxation_horizon = 100.0,
                                       autonomous = true)
    @test dqmom_solution.success
    @test length(dqmom_solution.residual) == length(dqmom_solution.final_state)
    @test length(observable_values(dqmom_solution, :d43)) == 1
    @test observable_values(dqmom_solution, :volume) == [2.0]

    # C starts at the feed value, but positive nucleation leaves a nonzero
    # population RHS. A concentration-only stop rule would accept this state.
    population_residual_problem = steady_state_test_problem(
        nucleation_rate = 3e5, growth_rate = 2e-7,
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    short_relaxation_error = try
        solve_steadystate(population_residual_problem;
            relaxation_horizon = 0.1, residual_reltol = 1e-12,
            residual_abstol = 1e-14, autonomous = true)
        nothing
    catch error_value
        error_value
    end
    @test short_relaxation_error isa SteadyStateConvergenceError
    short_relaxation = short_relaxation_error.solution
    @test !short_relaxation.success
    @test !short_relaxation.converged
    @test maximum(abs, short_relaxation.residual[1:5]) > 0.0
    @test CriSTool.OrdinaryDiffEq.SciMLBase.successful_retcode(short_relaxation.retcode)
end

@testset "Steady ideal MSMPR frozen moments and flow balance" begin
    ideal_problem = steady_state_test_problem(
        nucleation_rate = 3e5,
        growth_rate = 2e-7,
        feed_concentration = 1.0,
        initial_concentration_value = 1.0,
        initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ideal_solution = solve_steadystate(ideal_problem;
                                       relaxation_horizon = 1_000.0,
                                       autonomous = true)

    # Frozen closed-form raw-moment values for B=3e5 m⁻³s⁻¹,
    # G=2e-7 m/s, and tau=10 s from the September MSMPR plan.
    expected_moments = [3.0e6, 6.0, 2.4e-5, 1.44e-10, 1.152e-15]
    @test ideal_solution.success
    @test ideal_solution.final_state[1:5] ≈ expected_moments rtol = 2e-6
    @test observable_values(ideal_solution, :d43) ≈ [8.0e-6] rtol = 2e-6
    @test ideal_solution.concentration[1] ≈ 1.0 - 1.597968e-7 rtol = 2e-6
    @test ideal_solution.product_solid_mass_flow ≈
          0.2 * 1370.0 * 0.81 * expected_moments[4] rtol = 2e-6
    @test ideal_solution.product_dissolved_solute_flow ≈
          0.2 * ideal_solution.concentration[1] rtol = 1e-12
    @test ideal_solution.diagnostics.total_api_balance.relative_error < 1e-5
    @test ideal_solution.diagnostics.scaled_residual_norm <=
          ideal_solution.diagnostics.residual_limit

    exhausted_error = try
        solve_steadystate(ideal_problem;
            initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            relaxation_horizon = 0.01,
            residual_reltol = 1e-12, residual_abstol = 1e-14,
            autonomous = true)
        nothing
    catch error_value
        error_value
    end
    @test exhausted_error isa SteadyStateConvergenceError
    exhausted = exhausted_error.solution
    @test !exhausted.success
    @test !exhausted.converged
    @test length(observable_values(exhausted, :concentration)) == 1
end

@testset "Steady MSMPR ForwardDiff agrees with finite differences" begin
    function steady_d43_for_growth(growth_parameter)
        gradient_problem = steady_state_test_problem(
            nucleation_rate = 3e5,
            growth_rate = growth_parameter,
            feed_concentration = 1.0,
            initial_concentration_value = 1.0,
            initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        return only(observable_values(solve_steadystate(
            gradient_problem; relaxation_horizon = 1_000.0,
            autonomous = true), :d43))
    end

    autodiff_gradient = ForwardDiff.derivative(steady_d43_for_growth, 2e-7)
    finite_difference_gradient = FiniteDifferences.central_fdm(
        5, 1; max_range = 1e-8)(steady_d43_for_growth, 2e-7)
    @test autodiff_gradient ≈ 40.0 rtol = 2e-4
    @test finite_difference_gradient ≈ 40.0 rtol = 2e-4
    @test autodiff_gradient ≈ finite_difference_gradient rtol = 2e-4

    function steady_d43_for_remade_growth(growth_parameter)
        template_problem = steady_state_test_problem(
            nucleation_rate = 3e5,
            growth_rate = 2e-7,
            feed_concentration = 1.0,
            initial_concentration_value = 1.0,
            initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        ode_problem, algorithm = crystallisation_odeproblem(
            template_problem, (0.0, 1_000.0))
        candidate_parameters = CriSTool.ComponentArrays.ComponentArray(
            ; nucl = [3e5], gr = [growth_parameter], diss = Float64[])
        candidate_ode_problem = CriSTool.OrdinaryDiffEq.remake(
            ode_problem; p = candidate_parameters)
        steady_result = CriSTool._solve_steadystate_ode(
            template_problem, candidate_ode_problem, algorithm;
            steady_options = (; relaxation_horizon = 1_000.0,
                               autonomous = true))
        return only(observable_values(steady_result, :d43))
    end

    remade_parameter_gradient = ForwardDiff.derivative(
        steady_d43_for_remade_growth, 2e-7)
    remade_parameter_finite_difference = FiniteDifferences.central_fdm(
        5, 1; max_range = 1e-8)(steady_d43_for_remade_growth, 2e-7)
    @test remade_parameter_gradient ≈ 40.0 rtol = 2e-4
    @test remade_parameter_finite_difference ≈ 40.0 rtol = 2e-4
    @test remade_parameter_gradient ≈ remade_parameter_finite_difference rtol = 2e-4
end
