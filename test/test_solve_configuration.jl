# Analytical constant-growth raw moments and independent callback-state checks.
using ComponentArrays
struct SolvePolicyConstantGrowth <: AbstractFPScalarGrowthFunction
    nparams::Int
end
SolvePolicyConstantGrowth() = SolvePolicyConstantGrowth(1)
CriSTool.paramaxis(::SolvePolicyConstantGrowth) = ComponentArrays.Axis(speed = 1)
CriSTool.growthrate(::SolvePolicyConstantGrowth, kinetic_parameters,
                   configured_problem::CrystallisationProblem, ode_state, elapsed_time) =
    kinetic_parameters[1]

@testset "Shared direct/prepared solve configuration" begin
    disabled_nucleation = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                               nucleation_order = 1.0)
    growth_parameters = [1e-9]
    for tolerance_mode in (:scalar, :auto)
        qmom_problem = CrystallisationProblem(
            kinetics_nucleationfunction = disabled_nucleation,
            parameterset_nucleation = Float64[],
            kinetics_growthfunction = SolvePolicyConstantGrowth(),
            parameterset_growth = growth_parameters,
            initial_concentration = 20.0,
            initial_state = [1e12, 1e6, 1.0, 1e-6, 1e-12, 1e-18, 20.0],
            solver = QMOM(nquadrature = 3, tolerance_mode = tolerance_mode,
                          reltol = 1e-8, abstol = 1e-9))
        saved_times = [0.0, 100.0]
        ode_template, ode_algorithm = crystallisation_odeproblem(qmom_problem, saved_times)
        prepared_experiment = CriSTool.PreparedExperiment(qmom_problem, ode_template,
                                                          ode_algorithm, saved_times)
        direct_solution = CriSTool._simulatecrystallisation(qmom_problem, saved_times)
        prepared_solution = CriSTool._solve_prepared(prepared_experiment, growth_parameters)
        frozen_final_moments = [1e12, 1.1e6, 1.21, 1.331e-6, 1.4641e-12, 1.61051e-18]
        @test direct_solution.moments[:, end] ≈ frozen_final_moments rtol = 2e-6
        @test prepared_solution.moments[:, end] ≈ frozen_final_moments rtol = 2e-6
        @test prepared_solution.concentration ≈ direct_solution.concentration rtol = 1e-12
        @test prepared_solution.concentration[end] +
              qmom_problem.crystal_density * qmom_problem.volume_shape_factor *
              prepared_solution.moments[4, end] ≈
              20.0 + qmom_problem.crystal_density * qmom_problem.volume_shape_factor * 1e-6
        @test CriSTool._solve_prepared(prepared_experiment, growth_parameters).moments ==
              prepared_solution.moments
        moment_prediction = speed_multiplier ->
            CriSTool._solve_prepared(prepared_experiment,
                                     [speed_multiplier * 1e-9]).moments[2, end] / 1e6
        @test ForwardDiff.derivative(moment_prediction, 1.0) ≈ 0.1 rtol = 1e-6
        @test (moment_prediction(1.001) - moment_prediction(0.999)) / 0.002 ≈
              0.1 rtol = 1e-6
        concurrent_tasks = [Threads.@spawn CriSTool._solve_prepared(prepared_experiment,
                                                                    growth_parameters)
                            for task_index in 1:3]
        @test all(fetch(task).moments == prepared_solution.moments for task in concurrent_tasks)
    end

    extinction_problem = CrystallisationProblem(
        kinetics_nucleationfunction = disabled_nucleation,
        parameterset_nucleation = Float64[],
        kinetics_growthfunction = growth_dissolution(),
        parameterset_growth = [1000e-9 / 60, 0.0, 1.0],
        initial_concentration = 5.0,
        initial_state = [1e12, 1e6, 1.0, 1e-6, 1e-12, 1e-18, 5.0],
        saturation_model = ConstantSolubility(10.0),
        solid_mass_concentration_threshold = 5e-4,
        solver = QMOM(nquadrature = 3, tolerance_mode = :auto))
    extinction_times = [0.0, 3600.0]
    extinction_template, extinction_algorithm = crystallisation_odeproblem(
        extinction_problem, extinction_times)
    extinction_prepared = CriSTool.PreparedExperiment(extinction_problem,
        extinction_template, extinction_algorithm, extinction_times)
    for extinction_solution in (
        CriSTool._simulatecrystallisation(extinction_problem, extinction_times),
        CriSTool._solve_prepared(extinction_prepared, [1000e-9 / 60, 0.0, 1.0]))
        @test extinction_solution.final_state[1:6] == zeros(6)
        @test extinction_solution.concentration[end] ≈
              5.0 + extinction_problem.crystal_density *
                    extinction_problem.volume_shape_factor * 1e-6 rtol = 1e-7
    end

    callback_problem = CrystallisationProblem(
        kinetics_nucleationfunction = disabled_nucleation,
        parameterset_nucleation = Float64[],
        kinetics_growthfunction = SolvePolicyConstantGrowth(),
        parameterset_growth = [0.0],
        initial_concentration = 20.0,
        solvent_dynamics = (configured_problem, ode_state, elapsed_time, volume_rate) ->
            (; concentration = -1.0),
        solver = MoM(tolerance_mode = :auto))
    callback_template, callback_algorithm = crystallisation_odeproblem(callback_problem,
                                                                       [0.0, 2.0])
    callback_counts = Int[]
    callback_floor_minima = Float64[]
    factory = odeproblem -> begin
        per_solve_count = Ref(0)
        minimum_floor = Ref(Inf)
        CriSTool.DiscreteCallback((ode_state, elapsed_time, integrator) -> true,
            integrator -> begin
                per_solve_count[] += 1
                minimum_floor[] = min(minimum_floor[], minimum(integrator.opts.abstol))
                CriSTool.u_modified!(integrator, false)
            end;
            finalize = (callback, ode_state, elapsed_time, integrator) -> begin
                push!(callback_counts, per_solve_count[])
                push!(callback_floor_minima, minimum_floor[])
            end,
            save_positions = (false, false))
    end
    for repetition in 1:2
        result = CriSTool._solve_crystallisation_ode(callback_problem, callback_template,
                                                    callback_algorithm, [0.0, 2.0];
                                                    callback_factory = factory)
        @test result.u[end][end] ≈ 18.0 atol = 1e-9
    end
    budget_result = CriSTool._solve_crystallisation_ode(callback_problem,
        callback_template, CriSTool.OrdinaryDiffEq.Euler(), [0.0, 2.0];
        solve_options = (; dt = 0.25, adaptive = false, maxiters = 4))
    @test CriSTool.OrdinaryDiffEq.SciMLBase.ReturnCode.MaxIters ==
          budget_result.retcode
    sufficient_budget_result = CriSTool._solve_crystallisation_ode(callback_problem,
        callback_template, CriSTool.OrdinaryDiffEq.Euler(), [0.0, 2.0];
        solve_options = (; dt = 0.25, adaptive = false, maxiters = 8))
    @test sufficient_budget_result.retcode ==
          CriSTool.OrdinaryDiffEq.SciMLBase.ReturnCode.Success
    @test callback_counts[1] == callback_counts[2]
    @test all(callback_floor_minima .>= callback_problem.solver.abstol)
    @test_throws ArgumentError CriSTool._solve_crystallisation_ode(
        callback_problem, callback_template, callback_algorithm, [0.0, 2.0];
        solve_options = (; callback = nothing))
    @test_throws ArgumentError CriSTool._solve_crystallisation_ode(
        callback_problem, callback_template, callback_algorithm, [0.0, 2.0];
        solve_options = (; save_idxs = [1]))
end
