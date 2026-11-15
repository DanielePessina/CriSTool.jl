struct DQMOMOperationsZeroNucleation <: CriSTool.AbstractFPNucleationFunction
    nparams::Int
end
CriSTool.paramaxis(::DQMOMOperationsZeroNucleation) = CriSTool.ComponentArrays.Axis()
CriSTool.nucleationrate(::DQMOMOperationsZeroNucleation,
                        kinetic_parameters,
                        kinetic_problem,
                        numerical_state,
                        simulation_time) = zero(eltype(numerical_state))

struct DQMOMOperationsConstantGrowth <: AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::DQMOMOperationsConstantGrowth) =
    CriSTool.ComponentArrays.Axis(growth_rate = 1)
CriSTool.growthrate(::DQMOMOperationsConstantGrowth,
                    kinetic_parameters,
                    kinetic_problem,
                    numerical_state,
                    simulation_time) = kinetic_parameters[1]

const dqmom_operations_solver = DQMOM(nquadrature = 3,
                                      coordinate_scale = 1e-6,
                                      weight_scale = 1e12,
                                      reltol = 1e-10,
                                      abstol = 1e-12)
const dqmom_operations_seed = LogNormalInitialCrystals(
    mass_concentration = 0.2,
    d43 = 1.0e-6,
    geometric_std = 1.2)

function dqmom_operations_problem(reactor_operation;
                                  initial_crystals = dqmom_operations_seed,
                                  initial_concentration_value = 8.0,
                                  initial_solvent_state =
                                      (; concentration = initial_concentration_value),
                                  growth_rate_value = 0.0)
    operations_problem = CrystallisationProblem(
        operation = reactor_operation,
        solver = dqmom_operations_solver,
        kinetics_nucleationfunction = DQMOMOperationsZeroNucleation(0),
        kinetics_growthfunction = DQMOMOperationsConstantGrowth(1),
        kinetics_aggregationfunction = noaggregation(),
        kinetics_breakagefunction = nobreakage(),
        parameterset_nucleation = Float64[],
        parameterset_growth = [growth_rate_value],
        parameterset_aggregation = Float64[],
        parameterset_breakage = Float64[],
        initial_concentration = initial_concentration_value,
        initial_solvent_state = initial_solvent_state,
        saturation_model = ConstantSolubility(1.0))
    return CriSTool._problem_with_initial_state(
        operations_problem,
        initial_state_from_characteristics(operations_problem, initial_crystals))
end

function dqmom_operations_solve(operations_problem, final_time)
    save_times = [0.0, final_time]
    ode_problem, algorithm = crystallisation_odeproblem(operations_problem, save_times)
    ode_solution = CriSTool.solve(ode_problem, algorithm;
                                  saveat = save_times,
                                  reltol = 1e-10,
                                  abstol = 1e-12)
    return CriSTool._wrap_solution(operations_problem, ode_solution)
end

function dqmom_lognormal_moments(initial_crystals, operations_problem)
    log_spread = log(initial_crystals.geometric_std)
    log_median = log(initial_crystals.d43) - 3.5 * log_spread^2
    third_moment = initial_crystals.mass_concentration /
                   (operations_problem.crystal_density * operations_problem.volume_shape_factor)
    return [third_moment * exp((moment_index - 3) * log_median +
                               0.5 * (moment_index^2 - 9) * log_spread^2)
            for moment_index in 0:5]
end

@testset "DQMOM reactor transport preserves physical inventories" begin
    @testset "Fed-batch clear-feed dilution" begin
        initial_volume = 2.0
        inlet_flow = 0.5
        feed_concentration = 2.0
        final_time = 4.0
        initial_concentration_value = 8.0
        fed_operation = FedBatchOperation(
            initial_volume = initial_volume,
            inflow = inlet_flow,
            feed = CrystallisationFeed(concentration = feed_concentration))
        fed_problem = dqmom_operations_problem(fed_operation;
            initial_concentration_value)
        fed_solution = dqmom_operations_solve(fed_problem, final_time)

        initial_moments = dqmom_lognormal_moments(dqmom_operations_seed,
                                                  fed_problem)
        final_volume = initial_volume + inlet_flow * final_time
        expected_concentration = (initial_volume * initial_concentration_value +
                                  inlet_flow * final_time * feed_concentration) /
                                 final_volume
        @test fed_solution.success
        @test observable_values(fed_solution, :volume) ≈
              [initial_volume, final_volume] rtol = 1e-10
        @test fed_solution.concentration ≈
              [initial_concentration_value, expected_concentration] rtol = 1e-10
        @test fed_solution.moments[:, end] ≈
              (initial_volume / final_volume) .* initial_moments rtol = 2e-8
        @test fed_solution.nodes[:, end] ≈ fed_solution.nodes[:, 1] rtol = 2e-8
        @test length(fed_solution.final_state) == 8
        @test fed_solution.final_state[7] ≈ final_volume
        @test fed_solution.final_state[8] ≈ expected_concentration
        @test state_vars(fed_solution).volume ≈ [initial_volume, final_volume]
    end

    @testset "MSMPR seeded-feed exchange does not alter concentration" begin
        reactor_volume_value = 2.0
        inlet_flow = 0.5
        final_time = 4.0
        inlet_concentration = 2.0
        initial_concentration_value = 8.0
        inlet_crystals = LogNormalInitialCrystals(
            mass_concentration = 0.1,
            d43 = 2.0e-6,
            geometric_std = 1.1)
        msmpr_operation = MSMPROperation(
            volume = reactor_volume_value,
            inflow = inlet_flow,
            feed = CrystallisationFeed(concentration = inlet_concentration,
                                       crystals = inlet_crystals))
        msmpr_problem = dqmom_operations_problem(msmpr_operation;
            initial_concentration_value)
        msmpr_solution = dqmom_operations_solve(msmpr_problem, final_time)

        initial_moments = dqmom_lognormal_moments(dqmom_operations_seed,
                                                  msmpr_problem)
        inlet_moments = dqmom_lognormal_moments(inlet_crystals,
                                                msmpr_problem)
        residence_factor = exp(-inlet_flow * final_time / reactor_volume_value)
        expected_moments = inlet_moments .+
                           (initial_moments .- inlet_moments) .* residence_factor
        expected_concentration = inlet_concentration +
            (initial_concentration_value - inlet_concentration) * residence_factor

        @test msmpr_solution.success
        @test msmpr_solution.moments[:, end] ≈ expected_moments rtol = 5e-7
        @test msmpr_solution.concentration ≈
              [initial_concentration_value, expected_concentration] rtol = 1e-9
        @test msmpr_solution.reactor_state.volume ==
              [reactor_volume_value, reactor_volume_value]
        @test length(msmpr_solution.final_state) == 7
        @test msmpr_solution.final_state[end] ≈ expected_concentration
    end

    @testset "Fed-batch solvent indices follow the reactor block" begin
        transport_hook = (transport_problem, numerical_state, simulation_time, feed_values,
                          inlet_flow, volume_value) -> begin
            tank_values = solvent_state(transport_problem, numerical_state)
            (; pH = zero(tank_values.pH),
               tracer = inlet_flow / volume_value *
                        (feed_values.tracer - tank_values.tracer))
        end
        fed_operation = FedBatchOperation(
            initial_volume = 2.0,
            inflow = 0.5,
            feed = CrystallisationFeed(concentration = 2.0,
                                       solvent_state = (; tracer = 2.0),
                                       transport = transport_hook))
        fed_problem = dqmom_operations_problem(
            fed_operation;
            initial_solvent_state = (; concentration = 8.0, pH = 7.0, tracer = 6.0))
        fed_solution = dqmom_operations_solve(fed_problem, 4.0)

        @test fed_solution.concentration[end] ≈ 5.0 rtol = 1e-10
        @test fed_solution.solvent_state.pH == [7.0, 7.0]
        @test fed_solution.solvent_state.tracer ≈ [6.0, 4.0] rtol = 1e-10
        @test fed_solution.final_state[7:end] ≈ [4.0, 5.0, 7.0, 4.0]
        @test state_vars(fed_solution).volume ≈ [2.0, 4.0]
    end

    @testset "Internal crystallisation uses its own third-moment source" begin
        reactor_volume_value = 2.0
        inlet_flow = 0.5
        final_time = 4.0
        inlet_concentration = 2.0
        initial_concentration_value = 8.0
        growth_rate_value = 2.0e-8
        msmpr_operation = MSMPROperation(
            volume = reactor_volume_value,
            inflow = inlet_flow,
            feed = CrystallisationFeed(concentration = inlet_concentration))
        growth_problem = dqmom_operations_problem(msmpr_operation;
            initial_concentration_value, growth_rate_value)
        growth_solution = dqmom_operations_solve(growth_problem, final_time)

        initial_moments = dqmom_lognormal_moments(dqmom_operations_seed,
                                                  growth_problem)
        exchange_rate = inlet_flow / reactor_volume_value
        residence_factor = exp(-exchange_rate * final_time)
        integrated_growth_volume = growth_rate_value * 3.0 * residence_factor *
            (initial_moments[3] * final_time +
             growth_rate_value * initial_moments[2] * final_time^2 +
             growth_rate_value^2 * initial_moments[1] * final_time^3 / 3.0)
        expected_concentration = inlet_concentration +
            (initial_concentration_value - inlet_concentration) * residence_factor -
            growth_problem.crystal_density * growth_problem.volume_shape_factor *
            integrated_growth_volume

        @test growth_solution.success
        @test growth_solution.concentration[end] ≈ expected_concentration rtol = 2e-7
        @test growth_solution.concentration[end] <
              inlet_concentration +
              (initial_concentration_value - inlet_concentration) * residence_factor
    end

    @testset "ForwardDiff agrees with finite differences under flow" begin
        msmpr_operation = MSMPROperation(
            volume = 2.0,
            inflow = 0.25,
            feed = CrystallisationFeed(concentration = 2.0))
        function dqmom_operation_objective(normalized_growth_rate)
            _, solution = runsimulation(
                [normalized_growth_rate[1] * 1e-8];
                nucl = DQMOMOperationsZeroNucleation(0),
                gr = DQMOMOperationsConstantGrowth(1),
                agg = noaggregation(),
                br = nobreakage(),
                solver = dqmom_operations_solver,
                initial_concentration = 8.0,
                initial_crystals = dqmom_operations_seed,
                saturation_model = ConstantSolubility(1.0),
                operation = msmpr_operation,
                save_idx = [0.0, 4.0])
            return solution.concentration[end] + 1e6 * solution.d43[end]
        end

        finite_difference_backend = DI.AutoFiniteDifferences(
            FiniteDifferences.central_fdm(5, 1))
        forward_difference_backend = DI.AutoForwardDiff()
        test_parameter = [1.0]
        finite_difference_gradient = DI.gradient(dqmom_operation_objective,
                                                 finite_difference_backend,
                                                 test_parameter)
        forward_difference_gradient = DI.gradient(dqmom_operation_objective,
                                                   forward_difference_backend,
                                                   test_parameter)
        @test forward_difference_gradient ≈ finite_difference_gradient rtol = 5e-3 atol = 1e-8
    end
end
