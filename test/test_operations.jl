# Hand-computed inventory/mixing oracles; this file uses runtests.jl imports.
struct OperationZeroNucleation <: AbstractFPNucleationFunction
    nparams::Int
end
CriSTool.paramaxis(::OperationZeroNucleation) = CriSTool.ComponentArrays.Axis()
CriSTool.nucleationrate(::OperationZeroNucleation, parameters, operation_problem, numerical_state, simulation_time) = zero(eltype(numerical_state))
function _operation_test_problem(operation, solver; initial_concentration_value = 8.0,
                                  initial_crystals = nothing, growth_rate_value = 0.0)
    operation_problem = CrystallisationProblem(;
        operation, solver,
        kinetics_nucleationfunction = OperationZeroNucleation(0),
        parameterset_nucleation = Float64[],
        kinetics_growthfunction = growth_empirical(),
        parameterset_growth = [growth_rate_value, 1.0],
        initial_concentration = initial_concentration_value,
        saturation_model = ConstantSolubility(1.0))
    isnothing(initial_crystals) && return operation_problem
    return CriSTool._problem_with_initial_state(operation_problem,
        initial_state_from_characteristics(operation_problem, initial_crystals))
end

@testset "Operation hydraulics and independent mixing balances" begin
    @test_throws ArgumentError BatchOperation(volume = 0.0)
    @test_throws ArgumentError CrystallisationFeed(concentration = -1.0)
    @test_throws ArgumentError MSMPROperation(volume = 1.0, inflow = -1.0,
        feed = CrystallisationFeed(concentration = 1.0))
    clear_feed = CrystallisationFeed(concentration = 2.0)
    # V0=2, Q=0.5: V(4)=4, C(4)=(2*8+2*2)/4=5.
    # A seeded population with no sources halves every density moment.
    seed_characteristics = LogNormalInitialCrystals(mass_concentration = 3.0,
        d43 = 10e-6, geometric_std = 1.2)
    for numerical_solver in (MoM(), QMOM())
        fed_operation = FedBatchOperation(initial_volume = 2.0, inflow = 0.5,
                                         feed = clear_feed)
        fed_problem = _operation_test_problem(fed_operation, numerical_solver;
                                             initial_crystals = seed_characteristics)
        fed_ode, fed_algorithm = crystallisation_odeproblem(fed_problem, [0.0, 4.0])
        fed_raw = CriSTool.solve(fed_ode, fed_algorithm; saveat = [0.0, 4.0],
                                reltol = 1e-10, abstol = 1e-30)
        fed_result = CriSTool._wrap_solution(fed_problem, fed_raw)
        @test fed_result.success
        @test observable_values(fed_result, :volume) ≈ [2.0, 4.0] rtol = 1e-9
        @test fed_result.concentration ≈ [8.0, 5.0] rtol = 1e-9
        @test fed_result.final_state[CriSTool._population_state_range(fed_problem)] ≈
              0.5 .* fed_problem.initial_state[CriSTool._population_state_range(fed_problem)] rtol = 1e-8
        @test solvent_state(fed_problem, fed_result.final_state).concentration ≈ 5.0
        @test !hasproperty(fed_result.solvent_state, :volume)
        @test length(fed_result.final_state) == CriSTool._population_state_count(numerical_solver) + 2
        # Total solute inventory at t=4: 2*8 + (0.5*4)*2 + initial solid 2*3 = 26 kg.
        final_solid_concentration = fed_problem.crystal_density *
            fed_problem.volume_shape_factor * fed_result.final_state[4]
        @test 4.0 * (fed_result.concentration[end] + final_solid_concentration) ≈ 26.0 rtol = 1e-8

        # MSMPR tau=4; after four seconds C=2+6/e, all seed moments shrink by 1/e.
        msmpr_problem = _operation_test_problem(
            MSMPROperation(volume = 2.0, inflow = 0.5, feed = clear_feed), numerical_solver;
            initial_crystals = seed_characteristics)
        msmpr_ode, msmpr_algorithm = crystallisation_odeproblem(msmpr_problem, [0.0, 4.0])
        msmpr_raw = CriSTool.solve(msmpr_ode, msmpr_algorithm; saveat = [0.0, 4.0],
                                  reltol = 1e-10, abstol = 1e-30)
        msmpr_result = CriSTool._wrap_solution(msmpr_problem, msmpr_raw)
        @test msmpr_result.concentration[end] ≈ 4.207276647028654 rtol = 1e-9
        @test msmpr_result.final_state[CriSTool._population_state_range(msmpr_problem)] ≈
              0.36787944117144233 .* msmpr_problem.initial_state[CriSTool._population_state_range(msmpr_problem)] rtol = 1e-8
        @test reactor_vars(msmpr_result).volume == [2.0, 2.0]
        @test operation_flows(msmpr_problem, 4.0) == (; inflow = 0.5, outflow = 0.5)
    end
end

@testset "Seeded inlet and callable flow" begin
    inlet_seed = LogNormalInitialCrystals(mass_concentration = 3.0, d43 = 10e-6,
                                         geometric_std = 1.2)
    seeded_feed = CrystallisationFeed(concentration = 2.0, crystals = inlet_seed)
    for numerical_solver in (MoM(), QMOM())
        fed_problem = _operation_test_problem(
            FedBatchOperation(initial_volume = 2.0, inflow = simulation_time -> 0.25 * simulation_time,
                              feed = seeded_feed), numerical_solver)
        fed_ode, fed_algorithm = crystallisation_odeproblem(fed_problem, [0.0, 4.0])
        fed_raw = CriSTool.solve(fed_ode, fed_algorithm; saveat = [0.0, 4.0],
                                reltol = 1e-10, abstol = 1e-30)
        fed_result = CriSTool._wrap_solution(fed_problem, fed_raw)
        # Integrated feed volume = 0.125*4² = 2; half the final vessel is feed.
        @test reactor_vars(fed_result).volume[end] ≈ 4.0 rtol = 1e-9
        @test fed_result.concentration[end] ≈ 5.0 rtol = 1e-9
        @test fed_problem.crystal_density * fed_problem.volume_shape_factor *
              fed_result.final_state[4] ≈ 1.5 rtol = 1e-8
        @test fed_result.d43[end] ≈ 10e-6 rtol = 1e-8
    end
    extra_problem = CriSTool._copy_crystallisation_problem(
        _operation_test_problem(FedBatchOperation(initial_volume = 2.0, inflow = 0.5,
            feed = CrystallisationFeed(concentration = 2.0)), MoM());
        initial_solvent_state = (; concentration = 8.0, pH = 7.0))
    @test_throws ArgumentError CriSTool._validate_crystallisation_problem(extra_problem)
end

struct OperationConstantGrowth <: AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::OperationConstantGrowth) = CriSTool.ComponentArrays.Axis(operation_rate = 1)
CriSTool.growthrate(::OperationConstantGrowth, parameters, operation_problem, numerical_state, simulation_time) = parameters[1]

@testset "Fed-batch growth inventory and independent gradient" begin
    function operation_growth_concentration(growth_coefficient)
        growth_problem = CrystallisationProblem(;
            operation = FedBatchOperation(initial_volume = 2.0, inflow = 0.5,
                feed = CrystallisationFeed(concentration = 2.0)),
            solver = MoM(reltol = 1e-10, abstol = 1e-30),
            kinetics_nucleationfunction = OperationZeroNucleation(0),
            parameterset_nucleation = typeof(growth_coefficient)[],
            kinetics_growthfunction = OperationConstantGrowth(1),
            parameterset_growth = [growth_coefficient],
            initial_state = [2e9, 2e4, 0.2, 2e-6, 2e-11, 2.0, 8.0],
            saturation_model = ConstantSolubility(1.0))
        growth_ode, growth_algorithm = crystallisation_odeproblem(growth_problem, [0.0, 4.0])
        growth_raw = CriSTool.solve(growth_ode, growth_algorithm;
            saveat = [0.0, 4.0], reltol = 1e-10, abstol = 1e-30)
        return growth_raw[end, end]
    end
    # Known translating monodisperse seed, half-density at V=4. Exact derivative
    # of the third-moment inventory polynomial at G=1e-7 is -1440.301824.
    operation_ad_derivative = ForwardDiff.derivative(operation_growth_concentration, 1e-7)
    @test operation_ad_derivative ≈ -1440.301824 rtol = 1e-7
    operation_fd_derivative = FiniteDifferences.central_fdm(5, 1; max_range = 1e-8)(operation_growth_concentration, 1e-7)
    @test operation_fd_derivative ≈ -1440.301824 rtol = 1e-5
end

@testset "Explicit extra-variable transport and callable concentration" begin
    transport_hook = (transport_problem, numerical_state, simulation_time, feed_values,
                      inlet_flow, volume_value) -> begin
        tank_values = solvent_state(transport_problem, numerical_state)
        (; pH = zero(tank_values.pH),
           tracer = inlet_flow / volume_value * (feed_values.tracer - tank_values.tracer))
    end
    additional_feed = CrystallisationFeed(concentration = simulation_time -> 2.0 + simulation_time,
        solvent_state = (; tracer = 2.0), transport = transport_hook)
    additional_problem = CriSTool._copy_crystallisation_problem(
        _operation_test_problem(FedBatchOperation(initial_volume = 2.0, inflow = 0.5,
            feed = additional_feed), MoM());
        initial_solvent_state = (; concentration = 8.0, pH = 7.0, tracer = 6.0))
    additional_ode, additional_algorithm = crystallisation_odeproblem(additional_problem, [0.0, 4.0])
    additional_raw = CriSTool.solve(additional_ode, additional_algorithm;
        saveat = [0.0, 4.0], reltol = 1e-10, abstol = 1e-30)
    additional_result = CriSTool._wrap_solution(additional_problem, additional_raw)
    @test additional_result.concentration[end] ≈ 6.0 rtol = 1e-9
    @test additional_result.solvent_state.pH == [7.0, 7.0]
    @test additional_result.solvent_state.tracer[end] ≈ 4.0 rtol = 1e-9
end
