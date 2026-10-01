struct WENOCFLZeroBirth <: CriSTool.AbstractFPNucleationFunction
    nparams::Int
end
CriSTool.paramaxis(::WENOCFLZeroBirth) = CriSTool.ComponentArrays.Axis()
CriSTool.nucleationrate(::WENOCFLZeroBirth, kinetic_parameters,
                        reactor_problem, numerical_state, simulation_time) =
    zero(eltype(kinetic_parameters))

struct WENOCFLConstantGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::WENOCFLConstantGrowth) = CriSTool.ComponentArrays.Axis(transport_speed = 1)
CriSTool.growthrate(::WENOCFLConstantGrowth, kinetic_parameters,
                    reactor_problem, numerical_state, simulation_time) = kinetic_parameters[1]

struct WENOCFLLinearLengthGrowth <: CriSTool.AbstractFPLengthGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::WENOCFLLinearLengthGrowth) =
    CriSTool.ComponentArrays.Axis(transport_speed = 1)
CriSTool.growthrate_at_length(::WENOCFLLinearLengthGrowth, kinetic_parameters,
    reactor_problem, numerical_state, simulation_time, crystal_length) =
    kinetic_parameters[1] * (2 * crystal_length / 10e-6)
function CriSTool.growthrate!(destination, growth_model::WENOCFLLinearLengthGrowth,
    kinetic_parameters, reactor_problem, numerical_state, simulation_time, length_mesh)
    for length_index in eachindex(destination, length_mesh)
        destination[length_index] = CriSTool.growthrate_at_length(growth_model,
            kinetic_parameters, reactor_problem, numerical_state, simulation_time,
            length_mesh[length_index])
    end
    return destination
end

function weno_cfl_test_problem(; transport_speed = 1e-6, tolerance_mode = :scalar)
    return CrystallisationProblem(
        solver = WENO(meshsize = 10, lmax = 100e-6, tolerance_mode = tolerance_mode,
            timestepping_algorithm = transport_speed < 0 ? :ssprk43 : :auto),
        kinetics_nucleationfunction = WENOCFLZeroBirth(0), parameterset_nucleation = Float64[],
        kinetics_growthfunction = WENOCFLConstantGrowth(1), parameterset_growth = [transport_speed],
        saturation_model = ConstantSolubility(1.0), initial_concentration = 2.0)
end

@testset "WENO applies a solve-local physical timestep cap" begin
    for transport_speed in (1e-6, -1e-6)
        configured_transport = weno_cfl_test_problem(; transport_speed)
        accepted_times = Float64[0.0]
        observer_factory = odeproblem -> CriSTool.DiscreteCallback(
            (numerical_state, simulation_time, integrator) -> true,
            integrator -> push!(accepted_times, integrator.t);
            save_positions = (false, false))
        _, capped_result = runsimulation(configured_transport;
            save_idx = [0.0, 100.0], callback_factory = observer_factory)
        # Independent physical CFL: cell width 10 µm / speed 1 µm/s × 0.9 = 9 s.
        # Empty population makes the exact solution constant, so error control
        # alone permits large steps and cannot accidentally satisfy this guard.
        @test capped_result.concentration == [2.0, 2.0]
        @test length(accepted_times) > 10
        @test maximum(diff(accepted_times)) <= 9.0 + 1e-12

        empty!(accepted_times)
        push!(accepted_times, 0.0)
        runsimulation(configured_transport; save_idx = [0.0, 100.0],
            callback_factory = observer_factory, solve_options = (; dtmax = 2.0))
        @test maximum(diff(accepted_times)) <= 2.0 + 1e-12

        # Reusing the same configured problem must not retain the prior 2 s cap.
        empty!(accepted_times)
        push!(accepted_times, 0.0)
        runsimulation(configured_transport; save_idx = [0.0, 100.0],
            callback_factory = observer_factory)
        @test maximum(diff(accepted_times)) > 2.0
        @test maximum(diff(accepted_times)) <= 9.0 + 1e-12

        empty!(accepted_times)
        push!(accepted_times, 0.0)
        runsimulation(configured_transport; save_idx = [0.0, 100.0],
            callback_factory = observer_factory, solve_options = (; adaptive = false, dt = 50.0))
        @test maximum(diff(accepted_times)) <= 9.0 + 1e-12
    end
    auto_transport = weno_cfl_test_problem(tolerance_mode = :auto)
    _, auto_capped_result = runsimulation(auto_transport; save_idx = [0.0, 100.0])
    @test auto_capped_result.concentration == [2.0, 2.0]

    length_transport = CriSTool._copy_crystallisation_problem(weno_cfl_test_problem();
        kinetics_growthfunction = WENOCFLLinearLengthGrowth(1))
    vector_accepted_times = Float64[0.0]
    vector_observer = odeproblem -> CriSTool.DiscreteCallback(
        (numerical_state, simulation_time, integrator) -> true,
        integrator -> push!(vector_accepted_times, integrator.t);
        save_positions = (false, false))
    _, vector_capped_result = runsimulation(length_transport;
        save_idx = [0.0, 10.0], callback_factory = vector_observer)
    # Last centre = 95 µm, so Gmax = 19 µm/s and CFL = 9/19 s.
    @test maximum(diff(vector_accepted_times)) <= 9 / 19 + 1e-12
    @test vector_capped_result.concentration == [2.0, 2.0]
end

@testset "WENO timestep decisions preserve parameter gradients" begin
    seeded_transport = weno_cfl_test_problem(transport_speed = 2e-7)
    seeded_transport = CriSTool._copy_crystallisation_problem(seeded_transport;
        solver = WENO(meshsize = 80, lmax = 30e-6, reltol = 1e-6, abstol = 1e-10))
    seed_characteristics = LogNormalInitialCrystals(mass_concentration = 0.25,
        d43 = 6e-6, geometric_std = 1.25)
    seeded_transport = CriSTool._copy_crystallisation_problem(seeded_transport;
        initial_state = initial_state_from_characteristics(seeded_transport, seed_characteristics))
    concentration_objective = candidate_parameters -> last(last(runsimulation(
        2e-7 .* candidate_parameters, seeded_transport; save_idx = [0.0, 1.0])).concentration)
    autodiff_gradient = DI.gradient(concentration_objective, DI.AutoForwardDiff(), [1.0])
    difference_gradient = DI.gradient(concentration_objective,
        DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1)), [1.0])
    @test autodiff_gradient ≈ difference_gradient rtol = 2e-5
end
