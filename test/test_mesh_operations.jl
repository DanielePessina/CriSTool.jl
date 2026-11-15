struct MeshOperationZeroNucleation <: CriSTool.AbstractFPNucleationFunction
    nparams::Int
end
CriSTool.paramaxis(::MeshOperationZeroNucleation) = CriSTool.ComponentArrays.Axis()
CriSTool.nucleationrate(::MeshOperationZeroNucleation, parameters, problem, state, time) =
    zero(eltype(state))

struct MeshOperationConstantGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::MeshOperationConstantGrowth) =
    CriSTool.ComponentArrays.Axis(mesh_operation_rate = 1)
CriSTool.growthrate(::MeshOperationConstantGrowth, parameters, problem, state, time) =
    parameters[1]

function _mesh_operation_problem(operation, numerical_solver;
                                 initial_concentration_value = 8.0,
                                 initial_solvent_values = (; concentration = initial_concentration_value),
                                 initial_crystals = nothing,
                                 growth_rate_value = 0.0)
    mesh_problem = CrystallisationProblem(;
        operation,
        solver = numerical_solver,
        initial_concentration = initial_solvent_values.concentration,
        initial_solvent_state = initial_solvent_values,
        saturation_model = ConstantSolubility(1.0),
        kinetics_nucleationfunction = MeshOperationZeroNucleation(0),
        parameterset_nucleation = Float64[],
        kinetics_growthfunction = MeshOperationConstantGrowth(1),
        parameterset_growth = [growth_rate_value])
    isnothing(initial_crystals) && return mesh_problem
    return CriSTool._problem_with_initial_state(
        mesh_problem, initial_state_from_characteristics(mesh_problem, initial_crystals))
end

function _solve_mesh_operation(problem, saved_times)
    ode_problem, algorithm = crystallisation_odeproblem(problem, saved_times)
    ode_solution = CriSTool.solve(ode_problem, algorithm;
        saveat = saved_times, reltol = 1e-10, abstol = 1e-12)
    return ode_solution, CriSTool._wrap_solution(problem, ode_solution)
end

function _mesh_operation_trapezoid(time_values, samples)
    return sum((time_values[index + 1] - time_values[index]) *
               (samples[index + 1] + samples[index]) / 2
               for index in 1:(length(time_values) - 1))
end

@testset "FV and WENO reactor-operation transport" begin
    tank_seed = LogNormalInitialCrystals(mass_concentration = 2.0,
                                         d43 = 13e-6,
                                         geometric_std = 1.15)
    inlet_seed = LogNormalInitialCrystals(mass_concentration = 4.0,
                                          d43 = 25e-6,
                                          geometric_std = 1.12)
    saved_times = collect(range(0.0, 4.0; length = 9))

    for numerical_solver in (FiniteVol(meshsize = 36, lmax = 50e-6,
                                       reltol = 1e-10, abstol = 1e-12),
                             WENO(meshsize = 36, lmax = 50e-6,
                                  reltol = 1e-10, abstol = 1e-12))
        feed = CrystallisationFeed(concentration = 2.0, crystals = inlet_seed)
        tank_batch = _mesh_operation_problem(BatchOperation(volume = 2.0), numerical_solver;
                                             initial_crystals = tank_seed)
        feed_batch = _mesh_operation_problem(BatchOperation(volume = 2.0), numerical_solver;
                                             initial_crystals = inlet_seed)
        population_range = CriSTool._population_state_range(tank_batch)
        initial_population = tank_batch.initial_state[population_range]
        feed_population = initial_state_from_characteristics(feed_batch,
                                                              inlet_seed)[population_range]

        fed_problem = _mesh_operation_problem(
            FedBatchOperation(initial_volume = 2.0, inflow = 0.5, feed = feed),
            numerical_solver; initial_crystals = tank_seed)
        fed_raw, fed_solution = _solve_mesh_operation(fed_problem, saved_times)
        @test fed_solution.success
        @test size(fed_solution.numberdensity) ==
              (numerical_solver.meshsize, length(saved_times))
        @test length(fed_solution.final_state) ==
              numerical_solver.meshsize + 1 + length(propertynames(fed_problem.initial_solvent_state))
        expected_fed_population = hcat(((2.0 .* initial_population .+
                                         (0.5 * time_value) .* feed_population) ./
                                        (2.0 + 0.5 * time_value)
                                        for time_value in saved_times)...)
        @test fed_solution.numberdensity ≈ expected_fed_population rtol = 2e-8 atol = 1e-10
        @test fed_solution.concentration ≈
              [(2.0 * 8.0 + 0.5 * time_value * 2.0) / (2.0 + 0.5 * time_value)
               for time_value in saved_times] rtol = 2e-9
        @test reactor_vars(fed_solution).volume ≈ [2.0 + 0.5 * time_value
                                                   for time_value in saved_times] rtol = 2e-9
        @test all(iszero, reactor_vars(fed_solution).size_boundary_lower_solid_mass_flow)
        @test all(iszero, reactor_vars(fed_solution).size_boundary_upper_solid_mass_flow)

        msmpr_problem = _mesh_operation_problem(
            MSMPROperation(volume = 2.0, inflow = 0.5, feed = feed),
            numerical_solver; initial_crystals = tank_seed)
        _, msmpr_solution = _solve_mesh_operation(msmpr_problem, saved_times)
        washout_factor = exp.(-0.5 .* saved_times ./ 2.0)
        expected_msmpr_population = hcat((feed_population .+
                                          (initial_population .- feed_population) .* factor
                                          for factor in washout_factor)...)
        @test msmpr_solution.numberdensity ≈ expected_msmpr_population rtol = 2e-8 atol = 1e-10
        @test msmpr_solution.concentration ≈
              [2.0 + (8.0 - 2.0) * exp(-0.5 * time_value / 2.0)
               for time_value in saved_times] rtol = 2e-9
        @test reactor_vars(msmpr_solution).volume == fill(2.0, length(saved_times))
    end
end

@testset "Mesh operation layout and explicit solvent transport" begin
    transport_hook = (problem, state, time, feed_values, inlet_flow, volume) -> begin
        solvent_values = solvent_state(problem, state)
        (; pH = 0.0,
           tracer = inlet_flow / volume * (feed_values.tracer - solvent_values.tracer))
    end
    inlet_seed = LogNormalInitialCrystals(mass_concentration = 3.0,
                                          d43 = 20e-6,
                                          geometric_std = 1.1)
    feed = CrystallisationFeed(concentration = 2.0,
                               crystals = inlet_seed,
                               solvent_state = (; tracer = 2.0),
                               transport = transport_hook)
    saved_times = [0.0, 2.0, 4.0]

    for numerical_solver in (FiniteVol(meshsize = 32, lmax = 50e-6),
                             WENO(meshsize = 32, lmax = 50e-6))
        problem = _mesh_operation_problem(
            FedBatchOperation(initial_volume = 2.0, inflow = 0.5, feed = feed),
            numerical_solver;
            initial_solvent_values = (; concentration = 8.0, pH = 7.0, tracer = 6.0),
            initial_crystals = inlet_seed)
        _, result = _solve_mesh_operation(problem, saved_times)
        @test size(result.numberdensity) == (numerical_solver.meshsize, length(saved_times))
        @test result.concentration ≈ [8.0, 6.0, 5.0] rtol = 2e-8
        @test result.solvent_state.pH == fill(7.0, length(saved_times))
        @test result.solvent_state.tracer ≈ [6.0, 14.0 / 3.0, 4.0] rtol = 2e-8
        @test reactor_vars(result).volume ≈ [2.0, 3.0, 4.0] rtol = 2e-9
        @test all(iszero, reactor_vars(result).size_boundary_upper_solid_mass_flow)
    end
end

@testset "Moment and mesh operation transport share independent inlet oracles" begin
    moment_solver = MoM(nmoments = 4, reltol = 1e-10, abstol = 1e-30)
    tank_seed = LogNormalInitialCrystals(mass_concentration = 2.0,
                                         d43 = 12e-6,
                                         geometric_std = 1.2)
    inlet_seed = LogNormalInitialCrystals(mass_concentration = 5.0,
                                          d43 = 24e-6,
                                          geometric_std = 1.1)
    feed = CrystallisationFeed(concentration = 1.5, crystals = inlet_seed)
    fed_operation = FedBatchOperation(initial_volume = 3.0, inflow = 0.25, feed = feed)
    problem = _mesh_operation_problem(fed_operation, moment_solver;
                                      initial_concentration_value = 7.0,
                                      initial_crystals = tank_seed)
    feed_batch = _mesh_operation_problem(BatchOperation(volume = 3.0), moment_solver;
                                         initial_crystals = inlet_seed)
    population_range = CriSTool._population_state_range(problem)
    initial_moments = problem.initial_state[population_range]
    feed_moments = initial_state_from_characteristics(feed_batch, inlet_seed)[population_range]
    saved_times = collect(range(0.0, 4.0; length = 5))
    raw_solution, moment_solution = _solve_mesh_operation(problem, saved_times)
    expected_moments = hcat(((3.0 .* initial_moments .+
                              (0.25 * time_value) .* feed_moments) ./
                             (3.0 + 0.25 * time_value)
                             for time_value in saved_times)...)
    @test raw_solution[population_range, :] ≈ expected_moments rtol = 2e-9 atol = 1e-25
    @test moment_solution.concentration ≈
          [(3.0 * 7.0 + 0.25 * time_value * 1.5) / (3.0 + 0.25 * time_value)
           for time_value in saved_times] rtol = 2e-9
end

function _mesh_growth_boundary_balance(mesh_solver)
    numerical_solver, time_end = mesh_solver
    initial_crystals = GaussianInitialCrystals(mass_concentration = 4.0,
                                               d43 = 18e-6,
                                               standard_deviation = 0.5e-6)
    batch_problem = _mesh_operation_problem(BatchOperation(volume = 2.0),
        numerical_solver; initial_concentration_value = 10.0,
        initial_crystals, growth_rate_value = 0.5e-6)
    saved_times = collect(range(0.0, time_end; length = 121))
    _, result = _solve_mesh_operation(batch_problem, saved_times)
    @test result.success

    solid_mass_concentration = [
        batch_problem.crystal_density * batch_problem.volume_shape_factor *
        numerical_solver.cell_dL *
        sum(result.numberdensity[:, time_index] .*
            numerical_solver.cell_centre.^3)
        for time_index in eachindex(saved_times)]
    total_mass = 2.0 .* (result.concentration .+ solid_mass_concentration)
    boundary_flows = reactor_vars(result)
    total_boundary_mass_loss = _mesh_operation_trapezoid(
        saved_times,
        boundary_flows.size_boundary_lower_solid_mass_flow .+
        boundary_flows.size_boundary_upper_solid_mass_flow)
    total_inventory_loss = first(total_mass) - last(total_mass)
    relative_balance_error = abs(total_inventory_loss - total_boundary_mass_loss) /
                             first(total_mass)
    return (; relative_balance_error, total_boundary_mass_loss,
              total_inventory_loss, boundary_flows)
end

@testset "Mesh growth reports size-boundary mass loss and refines inventory balance" begin
    for solver_type in (FiniteVol, WENO)
        coarse_solver = solver_type(meshsize = 32, lmin = 0.0, lmax = 20e-6,
                                    reltol = 1e-9, abstol = 1e-12)
        fine_solver = solver_type(meshsize = 64, lmin = 0.0, lmax = 20e-6,
                                  reltol = 1e-9, abstol = 1e-12)
        coarse_balance = _mesh_growth_boundary_balance((coarse_solver, 6.0))
        fine_balance = _mesh_growth_boundary_balance((fine_solver, 6.0))

        @test all(iszero, coarse_balance.boundary_flows.size_boundary_lower_solid_mass_flow)
        @test all(iszero, fine_balance.boundary_flows.size_boundary_lower_solid_mass_flow)
        @test coarse_balance.total_boundary_mass_loss > 0.0
        @test fine_balance.total_boundary_mass_loss > 0.0
        @test coarse_balance.relative_balance_error < 0.03
        @test fine_balance.relative_balance_error < 0.03
        @test fine_balance.relative_balance_error < coarse_balance.relative_balance_error
    end
end
