struct WenoOutputBoundaryNucleation <: CriSTool.AbstractFPNucleationFunction
    nparams::Int
end
CriSTool.paramaxis(model::WenoOutputBoundaryNucleation) =
    CriSTool.ComponentArrays.Axis(weno_output_nucleation = model.nparams)
function CriSTool.nucleationrate(::WenoOutputBoundaryNucleation, parameters,
                                 problem, state, time)
    return parameters[1] * (1 + time / 10)
end

struct WenoOutputScalarGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(model::WenoOutputScalarGrowth) =
    CriSTool.ComponentArrays.Axis(weno_output_scalar_growth = model.nparams)
CriSTool.growthrate(::WenoOutputScalarGrowth, parameters, problem, state, time) =
    parameters[1]

struct WenoOutputLengthGrowth <: CriSTool.AbstractFPLengthGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(model::WenoOutputLengthGrowth) =
    CriSTool.ComponentArrays.Axis(weno_output_length_growth = model.nparams)
function CriSTool.growthrate!(destination::AbstractVector,
                              ::WenoOutputLengthGrowth, parameters, problem,
                              state, time, mesh::AbstractVector)
    cell_count = length(mesh)
    concentration = CriSTool.solvent_state(problem, state).concentration
    volume = CriSTool.reactor_volume(problem, state)
    @inbounds for cell_index in eachindex(mesh, destination)
        destination[cell_index] = parameters[cell_index] +
            parameters[cell_count + cell_index] * time +
            1e-9 * concentration + 1e-10 * volume + 1e-3 * mesh[cell_index]
    end
    return destination
end

function _weno_output_boundary_problem(growth_model, growth_parameters;
                                      operation = BatchOperation(volume = 2.5),
                                      nucleation_parameters = [8.0])
    return CrystallisationProblem(
        ; operation,
        solver = WENO(meshsize = 4, lmin = 1.0, lmax = 5.0),
        crystal_density = 1500.0,
        volume_shape_factor = 0.5,
        initial_concentration = 5.0,
        initial_solvent_state = (; concentration = 5.0),
        saturation_model = ConstantSolubility(1.0),
        kinetics_nucleationfunction = WenoOutputBoundaryNucleation(1),
        parameterset_nucleation = nucleation_parameters,
        kinetics_growthfunction = growth_model,
        parameterset_growth = growth_parameters)
end

function _weno_output_state(problem, numberdensity; volume = nothing,
                            concentration = 5.0)
    operation_state = problem.operation isa FedBatchOperation ? [volume] : Float64[]
    return vcat(numberdensity, operation_state, [concentration])
end

function _weno_output_saved_solution(problem, saved_times, saved_states;
                                      parameters = (; nucl = problem.parameterset_nucleation,
                                          gr = problem.parameterset_growth,
                                          diss = problem.parameterset_dissolution))
    return (; t = saved_times, u = saved_states, prob = (; p = parameters))
end

@testset "WENO boundary outputs match the signed face flux formulas" begin
    scalar_state = [2.0, 3.0, 4.0, 5.0]
    saved_times = [0.0, 2.0]
    scalar_problem = _weno_output_boundary_problem(WenoOutputScalarGrowth(1), [0.25])
    scalar_states = [_weno_output_state(scalar_problem, scalar_state;
                                        concentration = 5.0 + time)
                     for time in saved_times]
    scalar_ode_solution = _weno_output_saved_solution(scalar_problem, saved_times, scalar_states)
    scalar_flows = CriSTool._mesh_boundary_flow_state(
        scalar_problem.solver, scalar_problem, scalar_ode_solution)
    scalar_scale = scalar_problem.crystal_density * scalar_problem.volume_shape_factor
    expected_nucleation = [8.0 * (1 + time / 10) for time in saved_times]
    expected_upper_trace = scalar_state[end] +
                           0.5 * (scalar_state[end] - scalar_state[end - 1])
    @test scalar_flows.size_boundary_lower_solid_mass_flow ≈
          [-rate * scalar_problem.solver.cell_face[1]^3 * scalar_scale * 2.5
           for rate in expected_nucleation]
    @test scalar_flows.size_boundary_upper_solid_mass_flow ≈
          fill(0.25 * expected_upper_trace * scalar_problem.solver.cell_face[end]^3 *
               scalar_scale * 2.5, length(saved_times))

    negative_trace_state = [2.0, 3.0, 4.0, 1.0]
    negative_trace_flows = CriSTool._mesh_boundary_flow_state(
        scalar_problem.solver, scalar_problem,
        _weno_output_saved_solution(scalar_problem, [0.0], [_weno_output_state(scalar_problem, negative_trace_state)]))
    @test iszero(first(negative_trace_flows.size_boundary_upper_solid_mass_flow))

    dissolution_problem = _weno_output_boundary_problem(
        WenoOutputScalarGrowth(1), [-0.5])
    dissolution_flows = CriSTool._mesh_boundary_flow_state(
        dissolution_problem.solver, dissolution_problem,
        _weno_output_saved_solution(dissolution_problem, [0.0], [_weno_output_state(dissolution_problem, scalar_state)]))
    @test dissolution_flows.size_boundary_lower_solid_mass_flow ==
          [0.5 * scalar_state[1] * dissolution_problem.solver.cell_face[1]^3 *
           scalar_scale * 2.5]
    @test iszero(first(dissolution_flows.size_boundary_upper_solid_mass_flow))

    stagnant_problem = _weno_output_boundary_problem(WenoOutputScalarGrowth(1), [0.0])
    stagnant_flows = CriSTool._mesh_boundary_flow_state(
        stagnant_problem.solver, stagnant_problem,
        _weno_output_saved_solution(stagnant_problem, [0.0], [_weno_output_state(stagnant_problem, scalar_state)]))
    @test iszero(first(stagnant_flows.size_boundary_lower_solid_mass_flow))
    @test iszero(first(stagnant_flows.size_boundary_upper_solid_mass_flow))
end

@testset "WENO length-dependent boundary outputs use the configured operation volume" begin
    solver = WENO(meshsize = 4, lmin = 1.0, lmax = 5.0)
    fed_operation = FedBatchOperation(
        initial_volume = 2.0, inflow = 0.5,
        feed = CrystallisationFeed(concentration = 4.0))
    growth_parameters = [-0.2e-6, 0.1e-6, -0.1e-6, 0.3e-6,
                         0.2e-6, 0.0, 0.0, -0.2e-6]
    growth_model = WenoOutputLengthGrowth(length(growth_parameters))
    problem = _weno_output_boundary_problem(growth_model, growth_parameters;
                                             operation = fed_operation)
    saved_times = [0.0, 2.0]
    populations = ([2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0])
    volumes = [2.0, 3.0]
    concentrations = [5.0, 4.0]
    states = [_weno_output_state(problem, populations[index];
                                 volume = volumes[index],
                                 concentration = concentrations[index])
              for index in eachindex(saved_times)]
    flows = CriSTool._mesh_boundary_flow_state(
        solver, problem, _weno_output_saved_solution(problem, saved_times, states))

    # The custom growth law includes time, volume, concentration, and mesh
    # context. Its first/last rates have opposite signs at each saved time.
    expected_lower_face_flux = Float64[]
    expected_upper_face_flux = Float64[]
    for sample_index in eachindex(saved_times)
        simulation_time = saved_times[sample_index]
        mesh = solver.cell_centre
        context = 1e-9 * concentrations[sample_index] +
                  1e-10 * volumes[sample_index]
        lower_rate = growth_parameters[1] +
                     growth_parameters[5] * simulation_time +
                     context + 1e-3 * mesh[1]
        upper_rate = growth_parameters[4] +
                     growth_parameters[8] * simulation_time +
                     context + 1e-3 * mesh[end]
        lower_flux = lower_rate > 0 ? 8.0 * (1 + simulation_time / 10) :
                     lower_rate < 0 ? lower_rate * populations[sample_index][1] : 0.0
        upper_trace = populations[sample_index][end] +
                      0.5 * (populations[sample_index][end] -
                             populations[sample_index][end - 1])
        upper_flux = upper_rate > 0 ? upper_rate * upper_trace : 0.0
        push!(expected_lower_face_flux, lower_flux)
        push!(expected_upper_face_flux, upper_flux)
    end

    mass_scale = problem.crystal_density * problem.volume_shape_factor
    @test flows.size_boundary_lower_solid_mass_flow ≈
          [-expected_lower_face_flux[index] * solver.cell_face[1]^3 *
           mass_scale * volumes[index] for index in eachindex(saved_times)]
    @test flows.size_boundary_upper_solid_mass_flow ≈
          [expected_upper_face_flux[index] * solver.cell_face[end]^3 *
           mass_scale * volumes[index] for index in eachindex(saved_times)]
end

@testset "Fused mesh moments retain physical cell-width integration and AD" begin
    mesh = LinRange(1.0e-6, 5.0e-6, 3)
    numberdensity = [2.0 1.0; 1.0 3.0; 0.0 2.0]
    sizes = CriSTool._momentsizes(mesh, numberdensity)

    @test sizes.moment2 ≈ [22.0e-18, 156.0e-18] rtol = 1e-14 atol = 0.0
    @test sizes.d10 ≈ [(5 / 3) * 1e-6, (10 / 3) * 1e-6] rtol = 1e-14
    @test sizes.d32 ≈ [(29 / 11) * 1e-6, (166 / 39) * 1e-6] rtol = 1e-14
    @test sizes.d43 ≈ [(83 / 29) * 1e-6, 4.5e-6] rtol = 1e-14

    density_column = [2.0, 1.0, 0.0]
    moment2_derivative = ForwardDiff.derivative(1.0) do scale
        scaled_density = reshape(scale .* density_column, :, 1)
        return first(CriSTool._momentsizes(mesh, scaled_density).moment2)
    end
    @test moment2_derivative ≈ 22.0e-18 rtol = 1e-14 atol = 0.0
end

@testset "Mesh boundary flows use candidate kinetics, including signed dissolution" begin
    # Linear density [2,3,4,5] has upper face trace 5.5 on [1,5].
    # rho*kv*V = 1500*0.5*2.5 = 1875; upper length cubed = 125.
    for candidate_mesh_solver in (FiniteVol(meshsize = 4, lmin = 1.0, lmax = 5.0),
                                  WENO(meshsize = 4, lmin = 1.0, lmax = 5.0))
        boundary_template = CriSTool._copy_crystallisation_problem(
            _weno_output_boundary_problem(WenoOutputScalarGrowth(1), [0.25]);
            solver = candidate_mesh_solver)
        boundary_states = [[2.0, 3.0, 4.0, 5.0, 5.0]]
        positive_candidate = _weno_output_saved_solution(boundary_template, [0.0], boundary_states;
            parameters = (; nucl = [16.0], gr = [0.5], diss = Float64[]))
        candidate_flows = CriSTool._mesh_boundary_flow_state(
            candidate_mesh_solver, boundary_template, positive_candidate)
        @test candidate_flows.size_boundary_lower_solid_mass_flow == [-30_000.0]
        @test candidate_flows.size_boundary_upper_solid_mass_flow == [644_531.25]

        # A candidate reversing the template's positive growth must change
        # both boundary directions: lower outflow, zero upper inflow.
        negative_candidate = _weno_output_saved_solution(boundary_template, [0.0], boundary_states;
            parameters = (; nucl = [16.0], gr = [-0.5], diss = Float64[]))
        negative_flows = CriSTool._mesh_boundary_flow_state(
            candidate_mesh_solver, boundary_template, negative_candidate)
        @test negative_flows.size_boundary_lower_solid_mass_flow == [1_875.0]
        @test negative_flows.size_boundary_upper_solid_mass_flow == [0.0]

        candidate_boundary_derivative = ForwardDiff.derivative(0.5) do candidate_speed
            derivative_solution = _weno_output_saved_solution(boundary_template,
                [0.0], [zero(candidate_speed) .+ only(boundary_states)];
                parameters = (; nucl = [16.0], gr = [candidate_speed], diss = Float64[]))
            only(CriSTool._mesh_boundary_flow_state(candidate_mesh_solver,
                boundary_template, derivative_solution).size_boundary_upper_solid_mass_flow)
        end
        @test candidate_boundary_derivative == 1_289_062.5
    end
end
