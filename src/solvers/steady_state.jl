"""Result of relaxing an autonomous MSMPR to a numerically checked steady state.

`physical_result` is the solver-specific result wrapper at the final saved
state. Thus `observable_values`, `state_vars`, and `size_metrics` retain the
same names as a transient solution while each trajectory contains one value.
`final_state` uses the public physical solver coordinates; the native ODE state
is retained in diagnostics. `residual` is the native ODE right-hand side;
`scaled_residual` multiplies each component by the residence time and divides
by its characteristic state scale.
"""
@concrete struct CrystallisationSteadyStateSolution <: AbstractSolution
    physical_result
    final_state
    residual
    scaled_residual
    residual_norm
    scaled_residual_norm
    hydraulics::NamedTuple
    product_dissolved_solute_flow
    product_solid_mass_flow
    diagnostics::NamedTuple
    ode_stats
    retcode
    converged::Bool
    success::Bool
end

"""A direct steady solve ended without a validated physical equilibrium."""
struct SteadyStateConvergenceError{TSolution <: CrystallisationSteadyStateSolution} <: Exception
    solution::TSolution
end

function Base.showerror(io::IO, error::SteadyStateConvergenceError)
    result = error.solution
    print(io, "MSMPR steady-state solve failed: retcode=$(result.retcode), ",
          "converged=$(result.converged), scaled residual=", result.scaled_residual_norm,
          ", physical status=", result.diagnostics.physical_state_status,
          ", flow balance valid=", result.diagnostics.balance_valid, ".")
end

function Base.getproperty(solution::CrystallisationSteadyStateSolution,
                          name::Symbol)
    if name in fieldnames(typeof(solution))
        return getfield(solution, name)
    end
    return getproperty(getfield(solution, :physical_result), name)
end

function Base.propertynames(solution::CrystallisationSteadyStateSolution,
                            private::Bool = false)
    own_names = fieldnames(typeof(solution))
    result_names = propertynames(getfield(solution, :physical_result), private)
    return Tuple(unique((own_names..., result_names...)))
end

"""Named state trajectories carried by a one-point steady result."""
function state_vars(solution::CrystallisationSteadyStateSolution)
    result = solution.physical_result
    named_state = hasmethod(state_vars, Tuple{typeof(result)}) ?
                  state_vars(result) :
                  merge(result.solvent_state, reactor_vars(result))
    named_state = merge(named_state, reactor_vars(solution))
    return named_state
end

reactor_vars(solution::CrystallisationSteadyStateSolution) =
    (; volume = fill(solution.hydraulics.volume,
                     length(solution.physical_result.time)))

"""Particle-size metrics exposed by the solver-specific one-point result."""
function size_metrics(solution::CrystallisationSteadyStateSolution)
    result = solution.physical_result
    if hasmethod(size_metrics, Tuple{typeof(result)})
        return size_metrics(result)
    end
    return (; d10 = result.d10,
            d32 = result.d32,
            d43 = result.d43,
            moment2 = result.moment2)
end

"""
    observable_values(solution::CrystallisationSteadyStateSolution, name::Symbol)

Return a one-element trajectory for scalar state or size observables. Resolved
population fields retain their population-by-time matrix representation.
"""
function observable_values(solution::CrystallisationSteadyStateSolution,
                           name::Symbol)
    named_values = merge(state_vars(solution), reactor_vars(solution),
                         size_metrics(solution))
    hasproperty(named_values, name) ||
        throw(ArgumentError("No simulated observable named :$name for $(typeof(solution))."))
    value = getproperty(named_values, name)
    return value isa AbstractArray ? value : [value]
end

function _steady_state_autonomy_issues(problem::CrystallisationProblem)
    operation = problem.operation
    issues = Symbol[]
    operation.inflow isa Real || push!(issues, :inflow)
    all(value -> value isa Real, values(operation.feed.solvent_state)) ||
        push!(issues, :feed_profile)
    problem.temp_profile isa ConstantTemperature || push!(issues, :temperature_profile)
    if !(problem.saturation_model isa ConstantSolubility ||
         problem.saturation_model isa PolynomialSolubility)
        push!(issues, :saturation_model)
    end
    problem.solvent_dynamics === default_solvent_dynamics ||
        push!(issues, :solvent_dynamics)
    isnothing(operation.feed.transport) || push!(issues, :feed_transport)

    for (name, kinetic_model) in ((:nucleation_kinetics, problem.kinetics_nucleationfunction),
                                  (:growth_kinetics, problem.kinetics_growthfunction),
                                  (:dissolution_kinetics, problem.kinetics_dissolutionfunction),
                                  (:aggregation_kinetics, problem.kinetics_aggregationfunction),
                                  (:breakage_kinetics, problem.kinetics_breakagefunction))
        parentmodule(typeof(kinetic_model)) === parentmodule(typeof(problem)) ||
            push!(issues, name)
    end
    return unique(issues)
end

function _validate_steadystate_problem(problem::CrystallisationProblem;
                                       autonomous::Bool = false)
    problem.operation isa MSMPROperation ||
        throw(ArgumentError("solve_steadystate only supports fixed-volume MSMPROperation problems."))
    _validate_crystallisation_problem(problem)
    flow = operation_flows(problem.operation, 0.0).inflow
    flow > 0.0 ||
        throw(ArgumentError("A steady MSMPR solve requires a strictly positive constant inflow."))
    issues = _steady_state_autonomy_issues(problem)
    if !autonomous && !isempty(issues)
        issue_list = join(string.(issues), ", ")
        throw(ArgumentError("Steady-state solving requires autonomous operation profiles and kinetics; " *
                            "declare autonomous=true only when every opaque callable is time invariant. " *
                            "Profiles needing an explicit declaration: $issue_list."))
    end

    # The steady result always reports dissolved and third-moment solid product
    # flow. A lower-order MoM cannot supply that inventory independently.
    if problem.solver isa MoM && problem.solver.nmoments < 3
        throw(ArgumentError("A steady MoM result requires at least three tracked moments to report solid product flow."))
    end
    return nothing
end

function _steady_rhs(ode_problem, state, parameters, simulation_time)
    if OrdinaryDiffEq.SciMLBase.isinplace(ode_problem.f, length(state))
        derivative = similar(state)
        ode_problem.f(derivative, state, parameters, simulation_time)
        return derivative
    end
    return ode_problem.f(state, parameters, simulation_time)
end

function _steady_state_reference_scales(problem, ode_problem, user_scales)
    state = collect(ode_problem.u0)
    if !isnothing(user_scales)
        length(user_scales) == length(state) ||
            throw(ArgumentError("residual_scales has length $(length(user_scales)); " *
                                "expected $(length(state)) for the ODE state."))
        all(scale -> scale isa Real && isfinite(scale) && scale > 0.0, user_scales) ||
            throw(ArgumentError("residual_scales must contain finite positive state scales."))
        return collect(user_scales)
    end

    scales = abs.(state)
    feed_population = _operation_feed_population(problem)
    if !(problem.solver isa DQMOM)
        population_range = _population_state_range(problem)
        @inbounds for (state_index, population_index) in zip(population_range,
                                                              eachindex(feed_population))
            scales[state_index] = max(scales[state_index], abs(feed_population[population_index]))
        end
    end
    feed_concentration = _operation_feed_values(problem.operation.feed,
                                                 ode_problem.tspan[1]).concentration
    concentration_index = _solvent_state_index(problem, :concentration)
    scales[concentration_index] = max(scales[concentration_index],
                                       abs(feed_concentration))
    return scales
end

function _steady_scaled_residual(problem, ode_problem, state, parameters,
                                 simulation_time, residence_time, reference_scales;
                                 shared_population_scale::Bool = false)
    residual = _steady_rhs(ode_problem, state, parameters, simulation_time)
    scaled = similar(residual)
    population_range = _population_state_range(problem)
    population_scale = shared_population_scale ?
        max(maximum(abs, view(state, population_range)),
            maximum(reference_scales[population_range]), floatmin(Float64)) : zero(eltype(reference_scales))
    @inbounds for state_index in eachindex(residual, state, reference_scales)
        scale = shared_population_scale && state_index in population_range ? population_scale :
            max(abs(state[state_index]), reference_scales[state_index], floatmin(Float64))
        scaled[state_index] = abs(residual[state_index]) * residence_time / scale
    end
    return residual, scaled
end

function _steady_autostop_callback(problem, ode_problem, reference_scales,
                                   residence_time, minimum_relaxation,
                                   residual_reltol, residual_abstol;
                                   shared_population_scale::Bool = false)
    start_time = ode_problem.tspan[1]
    tolerance = residual_reltol + residual_abstol
    condition = (state, simulation_time, integrator) -> begin
        simulation_time - start_time >= minimum_relaxation || return false
        simulation_time > start_time || return false
        _, scaled = _steady_scaled_residual(problem, ode_problem, state,
                                            integrator.p, simulation_time,
                                            residence_time, reference_scales; shared_population_scale)
        return all(isfinite, scaled) && maximum(scaled) <= tolerance
    end
    return DiscreteCallback(condition,
                            integrator -> OrdinaryDiffEq.SciMLBase.terminate!(integrator);
                            save_positions = (false, false))
end

function _steady_moment3(problem, raw_state, physical_result)
    if problem.solver isa MoM
        return raw_state[4]
    elseif problem.solver isa QMOM || problem.solver isa DQMOM
        return physical_result.moments[4, end]
    end
    population = @view raw_state[_population_state_range(problem)]
    return momentcalculator(problem.solver.cell_centre, population, 3)
end

function _steady_feed_moment3(problem, feed_population)
    if problem.solver isa AbstractDiscretisedSolver
        return momentcalculator(problem.solver.cell_centre, feed_population, 3)
    end
    return feed_population[4]
end

function _steady_physical_constraints(problem, raw_state, physical_result)
    all(isfinite, raw_state) || return false, :nonfinite_state
    concentration = raw_state[_solvent_state_index(problem, :concentration)]
    concentration >= 0.0 || return false, :negative_concentration

    if problem.solver isa AbstractDiscretisedSolver
        all(value -> value >= 0.0, @view raw_state[_population_state_range(problem)]) ||
            return false, :negative_number_density
    elseif problem.solver isa DQMOM
        all(value -> isfinite(value) && value > 0.0, physical_result.weights) ||
            return false, :nonpositive_quadrature_weight
        all(value -> isfinite(value) && value > problem.solver.minimum_size,
            physical_result.nodes) || return false, :invalid_quadrature_node
    elseif problem.solver isa QMOM
        all(diagnostic -> diagnostic.status in (:ok, :empty, :deflated),
            physical_result.inversion_diagnostics) || return false, :unrealizable_moments
    elseif problem.solver isa MoM
        moments = @view raw_state[_population_state_range(problem)]
        all(value -> isfinite(value) && value >= 0.0, moments) ||
            return false, :negative_moment
        for index in 2:(length(moments) - 1)
            left = moments[index]^2
            right = moments[index - 1] * moments[index + 1]
            left <= right + 1e-10 * max(abs(left), abs(right), floatmin(Float64)) ||
                return false, :non_logconvex_moments
        end
    end
    return true, :ok
end

function _steady_product_diagnostics(problem, raw_state, physical_result,
                                     simulation_time)
    operation = problem.operation::MSMPROperation
    flows = operation_flows(operation, simulation_time)
    feed_values = _operation_feed_values(operation.feed, simulation_time)
    volume = reactor_volume(problem, raw_state)
    residence_time = volume / flows.inflow
    moment3 = _steady_moment3(problem, raw_state, physical_result)
    feed_population = _operation_feed_population(problem)
    feed_moment3 = _steady_feed_moment3(problem, feed_population)
    concentration = raw_state[_solvent_state_index(problem, :concentration)]
    density = problem.crystal_density * problem.volume_shape_factor
    inlet_dissolved_flow = flows.inflow * feed_values.concentration
    outlet_dissolved_flow = flows.outflow * concentration
    inlet_solid_flow = flows.inflow * density * feed_moment3
    outlet_solid_flow = flows.outflow * density * moment3
    total_inlet_flow = inlet_dissolved_flow + inlet_solid_flow
    total_outlet_flow = outlet_dissolved_flow + outlet_solid_flow
    boundary_values = reactor_vars(physical_result)
    lower_boundary_flow = hasproperty(boundary_values, :size_boundary_lower_solid_mass_flow) ?
        last(boundary_values.size_boundary_lower_solid_mass_flow) : zero(total_outlet_flow)
    upper_boundary_flow = hasproperty(boundary_values, :size_boundary_upper_solid_mass_flow) ?
        last(boundary_values.size_boundary_upper_solid_mass_flow) : zero(total_outlet_flow)
    size_boundary_flow = lower_boundary_flow + upper_boundary_flow
    balance_residual = total_inlet_flow - total_outlet_flow - size_boundary_flow
    balance_scale = max(abs(total_inlet_flow), abs(total_outlet_flow), abs(size_boundary_flow), floatmin(Float64))
    relative_balance_error = abs(balance_residual) / balance_scale
    hydraulics = (; volume, inflow = flows.inflow, outflow = flows.outflow,
                    residence_time)
    product = (; dissolved_solute = outlet_dissolved_flow,
                 solid_mass = outlet_solid_flow)
    balance = (; inlet_dissolved_solute = inlet_dissolved_flow,
                 inlet_solid_mass = inlet_solid_flow,
                 outlet_dissolved_solute = outlet_dissolved_flow,
                 outlet_solid_mass = outlet_solid_flow,
                 total_inlet_api = total_inlet_flow,
                 total_outlet_api = total_outlet_flow,
                 size_boundary_lower_solid_mass_flow = lower_boundary_flow,
                 size_boundary_upper_solid_mass_flow = upper_boundary_flow,
                 total_api_residual = balance_residual,
                 relative_error = relative_balance_error)
    return hydraulics, product, balance
end

function _steady_option(options::NamedTuple, name::Symbol, default)
    return hasproperty(options, name) ? getproperty(options, name) : default
end

"""
    _solve_steadystate_ode(problem, ode_problem, algorithm;
                           steady_options=(;), solve_options=(;),
                           callback_factory=nothing)

Internal prepared-loss entry point. `ode_problem` is authoritative: in
particular, its `p` may be a candidate-specific `ComponentArray` with Dual
values. The returned result records failure in `success`, `converged`, and
`diagnostics` instead of discarding the solver retcode.
"""
function _solve_steadystate_ode(problem::CrystallisationProblem, ode_problem,
                                algorithm;
                                steady_options::NamedTuple = (;),
                                solve_options::NamedTuple = (;),
                                callback_factory = nothing)
    allowed_options = (:relaxation_horizon, :minimum_relaxation,
                       :residual_reltol, :residual_abstol, :residual_scales,
                       :autonomous, :balance_reltol)
    all(option -> option in allowed_options, keys(steady_options)) ||
        throw(ArgumentError("steady_options contains an unsupported option."))

    explicitly_autonomous = _steady_option(steady_options, :autonomous, false)
    explicitly_autonomous isa Bool ||
        throw(ArgumentError("steady_options.autonomous must be Bool."))
    _validate_steadystate_problem(problem; autonomous = explicitly_autonomous)

    start_time, original_end_time = ode_problem.tspan
    default_horizon = original_end_time - start_time
    horizon = _steady_option(steady_options, :relaxation_horizon, default_horizon)
    horizon isa Real && isfinite(horizon) && horizon > 0.0 ||
        throw(ArgumentError("relaxation_horizon must be finite and strictly positive."))
    end_time = start_time + horizon
    active_ode_problem = end_time == original_end_time ? ode_problem :
                         remake(ode_problem; tspan = (start_time, end_time))

    minimum_relaxation = _steady_option(steady_options, :minimum_relaxation, 0.0)
    minimum_relaxation isa Real && isfinite(minimum_relaxation) &&
        minimum_relaxation >= 0.0 && minimum_relaxation <= horizon ||
        throw(ArgumentError("minimum_relaxation must lie between zero and relaxation_horizon."))
    residual_reltol = _steady_option(steady_options, :residual_reltol, 1e-10)
    residual_abstol = _steady_option(steady_options, :residual_abstol, 1e-12)
    all(tolerance -> tolerance isa Real && isfinite(tolerance) && tolerance >= 0.0,
        (residual_reltol, residual_abstol)) ||
        throw(ArgumentError("Residual tolerances must be finite and nonnegative."))
    residual_reltol + residual_abstol > 0.0 ||
        throw(ArgumentError("At least one residual tolerance must be strictly positive."))
    balance_reltol = _steady_option(steady_options, :balance_reltol, 1e-5)
    balance_reltol isa Real && isfinite(balance_reltol) && balance_reltol > 0.0 ||
        throw(ArgumentError("balance_reltol must be finite and strictly positive."))

    flow = operation_flows(problem.operation, start_time).inflow
    volume = problem.operation.volume
    residence_time = volume / flow
    supplied_residual_scales = _steady_option(steady_options, :residual_scales, nothing)
    reference_scales = _steady_state_reference_scales(problem, active_ode_problem, supplied_residual_scales)
    shared_population_scale = problem.solver isa AbstractDiscretisedSolver && supplied_residual_scales === nothing
    convergence_callback = _steady_autostop_callback(
        problem, active_ode_problem, reference_scales, residence_time,
        minimum_relaxation, residual_reltol, residual_abstol; shared_population_scale)

    combined_callback_factory = ode_problem_for_solve -> begin
        custom_callback = callback_factory === nothing ? nothing :
                          callback_factory(ode_problem_for_solve)
        CallbackSet(convergence_callback, custom_callback)
    end
    # A transient solver tolerance can be much looser than the requested
    # equilibrium residual. Steady relaxation defaults to consistent precision
    # and a hydraulic step bound; explicit caller solve options still win.
    steady_solve_options = merge((;
        reltol = min(problem.solver.reltol,
            (residual_reltol + residual_abstol) / (100 * sqrt(length(ode_problem.u0)))),
        dtmax = residence_time / 4), solve_options)
    raw_solution = _solve_crystallisation_ode(
        problem, active_ode_problem, algorithm, [end_time]; solve_options = steady_solve_options,
        callback_factory = combined_callback_factory)

    final_ode_state = raw_solution.u[end]
    raw_state = collect(final_ode_state)
    final_time = raw_solution.t[end]
    actual_parameters = raw_solution.prob.p
    residual, scaled_residual = _steady_scaled_residual(
        problem, active_ode_problem, final_ode_state, actual_parameters, final_time,
        residence_time, reference_scales; shared_population_scale)
    residual_norm = norm(residual, Inf)
    scaled_residual_norm = maximum(scaled_residual)
    solver_success = OrdinaryDiffEq.SciMLBase.successful_retcode(raw_solution.retcode)
    residual_limit = residual_reltol + residual_abstol
    converged = solver_success && all(isfinite, scaled_residual) &&
                scaled_residual_norm <= residual_limit

    applicable(_wrap_solution, problem, raw_solution) ||
        throw(ArgumentError("No physical result wrapper is available for $(typeof(problem.solver)) with the selected kinetics."))
    physical_result = _wrap_solution(problem, raw_solution)
    physical_valid, physical_status = _steady_physical_constraints(
        problem, raw_state, physical_result)
    hydraulics, product, balance = _steady_product_diagnostics(
        problem, raw_state, physical_result, final_time)
    balance_valid = balance.relative_error <= balance_reltol
    successful = solver_success && converged && physical_valid && balance_valid
    diagnostics = (; converged, residual_norm, scaled_residual_norm, residual_limit,
                    residual_scales = reference_scales,
                    population_scaling = shared_population_scale ? :density_peak : :per_component,
                    physical_state_valid = physical_valid,
                    physical_state_status = physical_status,
                    total_api_balance = balance,
                    balance_reltol,
                    balance_valid,
                    equilibrium_time = final_time,
                    ode_final_state = raw_state,
                    integration_options = steady_solve_options,
                    solver_success)

    return CrystallisationSteadyStateSolution(physical_result, physical_result.final_state,
                                               residual, scaled_residual,
                                               residual_norm,
                                               scaled_residual_norm,
                                               hydraulics,
                                               product.dissolved_solute,
                                               product.solid_mass,
                                               diagnostics,
                                               raw_solution.stats,
                                               raw_solution.retcode,
                                               converged,
                                               successful)
end

"""
    solve_steadystate(problem::CrystallisationProblem;
                      initial_guess=nothing, relaxation_horizon=nothing,
                      minimum_relaxation=0.0, residual_reltol=1e-10,
                      residual_abstol=1e-12, residual_scales=nothing,
                      solve_options=(;), algorithm=nothing,
                      callback_factory=nothing, autonomous=false)

Relax a constant, autonomous fixed-volume MSMPR to a stable steady state.
The solver integrates the generated ODE and stops early only when the scaled
full-state RHS residual is within tolerance. It returns a distinct
`CrystallisationSteadyStateSolution`; `success` is false if the relaxation
horizon, residual, physical constraints, or total-API flow balance fails.
If those checks fail, the direct call throws `SteadyStateConvergenceError`,
which carries the unsuccessful result and its diagnostics. All built-in state
and size observables contain one value on the result.

Opaque time profiles and user-defined kinetics require `autonomous=true` as an
explicit declaration that their behavior is time invariant. Batch and
fed-batch problems remain transient `runsimulation` workflows.
"""
function solve_steadystate(problem::CrystallisationProblem;
                           initial_guess::Union{Nothing, AbstractVector} = nothing,
                           relaxation_horizon = nothing,
                           minimum_relaxation::Real = 0.0,
                           residual_reltol::Real = 1e-10,
                           residual_abstol::Real = 1e-12,
                           residual_scales = nothing,
                           solve_options::NamedTuple = (;),
                           algorithm = nothing,
                           callback_factory = nothing,
                           autonomous::Bool = false,
                           balance_reltol::Real = 1e-5)
    _validate_steadystate_problem(problem; autonomous)
    flow = operation_flows(problem.operation, 0.0).inflow
    residence_time = problem.operation.volume / flow
    horizon = isnothing(relaxation_horizon) ? 100.0 * residence_time :
              relaxation_horizon
    horizon isa Real && isfinite(horizon) && horizon > 0.0 ||
        throw(ArgumentError("relaxation_horizon must be finite and strictly positive."))
    minimum_relaxation <= horizon ||
        throw(ArgumentError("minimum_relaxation must not exceed relaxation_horizon."))

    solve_problem = if isnothing(initial_guess)
        problem
    else
        guessed_problem = _copy_crystallisation_problem(
            problem; initial_state = collect(initial_guess))
        _validate_crystallisation_problem(guessed_problem)
        guessed_problem
    end
    ode_problem, default_algorithm = crystallisation_odeproblem(
        solve_problem, (0.0, horizon))
    steady_options = (; relaxation_horizon = horizon,
                       minimum_relaxation,
                       residual_reltol,
                       residual_abstol,
                       residual_scales,
                       autonomous,
                       balance_reltol)
    solution = _solve_steadystate_ode(
        solve_problem, ode_problem,
        algorithm === nothing ? default_algorithm : algorithm;
        steady_options, solve_options, callback_factory)
    solution.success || throw(SteadyStateConvergenceError(solution))
    return solution
end
