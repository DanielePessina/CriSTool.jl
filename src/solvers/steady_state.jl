"""
    SteadyStateOptions(; relaxation_horizon=nothing, minimum_relaxation=0.0,
                        residual_reltol=1e-10, residual_abstol=1e-12,
                        residual_scales=nothing, autonomous=false,
                        balance_reltol=1e-5)

Settings for an MSMPR steady-state relaxation, validated on construction.

- `relaxation_horizon` (s): longest relaxation time; `nothing` uses 100
  residence times in `solve_steadystate` and the template ODE span in prepared
  inference.
- `minimum_relaxation` (s): earliest time at which the residual stop may fire.
- `residual_reltol`, `residual_abstol`: the run is converged when every scaled
  RHS component is at most their sum.
- `residual_scales`: optional positive per-state residual scales.
- `autonomous`: declare that opaque profiles and user kinetics are time
  invariant.
- `balance_reltol`: accepted relative total-API flow-balance error.
"""
struct SteadyStateOptions
    relaxation_horizon::Union{Nothing, Float64}
    minimum_relaxation::Float64
    residual_reltol::Float64
    residual_abstol::Float64
    residual_scales::Union{Nothing, Vector{Float64}}
    autonomous::Bool
    balance_reltol::Float64
    function SteadyStateOptions(relaxation_horizon, minimum_relaxation,
                                residual_reltol, residual_abstol,
                                residual_scales, autonomous, balance_reltol)
        isnothing(relaxation_horizon) ||
            (relaxation_horizon isa Real && isfinite(relaxation_horizon) &&
             relaxation_horizon > 0.0) ||
            throw(ArgumentError("relaxation_horizon must be finite and strictly positive."))
        minimum_relaxation isa Real && isfinite(minimum_relaxation) &&
            minimum_relaxation >= 0.0 ||
            throw(ArgumentError("minimum_relaxation must be finite and nonnegative."))
        isnothing(relaxation_horizon) || minimum_relaxation <= relaxation_horizon ||
            throw(ArgumentError("minimum_relaxation must not exceed relaxation_horizon."))
        all(tolerance -> tolerance isa Real && isfinite(tolerance) && tolerance >= 0.0,
            (residual_reltol, residual_abstol)) ||
            throw(ArgumentError("Residual tolerances must be finite and nonnegative."))
        residual_reltol + residual_abstol > 0.0 ||
            throw(ArgumentError("At least one residual tolerance must be strictly positive."))
        isnothing(residual_scales) ||
            all(scale -> scale isa Real && isfinite(scale) && scale > 0.0, residual_scales) ||
            throw(ArgumentError("residual_scales must contain finite positive state scales."))
        autonomous isa Bool || throw(ArgumentError("autonomous must be Bool."))
        balance_reltol isa Real && isfinite(balance_reltol) && balance_reltol > 0.0 ||
            throw(ArgumentError("balance_reltol must be finite and strictly positive."))
        return new(relaxation_horizon, minimum_relaxation, residual_reltol,
                   residual_abstol,
                   isnothing(residual_scales) ? nothing : collect(Float64, residual_scales),
                   autonomous, balance_reltol)
    end
end

function SteadyStateOptions(; relaxation_horizon = nothing, minimum_relaxation = 0.0,
                            residual_reltol = 1e-10, residual_abstol = 1e-12,
                            residual_scales = nothing, autonomous = false,
                            balance_reltol = 1e-5)
    return SteadyStateOptions(relaxation_horizon, minimum_relaxation, residual_reltol,
                              residual_abstol, residual_scales, autonomous,
                              balance_reltol)
end


"""
Result of relaxing an autonomous MSMPR to a numerically checked steady state.

`steady.equilibrium` holds the equilibrium values as one named tuple:
`time`, `concentration`, `d43`, `d32`, other solvent and size scalars, and
population vectors such as `moments` or `numberdensity`. The underlying
one-sample physical solver result is `steady.solution`; `observable_values`,
`state_vars` and `size_metrics` return its one-sample trajectories, so a
steady result can be used wherever a transient result is expected.

Fields:
- `solution`: one-sample `CrystallisationSolution` at the equilibrium time.
- `equilibrium`: the same values as scalars (and population vectors).
- `residual`, `scaled_residual`: native ODE right-hand side and its components
  multiplied by the residence time and divided by their state scale.
- `hydraulics`: `(; volume, inflow, outflow, residence_time)`.
- `product`: `(; dissolved_solute_flow, solid_mass_flow)` outlet flows (kg/s).
- `diagnostics`: convergence (`converged`, residual norms and limit), physical
  state, total-API balance and integration details. For discretised solvers,
  `population_density_diagnostics` records the raw density minimum, peak scale,
  negative-cell count and machine-roundoff tolerance; accepted undershoots stay
  in the raw state and are labeled in `physical_state_status`.
- `ode_stats`, `retcode`: from the relaxation solve.
- `success`: solver, residual, physical and balance checks all passed.
"""
@concrete struct CrystallisationSteadyStateSolution <: AbstractSolution
    solution
    equilibrium
    residual
    scaled_residual
    hydraulics
    product
    diagnostics
    ode_stats
    retcode
    success::Bool
end

"""A direct steady solve ended without a validated physical equilibrium."""
struct SteadyStateConvergenceError{TSolution <: CrystallisationSteadyStateSolution} <: Exception
    solution::TSolution
end

function Base.showerror(io::IO, error::SteadyStateConvergenceError)
    result = error.solution
    print(io, "MSMPR steady-state solve failed: retcode=$(result.retcode), ",
          "converged=$(result.diagnostics.converged), scaled residual=",
          result.diagnostics.scaled_residual_norm,
          ", physical status=", result.diagnostics.physical_state_status,
          ", flow balance valid=", result.diagnostics.balance_valid, ".")
end

# A steady result has one saved sample: trajectories collapse to that value
# and population-by-time matrices to their single column.
_steady_point_value(value::AbstractVector) = length(value) == 1 ? only(value) : value
_steady_point_value(value::AbstractMatrix) = size(value, 2) == 1 ? vec(value) : value
_steady_point_value(value) = value

# Equilibrium values of a one-sample solution: `time`, the solvent and reactor
# variables, population arrays as vectors and size metrics as scalars.
_equilibrium_values(endpoint_solution) =
    map(_steady_point_value, merge((; time = endpoint_solution.time), state_vars(endpoint_solution),
                                   size_metrics(endpoint_solution)))

state_vars(steady::CrystallisationSteadyStateSolution) = state_vars(steady.solution)
reactor_vars(steady::CrystallisationSteadyStateSolution) = reactor_vars(steady.solution)
size_metrics(steady::CrystallisationSteadyStateSolution) = size_metrics(steady.solution)

"""
    observable_values(steady::CrystallisationSteadyStateSolution, name::Symbol)

One-sample trajectory of a state or size observable at the equilibrium.
Resolved population fields keep their population-by-time matrix shape.
"""
observable_values(steady::CrystallisationSteadyStateSolution, name::Symbol) =
    observable_values(steady.solution, name)

function _steady_state_autonomy_issues(problem::CrystallisationProblem)
    operation = problem.operation
    issues = Symbol[]
    operation.inflow isa Real || push!(issues, :inflow)
    operation.feed.concentration isa Real &&
        all(value -> value isa Real, values(operation.feed.solvent_state)) ||
        push!(issues, :feed_profile)
    _is_time_invariant(problem.temp_profile) || push!(issues, :temperature_profile)
    _is_time_invariant(problem.model.system.saturation_model) || push!(issues, :saturation_model)
    problem.solvent_dynamics === default_solvent_dynamics ||
        push!(issues, :solvent_dynamics)
    isnothing(operation.feed.transport) || push!(issues, :feed_transport)

    for (name, kinetic_model) in ((:nucleation_kinetics, problem.model.nucleation.law),
                                  (:growth_kinetics, problem.model.growth.law),
                                  (:dissolution_kinetics, problem.model.dissolution.law),
                                  (:aggregation_kinetics, problem.model.aggregation.law),
                                  (:breakage_kinetics, problem.model.breakage.law))
        _is_time_invariant(kinetic_model) || push!(issues, name)
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

    _validate_steady_solver(problem.solver)
    return nothing
end

# The steady result always reports dissolved and third-moment solid product
# flow. A lower-order MoM cannot supply that inventory independently.
_validate_steady_solver(solver::AbstractSolver) = nothing
function _validate_steady_solver(solver::MoM)
    solver.nmoments >= 3 ||
        throw(ArgumentError("A steady MoM result requires at least three tracked moments to report solid product flow."))
    return nothing
end

function _steady_rhs(ode_problem, state, parameters, simulation_time)
    if SciMLBase.isinplace(ode_problem.f, length(state))
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
    _include_feed_population_scales!(scales, problem.solver, problem,
                                     _operation_feed_population(problem))
    feed_concentration = _feed_concentration(problem.operation.feed, ode_problem.tspan[1])
    concentration_index = _solvent_state_index(problem, :concentration)
    scales[concentration_index] = max(scales[concentration_index],
                                       abs(feed_concentration))
    return scales
end

"""
    _include_feed_population_scales!(scales, solver, problem, feed_population)

Raise population residual scales to the feed population. DQMOM feeds are
physical moments rather than weight/node states, so they do not apply.
"""
function _include_feed_population_scales!(scales, solver::AbstractSolver, problem, feed_population)
    @inbounds for (state_index, population_index) in zip(_population_state_range(problem),
                                                          eachindex(feed_population))
        scales[state_index] = max(scales[state_index], abs(feed_population[population_index]))
    end
    return scales
end
_include_feed_population_scales!(scales, solver::DQMOM, problem, feed_population) = scales

function _steady_scaled_residual(problem, ode_problem, state, parameters,
                                 simulation_time, residence_time, reference_scales;
                                 shared_population_scale::Bool = false)
    residual = _steady_rhs(ode_problem, state, parameters, simulation_time)
    scaled = similar(residual)
    population_range = _population_state_range(problem)
    population_scale = shared_population_scale ?
        max(maximum(abs, view(state, population_range)),
            maximum(view(reference_scales, population_range)), floatmin(Float64)) : zero(eltype(reference_scales))
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
                            integrator -> SciMLBase.terminate!(integrator);
                            save_positions = (false, true))
end

"""Third raw moment of a population vector in the solver's coordinates (raw moments for moment solvers)."""
_third_raw_moment(solver::AbstractDiscretisedSolver, numberdensity) =
    momentcalculator(solver.cell_centre, numberdensity, 3)
_third_raw_moment(solver::AbstractMomentSolver, raw_moments) = raw_moments[4]

"""Final saved population of a physical result, as densities or raw moments."""
_final_population_values(population::MeshPopulation) = population.numberdensity[:, end]
_final_population_values(population::Union{MomentPopulation, QuadraturePopulation}) =
    population.moments[:, end]

"""Raw density extremes and roundoff tolerance of a mesh steady state; `nothing` otherwise."""
_steady_population_density_diagnostics(solver::AbstractMomentSolver, problem, raw_state) = nothing
function _steady_population_density_diagnostics(solver::AbstractDiscretisedSolver, problem, raw_state)
    population_density = @view raw_state[_population_state_range(problem)]
    minimum_density = minimum(population_density)
    density_scale = maximum(abs, population_density)
    # Density can span many orders of magnitude at the finite-domain tail. Allow
    # only 128 machine epsilons relative to its peak; preserve the raw state and
    # report any accepted undershoot instead of clipping it or using solve tol.
    scalar_type = eltype(raw_state) <: AbstractFloat ? eltype(raw_state) : Float64
    roundoff_tolerance = 128 * eps(scalar_type) *
                         max(density_scale, floatmin(scalar_type))
    negative_count = count(value -> value < zero(value), population_density)

    return (; minimum = minimum_density,
              scale = density_scale,
              roundoff_tolerance,
              negative_count)
end

function _steady_physical_constraints(problem, raw_state, physical_result)
    all(isfinite, raw_state) || return false, :nonfinite_state
    concentration = raw_state[_solvent_state_index(problem, :concentration)]
    concentration >= 0.0 || return false, :negative_concentration
    return _steady_population_constraints(problem.solver, problem, raw_state, physical_result)
end

"""`(valid, status)` of the solver's population at a candidate steady state."""
function _steady_population_constraints(solver::AbstractDiscretisedSolver, problem, raw_state,
                                        physical_result)
    density_diagnostics = _steady_population_density_diagnostics(solver, problem, raw_state)
    density_diagnostics.negative_count > 0 || return true, :ok
    density_diagnostics.minimum >= -density_diagnostics.roundoff_tolerance ||
        return false, :negative_number_density
    return true, :roundoff_negative_number_density
end

function _steady_population_constraints(solver::DQMOM, problem, raw_state, physical_result)
    population = physical_result.population
    all(value -> isfinite(value) && value > 0.0, population.weights) ||
        return false, :nonpositive_quadrature_weight
    all(value -> isfinite(value) && value > solver.minimum_size, population.nodes) ||
        return false, :invalid_quadrature_node
    return true, :ok
end

function _steady_population_constraints(solver::QMOM, problem, raw_state, physical_result)
    all(diagnostic -> diagnostic.status in (:ok, :empty, :deflated),
        physical_result.population.diagnostics) || return false, :unrealizable_moments
    return true, :ok
end

function _steady_population_constraints(solver::MoM, problem, raw_state, physical_result)
    moments = @view raw_state[_population_state_range(problem)]
    all(value -> isfinite(value) && value >= 0.0, moments) ||
        return false, :negative_moment
    for index in 2:(length(moments) - 1)
        left = moments[index]^2
        right = moments[index - 1] * moments[index + 1]
        left <= right + 1e-10 * max(abs(left), abs(right), floatmin(Float64)) ||
            return false, :non_logconvex_moments
    end
    return true, :ok
end

function _steady_product_diagnostics(problem, raw_state, physical_result,
                                     simulation_time)
    operation = problem.operation::MSMPROperation
    flows = operation_flows(operation, simulation_time)
    volume = reactor_volume(problem, raw_state)
    residence_time = volume / flows.inflow
    moment3 = _third_raw_moment(problem.solver, _final_population_values(physical_result.population))
    feed_moment3 = _third_raw_moment(problem.solver, _operation_feed_population(problem))
    concentration = raw_state[_solvent_state_index(problem, :concentration)]
    density = problem.model.system.crystal_density * problem.model.system.volume_shape_factor
    inlet_dissolved_flow = flows.inflow * _feed_concentration(operation.feed, simulation_time)
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
    product = (; dissolved_solute_flow = outlet_dissolved_flow,
                 solid_mass_flow = outlet_solid_flow)
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

_default_steady_relaxation_horizon(problem) =
    100.0 * (problem.operation.volume / operation_flows(problem.operation, 0.0).inflow)

"""
    _solve_steadystate_ode(problem, ode_problem, algorithm;
                           steady_options=SteadyStateOptions(), solve_options=(;),
                           callback_factory=nothing)

Internal prepared-loss entry point. `ode_problem` is authoritative: in
particular, its `p` may be a candidate-specific `ComponentArray` with Dual
values. Without an explicit `relaxation_horizon` the ODE span is the horizon.
The returned result records failure in `success` and `diagnostics` instead of
discarding the solver retcode.
"""
function _solve_steadystate_ode(problem::CrystallisationProblem, ode_problem,
                                algorithm;
                                steady_options::SteadyStateOptions = SteadyStateOptions(),
                                solve_options::NamedTuple = (;),
                                callback_factory = nothing,
                                reconstruction_failure_handler = nothing)
    options = steady_options
    _validate_steadystate_problem(problem; autonomous = options.autonomous)

    start_time, original_end_time = ode_problem.tspan
    horizon = something(options.relaxation_horizon, original_end_time - start_time)
    options.minimum_relaxation <= horizon ||
        throw(ArgumentError("minimum_relaxation must lie between zero and relaxation_horizon."))
    end_time = start_time + horizon
    active_ode_problem = end_time == original_end_time ? ode_problem :
                         remake(ode_problem; tspan = (start_time, end_time))

    residual_limit = options.residual_reltol + options.residual_abstol
    flow = operation_flows(problem.operation, start_time).inflow
    residence_time = problem.operation.volume / flow
    reference_scales = _steady_state_reference_scales(problem, active_ode_problem,
                                                      options.residual_scales)
    shared_population_scale = problem.solver isa AbstractDiscretisedSolver &&
                              isnothing(options.residual_scales)
    convergence_callback = _steady_autostop_callback(
        problem, active_ode_problem, reference_scales, residence_time,
        options.minimum_relaxation, options.residual_reltol, options.residual_abstol;
        shared_population_scale)

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
            residual_limit / (100 * sqrt(length(ode_problem.u0)))),
        dtmax = residence_time / 4), solve_options)
    raw_solution = _solve_crystallisation_ode(
        problem, active_ode_problem, algorithm, [end_time]; solve_options = steady_solve_options,
        callback_factory = combined_callback_factory)

    final_ode_state = raw_solution.u[end]
    raw_state = collect(final_ode_state)
    final_time = raw_solution.t[end]
    residual, scaled_residual = _steady_scaled_residual(
        problem, active_ode_problem, final_ode_state, raw_solution.prob.p, final_time,
        residence_time, reference_scales; shared_population_scale)
    scaled_residual_norm = maximum(scaled_residual)
    solver_success = SciMLBase.successful_retcode(raw_solution.retcode)
    converged = solver_success && all(isfinite, scaled_residual) &&
                scaled_residual_norm <= residual_limit

    final_sample_index = lastindex(raw_solution.u)
    equilibrium_sample = SciMLBase.solution_slice(raw_solution,
        final_sample_index:final_sample_index)
    physical_result, reconstructed = _wrap_prepared_result(problem, equilibrium_sample;
        reconstruction_failure_handler)
    reconstructed || return physical_result
    physical_valid, physical_status = _steady_physical_constraints(
        problem, raw_state, physical_result)
    hydraulics, product, balance = _steady_product_diagnostics(
        problem, raw_state, physical_result, final_time)
    balance_valid = balance.relative_error <= options.balance_reltol
    diagnostics = (; converged,
                    residual_norm = norm(residual, Inf),
                    scaled_residual_norm, residual_limit,
                    residual_scales = reference_scales,
                    population_scaling = shared_population_scale ? :density_peak : :per_component,
                    physical_state_valid = physical_valid,
                    physical_state_status = physical_status,
                    population_density_diagnostics =
                        _steady_population_density_diagnostics(problem.solver, problem, raw_state),
                    total_api_balance = balance,
                    balance_reltol = options.balance_reltol,
                    balance_valid,
                    equilibrium_time = final_time,
                    ode_final_state = raw_state,
                    integration_options = steady_solve_options,
                    solver_success)

    return CrystallisationSteadyStateSolution(physical_result, _equilibrium_values(physical_result),
                                               residual, scaled_residual,
                                               hydraulics, product, diagnostics,
                                               raw_solution.stats, raw_solution.retcode,
                                               solver_success && converged &&
                                                   physical_valid && balance_valid)
end

"""
    solve_steadystate(problem::CrystallisationProblem;
                      options=SteadyStateOptions(), initial_guess=nothing,
                      algorithm=nothing, solve_options=(;),
                      callback_factory=nothing)

Relax a constant, autonomous fixed-volume MSMPR to a stable steady state.
The solver integrates the generated ODE and stops early only when the scaled
full-state RHS residual is within tolerance. It returns a distinct
`CrystallisationSteadyStateSolution`; `success` is false if the relaxation
horizon, residual, physical constraints, or total-API flow balance fails.
If those checks fail, the direct call throws `SteadyStateConvergenceError`,
which carries the unsuccessful result and its diagnostics. `steady.equilibrium`
holds the equilibrium values (`steady.equilibrium.concentration`, `.d43`).

Opaque time profiles and user-defined kinetics require
`SteadyStateOptions(autonomous = true)` as an explicit declaration that their
behavior is time invariant. Batch and fed-batch problems remain transient
`runsimulation` workflows.
"""
function solve_steadystate(problem::CrystallisationProblem;
                           options::SteadyStateOptions = SteadyStateOptions(),
                           initial_guess::Union{Nothing, AbstractVector} = nothing,
                           algorithm = nothing,
                           solve_options::NamedTuple = (;),
                           callback_factory = nothing)
    _validate_steadystate_problem(problem; autonomous = options.autonomous)
    horizon = something(options.relaxation_horizon, _default_steady_relaxation_horizon(problem))
    options.minimum_relaxation <= horizon ||
        throw(ArgumentError("minimum_relaxation must not exceed relaxation_horizon."))

    solve_problem = if isnothing(initial_guess)
        problem
    else
        guessed_problem = _problem_with_initial_state(problem, collect(initial_guess))
        _validate_crystallisation_problem(guessed_problem)
        guessed_problem
    end
    ode_problem, default_algorithm = crystallisation_odeproblem(
        solve_problem, (0.0, horizon))
    steady = _solve_steadystate_ode(
        solve_problem, ode_problem, something(algorithm, default_algorithm);
        steady_options = options, solve_options, callback_factory)
    steady.success || throw(SteadyStateConvergenceError(steady))
    return steady
end
