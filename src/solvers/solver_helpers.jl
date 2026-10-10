###### Solver functions ######

"""
    crystallisation_odeproblem(problem, saveat, kinetic_values=kinetic_parameter_vector(problem))
        -> (odeproblem, default_algorithm)

Build the ODE problem integrated by `problem.solver` over `saveat[1]..saveat[end]`,
together with that solver's default OrdinaryDiffEq algorithm (or the
configured `timestepping_algorithm`). `kinetic_values` is the composite kinetic
parameter vector on `paramaxis(problem)` and becomes the ODE parameter `p`; its
element type sets the state element type (e.g. ForwardDiff duals). Each solver
type provides one `crystallisation_odeproblem(solver, problem, saveat, kinetic_values)` method.
"""
crystallisation_odeproblem(problem::CrystallisationProblem, saveat,
                           kinetic_values = kinetic_parameter_vector(problem)) =
    crystallisation_odeproblem(problem.solver, problem, saveat, kinetic_values)

"""Configured algorithm of `solver`, or `default_algorithm` when it is `nothing`."""
_timestepping_algorithm(solver::AbstractSolver, default_algorithm) =
    something(solver.timestepping_algorithm, default_algorithm)

"""
    _size_boundary_flow_state(solver, problem, ode_solution) -> NamedTuple

Signed lower/upper size-boundary solid-mass flows (kg/s) of a mesh solution;
moment and quadrature solvers have no size boundary.
"""
_size_boundary_flow_state(solver::AbstractDiscretisedSolver, problem, ode_solution) =
    _mesh_boundary_flow_state(solver, problem, ode_solution)
_size_boundary_flow_state(solver::AbstractMomentSolver, problem, ode_solution) = (;)

"""Physical solution of an ODE solution; one `_wrap_solution(solver, ...)` method per solver."""
_wrap_solution(problem::CrystallisationProblem, ode_solution) =
    _wrap_solution(problem.solver, problem, ode_solution,
                   _size_boundary_flow_state(problem.solver, problem, ode_solution))

"""Return `(result, reconstructed)`, classifying only package reconstruction failures."""
function _wrap_prepared_result(configured_problem, ode_solution;
                               reconstruction_failure_handler = nothing)
    # Mesh boundary diagnostics re-enter user rate laws at interpolated saved
    # times. Evaluate them before guarding package-owned reconstruction.
    boundary_flow_state = _size_boundary_flow_state(configured_problem.solver,
                                                    configured_problem, ode_solution)
    try
        physical_result = _wrap_solution(configured_problem.solver, configured_problem,
                                         ode_solution, boundary_flow_state)
        return physical_result, true
    catch reconstruction_error
        reconstruction_error isa DomainError || rethrow()
        reconstruction_failure_handler === nothing && rethrow()
        return reconstruction_failure_handler(ode_solution, reconstruction_error), false
    end
end

"""
    weno_flux(y::AbstractArray{T}, i::Integer) where {T<:Real}

Calculate the WENO (Weighted Essentially Non-Oscillatory) reconstruction of the flux at cell interface i+1/2.

Uses a 5th order WENO scheme with three candidate stencils for high accuracy in smooth regions while avoiding
oscillations near discontinuities.

# Arguments
- `y`: Array of cell-averaged values
- `i`: Cell index for reconstruction

# Returns
- Reconstructed value at cell interface i+1/2
"""
weno_flux(cell_averages::AbstractArray{<:Real}, left_cell_index::Integer) =
    _weno_flux_left(cell_averages, left_cell_index)

"""
    _get_initial_state(problem) -> Vector

The configured `initial_state`, or an empty population followed by the
operation and solvent initial values.
"""
_get_initial_state(problem) =
    [something(problem.initial_population, zeros(_population_state_count(problem.solver)));
     _operation_initial_state(problem.operation); collect(values(problem.initial_solvent_state))]

@inline function _concentration_depletion(cell_dL, numberdensity, scalar_growth_rate, cell_centre)
    derivative_type = promote_type(eltype(numberdensity), typeof(scalar_growth_rate))
    acc = zero(derivative_type)
    @inbounds for i in eachindex(numberdensity, cell_centre)
        acc += cell_dL * numberdensity[i] * scalar_growth_rate *
               (cell_centre[i] * cell_centre[i])
    end
    return acc
end

@inline function _concentration_depletion(cell_dL, numberdensity,
                                           net_growth_rates::AbstractVector,
                                           cell_centre)
    length(numberdensity) == length(net_growth_rates) == length(cell_centre) ||
        throw(ArgumentError("numberdensity, net_growth_rates, and cell_centre must have the same length."))
    derivative_type = promote_type(eltype(numberdensity), eltype(net_growth_rates))
    acc = zero(derivative_type)
    @inbounds for i in eachindex(numberdensity, net_growth_rates, cell_centre)
        acc += cell_dL * numberdensity[i] * net_growth_rates[i] *
               (cell_centre[i] * cell_centre[i])
    end
    return acc
end

@inline _rate_zero(growth_rate::Number) = zero(growth_rate)
@inline _rate_zero(net_growth_rates::AbstractVector) = zero(eltype(net_growth_rates))

"""
    _fused_solvent_coupling_enabled(problem)

True when the solvent block is exactly one `:concentration` variable driven by
`DefaultSolventDynamics`.  In that case the discretised RHS can compute the
concentration depletion inside the divergence sweep (`_fused_depletion_divergence!`)
instead of a separate pass, and write the solvent slot directly.

The gate is a compile-time constant for a concrete problem type, so the
`if` in the solver closures folds to the fused path with zero runtime cost.
"""
@inline _fused_solvent_coupling_enabled(problem) =
    problem.solvent_dynamics isa DefaultSolventDynamics &&
    propertynames(problem.initial_solvent_state) == (:concentration,)

"""Scalar-growth fused divergence + concentration-depletion sweep.

The depletion sum is bit-identical to `_concentration_depletion(cell_dL,
numberdensity, scalar_growth_rate, cell_centre)` (same expression, same
accumulation order), so the produced concentration rate matches
`DefaultSolventDynamics` exactly.
"""
@inline function _fused_depletion_divergence!(dstdt_nd_view, flux, numberdensity,
                                              cell_centre, cell_dL,
                                              scalar_growth_rate)
    acc = zero(promote_type(eltype(numberdensity), typeof(scalar_growth_rate)))
    @inbounds for i in eachindex(dstdt_nd_view)
        dstdt_nd_view[i] = -(flux[i + 1] - flux[i]) / cell_dL
        acc += cell_dL * numberdensity[i] * scalar_growth_rate *
               (cell_centre[i] * cell_centre[i])
    end
    return acc
end

"""Mesh-aligned-growth fused divergence + concentration-depletion sweep."""
@inline function _fused_depletion_divergence!(dstdt_nd_view, flux, numberdensity,
                                              cell_centre, cell_dL,
                                              net_growth_rates::AbstractVector)
    length(numberdensity) == length(net_growth_rates) == length(cell_centre) ||
        throw(ArgumentError("numberdensity, net_growth_rates, and cell_centre must have the same length."))
    acc = zero(promote_type(eltype(numberdensity), eltype(net_growth_rates)))
    @inbounds for i in eachindex(dstdt_nd_view)
        dstdt_nd_view[i] = -(flux[i + 1] - flux[i]) / cell_dL
        acc += cell_dL * numberdensity[i] * net_growth_rates[i] *
               (cell_centre[i] * cell_centre[i])
    end
    return acc
end

@inline function _signed_upwind_flux(growth_rate, left_density, right_density)
    if growth_rate > zero(growth_rate)
        return growth_rate * left_density
    elseif growth_rate < zero(growth_rate)
        return growth_rate * right_density
    end
    return zero(growth_rate)
end

@inline function _signed_cfl(cell_dL, growth_rate; courant = 0.99)
    speed = abs(growth_rate)
    return speed == zero(speed) ? Inf : courant * cell_dL / speed
end

@inline function _signed_cfl(cell_dL, net_growth_rates::AbstractVector; courant = 0.99)
    isempty(net_growth_rates) && return Inf
    speed = maximum(abs, net_growth_rates)
    return speed == zero(speed) ? Inf : courant * cell_dL / speed
end

"""
    _fill_signed_first_order_flux!(flux, numberdensity, net_growth_rates,
                                   nucleation_rate)

Fill conservative first-order upwind fluxes for a mesh-aligned, possibly
sign-changing growth rate.  Nucleation is a lower-boundary inflow only when
the lower-face growth rate is positive; the opposite boundary is a zero-inflow
boundary for negative growth rate.
"""
function _fill_signed_first_order_flux!(flux::AbstractVector,
                                        numberdensity::AbstractVector,
                                        net_growth_rates::AbstractVector,
                                        nucleation_rate)
    n_cells = length(numberdensity)
    length(flux) == n_cells + 1 ||
        throw(ArgumentError("flux must have one more entry than numberdensity."))
    length(net_growth_rates) == n_cells ||
        throw(ArgumentError("net_growth_rates and numberdensity must have the same length."))

    @inbounds begin
        lower_rate = net_growth_rates[1]
        flux[1] = lower_rate > zero(lower_rate) ? nucleation_rate :
                  lower_rate < zero(lower_rate) ? lower_rate * numberdensity[1] :
                  zero(lower_rate)

        for face in 2:n_cells
            face_rate = 0.5 * (net_growth_rates[face - 1] + net_growth_rates[face])
            flux[face] = _signed_upwind_flux(face_rate,
                                             numberdensity[face - 1],
                                             numberdensity[face])
        end

        upper_rate = net_growth_rates[end]
        flux[end] = upper_rate > zero(upper_rate) ? upper_rate * numberdensity[end] :
                    zero(upper_rate)
    end
    return flux
end

"""
    _parameter_eltype(odeproblem.p)

Scalar element type carried by an ODE problem's parameter container, or
`Float64` for parameter containers without an element type.
"""
_parameter_eltype(ode_parameters::AbstractArray) = eltype(ode_parameters)
_parameter_eltype(ode_parameters) = Float64

"""
    _auto_abstol_opts(solver, u0, absolute_floor, scalar_type=Float64; reltol=solver.reltol)

Allocate solve-local component tolerances and an AutoAbstol callback. With
`solver.auto_abstol`, each component retains its physical absolute floor while its tolerance
tracks its own largest magnitude. Scalar mode leaves the supplied floor intact.

The `AutoAbstol` mutable magnitude cache and the `abstol` buffer it writes back
into are built at `scalar_type`. When ODE parameter derivatives promote the
integrated state to `ForwardDiff.Dual`, both buffers must accept Dual
assignment; promoting them keeps the callback derivative-safe while preserving
the physical per-component floors and per-solve cache isolation.
"""
function _auto_abstol_opts(solver::AbstractSolver, u0::AbstractVector,
                           absolute_floor, scalar_type::Type = Float64;
                           reltol = solver.reltol)
    if solver.auto_abstol
        component_floor = absolute_floor isa AbstractVector ? copy(absolute_floor) :
                          fill(float(absolute_floor), length(u0))
        promoted_floor = scalar_type.(component_floor)
        return promoted_floor,
               AutoAbstol(false; init_curmax = promoted_floor ./ reltol)
    end
    return absolute_floor isa AbstractVector ? copy(absolute_floor) : absolute_floor, nothing
end

_solver_absolute_floor(configured_problem::CrystallisationProblem, odeproblem) =
    _solver_absolute_floor(configured_problem, odeproblem, configured_problem.solver)

_solver_absolute_floor(configured_problem, odeproblem, configured_solver::AbstractSolver) =
    configured_solver.abstol

function _solver_absolute_floor(configured_problem, odeproblem, qmom_solver::QMOM)
    n_moments = moment_count(qmom_solver)
    moment_zero = abs(odeproblem.u0[1])
    # A scalar tolerance cannot resolve SI raw moments with disparate dimensions.
    moment_tolerances = [max(qmom_solver.abstol *
                             max(moment_zero * qmom_solver.coordinate_scale^index,
                                 eps(Float64) * qmom_solver.coordinate_scale^index),
                             eps(Float64) * qmom_solver.coordinate_scale^index)
                         for index in 0:(n_moments - 1)]
    return vcat(moment_tolerances,
                fill(qmom_solver.abstol, length(odeproblem.u0) - n_moments))
end

function _validate_crystallisation_solve_options(solve_options::NamedTuple)
    for option_name in keys(solve_options)
        option_name in (:u0, :p, :tspan, :saveat, :save_idxs, :callback,
                        :merge_callbacks, :save_start, :save_end, :save_everystep,
                        :initialize_save, :timeseries_steps) &&
            throw(ArgumentError("solve_options cannot set $option_name; preserve the prepared state/time contract and use callback_factory for callbacks."))
    end
    return solve_options
end

"""Algorithm hints: the non-stiff MoM closure needs none, other solvers are stiff."""
_solver_alg_hints(solver::MoM) = (;)
_solver_alg_hints(solver::AbstractSolver) = (alg_hints = [:stiff],)

"""
    _solve_crystallisation_ode(problem, odeproblem, algorithm, saveat;
                              solve_options=(;), callback_factory=nothing)

Shared direct/prepared option policy. `callback_factory(odeproblem)` constructs
one fresh user callback per evaluation, composed with package safety callbacks.
Iteration and elapsed-time limits may be supplied explicitly in `solve_options`.
"""
function _solve_crystallisation_ode(problem::CrystallisationProblem, odeproblem,
                                    algorithm, saveat;
                                    solve_options::NamedTuple = (;),
                                    callback_factory = nothing)
    _validate_crystallisation_solve_options(solve_options)
    solver_defaults = merge((reltol = problem.solver.reltol,
                             abstol = _solver_absolute_floor(problem, odeproblem),
                             dense = false, maxiters = CRISTOOL_MAX_SOLVER_ITERS),
                            _solver_alg_hints(problem.solver))
    selected_options = merge(solver_defaults, solve_options)
    scalar_type = promote_type(eltype(odeproblem.u0), _parameter_eltype(odeproblem.p))
    abstol, auto_callback = _auto_abstol_opts(problem.solver, odeproblem.u0,
                                              selected_options.abstol, scalar_type;
                                              reltol = selected_options.reltol)
    user_callback = callback_factory === nothing ? nothing : callback_factory(odeproblem)
    solve_callback = CallbackSet(auto_callback, user_callback)
    # Templates are reused; callbacks can hold mutable event state. Do not share
    # those states across parameter candidates or concurrent evaluations.
    original_callback = get(odeproblem.kwargs, :callback, nothing)
    solve_problem = isnothing(original_callback) ? odeproblem :
                    remake(odeproblem; callback = deepcopy(original_callback))
    return solve(solve_problem, algorithm; selected_options..., abstol = abstol,
                 saveat = saveat, callback = solve_callback)
end

"""Check that save times are finite, nonnegative SI seconds in increasing order."""
function _validate_save_times(save_times)
    length(save_times) >= 1 ||
        throw(ArgumentError("save_idx must contain at least one time."))
    all(time -> time isa Real && isfinite(time) && time >= 0, save_times) ||
        throw(ArgumentError("save_idx must contain finite nonnegative SI times in seconds."))
    length(save_times) == 1 || all(diff(collect(save_times)) .> 0) ||
        throw(ArgumentError("save_idx must be strictly increasing."))
    return save_times
end

"""
    _simulatecrystallisation(problem, saveat, kinetic_values=kinetic_parameter_vector(problem);
                             algorithm=nothing, solve_options=(;), callback_factory=nothing)

Solve `problem` at `saveat` with the solver selected by `problem.solver` and
return its physical solution. Each solver type provides the two methods
`crystallisation_odeproblem(solver, problem, saveat, kinetic_values) -> (odeproblem, default_algorithm)`
and `_wrap_solution(solver, problem, ode_solution, boundary_flow_state)`.
"""
function _simulatecrystallisation(problem::CrystallisationProblem, saveat,
                                  kinetic_values = kinetic_parameter_vector(problem);
                                  algorithm = nothing,
                                  solve_options::NamedTuple = (;),
                                  callback_factory = nothing)
    ode_problem, default_algorithm = crystallisation_odeproblem(problem, saveat, kinetic_values)
    ode_solution = _solve_crystallisation_ode(problem, ode_problem,
                                              something(algorithm, default_algorithm),
                                              saveat; solve_options, callback_factory)
    return _wrap_solution(problem, ode_solution)
end
