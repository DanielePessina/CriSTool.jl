###### Solver functions ######

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
function weno_flux(y::AbstractArray{T}, i::Integer) where {T <: Real}
    # Constants for WENO scheme
    ε = T(CRISTOOL_WENO_EPSILON)
    γ₁, γ₂, γ₃ = T(0.3), T(0.6), T(0.1)
    c13_12 = T(13 / 12)
    c1_4 = T(1 / 4)

    # Precompute commonly used differences to avoid redundant calculations
    @inbounds begin
        d1 = y[i + 2] - y[i + 1]
        d2 = y[i + 1] - y[i]
        d3 = y[i] - y[i - 1]
        d4 = y[i - 1] - y[i - 2]

        # Calculate candidate stencils (q values)
        q₁ = muladd(T(5 / 6), y[i + 1], muladd(T(1 / 3), y[i], -T(1 / 6) * y[i + 2]))
        q₂ = muladd(T(5 / 6), y[i], muladd(T(1 / 3), y[i + 1], -T(1 / 6) * y[i - 1]))
        q₃ = muladd(T(11 / 6), y[i], muladd(-T(7 / 6), y[i - 1], T(1 / 3) * y[i - 2]))

        # Calculate smoothness indicators (β values)
        β₁ = muladd(c13_12, (d1 - d2)^2, c1_4 * (d1 + d2)^2)
        β₂ = muladd(c13_12, d2^2, c1_4 * (d3 + d1)^2)
        β₃ = muladd(c13_12, d4^2, c1_4 * (3d3 + d4)^2)

        # Calculate non-linear weights
        α₁ = γ₁ / (ε + β₁)^2
        α₂ = γ₂ / (ε + β₂)^2
        α₃ = γ₃ / (ε + β₃)^2

        # Normalize weights
        α_sum_inv = 1 / (α₁ + α₂ + α₃)
        ω₁ = α₁ * α_sum_inv
        ω₂ = α₂ * α_sum_inv
        ω₃ = α₃ * α_sum_inv

        # Compute final reconstruction
        return muladd(ω₁, q₁, muladd(ω₂, q₂, ω₃ * q₃))
    end
end

"""
    _get_initial_state(CryProblem) -> Vector

Get initial state vector for simulation.

Returns zeros for number density/moments plus initial concentration,
or the user-provided initial_state if specified.

# Arguments
- `CryProblem`: Crystallisation problem specification

# Returns
- Initial state vector [number_density_or_moments..., concentration]
"""
_get_initial_state(CryProblem) = isnothing(CryProblem.initial_state) ?
                                     _zero_state(CryProblem.solver, CryProblem) :
                                     CryProblem.initial_state

_zero_state(solver::AbstractDiscretisedSolver, CryProblem) =
    [zeros(solver.meshsize); _operation_initial_state(CryProblem.operation); collect(values(CryProblem.initial_solvent_state))]
_zero_state(solver::MoM, CryProblem) =
    [zeros(solver.nmoments + 1); _operation_initial_state(CryProblem.operation); collect(values(CryProblem.initial_solvent_state))]
_zero_state(solver::QMOM, CryProblem) =
    [zeros(moment_count(solver)); _operation_initial_state(CryProblem.operation); collect(values(CryProblem.initial_solvent_state))]
_zero_state(solver::DQMOM, CryProblem) =
    [zeros(2 * solver.nquadrature); _operation_initial_state(CryProblem.operation); collect(values(CryProblem.initial_solvent_state))]

"""
    _build_timestepping_algorithm(algorithm_type; step_limiter=nothing, stage_limiter=nothing)

Build an ODE solver algorithm with optional step and stage limiters.

# Arguments
- `algorithm_type`: ODE solver type (e.g., Tsit5, SSPRK43)
- `step_limiter`: Optional step limiter callback
- `stage_limiter`: Optional stage limiter callback

# Returns
- Configured ODE solver algorithm instance
"""
@inline function _build_timestepping_algorithm(algorithm_type;
                                                step_limiter = nothing,
                                                stage_limiter = nothing)
    if isnothing(step_limiter) && isnothing(stage_limiter)
        return algorithm_type()
    end
    if isnothing(step_limiter)
        return algorithm_type(stage_limiter! = stage_limiter)
    end
    if isnothing(stage_limiter)
        return algorithm_type(step_limiter! = step_limiter)
    end
    return algorithm_type(step_limiter! = step_limiter, stage_limiter! = stage_limiter)
end

"""
    _build_timestepping_algorithm_step_only(algorithm_type; step_limiter=nothing)

Build an ODE solver algorithm with optional step limiter only.

Used for solvers that don't support stage limiters (e.g., implicit methods).

# Arguments
- `algorithm_type`: ODE solver type
- `step_limiter`: Optional step limiter callback

# Returns
- Configured ODE solver algorithm instance
"""
@inline function _build_timestepping_algorithm_step_only(algorithm_type;
                                                         step_limiter = nothing)
    if isnothing(step_limiter)
        return algorithm_type()
    end
    return algorithm_type(step_limiter! = step_limiter)
end

"""
    _timestepping_algorithm(::Val{:tsit5}, step_limiter, stage_limiter)

Build Tsit5 algorithm with limiters.
"""
@inline _timestepping_algorithm(::Val{:tsit5}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm(Tsit5; step_limiter = step_limiter,
                                  stage_limiter = stage_limiter)

"""
    _timestepping_algorithm(::Val{:ssprk43}, step_limiter, stage_limiter)

Build SSPRK43 algorithm with limiters.
"""
@inline _timestepping_algorithm(::Val{:ssprk43}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm(SSPRK43; step_limiter = step_limiter,
                                  stage_limiter = stage_limiter)

"""
    _timestepping_algorithm(::Val{:kvaerno5}, step_limiter, stage_limiter)

Build Kvaerno5 algorithm with step limiter only.
"""
@inline _timestepping_algorithm(::Val{:kvaerno5}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm_step_only(Kvaerno5; step_limiter = step_limiter)

"""
    _timestepping_algorithm(::Val{:kencarp4}, step_limiter, stage_limiter)

Build KenCarp4 algorithm with step limiter only.
"""
@inline _timestepping_algorithm(::Val{:kencarp4}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm_step_only(KenCarp4; step_limiter = step_limiter)

"""
    _resolve_timestepping_algorithm(solver::AbstractSolver, default_algorithm::Symbol;
                                    step_limiter=nothing, stage_limiter=nothing)

Resolve and build the timestepping algorithm for a solver.

Uses the solver's configured algorithm if not `:auto`, otherwise uses the default.

# Arguments
- `solver`: Solver with timestepping_algorithm field
- `default_algorithm`: Default algorithm symbol if solver uses :auto
- `step_limiter`: Optional step limiter callback
- `stage_limiter`: Optional stage limiter callback

# Returns
- Configured ODE solver algorithm instance
"""
@inline function _resolve_timestepping_algorithm(solver::AbstractSolver,
                                                 default_algorithm::Symbol;
                                                 step_limiter = nothing,
                                                 stage_limiter = nothing)
    alg_symbol = solver.timestepping_algorithm === :auto ? default_algorithm :
                 solver.timestepping_algorithm
    return _timestepping_algorithm(Val(alg_symbol), step_limiter, stage_limiter)
end

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

Allocate solve-local component tolerances and an AutoAbstol callback. In auto
mode, each component retains its physical absolute floor while its tolerance
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
    if solver.tolerance_mode === :auto
        component_floor = absolute_floor isa AbstractVector ? copy(absolute_floor) :
                          fill(float(absolute_floor), length(u0))
        promoted_floor = scalar_type.(component_floor)
        return promoted_floor,
               AutoAbstol(false; init_curmax = promoted_floor ./ reltol)
    end
    return absolute_floor isa AbstractVector ? copy(absolute_floor) : absolute_floor, nothing
end

_solver_absolute_floor(problem::CrystallisationProblem, odeproblem) = problem.solver.abstol

function _solver_absolute_floor(problem::CrystallisationProblem{NuclF, GrF, BrF,
                                                                 AggF, QMOM},
                                 odeproblem) where {NuclF, GrF, BrF, AggF}
    qmom_solver = problem.solver
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
    solver_defaults = (reltol = problem.solver.reltol,
                       abstol = _solver_absolute_floor(problem, odeproblem),
                       dense = false, maxiters = CRISTOOL_MAX_SOLVER_ITERS)
    if !(problem.solver isa MoM)
        solver_defaults = merge(solver_defaults, (alg_hints = [:stiff],))
    end
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
