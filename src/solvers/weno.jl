"""
    _weno5_trace(y₋₂, y₋₁, y₀, y₊₁, y₊₂)

Fifth-order Jiang--Shu WENO trace at the face immediately to the right of
`y₀` from the five-point stencil. Calling it with the stencil reversed gives
the mirrored (right) trace at the face to the left of `y₀`. Scalar arguments
only, so it is allocation free (including ForwardDiff evaluations).
"""
@inline function _weno5_trace(y₋₂::T, y₋₁::T, y₀::T, y₊₁::T, y₊₂::T) where {T <: Real}
    ε = T(CRISTOOL_WENO_EPSILON)
    # Linear weights for the left, centred, and right candidate stencils.
    γ₀, γ₁, γ₂ = T(0.1), T(0.6), T(0.3)
    c13_12 = T(13 / 12)
    c1_4 = T(1 / 4)

    q₀ = muladd(T(1 / 3), y₋₂,
                muladd(-T(7 / 6), y₋₁, T(11 / 6) * y₀))
    q₁ = muladd(-T(1 / 6), y₋₁,
                muladd(T(5 / 6), y₀, T(1 / 3) * y₊₁))
    q₂ = muladd(T(1 / 3), y₀,
                muladd(T(5 / 6), y₊₁, -T(1 / 6) * y₊₂))

    β₀ = muladd(c13_12, (y₋₂ - 2y₋₁ + y₀)^2,
                c1_4 * (y₋₂ - 4y₋₁ + 3y₀)^2)
    β₁ = muladd(c13_12, (y₋₁ - 2y₀ + y₊₁)^2,
                c1_4 * (y₋₁ - y₊₁)^2)
    β₂ = muladd(c13_12, (y₀ - 2y₊₁ + y₊₂)^2,
                c1_4 * (3y₀ - 4y₊₁ + y₊₂)^2)

    α₀ = γ₀ / (ε + β₀)^2
    α₁ = γ₁ / (ε + β₁)^2
    α₂ = γ₂ / (ε + β₂)^2
    α_sum_inv = 1 / (α₀ + α₁ + α₂)
    ω₀ = α₀ * α_sum_inv
    ω₁ = α₁ * α_sum_inv
    ω₂ = α₂ * α_sum_inv

    return muladd(ω₀, q₀, muladd(ω₁, q₁, ω₂ * q₂))
end

"""
    _weno_flux_left(numberdensity, left_cell_index)

Reconstruct the left trace at the face immediately to the right of
`left_cell_index` with fifth-order Jiang--Shu WENO.  This is kept local to
the WENO solver so the FV implementation and its established limiter remain
unchanged.
"""
@inline function _weno_flux_left(y::AbstractArray{T}, left_cell_index::Integer) where {T <: Real}
    @inbounds return _weno5_trace(y[left_cell_index - 2], y[left_cell_index - 1],
                                  y[left_cell_index], y[left_cell_index + 1],
                                  y[left_cell_index + 2])
end

"""
    _weno_flux_right(numberdensity, right_cell_index)

Reconstruct the right trace at the face immediately to the left of
`right_cell_index`: the mirror image of [`_weno_flux_left`](@ref), obtained by
reversing the stencil. The positive-growth WENO flux is a left trace
(information travels towards increasing crystal length); a negative growth
rate needs the right trace.

The padded array contains two ghost entries on either side of the physical
mesh.  Reading the stencil by index, instead of reversing a temporary array,
is important because this function is called from the ODE RHS and must remain
allocation free (including ForwardDiff evaluations).
"""
@inline function _weno_flux_right(y::AbstractArray{T}, right_cell_index::Integer) where {T <: Real}
    @inbounds return _weno5_trace(y[right_cell_index + 2], y[right_cell_index + 1],
                                  y[right_cell_index], y[right_cell_index - 1],
                                  y[right_cell_index - 2])
end

"""
    _weno_nonnegative_state(high_order_state, upwind_state)

Apply the local positivity correction to a reconstructed interface state.
The physical growth rate is deliberately absent from this function: only
the numerical state is corrected.  If WENO produces a negative trace, blend
that trace towards the corresponding first-order upwind state until the
trace is exactly zero.  This is the minimal conservative fallback for an
interface; it preserves the WENO stencil everywhere its reconstruction is
already nonnegative and never clips or changes the signed growth rate.
"""
@inline function _weno_nonnegative_state(high_order_state, upwind_state)
    # A valid ODE stage should have a nonnegative upwind state.  Treat a tiny
    # stage undershoot as an empty state rather than allowing it to re-enter
    # through a high-order interface flux.  Do not cap a positive trace by
    # the upwind-cell value: that would silently turn a smooth WENO
    # reconstruction into a first-order method and is not a positivity
    # condition.
    upwind_state <= zero(upwind_state) && return zero(high_order_state + upwind_state)
    high_order_state >= zero(high_order_state) && return high_order_state

    # Negative trace h < 0 < u: the blend weight θ = -h / (u - h) lies in
    # (0, 1) and gives θu + (1 - θ)h = 0 exactly, so the limited state is
    # zero. Returning it directly avoids evaluating the blend, whose
    # ForwardDiff quotient rule squares u - h: on a far-tail face (u ~ 1e-162)
    # that square underflows to zero and turns zero partials into NaN, which
    # stops every dual solve as `Unstable`.
    return zero(high_order_state + upwind_state)
end

"""
    _fill_weno_padding!(padding, numberdensity, lower_ghost)

Copy `numberdensity` into the interior of the two-ghost-cell `padding` buffer.
Both lower ghosts take `lower_ghost`; both upper ghosts are zero (no incoming
population at lmax).
"""
@inline function _fill_weno_padding!(padding::AbstractVector, numberdensity::AbstractVector,
                                     lower_ghost)
    n_cells = length(numberdensity)
    @inbounds begin
        padding[1] = lower_ghost
        padding[2] = lower_ghost
        padding[3:(n_cells + 2)] .= numberdensity
        padding[n_cells + 3] = zero(eltype(padding))
        padding[n_cells + 4] = zero(eltype(padding))
    end
    return padding
end

"""Signed lower-face flux: nucleation inflow for growth, outflow `rate * n` for dissolution."""
@inline _weno_signed_lower_flux(growth_rate, nucleation_rate, lowest_density) =
    growth_rate > zero(growth_rate) ? nucleation_rate :
    growth_rate < zero(growth_rate) ? growth_rate * lowest_density : zero(growth_rate)

"""
    _weno_upper_outflow_flux(numberdensity, upper_growth_rate)

Second-order outflow extrapolation of the number flux at lmax, with the same
nonnegative trace correction as the interior WENO faces.
"""
@inline function _weno_upper_outflow_flux(numberdensity, upper_growth_rate)
    upwind_state = last(numberdensity)
    high_order_state = length(numberdensity) == 1 ? upwind_state :
                       upwind_state + 0.5 * (upwind_state - numberdensity[end - 1])
    return upper_growth_rate * _weno_nonnegative_state(high_order_state, upwind_state)
end

"""Fill WENO fluxes for a scalar negative signed growth rate."""
function _fill_signed_weno_flux!(flux::AbstractVector,
                                 numberdensity::AbstractVector,
                                 scalar_growth_rate,
                                 nucleation_rate,
                                 ndens_pad_cache::AbstractVector)
    n_cells = length(numberdensity)
    length(flux) == n_cells + 1 ||
        throw(ArgumentError("flux must have one more entry than numberdensity."))
    length(ndens_pad_cache) == n_cells + 4 ||
        throw(ArgumentError("WENO density padding has the wrong length."))

    # Zero-gradient lower ghost cells provide the one-sided stencil just
    # inside the outflow boundary.  Zero upper ghosts enforce no inflow at
    # lmax for dissolution.
    _fill_weno_padding!(ndens_pad_cache, numberdensity, numberdensity[1])
    @inbounds begin
        flux[1] = scalar_growth_rate * numberdensity[1]
        for face_index in 2:n_cells
            # The right cell of face i+1/2 is at padded index i+2.
            reconstructed_state = _weno_flux_right(ndens_pad_cache, face_index + 2)
            flux[face_index] = scalar_growth_rate *
                               _weno_nonnegative_state(reconstructed_state,
                                                       numberdensity[face_index])
        end
        flux[n_cells + 1] = zero(scalar_growth_rate)
    end
    return flux
end

"""Fill WENO fluxes for a mesh-aligned, possibly sign-changing growth rate."""
function _fill_signed_weno_flux!(flux::AbstractVector,
                                 numberdensity::AbstractVector,
                                 net_growth_rates::AbstractVector,
                                 nucleation_rate,
                                 ndens_pad_cache::AbstractVector)
    n_cells = length(numberdensity)
    length(flux) == n_cells + 1 ||
        throw(ArgumentError("flux must have one more entry than numberdensity."))
    length(net_growth_rates) == n_cells ||
        throw(ArgumentError("net_growth_rates and numberdensity must have the same length."))
    length(ndens_pad_cache) == n_cells + 4 ||
        throw(ArgumentError("WENO density padding has the wrong length."))

    # One-sided, zero-gradient lower padding is used for positive and
    # negative interior traces.  The upper padding is zero because the
    # physical boundary has no incoming population for negative growth.
    _fill_weno_padding!(ndens_pad_cache, numberdensity, numberdensity[1])
    @inbounds begin
        flux[1] = _weno_signed_lower_flux(net_growth_rates[1], nucleation_rate,
                                          numberdensity[1])

        for face_index in 2:n_cells
            face_growth_rate = 0.5 * (net_growth_rates[face_index - 1] +
                                      net_growth_rates[face_index])
            if face_growth_rate > zero(face_growth_rate)
                reconstructed_state = _weno_flux_left(ndens_pad_cache, face_index + 1)
                upwind_state = numberdensity[face_index - 1]
                flux[face_index] = face_growth_rate *
                                   _weno_nonnegative_state(reconstructed_state,
                                                           upwind_state)
            elseif face_growth_rate < zero(face_growth_rate)
                right_cell_index = face_index + 2
                reconstructed_state = _weno_flux_right(ndens_pad_cache,
                                                       right_cell_index)
                upwind_state = numberdensity[face_index]
                flux[face_index] = face_growth_rate *
                                   _weno_nonnegative_state(reconstructed_state,
                                                           upwind_state)
            else
                flux[face_index] = zero(face_growth_rate)
            end
        end

        upper_growth_rate = net_growth_rates[n_cells]
        flux[n_cells + 1] = upper_growth_rate > zero(upper_growth_rate) ?
                            _weno_upper_outflow_flux(numberdensity, upper_growth_rate) :
                            zero(upper_growth_rate)
    end
    return flux
end

@inline function _weno_nonempty_left_state(padded_density, left_cell_index,
                                          upwind_density)
    # The existing positivity correction returns zero for an empty or
    # negative upwind cell, irrespective of the high-order trace.  Avoid
    # constructing a five-point stencil whose result will be discarded.
    upwind_density <= zero(upwind_density) && return zero(upwind_density)
    return _weno_nonnegative_state(weno_flux(padded_density, left_cell_index),
                                  upwind_density)
end

"""Fill positive scalar-growth WENO fluxes with local positivity correction."""
@inline function _fill_positive_weno_flux!(flux::AbstractVector,
                                           numberdensity::AbstractVector,
                                           scalar_growth_rate,
                                           nucleation_rate,
                                           ndens_pad_cache::AbstractVector)
    # Boundary flux B=G*n has density B/G, not the units of B itself.
    _fill_weno_padding!(ndens_pad_cache, numberdensity, nucleation_rate / scalar_growth_rate)
    @inbounds begin
        flux[1] = nucleation_rate
        flux[2] = scalar_growth_rate *
                  _weno_nonempty_left_state(ndens_pad_cache, 3, numberdensity[1])
        for cell_index in 3:length(numberdensity)
            flux[cell_index] = scalar_growth_rate *
                _weno_nonempty_left_state(ndens_pad_cache, cell_index + 1,
                                         numberdensity[cell_index - 1])
        end
        flux[end] = _weno_upper_outflow_flux(numberdensity, scalar_growth_rate)
    end
    return flux
end

"""
    _fill_weno_flux!(flux, numberdensity, net_growth, nucleation_rate, ndens_pad_cache)

WENO face fluxes for the transport rate shape. A scalar rate takes the
positive-growth fast path (lower-boundary nucleation inflow), the mirrored
right traces for negative growth, or zero flux; a per-cell rate field selects
the left or right trace face by face from its local sign.
"""
@inline function _fill_weno_flux!(flux, numberdensity, scalar_growth_rate::Number,
                                  nucleation_rate, ndens_pad_cache)
    if scalar_growth_rate > zero(scalar_growth_rate)
        _fill_positive_weno_flux!(flux, numberdensity, scalar_growth_rate,
                                  nucleation_rate, ndens_pad_cache)
    elseif scalar_growth_rate < zero(scalar_growth_rate)
        # Negative growth is outflow at lmin and zero inflow at lmax. Every
        # interior face uses the mirrored fifth-order right trace; only a
        # locally negative reconstruction blends to first-order upwinding
        # through _weno_nonnegative_state.
        _fill_signed_weno_flux!(flux, numberdensity, scalar_growth_rate,
                                nucleation_rate, ndens_pad_cache)
    else
        fill!(flux, zero(scalar_growth_rate))
    end
    return flux
end

# Length-dependent transport remains a WENO transport solve: each face selects
# the left or mirrored right fifth-order trace from the sign of its local net
# growth rate. The first-order state is only a local positivity blend.
@inline _fill_weno_flux!(flux, numberdensity, net_growth_rates::AbstractVector,
                         nucleation_rate, ndens_pad_cache) =
    _fill_signed_weno_flux!(flux, numberdensity, net_growth_rates, nucleation_rate,
                            ndens_pad_cache)

"""
    _weno_boundary_fluxes(numberdensity, net_growth, nucleation_rate) -> (lower, upper)

The lower- and upper-face fluxes that [`_fill_weno_flux!`](@ref) produces,
without reconstructing the interior faces.
"""
function _weno_boundary_fluxes(numberdensity, scalar_growth_rate::Number, nucleation_rate)
    if scalar_growth_rate > zero(scalar_growth_rate)
        # The positive scalar path's lower flux is the nucleation inflow.
        return nucleation_rate, _weno_upper_outflow_flux(numberdensity, scalar_growth_rate)
    elseif scalar_growth_rate < zero(scalar_growth_rate)
        # Signed scalar dissolution exits through lmin and has no incoming
        # population at lmax.
        return scalar_growth_rate * first(numberdensity), zero(scalar_growth_rate)
    end
    return zero(scalar_growth_rate), zero(scalar_growth_rate)
end

function _weno_boundary_fluxes(numberdensity, net_growth_rates::AbstractVector,
                               nucleation_rate)
    upper_growth_rate = last(net_growth_rates)
    lower_boundary_flux = _weno_signed_lower_flux(first(net_growth_rates), nucleation_rate,
                                                  first(numberdensity))
    upper_boundary_flux = upper_growth_rate > zero(upper_growth_rate) ?
                          _weno_upper_outflow_flux(numberdensity, upper_growth_rate) :
                          zero(upper_growth_rate)
    return lower_boundary_flux, upper_boundary_flux
end

_mesh_boundary_fluxes(::WENO, flux_cache, numberdensity, net_growth, nucleation_rate) =
    _weno_boundary_fluxes(numberdensity, net_growth, nucleation_rate)

_mesh_solver_cache(::WENO, problem) = DiffCache(zeros(problem.solver.meshsize + 4))

"""Per-cell net growth buffer of the CFL callback; a scalar growth law needs none."""
_weno_cfl_growth_cache(::Val{false}, problem) = nothing
_weno_cfl_growth_cache(::Val{true}, problem) = DiffCache(zeros(problem.solver.meshsize))

"""
    _weno_cfl_callback(problem, time_scalar_type) -> DiscreteCallback

Fresh CFL step limiter for one WENO solve. Stage/step limiter hooks discard
returned values. This solve-local callback instead caps the next step from the
accepted-state transport rate, including initialization. For varying rates
this is a local CFL estimate, not a guarantee about every intermediate stage.
"""
function _weno_cfl_callback(CryProblem::CrystallisationProblem, time_scalar_type::Type)
    growth_layout = _mesh_growth_layout(CryProblem)
    growth_rate_cache_dc = _weno_cfl_growth_cache(growth_layout, CryProblem)
    weno_cfl_bound(numerical_state, kinetic_parameters, simulation_time) =
        _signed_cfl(CryProblem.solver.cell_dL,
                    _mesh_net_growth(growth_layout, growth_rate_cache_dc, kinetic_parameters,
                                     CryProblem, numerical_state, simulation_time);
                    courant = 0.9)
    configured_dtmax = Ref(time_scalar_type(Inf))
    bounded_weno_cfl = (numerical_state, kinetic_parameters, simulation_time) ->
        min(configured_dtmax[], SciMLBase.value(
            weno_cfl_bound(numerical_state, kinetic_parameters, simulation_time)))
    cfl_step_callback = StepsizeLimiter(bounded_weno_cfl;
        safety_factor = one(time_scalar_type), cached_dtcache = zero(time_scalar_type))
    initialize_weno_cfl = (callback, numerical_state, simulation_time, integrator) -> begin
        # Preserve an explicit solve_options.dtmax, even when a slower rate
        # would permit a larger CFL step later in the simulation.
        configured_dtmax[] = integrator.opts.dtmax
        cfl_step_callback.initialize(callback, numerical_state, simulation_time, integrator)
    end
    return DiscreteCallback(cfl_step_callback.condition,
        cfl_step_callback.affect!; initialize = initialize_weno_cfl,
        save_positions = (false, false))
end

@inline _fill_mesh_flux!(::WENO, padding, face_flux, numberdensity, net_growth,
                         nucleation_rate) =
    _fill_weno_flux!(face_flux, numberdensity, net_growth, nucleation_rate, padding)

function crystallisation_odeproblem(solver::WENO, CryProblem::CrystallisationProblem, saveat, kinetic_values)
    flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    growth_rate_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize))
    weno_rhs! = _mesh_population_rhs(CryProblem, flux_cache_dc, growth_rate_cache_dc,
                                     _mesh_solver_cache(solver, CryProblem))

    utyped = eltype(kinetic_values).(_get_initial_state(CryProblem))
    signed_growth_transport = _dissolution_enabled(CryProblem)
    ODEprob = ODEProblem(weno_rhs!, utyped, (saveat[1], saveat[end]), kinetic_values)
    return (ODEprob, _timestepping_algorithm(solver, signed_growth_transport ? SSPRK43() : Tsit5()))
end

_solver_callbacks(solver::WENO, problem::CrystallisationProblem, odeproblem) =
    _weno_cfl_callback(problem, typeof(float(first(odeproblem.tspan))))
