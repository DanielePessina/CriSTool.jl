"""Fused divergence sweep + aggregation/breakage sources + solvent tail.

When the default single-concentration solvent coupling is active, the
concentration depletion is accumulated inside the divergence pass (one array
sweep instead of two).  Otherwise the sweep and the `_solvent_derivatives`
pass run exactly as before.
"""
@inline function _weno_divergence_sweep!(dstdt, st, p, t, flux_cache, CryProblem,
                                         growth_rate)
    dstdt_nd_view = crystal_state(CryProblem, dstdt)
    numberdensity = crystal_state(CryProblem, st)
    if _fused_solvent_coupling_enabled(CryProblem)
        fused_depletion = _fused_depletion_divergence!(dstdt_nd_view, flux_cache,
                                                       numberdensity,
                                                       CryProblem.solver.cell_centre,
                                                       CryProblem.solver.cell_dL,
                                                       growth_rate)
    else
        for i in eachindex(dstdt_nd_view)
            dstdt_nd_view[i] = -(flux_cache[i + 1] - flux_cache[i]) /
                               CryProblem.solver.cell_dL
        end
        fused_depletion = nothing
    end
    _weno_apply_sources!(dstdt, st, p, t, CryProblem)
    _weno_solvent_tail!(dstdt, st, p, t, CryProblem, growth_rate, fused_depletion)
    return nothing
end

function _weno_apply_sources!(dstdt, st, p, t, CryProblem)
    dstdt_nd_view = crystal_state(CryProblem, dstdt)
    agg_rate = aggregationrate(CryProblem.kinetics_aggregationfunction, p.agg,
                               CryProblem, st, t)
    br_rate = breakagerate(CryProblem.kinetics_breakagefunction, p.br,
                           CryProblem, st, t)

    if !(typeof(agg_rate) <: Real && agg_rate == 0.0)
        dstdt_nd_view .+= agg_rate
    end
    if !(typeof(br_rate) <: Real && br_rate == 0.0)
        dstdt_nd_view .+= br_rate
    end
    return nothing
end

"""Fused default-solvent RHS tail for the WENO sweeps.

The depletion came from the fused divergence sweep; the solvent slot receives
`-(3 * kv * rho * depletion)` with the same operation order as
`DefaultSolventDynamics` (bit-identical).  Non-default solvent couplings keep
the separate `_solvent_derivatives` pass.
"""
@inline function _weno_solvent_tail!(dstdt, st, p, t, CryProblem, growth_rate,
                                     fused_depletion)
    if fused_depletion === nothing
        solvent_rates = _solvent_derivatives(CryProblem, st, t, growth_rate)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)
    else
        depletion_coupled = 3 * CryProblem.volume_shape_factor *
                            CryProblem.crystal_density * fused_depletion
        dstdt[end] = -depletion_coupled
    end
    return nothing
end

"""
    _weno_flux_left(numberdensity, left_cell_index)

Reconstruct the left trace at the face immediately to the right of
`left_cell_index` with fifth-order Jiang--Shu WENO.  This is kept local to
the WENO solver so the FV implementation and its established limiter remain
unchanged.
"""
@inline function _weno_flux_left(y::AbstractArray{T}, left_cell_index::Integer) where {T <: Real}
    ε = T(CRISTOOL_WENO_EPSILON)
    # Linear weights for the left, centred, and right candidate stencils.
    γ₀, γ₁, γ₂ = T(0.1), T(0.6), T(0.3)
    c13_12 = T(13 / 12)
    c1_4 = T(1 / 4)

    @inbounds begin
        y₋₂ = y[left_cell_index - 2]
        y₋₁ = y[left_cell_index - 1]
        y₀ = y[left_cell_index]
        y₊₁ = y[left_cell_index + 1]
        y₊₂ = y[left_cell_index + 2]

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
end

"""
    _weno_flux_right(numberdensity, right_cell_index)

Reconstruct the right trace at the face immediately to the left of
`right_cell_index` with the mirror image of [`weno_flux`](@ref).  The
positive-growth WENO flux is a left trace (information travels towards
increasing crystal length); a negative growth rate needs the right trace.

The padded array contains two ghost entries on either side of the physical
mesh.  Keeping the implementation index based, instead of reversing a
temporary array, is important because this function is called from the ODE
RHS and must remain allocation free (including ForwardDiff evaluations).
"""
@inline function _weno_flux_right(y::AbstractArray{T}, right_cell_index::Integer) where {T <: Real}
    ε = T(CRISTOOL_WENO_EPSILON)
    γ₀, γ₁, γ₂ = T(0.1), T(0.6), T(0.3)
    c13_12 = T(13 / 12)
    c1_4 = T(1 / 4)

    @inbounds begin
        y₋₂ = y[right_cell_index - 2]
        y₋₁ = y[right_cell_index - 1]
        y₀ = y[right_cell_index]
        y₊₁ = y[right_cell_index + 1]
        y₊₂ = y[right_cell_index + 2]

        # Candidate right traces are the mirror image of the left traces.
        # q₀ is the right-most stencil, q₁ the centred stencil, and q₂ the
        # left-most stencil at the face immediately before y₀.
        q₀ = muladd(T(1 / 3), y₊₂,
                    muladd(-T(7 / 6), y₊₁, T(11 / 6) * y₀))
        q₁ = muladd(-T(1 / 6), y₊₁,
                    muladd(T(5 / 6), y₀, T(1 / 3) * y₋₁))
        q₂ = muladd(T(1 / 3), y₀,
                    muladd(T(5 / 6), y₋₁, -T(1 / 6) * y₋₂))

        β₀ = muladd(c13_12, (y₊₂ - 2y₊₁ + y₀)^2,
                    c1_4 * (y₊₂ - 4y₊₁ + 3y₀)^2)
        β₁ = muladd(c13_12, (y₊₁ - 2y₀ + y₋₁)^2,
                    c1_4 * (y₊₁ - y₋₁)^2)
        β₂ = muladd(c13_12, (y₀ - 2y₋₁ + y₋₂)^2,
                    c1_4 * (3y₀ - 4y₋₁ + y₋₂)^2)

        α₀ = γ₀ / (ε + β₀)^2
        α₁ = γ₁ / (ε + β₁)^2
        α₂ = γ₂ / (ε + β₂)^2
        α_sum_inv = 1 / (α₀ + α₁ + α₂)
        ω₀ = α₀ * α_sum_inv
        ω₁ = α₁ * α_sum_inv
        ω₂ = α₂ * α_sum_inv

        return muladd(ω₀, q₀, muladd(ω₁, q₁, ω₂ * q₂))
    end
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

    # Linear flux/state blending coefficient.  The resulting state is zero
    # (up to roundoff) at the positivity boundary; the final guard prevents a
    # negative last bit from leaking into the flux.
    blend = -high_order_state / (upwind_state - high_order_state)
    blend = blend < zero(blend) ? zero(blend) :
            blend > one(blend) ? one(blend) : blend
    limited_state = muladd(blend, upwind_state,
                           (one(blend) - blend) * high_order_state)
    return limited_state < zero(limited_state) ? zero(limited_state) : limited_state
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
    @inbounds begin
        ndens_pad_cache[1] = numberdensity[1]
        ndens_pad_cache[2] = numberdensity[1]
        ndens_pad_cache[3:(n_cells + 2)] .= numberdensity
        ndens_pad_cache[n_cells + 3] = zero(eltype(ndens_pad_cache))
        ndens_pad_cache[n_cells + 4] = zero(eltype(ndens_pad_cache))

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
    @inbounds begin
        ndens_pad_cache[1] = numberdensity[1]
        ndens_pad_cache[2] = numberdensity[1]
        ndens_pad_cache[3:(n_cells + 2)] .= numberdensity
        ndens_pad_cache[n_cells + 3] = zero(eltype(ndens_pad_cache))
        ndens_pad_cache[n_cells + 4] = zero(eltype(ndens_pad_cache))

        lower_growth_rate = net_growth_rates[1]
        flux[1] = lower_growth_rate > zero(lower_growth_rate) ? nucleation_rate :
                   lower_growth_rate < zero(lower_growth_rate) ?
                   lower_growth_rate * numberdensity[1] : zero(lower_growth_rate)

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
        if upper_growth_rate > zero(upper_growth_rate)
            # Retain the established second-order outflow extrapolation at
            # lmax, with the same nonnegative trace correction as the
            # interior WENO faces.
            upwind_state = numberdensity[n_cells]
            high_order_state = n_cells == 1 ? upwind_state :
                               upwind_state + 0.5 * (upwind_state -
                                                     numberdensity[n_cells - 1])
            flux[n_cells + 1] = upper_growth_rate *
                               _weno_nonnegative_state(high_order_state,
                                                       upwind_state)
        else
            flux[n_cells + 1] = zero(upper_growth_rate)
        end
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
    @inbounds begin
        # Boundary flux B=G*n has density B/G, not the units of B itself.
        ndens_pad_cache[1] = nucleation_rate / scalar_growth_rate
        ndens_pad_cache[2] = nucleation_rate / scalar_growth_rate
        ndens_pad_cache[3:(end - 2)] .= numberdensity
        ndens_pad_cache[end - 1] = zero(eltype(ndens_pad_cache))
        ndens_pad_cache[end] = zero(eltype(ndens_pad_cache))

        flux[1] = nucleation_rate
        flux[2] = scalar_growth_rate *
                  _weno_nonempty_left_state(ndens_pad_cache, 3, numberdensity[1])
        for cell_index in 3:length(numberdensity)
            flux[cell_index] = scalar_growth_rate *
                _weno_nonempty_left_state(ndens_pad_cache, cell_index + 1,
                                         numberdensity[cell_index - 1])
        end
        high_order_state = numberdensity[end] +
                           0.5 * (numberdensity[end] - numberdensity[end - 1])
        flux[end] = scalar_growth_rate *
            _weno_nonnegative_state(high_order_state, numberdensity[end])
    end
    return flux
end

"""Boundary solid-mass flow samples using the exact WENO boundary fluxes.

The signed lower and upper flows are positive out of the size domain and use
kg/s. Batch operation volume is the explicit reactor volume; the default
1 m³ value is its documented unit-volume reference.
"""
function _mesh_boundary_flow_state(solver::WENO, problem, ode_solution)
    saved_times = ode_solution.t
    sample_state = first(ode_solution.u)
    lower_mass_flow = Vector{typeof(zero(eltype(sample_state)))}(undef, length(saved_times))
    upper_mass_flow = similar(lower_mass_flow)
    growth_rate_cache = similar(sample_state, solver.meshsize)
    # Prepared inference remakes p without replacing the physical template.
    parameters = ode_solution.prob.p
    lower_length = first(solver.cell_face)
    upper_length = last(solver.cell_face)
    mass_scale = problem.crystal_density * problem.volume_shape_factor

    for time_index in eachindex(saved_times)
        numerical_state = ode_solution.u[time_index]
        numberdensity = crystal_state(problem, numerical_state)
        simulation_time = saved_times[time_index]
        nucleation_rate_value = nucleationrate(problem.kinetics_nucleationfunction,
                                               parameters.nucl, problem,
                                               numerical_state, simulation_time)

        if problem.kinetics_growthfunction isa Union{AbstractFPScalarGrowthFunction,
                                                     AbstractFPScalarDissolutionFunction} &&
           !(problem.kinetics_dissolutionfunction isa AbstractFPLengthDissolutionFunction)
            scalar_growth_rate = net_growth_rate(
                problem.kinetics_growthfunction, parameters.gr,
                problem.kinetics_dissolutionfunction, parameters.diss,
                problem, numerical_state, simulation_time)
            if scalar_growth_rate > zero(scalar_growth_rate)
                # Only the boundary values are reported. The positive scalar
                # path's lower flux is the nucleation inflow, and its upper
                # flux uses the same second-order outflow trace as the RHS.
                lower_boundary_flux = nucleation_rate_value
                upwind_state = numberdensity[end]
                high_order_state = upwind_state +
                                   0.5 * (upwind_state - numberdensity[end - 1])
                upper_boundary_flux = scalar_growth_rate *
                    _weno_nonnegative_state(high_order_state, upwind_state)
            elseif scalar_growth_rate < zero(scalar_growth_rate)
                # Signed scalar dissolution exits through lmin and has no
                # incoming population at lmax.
                lower_boundary_flux = scalar_growth_rate * first(numberdensity)
                upper_boundary_flux = zero(scalar_growth_rate)
            else
                lower_boundary_flux = zero(scalar_growth_rate)
                upper_boundary_flux = zero(scalar_growth_rate)
            end
        else
            # Preserve the model's full mesh, state, and time context. Custom
            # length-dependent laws need not support pointwise endpoint calls.
            net_growth_rate!(growth_rate_cache,
                             problem.kinetics_growthfunction, parameters.gr,
                             problem.kinetics_dissolutionfunction, parameters.diss,
                             problem, numerical_state, simulation_time,
                             solver.cell_centre)

            lower_growth_rate = first(growth_rate_cache)
            lower_boundary_flux = lower_growth_rate > zero(lower_growth_rate) ?
                nucleation_rate_value :
                lower_growth_rate < zero(lower_growth_rate) ?
                lower_growth_rate * first(numberdensity) : zero(lower_growth_rate)

            upper_growth_rate = last(growth_rate_cache)
            if upper_growth_rate > zero(upper_growth_rate)
                upwind_state = last(numberdensity)
                high_order_state = length(numberdensity) == 1 ? upwind_state :
                    upwind_state + 0.5 * (upwind_state - numberdensity[end - 1])
                upper_boundary_flux = upper_growth_rate *
                    _weno_nonnegative_state(high_order_state, upwind_state)
            else
                upper_boundary_flux = zero(upper_growth_rate)
            end
        end

        reactor_volume_value = _mesh_reactor_volume(problem, numerical_state)
        # The old full flux buffer converted each face value to the solution
        # element type before applying the physical mass-flow scaling.
        lower_boundary_flux = convert(eltype(sample_state), lower_boundary_flux)
        upper_boundary_flux = convert(eltype(sample_state), upper_boundary_flux)
        lower_mass_flow[time_index] = -lower_boundary_flux * lower_length^3 *
                                      mass_scale * reactor_volume_value
        upper_mass_flow[time_index] = upper_boundary_flux * upper_length^3 *
                                      mass_scale * reactor_volume_value
    end

    return (; size_boundary_lower_solid_mass_flow = lower_mass_flow,
              size_boundary_upper_solid_mass_flow = upper_mass_flow)
end

function _weno_rhs!(dstdt, st, p, t, CryProblem,
                    growthfunction::Union{AbstractFPScalarGrowthFunction,
                                          AbstractFPScalarDissolutionFunction},
                    flux_cache, ndens_pad_cache, growth_rate_cache)
    numberdensity = crystal_state(CryProblem, st)
    if CryProblem.kinetics_dissolutionfunction isa AbstractFPLengthDissolutionFunction
        net_growth_rates = get_tmp(growth_rate_cache, st)
        net_growth_rate!(net_growth_rates,
                               growthfunction,
                               p.gr,
                               CryProblem.kinetics_dissolutionfunction,
                               p.diss,
                               CryProblem,
                               st,
                               t,
                               CryProblem.solver.cell_centre)
        inflowbc = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                                  CryProblem, st, t)
        _fill_signed_weno_flux!(flux_cache,
                                numberdensity,
                                net_growth_rates,
                                inflowbc,
                                ndens_pad_cache)
        _weno_divergence_sweep!(dstdt, st, p, t, flux_cache, CryProblem, net_growth_rates)
        return nothing
    end

    scalar_growth_rate = net_growth_rate(growthfunction,
                                         p.gr,
                                         CryProblem.kinetics_dissolutionfunction,
                                         p.diss,
                                         CryProblem,
                                         st,
                                         t)
    inflowbc = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                              CryProblem, st, t)

    if scalar_growth_rate > zero(scalar_growth_rate)
        # Positive growth fast path: preserve the existing WENO reconstruction
        # and lower-boundary nucleation convention.
        _fill_positive_weno_flux!(flux_cache, numberdensity,
                                  scalar_growth_rate, inflowbc,
                                  ndens_pad_cache)
    elseif scalar_growth_rate < zero(scalar_growth_rate)
        # Negative growth is outflow at lmin and zero-inflow at lmax.  Use
        # the mirrored fifth-order WENO right trace at every interior face;
        # only a locally negative reconstruction blends to first-order
        # upwinding through _weno_nonnegative_state.
        _fill_signed_weno_flux!(flux_cache,
                                numberdensity,
                                scalar_growth_rate,
                                inflowbc,
                                ndens_pad_cache)
    else
        fill!(flux_cache, zero(scalar_growth_rate))
    end

    _weno_divergence_sweep!(dstdt, st, p, t, flux_cache, CryProblem, scalar_growth_rate)
    return nothing
end

function _weno_rhs!(dstdt, st, p, t, CryProblem,
                    growthfunction::Union{AbstractFPLengthGrowthFunction,
                                          AbstractFPLengthDissolutionFunction},
                    flux_cache, ndens_pad_cache, growth_rate_cache)
    numberdensity = crystal_state(CryProblem, st)
    net_growth_rates = get_tmp(growth_rate_cache, st)
    net_growth_rate!(net_growth_rates,
                           growthfunction,
                           p.gr,
                           CryProblem.kinetics_dissolutionfunction,
                           p.diss,
                           CryProblem,
                           st,
                           t,
                           CryProblem.solver.cell_centre)
    inflowbc = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                              CryProblem, st, t)

    # Length-dependent transport remains a WENO transport solve: each face
    # selects the left or mirrored right fifth-order trace from the sign of
    # its local net growth rate.  The first-order state is only a local
    # positivity blend and never replaces the scheme globally.
    _fill_signed_weno_flux!(flux_cache,
                            numberdensity,
                            net_growth_rates,
                            inflowbc,
                            ndens_pad_cache)

    _weno_divergence_sweep!(dstdt, st, p, t, flux_cache, CryProblem, net_growth_rates)
    return nothing
end

function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  saveat) where {NuclF <: AbstractNucleationFunction,
                                                 GrF <: AbstractGrowthFunction,
                                                 BrF <: AbstractBreakageFunction,
                                                 AggF <: AbstractAggregationFunction,
                                                 NuP <: AbstractVector{<:Real},
                                                 GrP <: AbstractVector{<:Real},
                                                 BrP <: AbstractVector{<:Real},
                                                 AggP <: AbstractVector{<:Real},
                                                 TP <: AbstractTemperature}
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility.
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))
    _growth_rate_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize))
    feed_population = _operation_feed_population(CryProblem)

    function WENO_Model(dstdt, st, p, t)
        _weno_rhs!(dstdt,
                   st,
                   p,
                   t,
                   CryProblem,
                   CryProblem.kinetics_growthfunction,
                   get_tmp(_flux_cache_dc, st),
                   get_tmp(_ndens_pad_cache_dc, st),
                   _growth_rate_cache_dc)
        _add_operation_transport!(dstdt, st, CryProblem, t, feed_population)
        return nothing
    end

    θ = (;
         nucl = CryProblem.parameterset_nucleation,
         gr = CryProblem.parameterset_growth,
         diss = CryProblem.parameterset_dissolution,
         br = CryProblem.parameterset_breakage,
         agg = CryProblem.parameterset_aggregation)

    ET = eltype(CryProblem.parameterset_nucleation)
    utyped = ET.(_get_initial_state(CryProblem))
    function weno_cfl_bound(numerical_state, kinetic_parameters, simulation_time)
        growthfunction = CryProblem.kinetics_growthfunction
        if growthfunction isa Union{AbstractFPLengthGrowthFunction,
                                    AbstractFPLengthDissolutionFunction} ||
           CryProblem.kinetics_dissolutionfunction isa AbstractFPLengthDissolutionFunction
            net_growth_rates = get_tmp(_growth_rate_cache_dc, numerical_state)
            net_growth_rate!(net_growth_rates,
                                   growthfunction,
                                   kinetic_parameters.gr,
                                   CryProblem.kinetics_dissolutionfunction,
                                   kinetic_parameters.diss,
                                   CryProblem,
                                   numerical_state,
                                   simulation_time,
                                   CryProblem.solver.cell_centre)
            return _signed_cfl(CryProblem.solver.cell_dL, net_growth_rates; courant = 0.9)
        end
        return _signed_cfl(CryProblem.solver.cell_dL,
                           net_growth_rate(growthfunction,
                                                  kinetic_parameters.gr,
                                                  CryProblem.kinetics_dissolutionfunction,
                                                  kinetic_parameters.diss,
                                                  CryProblem,
                                                  numerical_state,
                                                  simulation_time);
                           courant = 0.9)
    end
    signed_growth_transport =
        !(CryProblem.kinetics_dissolutionfunction isa nodissolution) ||
        (CryProblem.kinetics_growthfunction isa AbstractFPScalarDissolutionFunction) ||
        (CryProblem.kinetics_growthfunction isa AbstractFPLengthDissolutionFunction)
    default_algorithm = signed_growth_transport ? :ssprk43 : :tsit5
    # Stage/step limiter hooks discard returned values.  A solve-local
    # callback instead caps the next step from the accepted-state transport
    # rate, including initialization.  For varying rates this is a local
    # CFL estimate, not a guarantee about every intermediate stage.
    time_scalar_type = typeof(float(saveat[1]))
    configured_dtmax = Ref(time_scalar_type(Inf))
    bounded_weno_cfl = (numerical_state, kinetic_parameters, simulation_time) ->
        min(configured_dtmax[], OrdinaryDiffEq.SciMLBase.value(
            weno_cfl_bound(numerical_state, kinetic_parameters, simulation_time)))
    cfl_step_callback = StepsizeLimiter(bounded_weno_cfl;
        safety_factor = one(time_scalar_type), cached_dtcache = zero(time_scalar_type))
    initialize_weno_cfl = (callback, numerical_state, simulation_time, integrator) -> begin
        # Preserve an explicit solve_options.dtmax, even when a slower rate
        # would permit a larger CFL step later in the simulation.
        configured_dtmax[] = integrator.opts.dtmax
        cfl_step_callback.initialize(callback, numerical_state, simulation_time, integrator)
    end
    solve_cfl_callback = DiscreteCallback(cfl_step_callback.condition,
        cfl_step_callback.affect!; initialize = initialize_weno_cfl,
        save_positions = (false, false))
    ODEprob = ODEProblem(WENO_Model, utyped, (saveat[1], saveat[end]), θ;
                         callback = solve_cfl_callback)
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, default_algorithm)
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  sol; boundary_flow_state = nothing) where {NuclF <:
                                                                            AbstractNucleationFunction,
                                                                            GrF <:
                                                                            AbstractGrowthFunction,
                                                                            BrF <:
                                                                            AbstractBreakageFunction,
                                                                            AggF <:
                                                                            AbstractAggregationFunction,
                                                                            NuP <:
                                                                            AbstractVector{<:Real},
                                                                            GrP <:
                                                                            AbstractVector{<:Real},
                                                                            BrP <:
                                                                            AbstractVector{<:Real},
                                                                            AggP <:
                                                                            AbstractVector{<:Real},
                                                                            TP <:
                                                                            AbstractTemperature}
    nd_matrix = sol[_population_state_range(CryProblem), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre, nd_matrix,
                                              CryProblem.volume_shape_factor)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)
    reactor_state = merge(_reactor_solution_state(CryProblem, sol),
                          boundary_flow_state === nothing ?
                              _mesh_boundary_flow_state(CryProblem.solver, CryProblem, sol) :
                              boundary_flow_state)

    return CrystallisationFVSolution(sol.t, ### Will eventually have to be changed to discretised solution
                                     solvent_solution_state.concentration,
                                     nd_matrix,
                                     vol_weighted_dens,
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens, 0.1),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens, 0.5),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens, 0.9),
                                     moments.d10,
                                     moments.d32,
                                     moments.d43,
                                     moments.moment2,
                                     solvent_solution_state,
                                     sol[:, end],
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode),
                                     reactor_state)
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  saveat; algorithm = nothing, solve_options::NamedTuple = (;),
                                  callback_factory = nothing)::CrystallisationFVSolution where {NuclF <:
                                                                            AbstractNucleationFunction,
                                                                            GrF <:
                                                                            AbstractGrowthFunction,
                                                                            BrF <:
                                                                            AbstractBreakageFunction,
                                                                            AggF <:
                                                                            AbstractAggregationFunction,
                                                                            NuP <:
                                                                            AbstractVector{<:Real},
                                                                            GrP <:
                                                                            AbstractVector{<:Real},
                                                                            BrP <:
                                                                            AbstractVector{<:Real},
                                                                            AggP <:
                                                                            AbstractVector{<:Real},
                                                                            TP <:
                                                                            AbstractTemperature}
    ODEprob, tstep_solver = crystallisation_odeproblem(CryProblem, saveat)
    ODEsol = _solve_crystallisation_ode(CryProblem, ODEprob,
                                          algorithm === nothing ? tstep_solver : algorithm,
                                          saveat; solve_options, callback_factory)
    return _wrap_solution(CryProblem, ODEsol)
end
