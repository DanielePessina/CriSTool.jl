@inline _fv_scalar_net_growth_rate(growthfunction,
                                   growth_parameters,
                                   ::nodissolution,
                                   dissolution_parameters,
                                   problem,
                                   state,
                                   time) = growthrate(growthfunction,
                                                      growth_parameters,
                                                      problem,
                                                      state,
                                                      time)

@inline _fv_scalar_net_growth_rate(growthfunction,
                                   growth_parameters,
                                   dissolutionfunction::AbstractDissolutionFunction,
                                   dissolution_parameters,
                                   problem,
                                   state,
                                   time) = net_growth_rate(growthfunction,
                                                           growth_parameters,
                                                           dissolutionfunction,
                                                           dissolution_parameters,
                                                           problem,
                                                           state,
                                                           time)

"""Fill the legacy high-resolution FV scalar-growth flux in place."""
@inline function _fill_fv_scalar_flux!(flux::AbstractVector,
                                       numberdensity::AbstractVector,
                                       scalar_growth_rate,
                                       nucleation_rate)
    if scalar_growth_rate > zero(scalar_growth_rate)
        # Preserve the established high-resolution lower-boundary/nucleation
        # convention and OSPRE limiter at every interior face.
        flux[1] = nucleation_rate
        flux[2] = scalar_growth_rate * 0.5 * (numberdensity[1] + numberdensity[2])

        @inbounds for cell_index in 3:length(numberdensity)
            gradient_upstream = numberdensity[cell_index - 1] - numberdensity[cell_index - 2]
            gradient_downstream = numberdensity[cell_index] - numberdensity[cell_index - 1]
            gradient_floor = eps(eltype(numberdensity))
            gradient_denominator = abs(gradient_downstream) > gradient_floor ?
                gradient_downstream :
                (gradient_downstream < zero(gradient_downstream) ? -gradient_floor : gradient_floor)
            gradient_ratio = gradient_upstream / gradient_denominator
            flux[cell_index] = scalar_growth_rate *
                               (numberdensity[cell_index - 1] +
                                0.5 * fluxlimiter_ospre(gradient_ratio) *
                                gradient_downstream)
        end

        flux[length(numberdensity) + 1] = scalar_growth_rate *
                                          (numberdensity[end] +
                                           0.5 * (numberdensity[end] -
                                                  numberdensity[end - 1]))
    elseif scalar_growth_rate < zero(scalar_growth_rate)
        # Negative growth is outflow at L=lmin and zero-inflow at lmax.
        @inbounds begin
            flux[1] = scalar_growth_rate * numberdensity[1]
            for face_index in 2:length(numberdensity)
                flux[face_index] = scalar_growth_rate * numberdensity[face_index]
            end
            flux[end] = zero(scalar_growth_rate)
        end
    else
        fill!(flux, zero(scalar_growth_rate))
    end
    return flux
end

@inline _mesh_reactor_volume(problem, state) = reactor_volume(problem, state)

"""Boundary solid-mass flow samples using the exact FV face fluxes.

The signed lower and upper flows are positive out of the size domain and use
kg/s. Batch operation volume is the explicit reactor volume; the default
1 m³ value is its documented unit-volume reference.
"""
function _mesh_boundary_flow_state(solver::FiniteVol, problem, ode_solution)
    saved_times = ode_solution.t
    sample_state = first(ode_solution.u)
    lower_mass_flow = Vector{typeof(zero(eltype(sample_state)))}(undef, length(saved_times))
    upper_mass_flow = similar(lower_mass_flow)
    flux_cache = similar(sample_state, solver.meshsize + 1)
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
            scalar_growth_rate = _fv_scalar_net_growth_rate(
                problem.kinetics_growthfunction, parameters.gr,
                problem.kinetics_dissolutionfunction, parameters.diss,
                problem, numerical_state, simulation_time)
            _fill_fv_scalar_flux!(flux_cache, numberdensity,
                                  scalar_growth_rate, nucleation_rate_value)
        else
            net_growth_rate!(growth_rate_cache,
                             problem.kinetics_growthfunction, parameters.gr,
                             problem.kinetics_dissolutionfunction, parameters.diss,
                             problem, numerical_state, simulation_time,
                             solver.cell_centre)
            _fill_signed_first_order_flux!(flux_cache, numberdensity,
                                           growth_rate_cache,
                                           nucleation_rate_value)
        end

        reactor_volume_value = _mesh_reactor_volume(problem, numerical_state)
        lower_mass_flow[time_index] = -flux_cache[1] * lower_length^3 *
                                      mass_scale * reactor_volume_value
        upper_mass_flow[time_index] = flux_cache[end] * upper_length^3 *
                                      mass_scale * reactor_volume_value
    end

    return (; size_boundary_lower_solid_mass_flow = lower_mass_flow,
              size_boundary_upper_solid_mass_flow = upper_mass_flow)
end

"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., FiniteVol, ...}, saveat) -> CrystallisationFVSolution

Simulate crystallisation using Finite Volume solver with scalar growth.

Internal function that solves the population balance equation using high-resolution
finite volume method with flux limiters.

# Arguments
- `CryProblem`: Crystallisation problem with FiniteVol solver and scalar growth function
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationFVSolution` containing time, concentration, number density, and size quantiles
"""

    function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
           BrP, AggP, TP, SM, SS, SD, DissF, DissP},
                                   saveat) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             Union{AbstractFPScalarGrowthFunction,
                                                                                   AbstractFPScalarDissolutionFunction},
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
                  TP <: AbstractTemperature,
                  SM <: AbstractSolubilityModel,
                  SS <: NamedTuple,
                  SD,
                  DissF <: AbstractDissolutionFunction,
                  DissP <: AbstractVector{<:Real}}

    # Pre-allocate flux cache using DiffCache for ForwardDiff compatibility
    # DiffCache automatically handles type conversion for dual numbers during AD
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _growth_rate_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize))
    feed_population = _operation_feed_population(CryProblem)

    function HRFV_FLWmodel(dstdt, st, p, t)
        # Get properly-typed cache based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)

    numberdensity = crystal_state(CryProblem, st)

        cell_centre = CryProblem.solver.cell_centre
        dissolutionfunction = CryProblem.kinetics_dissolutionfunction

        if DissF <: AbstractFPLengthDissolutionFunction
            net_growth_rates = get_tmp(_growth_rate_cache_dc, st)
            net_growth_rate!(net_growth_rates,
                                   CryProblem.kinetics_growthfunction,
                                   p.gr,
                                   dissolutionfunction,
                                   p.diss,
                                   CryProblem,
                                   st,
                                   t,
                                   cell_centre)
            _fill_signed_first_order_flux!(_flux_cache,
                                           numberdensity,
                                           net_growth_rates,
                                           nucleationrate(CryProblem.kinetics_nucleationfunction,
                                                          p.nucl,
                                                          CryProblem,
                                                          st,
                                                          t))
            scalar_growth_rate = net_growth_rates
        else
            # Keep the legacy no-dissolution scalar path on the original
            # growth-rate seam.  Besides avoiding an unnecessary signed-rate
            # dispatch, this preserves the established FV RHS cost exactly;
            # the independent contribution is evaluated only when a model was
            # explicitly supplied in the `diss` slot.
            scalar_growth_rate = _fv_scalar_net_growth_rate(
                CryProblem.kinetics_growthfunction,
                p.gr,
                dissolutionfunction,
                p.diss,
                CryProblem,
                st,
                t)
            _fill_fv_scalar_flux!(_flux_cache, numberdensity, scalar_growth_rate,
                                  nucleationrate(CryProblem.kinetics_nucleationfunction,
                                                 p.nucl, CryProblem, st, t))
        end

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)

        if _fused_solvent_coupling_enabled(CryProblem)
            depletion = _fused_depletion_divergence!(dstdt_nd_view, _flux_cache,
                                                     numberdensity,
                                                     CryProblem.solver.cell_centre,
                                                     CryProblem.solver.cell_dL,
                                                     scalar_growth_rate)
        else
            for i in 1:length(dstdt_nd_view) # i is cell index
                dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                                   CryProblem.solver.cell_dL
            end
        end

        # Add aggregation and breakage terms
        agg_rate = aggregationrate(CryProblem.kinetics_aggregationfunction,
                                   p.agg,
                                   CryProblem, st, t)
        br_rate = breakagerate(CryProblem.kinetics_breakagefunction,
                               p.br,
                               CryProblem, st, t)

        if !(typeof(agg_rate) <: Real && agg_rate == 0.0)
            dstdt_nd_view .+= agg_rate
        end
        if !(typeof(br_rate) <: Real && br_rate == 0.0)
            dstdt_nd_view .+= br_rate
        end

        if _fused_solvent_coupling_enabled(CryProblem)
            depletion_coupled = 3 * CryProblem.volume_shape_factor *
                                CryProblem.crystal_density * depletion
            dstdt[end] = -depletion_coupled
        else
            solvent_rates = _solvent_derivatives(CryProblem, st, t, scalar_growth_rate)
            _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)
        end

        _add_operation_transport!(dstdt, st, CryProblem, t, feed_population)

        return nothing
    end

    # function CFLcallback(c)
    #     return CryProblem.solver.cell_dL[1] /
    #            growthrate(CryProblem.kinetics_growthfunction,
    #                       CryProblem.parameterset_growth,
    #                       c / _get_saturationconcentration(CryProblem.temp_profile, t),
    #                       CryProblem, temperature(CryProblem.temp_profile, t),
    #                       zeros(Float64, CryProblem.solver.meshsize))
    # end
    function CFLcallback(u, integrator, p, t)
        if DissF <: AbstractFPLengthDissolutionFunction
            net_growth_rates = get_tmp(_growth_rate_cache_dc, u)
            net_growth_rate!(net_growth_rates,
                                   CryProblem.kinetics_growthfunction,
                                   p.gr,
                                   CryProblem.kinetics_dissolutionfunction,
                                   p.diss,
                                   CryProblem,
                                   u,
                                   t,
                                   CryProblem.solver.cell_centre)
            return _signed_cfl(CryProblem.solver.cell_dL, net_growth_rates)
        end
        scalar_growth_rate = _fv_scalar_net_growth_rate(
            CryProblem.kinetics_growthfunction,
            p.gr,
            CryProblem.kinetics_dissolutionfunction,
            p.diss,
            CryProblem,
            u,
            t)
        return _signed_cfl(CryProblem.solver.cell_dL, scalar_growth_rate)
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       diss = CryProblem.parameterset_dissolution,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)

    # Construct initial conditions with the correct element type
    u0_typed = eltype(CryProblem.parameterset_nucleation).(_get_initial_state(CryProblem))

    # println("Initial state shape: ", size(u0_typed))

    ODEprob = ODEProblem(HRFV_FLWmodel,
                         u0_typed, # eltype-matched plain state (do NOT convert to the params type)
                         (saveat[1], saveat[end]),
                         θ)
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5;
                                        step_limiter = CFLcallback);
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                       FiniteVol, NuP, GrP,
                                                                       BrP, AggP, TP},
                                    sol; boundary_flow_state = nothing) where {NuclF <:
                                                                              AbstractNucleationFunction,
                                                                              GrF <:
                                                                              Union{AbstractFPScalarGrowthFunction,
                                                                                    AbstractFPScalarDissolutionFunction},
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
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre,
                                              nd_matrix,
                                              CryProblem.volume_shape_factor)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)
    reactor_state = merge(_reactor_solution_state(CryProblem, sol),
                          boundary_flow_state === nothing ?
                              _mesh_boundary_flow_state(CryProblem.solver, CryProblem, sol) :
                              boundary_flow_state)

    return CrystallisationFVSolution(sol.t,
                                     solvent_solution_state.concentration,
                                     nd_matrix,
                                     vol_weighted_dens,
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.1),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.5),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.9),
                                     moments.d10,
                                     moments.d32,
                                     moments.d43,
                                     moments.moment2,
                                     solvent_solution_state,
                                     vec(sol[:, end]),
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode),
                                     reactor_state)
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat; algorithm = nothing, solve_options::NamedTuple = (;),
                                  callback_factory = nothing)::CrystallisationFVSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             Union{AbstractFPScalarGrowthFunction,
                                                                                   AbstractFPScalarDissolutionFunction},
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

"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., FiniteVol, ...}, saveat) -> CrystallisationFVSolution

Simulate crystallisation using Finite Volume solver with length-dependent growth.

Internal function that solves the population balance equation using high-resolution
finite volume method with size-dependent growth rates.

# Arguments
- `CryProblem`: Crystallisation problem with FiniteVol solver and length-based growth function
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationFVSolution` containing time, concentration, number density, and size quantiles
"""
    function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             Union{AbstractFPLengthGrowthFunction,
                                                                                   AbstractFPLengthDissolutionFunction},
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
    # Pre-allocate flux cache using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _growth_rate_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize))
    feed_population = _operation_feed_population(CryProblem)

    function HRFV_FLWmodel(dstdt, st, p, t)
        # Get properly-typed cache based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)

        numberdensity = crystal_state(CryProblem, st)

        cell_centre = CryProblem.solver.cell_centre

        net_growth_rates = get_tmp(_growth_rate_cache_dc, st)
        net_growth_rate!(net_growth_rates,
                               CryProblem.kinetics_growthfunction,
                               p.gr,
                               CryProblem.kinetics_dissolutionfunction,
                               p.diss,
                               CryProblem,
                               st,
                               t,
                               cell_centre)

        _fill_signed_first_order_flux!(_flux_cache,
                                       numberdensity,
                                       net_growth_rates,
                                       nucleationrate(CryProblem.kinetics_nucleationfunction,
                                                      p.nucl,
                                                      CryProblem,
                                                      st,
                                                      t))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)

        if _fused_solvent_coupling_enabled(CryProblem)
            depletion = _fused_depletion_divergence!(dstdt_nd_view, _flux_cache,
                                                     numberdensity,
                                                     CryProblem.solver.cell_centre,
                                                     CryProblem.solver.cell_dL,
                                                     net_growth_rates)
        else
            for i in 1:length(dstdt_nd_view) # i is cell index
                dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                                   CryProblem.solver.cell_dL
            end
        end

        # Add aggregation and breakage terms
        agg_rate = aggregationrate(CryProblem.kinetics_aggregationfunction,
                                   p.agg,
                                   CryProblem, st, t)
        br_rate = breakagerate(CryProblem.kinetics_breakagefunction,
                               p.br,
                               CryProblem, st, t)

        if !(typeof(agg_rate) <: Real && agg_rate == 0.0)
            dstdt_nd_view .+= agg_rate
        end
        if !(typeof(br_rate) <: Real && br_rate == 0.0)
            dstdt_nd_view .+= br_rate
        end

        if _fused_solvent_coupling_enabled(CryProblem)
            depletion_coupled = 3 * CryProblem.volume_shape_factor *
                                CryProblem.crystal_density * depletion
            dstdt[end] = -depletion_coupled
        else
            solvent_rates = _solvent_derivatives(CryProblem, st, t, net_growth_rates)
            _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)
        end

        _add_operation_transport!(dstdt, st, CryProblem, t, feed_population)

        return nothing
    end

    function CFLcallback(u, integrator, p, t)
        net_growth_rates = get_tmp(_growth_rate_cache_dc, u)
        net_growth_rate!(net_growth_rates,
                               CryProblem.kinetics_growthfunction,
                               p.gr,
                               CryProblem.kinetics_dissolutionfunction,
                               p.diss,
                               CryProblem,
                               u,
                               t,
                               CryProblem.solver.cell_centre)
        return _signed_cfl(CryProblem.solver.cell_dL, net_growth_rates)
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       diss = CryProblem.parameterset_dissolution,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)

    # Construct initial conditions with the correct element type
    u0_typed = eltype(CryProblem.parameterset_nucleation).(_get_initial_state(CryProblem))

    ODEprob = ODEProblem(HRFV_FLWmodel,
                         u0_typed, # eltype-matched plain state (do NOT convert to the params type)
                         (saveat[1], saveat[end]),
                         θ)
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :ssprk43;
                                        step_limiter = CFLcallback);
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   sol; boundary_flow_state = nothing) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             Union{AbstractFPLengthGrowthFunction,
                                                                                   AbstractFPLengthDissolutionFunction},
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
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre,
                                              nd_matrix,
                                              CryProblem.volume_shape_factor)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)
    reactor_state = merge(_reactor_solution_state(CryProblem, sol),
                          boundary_flow_state === nothing ?
                              _mesh_boundary_flow_state(CryProblem.solver, CryProblem, sol) :
                              boundary_flow_state)

    return CrystallisationFVSolution(sol.t,
                                     solvent_solution_state.concentration,
                                     nd_matrix,
                                     vol_weighted_dens,
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.1),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.5),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.9),
                                     moments.d10,
                                     moments.d32,
                                     moments.d43,
                                     moments.moment2,
                                     solvent_solution_state,
                                     vec(sol[:, end]),
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode),
                                     reactor_state)
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat; algorithm = nothing, solve_options::NamedTuple = (;),
                                  callback_factory = nothing)::CrystallisationFVSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             Union{AbstractFPLengthGrowthFunction,
                                                                                   AbstractFPLengthDissolutionFunction},
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
    # Pre-allocate flux cache using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))

    ODEprob, tstep_solver = crystallisation_odeproblem(CryProblem, saveat)
    ODEsol = _solve_crystallisation_ode(CryProblem, ODEprob,
                                          algorithm === nothing ? tstep_solver : algorithm,
                                          saveat; solve_options, callback_factory)
    return _wrap_solution(CryProblem, ODEsol)
end
