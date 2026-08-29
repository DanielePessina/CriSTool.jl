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
                                                                      BrP, AggP, TP},
                                   saveat) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPScalarGrowthFunction,
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
    # DiffCache automatically handles type conversion for dual numbers during AD
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))

    function HRFV_FLWmodel(dstdt, st, p, t)
        # Get properly-typed cache based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)

    numberdensity = crystal_state(CryProblem, st)

        cell_centre = CryProblem.solver.cell_centre

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, st, t)

        # Calculate flux into _flux_cache
        _flux_cache[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                                        p.nucl,
                                        CryProblem, st, t) ## Inflow

        _flux_cache[2] = scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2])

        for idx_cell in 3:length(numberdensity) # Corresponds to _flux_cache[idx_cell]
            grad_up = numberdensity[idx_cell - 1] - numberdensity[idx_cell - 2]
            grad_down = numberdensity[idx_cell] - numberdensity[idx_cell - 1]
            r = grad_up / max(eps(eltype(st)), grad_down)
            _flux_cache[idx_cell] = scalargrowth * (numberdensity[idx_cell - 1] +
                                      0.5 *
                                      fluxlimiter_ospre(r) *
                                      grad_down)
        end

        _flux_cache[length(numberdensity) + 1] = scalargrowth * (numberdensity[end] +
                                                   0.5 * (numberdensity[end] -
                                                    numberdensity[end - 1]))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)

        for i in 1:length(dstdt_nd_view) # i is cell index
            dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                               CryProblem.solver.cell_dL
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

        solvent_rates = _solvent_derivatives(CryProblem, st, t, scalargrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

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
        return 0.99 * CryProblem.solver.cell_dL[1] /
               growthrate(CryProblem.kinetics_growthfunction,
                          p.gr, CryProblem, u, t)
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
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
                                    sol) where {NuclF <:
                                                                              AbstractNucleationFunction,
                                                                              GrF <:
                                                                              AbstractFPScalarGrowthFunction,
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

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    nd_matrix = sol[1:(end - n_solvent), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre,
                                              nd_matrix,
                                              CryProblem.kv)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

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
                                     moments.mu2,
                                     solvent_solution_state,
                                     vec(sol[:, end]),
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat)::CrystallisationFVSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPScalarGrowthFunction,
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
    ODEsol = solve(ODEprob,
                   tstep_solver;
                   reltol = CryProblem.solver.reltol,
                   abstol = CryProblem.solver.abstol,
                   dense = false,
                   alg_hints = [:stiff],
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS,)
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
                                                                             AbstractFPLengthGrowthFunction,
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

    function HRFV_FLWmodel(dstdt, st, p, t)
        # Get properly-typed cache based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)

        numberdensity = crystal_state(CryProblem, st)

        cell_centre = CryProblem.solver.cell_centre

        lengthbasedgrowth = growthrate(CryProblem.kinetics_growthfunction,
                                       p.gr, CryProblem, st, t)

        # Calculate flux into _flux_cache
        _flux_cache[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                                        p.nucl, CryProblem, st, t) ## Inflow

        # Central difference for flux at first interior interface
        g_interface = 0.5 * (lengthbasedgrowth[1] + lengthbasedgrowth[2])
        _flux_cache[2] = g_interface * 0.5 * (numberdensity[1] + numberdensity[2])

        # High-resolution scheme for other interior interfaces
        for idx_cell in 3:length(numberdensity) # Corresponds to _flux_cache[idx_cell]
            g_interface = 0.5 *
                          (lengthbasedgrowth[idx_cell - 1] + lengthbasedgrowth[idx_cell])
            grad_up = numberdensity[idx_cell - 1] - numberdensity[idx_cell - 2]
            grad_down = numberdensity[idx_cell] - numberdensity[idx_cell - 1]
            r = grad_up / max(eps(eltype(st)), grad_down)
            reconstructed_n = numberdensity[idx_cell - 1] +
                              0.5 *
                              fluxlimiter_ospre(r) *
                              grad_down
            _flux_cache[idx_cell] = g_interface * reconstructed_n
        end

        # Outflow boundary condition
        _flux_cache[length(numberdensity) + 1] = lengthbasedgrowth[end] *
                                                 (numberdensity[end] +
                                                  0.5 * (numberdensity[end] -
                                                   numberdensity[end - 1]))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)

        for i in 1:length(dstdt_nd_view) # i is cell index
            dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                               CryProblem.solver.cell_dL
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

        solvent_rates = _solvent_derivatives(CryProblem, st, t, lengthbasedgrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

        return nothing
    end

    function CFLcallback(u, integrator, p, t)
        g_vec = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                           CryProblem, u, t)
        return 0.99 * CryProblem.solver.cell_dL / maximum(g_vec)
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
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
                                   sol) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPLengthGrowthFunction,
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

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    nd_matrix = sol[1:(end - n_solvent), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre,
                                              nd_matrix,
                                              CryProblem.kv)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

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
                                     moments.mu2,
                                     solvent_solution_state,
                                     vec(sol[:, end]),
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat)::CrystallisationFVSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPLengthGrowthFunction,
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
    ODEsol = solve(ODEprob,
                   tstep_solver;
                   reltol = CryProblem.solver.reltol,
                   abstol = CryProblem.solver.abstol,
                   dense = false,
                   alg_hints = [:stiff],
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS,)
    return _wrap_solution(CryProblem, ODEsol)
end
