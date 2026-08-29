"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., WENO, ...}, saveat) -> CrystallisationFVSolution

Simulate crystallisation using WENO (Weighted Essentially Non-Oscillatory) solver.

Internal function that solves the population balance equation using 5th-order WENO
reconstruction for high accuracy near discontinuities.

# Arguments
- `CryProblem`: Crystallisation problem with WENO solver
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationFVSolution` containing time, concentration, number density, and size quantiles
"""
function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  saveat) where {NuclF <:
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
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))

    function WENO_Model(dstdt, st, p, t)
        # Get properly-typed caches based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)
        _ndens_pad_cache = get_tmp(_ndens_pad_cache_dc, st)

        numberdensity = crystal_state(CryProblem, st)
        cell_centre = CryProblem.solver.cell_centre
        temp = temperature(CryProblem.temp_profile, t)
        S = supersaturation(CryProblem, st, t)

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, st, t)
        inflowbc = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                                  CryProblem, st, t)

        # Fill padded density cache
        _ndens_pad_cache[1] = inflowbc
        _ndens_pad_cache[2] = inflowbc
        _ndens_pad_cache[3:(end - 2)] .= numberdensity
        _ndens_pad_cache[end - 1] = zero(eltype(st))
        _ndens_pad_cache[end] = zero(eltype(st))

        # Calculate flux into _flux_cache
        _flux_cache[1] = inflowbc
        _flux_cache[2] = scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2])
        for i in 3:length(numberdensity)
            _flux_cache[i] = scalargrowth * weno_flux(_ndens_pad_cache, i + 1)
        end
        _flux_cache[end] = scalargrowth * (numberdensity[end] +
                            0.5 * (numberdensity[end] - numberdensity[end - 1]))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)
        for i in 1:length(dstdt_nd_view)
            dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                               CryProblem.solver.cell_dL
        end

        # Add aggregation and breakage terms
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

        solvent_rates = _solvent_derivatives(CryProblem, st, t, scalargrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

        return nothing
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)

    ET = eltype(CryProblem.parameterset_nucleation) # Assumes this reflects TPara
    utyped = ET.(_get_initial_state(CryProblem)) # Convert initial state to correct type
    ODEprob = ODEProblem(WENO_Model,
                         utyped, # NuP is type of parameterset_nucleation
                         (saveat[1], saveat[end]),
                         θ)

    function CFLcallback!(u, integrator, p, t)
        return 0.9 * CryProblem.solver.cell_dL[1] /
               growthrate(CryProblem.kinetics_growthfunction,
                          p.gr, CryProblem, u, t)
    end
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5;
                                                   stage_limiter = CFLcallback!);
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  sol) where {NuclF <:
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
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    nd_matrix = sol[1:(end - n_solvent), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre, nd_matrix,
                                              CryProblem.kv)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

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
                                     moments.mu2,
                                     solvent_solution_state,
                                     sol[:, end],
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  saveat)::CrystallisationFVSolution where {NuclF <:
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
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))

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
