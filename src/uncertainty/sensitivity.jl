"""
Functions for forward sensitivity analysis of crystallisation models using SciMLSensitivity.
"""

function forwardsensitivity(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                               noaggregation, MoM, NuP, GrP,
                                                               BrP, AggP, TP, SM},
                            saveat) where {NuclF <: AbstractNucleationFunction,
                                           GrF <: AbstractGrowthFunction,
                                           NuP <: AbstractVector{<:Real},
                                           GrP <: AbstractVector{<:Real},
                                           BrP <: AbstractVector{<:Real},
                                           AggP <: AbstractVector{<:Real},
                                           TP <: AbstractTemperature,
                                           SM <: AbstractSolubilityModel}

    function MoM_model(du, u, p, t)
        scalargrowth = growthrate(CryProblem.kinetics_growthfunction,
                                  p.gr, CryProblem, u, t)

        n_mom = CryProblem.solver.nmoments
        @assert n_mom >= 2 "MoM solver requires nmoments >= 2 (concentration closure uses µ2)"

        du[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                               p.nucl, CryProblem, u, t)
        for k in 2:(n_mom + 1)
            du[k] = (k - 1) * scalargrowth * u[k - 1]
        end
        solvent_rates = _solvent_derivatives(CryProblem, u, t, scalargrowth)
        _write_solvent_derivatives!(du, CryProblem, solvent_rates)
    end

    θ = ComponentArray(nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth)

    ET = eltype(CryProblem.parameterset_nucleation)
    u0_typed = ET.(_get_initial_state(CryProblem))
    ODEprob = ODEForwardSensitivityProblem(MoM_model,
                                           u0_typed,
                                           (saveat[1], saveat[end]),
                                           θ)

    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5)
    sol = solve(ODEprob,
                tstep_solver;
                saveat = saveat,
                reltol = CryProblem.solver.reltol,
                abstol = CryProblem.solver.abstol,)

    return sol
end

function forwardsensitivity(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                               FiniteVol, NuP, GrP, BrP,
                                                               AggP, TP, SM},
                            saveat) where {NuclF <: AbstractNucleationFunction,
                                           GrF <: AbstractGrowthFunction,
                                           BrF <: AbstractBreakageFunction,
                                           AggF <: AbstractAggregationFunction,
                                           NuP <: AbstractVector{<:Real},
                                           GrP <: AbstractVector{<:Real},
                                           BrP <: AbstractVector{<:Real},
                                           AggP <: AbstractVector{<:Real},
                                           TP <: AbstractTemperature,
                                           SM <: AbstractSolubilityModel}

    function HRFV_FLWmodel(dstdt, st, p, t)

        numberdensity = crystal_state(CryProblem, st)
        # Parameter forward sensitivities keep the state Float64 while `p`
        # carries Dual values, so a state-keyed DiffCache would return a
        # Float64 buffer and discard derivative information.
        flux = Vector{promote_type(eltype(st), eltype(p))}(undef,
                                                            length(numberdensity) + 1)

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, st, t)

        if scalargrowth > zero(scalargrowth)
            flux[1] = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                                     CryProblem, st, t)
            flux[2] = scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2])
            for index in 3:length(numberdensity)
                grad_up = numberdensity[index - 1] - numberdensity[index - 2]
                grad_down = numberdensity[index] - numberdensity[index - 1]
                r = grad_up / max(eps(eltype(st)), grad_down)
                flux[index] = scalargrowth * (numberdensity[index - 1] +
                             0.5 * fluxlimiter_ospre(r) * grad_down)
            end
            flux[end] = scalargrowth * (numberdensity[end] +
                                        0.5 * (numberdensity[end] - numberdensity[end - 1]))
        elseif scalargrowth < zero(scalargrowth)
            flux[1] = scalargrowth * numberdensity[1]
            @inbounds for face in 2:length(numberdensity)
                flux[face] = scalargrowth * numberdensity[face]
            end
            flux[end] = zero(scalargrowth)
        else
            fill!(flux, zero(scalargrowth))
        end

        @inbounds for index in eachindex(numberdensity)
            dstdt[index] = -(flux[index + 1] - flux[index]) /
                           CryProblem.solver.cell_dL
        end
        solvent_rates = _solvent_derivatives(CryProblem, st, t, scalargrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

    end

    θ = ComponentArray(nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)
    ODEprob = ODEForwardSensitivityProblem(HRFV_FLWmodel,
                                           Float64.(_get_initial_state(CryProblem)),
                                           (0.0, saveat[end]), θ)

    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5)
    ODEsol = solve(ODEprob, tstep_solver;
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS)

    return ODEsol
end
