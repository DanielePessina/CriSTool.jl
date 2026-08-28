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
                                           SM <: AbstractSaturationModel}

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
        du[end] = -3 * CryProblem.kv * CryProblem.ρ * scalargrowth * u[3]
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
                                           SM <: AbstractSaturationModel}

    fluxlimiter_ospre(r) = (1.5(r^2) + r) / (r^2 + r + 1)

    function HRFV_FLWmodel(dstdt, st, p, t)

        numberdensity = @view st[1:(end - 1)]

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, st, t)

        flux = vcat(nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                                   CryProblem, st, t), ## Inflow
                    scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2]),
                    [scalargrowth * (numberdensity[i-1] +
                      0.5 *
                      fluxlimiter_ospre((numberdensity[i-1] - numberdensity[i-2] + 1e-12) /
                                        (numberdensity[i] - numberdensity[i-1] + 1e-12)) *
                      (numberdensity[i] - numberdensity[i-1]))
                     for i in 3:length(numberdensity)],
                    scalargrowth *
                    (numberdensity[end] + 0.5 * (numberdensity[end] - numberdensity[end-1])))

        dstdt[1:(end - 1)] = -diff(flux) / CryProblem.solver.cell_dL
        dstdt[end] = -CryProblem.kv * CryProblem.ρ *
                     (3 * sum(CryProblem.solver.cell_dL .* numberdensity .* scalargrowth .*
                          CryProblem.solver.cell_centre .^ 2))

    end

    θ = ComponentArray(nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)
    ODEprob = ODEForwardSensitivityProblem(HRFV_FLWmodel,
                                           [zeros(Float64,
                                                  CryProblem.solver.meshsize);
                                            CryProblem.initial_concentration],
                                           (0.0, saveat[end]), θ)

    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5)
    ODEsol = solve(ODEprob, tstep_solver;
                   saveat = saveat,
                   maxiters = 1e8)

    return ODEsol
end
