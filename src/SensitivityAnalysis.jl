"""
Functions for forward sensitivity analysis of crystallisation models using SciMLSensitivity.
"""

function forwardsensitivity(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                               noaggregation, MoM, NuP, GrP,
                                                               BrP, AggP},
                            saveat) where {NuclF <: AbstractNucleationFunction,
                                           GrF <: AbstractGrowthFunction,
                                           NuP <: AbstractVector{<:Real},
                                           GrP <: AbstractVector{<:Real},
                                           BrP <: AbstractVector{<:Real},
                                           AggP <: AbstractVector{<:Real}}

    function MoM_model(du, u, p, t)
        numberdensity = @view u[1:(end - 1)]
        temp = temperature(CryProblem.temp_profile, t)
        sat_conc = _get_saturationconcentration(CryProblem.temp_profile, t)
        supersat = u[end] / sat_conc

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction,
                                  p.gr,
                                  supersat,
                                  CryProblem,
                                  temp,
                                  CryProblem.loading,
                                  numberdensity)

        du[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                               p.nucl,
                               supersat,
                               CryProblem,
                               temp,
                               CryProblem.loading,
                               numberdensity)

        du[2] = scalargrowth * u[1]
        du[3] = 2 * scalargrowth * u[2]
        du[4] = 3 * scalargrowth * u[3]
        du[5] = 4 * scalargrowth * u[4]
        du[6] = -3 * CryProblem.kv * CryProblem.ρ * scalargrowth * u[3]
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
                                                               AggP},
                            saveat) where {NuclF <: AbstractNucleationFunction,
                                           GrF <: AbstractGrowthFunction,
                                           BrF <: AbstractBreakageFunction,
                                           AggF <: AbstractAggregationFunction,
                                           NuP <: AbstractVector{<:Real},
                                           GrP <: AbstractVector{<:Real},
                                           BrP <: AbstractVector{<:Real},
                                           AggP <: AbstractVector{<:Real}}

    fluxlimiter_sb(r) = max(0, min(1, 2 * r), min(2, r))
    fluxlimiter_vl(r) = (abs(r) + r) / (1 + abs(r))
    fluxlimiter_ospre(r) = (1.5(r^2) + r) / (r^2 + r + 1)

    function HRFV_FLWmodel(dstdt, st, p, t)

        numberdensity = @view st[1:(end - 1)]

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  st[end] / CryProblem.saturation_concentration, CryProblem,
                                  numberdensity)

        flux = vcat(nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                                   st[end] / CryProblem.saturation_concentration,
                                   CryProblem, numberdensity), ## Inflow
                    scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2]),
                    [scalargrowth * (numberdensity[i-1] +
                      0.5 *
                      fluxlimiter_ospre((numberdensity[i-1] - numberdensity[i-2] + 1e-12) /
                                        (numberdensity[i] - numberdensity[i-1] + 1e-12)) *
                      (numberdensity[i] - numberdensity[i-1]))
                     for i in 3:length(numberdensity)],
                    scalargrowth *
                    (numberdensity[end] + 0.5 * (numberdensity[end] - numberdensity[end-1])))

        dstdt[1:(end - 1)] = -diff(flux) / CryProblem.solver.cell_dL[1]
        dstdt[end] = -CryProblem.kv * CryProblem.ρ *
                     (3 * sum(CryProblem.solver.cell_dL .* numberdensity .* scalargrowth .*
                          CryProblem.solver.cell_centre .^ 2))

    end

    CFLcallback(c) = CryProblem.solver.cell_dL[1] /
                     growthrate(CryProblem.kinetics_growthfunction,
                                CryProblem.parameterset_growth,
                                c / CryProblem.saturation_concentration, CryProblem,
                                zeros(Float64, CryProblem.solver.meshsize))
    CFLcallback!(u, integrator, p,
                 t) = 0.7 * CryProblem.solver.cell_dL[1] /
                      growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                 u[end] / CryProblem.saturation_concentration, CryProblem,
                                 @view u[1:(end - 1)])

    θ = ComponentArray(nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)
    ODEprob = ODEForwardSensitivityProblem(HRFV_FLWmodel,
                                           convert(NuP,
                                                   [zeros(Float64,
                                                          CryProblem.solver.meshsize);
                                                    CryProblem.initial_concentration]),
                                           (0.0, saveat[end]), θ)

    ODEsol = solve(ODEprob,
                   Tsit5(),
                   # reltol=[ones(Float64, CryProblem.solver.meshsize) * 1e-1; 1e-3],
                   # abstol=[ones(Float64, CryProblem.solver.meshsize) * 1e-4; 1e-6],
                   saveat = saveat,
                   maxiters = 1e8)

    return ODEsol
end
