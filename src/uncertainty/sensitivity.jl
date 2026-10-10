"""
Functions for forward sensitivity analysis of crystallisation models using SciMLSensitivity.
"""

"""
    forwardsensitivity(problem, saveat)

Forward parameter sensitivities of a batch `MoM` or scalar-growth `FiniteVol`
problem, solved with SciMLSensitivity's `ODEForwardSensitivityProblem`.
"""
forwardsensitivity(CryProblem::CrystallisationProblem, saveat) =
    forwardsensitivity(CryProblem.solver, CryProblem, saveat)
forwardsensitivity(solver::AbstractSolver, CryProblem::CrystallisationProblem, saveat) =
    throw(ArgumentError("forwardsensitivity supports MoM and FiniteVol, not $(typeof(solver))."))

function forwardsensitivity(solver::MoM, CryProblem::CrystallisationProblem, saveat)
    _validate_solver_kinetics(solver, CryProblem)
    CryProblem.operation isa BatchOperation ||
        throw(ArgumentError("forwardsensitivity currently supports batch operations; differentiate runsimulation for flow operations."))

    function MoM_model(du, u, p, t)
        scalargrowth = net_growth_rate(CryProblem.kinetics_growthfunction,
                                       p.gr,
                                       CryProblem.kinetics_dissolutionfunction,
                                       p.diss, CryProblem, u, t)

        n_mom = solver.nmoments
        du[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                               p.nucl, CryProblem, u, t)
        for k in 2:(n_mom + 1)
            du[k] = (k - 1) * scalargrowth * u[k - 1]
        end
        solvent_rates = _solvent_derivatives(CryProblem, u, t, scalargrowth)
        _write_solvent_derivatives!(du, CryProblem, solvent_rates)
    end

    θ = ComponentArray(nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       diss = CryProblem.parameterset_dissolution)

    ET = eltype(CryProblem.parameterset_nucleation)
    u0_typed = ET.(_get_initial_state(CryProblem))
    ODEprob = ODEForwardSensitivityProblem(MoM_model,
                                           u0_typed,
                                           (saveat[1], saveat[end]),
                                           θ)

    sol = solve(ODEprob,
                _timestepping_algorithm(solver, Tsit5());
                saveat = saveat,
                reltol = CryProblem.solver.reltol,
                abstol = CryProblem.solver.abstol,)

    return sol
end

function forwardsensitivity(solver::FiniteVol, CryProblem::CrystallisationProblem, saveat)
    CryProblem.operation isa BatchOperation ||
        throw(ArgumentError("forwardsensitivity currently supports batch operations; differentiate runsimulation for flow operations."))
    _has_binary_sources(CryProblem) &&
        throw(ArgumentError("FiniteVol forwardsensitivity does not support aggregation or breakage."))
    _uses_length_rates(CryProblem) &&
        throw(ArgumentError("FiniteVol forwardsensitivity does not support length-dependent growth or dissolution."))

    function HRFV_FLWmodel(dstdt, st, p, t)

        numberdensity = crystal_state(CryProblem, st)
        # Parameter forward sensitivities keep the state Float64 while `p`
        # carries Dual values, so a state-keyed DiffCache would return a
        # Float64 buffer and discard derivative information.
        flux = Vector{promote_type(eltype(st), eltype(p))}(undef,
                                                            length(numberdensity) + 1)

        scalargrowth = net_growth_rate(CryProblem.kinetics_growthfunction, p.gr,
                                       CryProblem.kinetics_dissolutionfunction,
                                       p.diss, CryProblem, st, t)

        boundary_nucleation_rate = scalargrowth > zero(scalargrowth) ?
            nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                           CryProblem, st, t) : zero(scalargrowth)
        _fill_fv_scalar_flux!(flux, numberdensity, scalargrowth, boundary_nucleation_rate)

        @inbounds for index in eachindex(numberdensity)
            dstdt[index] = -(flux[index + 1] - flux[index]) /
                           CryProblem.solver.cell_dL
        end
        solvent_rates = _solvent_derivatives(CryProblem, st, t, scalargrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

    end

    θ = ComponentArray(nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       diss = CryProblem.parameterset_dissolution,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)
    ODEprob = ODEForwardSensitivityProblem(HRFV_FLWmodel,
                                           Float64.(_get_initial_state(CryProblem)),
                                           (0.0, saveat[end]), θ)

    ODEsol = solve(ODEprob, _timestepping_algorithm(solver, Tsit5());
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS)

    return ODEsol
end
