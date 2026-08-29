"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., MoM, ...}, saveat) -> CrystallisationMoMSolution

Simulate crystallisation using Method of Moments solver.

Internal function that solves the moment equations for nucleation and growth
without aggregation or breakage.

# Arguments
- `CryProblem`: Crystallisation problem with MoM solver
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationMoMSolution` containing time, concentration, and moment-derived sizes
"""
@inline function _mom_rhs(CryProblem, scalargrowth, B, solvent_rates, u::SVector{6})
    return SVector(B, scalargrowth * u[1], 2 * scalargrowth * u[2],
                   3 * scalargrowth * u[3], 4 * scalargrowth * u[4],
                   solvent_rates[1])
end

@inline function _mom_rhs(CryProblem, scalargrowth, B, solvent_rates,
                          u::SVector{N}) where {N}
    n_states = N
    n_solvent = length(solvent_rates)
    n_population = n_states - n_solvent
    return SVector(ntuple(Val(n_states)) do k
        k <= n_population ?
            (k == 1 ? B : (k - 1) * scalargrowth * u[k - 1]) :
            solvent_rates[k - n_population]
    end)
end

function _solvent_derivatives(problem::CrystallisationProblem, state, time, growth)
    return _solvent_derivatives(problem, state, time, growth,
                                problem.solvent_dynamics)
end

function _solvent_derivatives(problem::CrystallisationProblem, state, time, growth,
                              solvent_dynamics)
    rates = solvent_dynamics(problem, state, time, growth)
    names = propertynames(problem.initial_solvent_state)
    if rates isa NamedTuple
        return ntuple(index -> begin
            name = names[index]
            hasproperty(rates, name) ? getproperty(rates, name) : zero(growth)
        end, Val(length(names)))
    end
    length(rates) == length(names) ||
        throw(ArgumentError("solvent_dynamics must return one rate per initial solvent variable."))
    return rates
end

function _write_solvent_derivatives!(dst, problem::CrystallisationProblem, rates)
    n_solvent = length(propertynames(problem.initial_solvent_state))
    first_solvent = length(dst) - n_solvent + 1
    @inbounds for index in 1:n_solvent
        dst[first_solvent + index - 1] = rates[index]
    end
    return nothing
end

function _solvent_solution_state(problem::CrystallisationProblem, solution)
    names = propertynames(problem.initial_solvent_state)
    n_solvent = length(names)
    first_solvent = size(solution, 1) - n_solvent + 1
    values = ntuple(index -> solution[first_solvent + index - 1, :], Val(n_solvent))
    return NamedTuple{names}(values)
end

function _solvent_solution_index(problem::CrystallisationProblem, solution, name::Symbol)
    position = findfirst(==(name), propertynames(problem.initial_solvent_state))
    position === nothing && throw(ArgumentError("Unknown solvent-state variable :$name."))
    return size(solution, 1) - length(propertynames(problem.initial_solvent_state)) + position
end

function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                                       noaggregation, MoM,
                                                                       NuP, GrP, BrP, AggP,
                                                                       TP},
                                    saveat) where {NuclF <:
                                                   AbstractNucleationFunction,
                                                   GrF <:
                                                   AbstractGrowthFunction,
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
    n_mom = CryProblem.solver.nmoments
    @assert n_mom >= 2 "MoM solver requires nmoments >= 2 (concentration closure uses µ2)"

    function MoM_model(u, p, t)
        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, u, t)
        B = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                           CryProblem, u, t)
        solvent_rates = _solvent_derivatives(CryProblem, u, t, scalargrowth)
        return _mom_rhs(CryProblem, scalargrowth, B, solvent_rates, u)
    end

    θ = ComponentArray(;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth)

    ET = eltype(CryProblem.parameterset_nucleation)
    n_mom = CryProblem.solver.nmoments
    u0_vec = ET.(_get_initial_state(CryProblem))
    n_states = n_mom + 1 + length(propertynames(CryProblem.initial_solvent_state))
    u0_typed = SVector(ntuple(k -> u0_vec[k], Val(n_states)))
    ODEprob = ODEProblem(MoM_model, u0_typed, (saveat[1], saveat[end]), θ)

    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5)
    return (ODEprob, tstep_solver)
end

function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                           noaggregation, MoM,
                                                           NuP, GrP, BrP, AggP,
                                                           TP},
                        sol) where {NuclF <: AbstractNucleationFunction,
                                    GrF <: AbstractGrowthFunction,
                                    NuP <: AbstractVector{<:Real},
                                    GrP <: AbstractVector{<:Real},
                                    BrP <: AbstractVector{<:Real},
                                    AggP <: AbstractVector{<:Real},
                                    TP <: AbstractTemperature}
    final_state = collect(sol[:, end])

    n_mom = CryProblem.solver.nmoments
    n_states = n_mom + 1 + length(propertynames(CryProblem.initial_solvent_state))
    # Fixed moment indices: state k holds µ_{k-1}; µ2 = state 3, µ3 = state 4,
    # µ4 = state 5. Higher moments (if any) do not change these metrics.
    d32 = n_mom >= 3 ? CRISTOOL_MICROMETER_SCALE .* sol[4, :] ./
                       (sol[3, :] .+ CRISTOOL_MOMENT_RATIO_FLOOR) :
          fill(NaN, length(sol.t))
    d43 = n_mom >= 4 ? CRISTOOL_MICROMETER_SCALE .* sol[5, :] ./
                       (sol[4, :] .+ CRISTOOL_MOMENT_RATIO_FLOOR) :
          fill(NaN, length(sol.t))
    mu2 = n_mom >= 2 ? sol[3, :] : fill(NaN, length(sol.t))
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

    return CrystallisationMoMSolution(sol.t,
                                      solvent_solution_state.concentration,
                                      CRISTOOL_MICROMETER_SCALE * sol[2, :] ./
                                      (sol[1, :] .+ CRISTOOL_MOMENT_RATIO_FLOOR),
                                      d32,
                                      d43,
                                      mu2,
                                      solvent_solution_state,
                                      final_state,
                                      sol.stats,
                                      OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end

function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                                     noaggregation, MoM,
                                                                     NuP, GrP, BrP, AggP,
                                                                     TP},
                                  saveat)::CrystallisationMoMSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractGrowthFunction,
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
    sol = solve(ODEprob,
                tstep_solver;
                saveat = saveat,
                reltol = CryProblem.solver.reltol,
                abstol = CryProblem.solver.abstol,)
    return _wrap_solution(CryProblem, sol)
end
