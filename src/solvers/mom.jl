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
@inline function _mom_rhs(CryProblem, scalar_growth_rate, B, solvent_rates, u::SVector{6})
    return SVector(B, scalar_growth_rate * u[1], 2 * scalar_growth_rate * u[2],
                   3 * scalar_growth_rate * u[3], 4 * scalar_growth_rate * u[4],
                   solvent_rates[1])
end

@inline function _mom_rhs(CryProblem, scalar_growth_rate, B, solvent_rates,
                          u::SVector{N}) where {N}
    n_states = N
    n_solvent = length(solvent_rates)
    n_population = n_states - n_solvent
    return SVector(ntuple(Val(n_states)) do k
        k <= n_population ?
            (k == 1 ? B : (k - 1) * scalar_growth_rate * u[k - 1]) :
            solvent_rates[k - n_population]
    end)
end

function _mom_extinction_callback(CryProblem::CrystallisationProblem)
    # Extinction handling is only meaningful for explicitly signed scalar
    # kinetics.  Keeping the callback off the ordinary positive-growth path
    # avoids changing its event/counter behavior.
    dissolution_enabled = !(CryProblem.kinetics_dissolutionfunction isa nodissolution) ||
                          (CryProblem.kinetics_growthfunction isa
                           AbstractFPScalarDissolutionFunction)
    dissolution_enabled ||
        return nothing

    n_mom = CryProblem.solver.nmoments
    n_mom < 3 && return nothing

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    n_population = n_mom + 1
    n_states = n_population + n_solvent
    solvent_names = propertynames(CryProblem.initial_solvent_state)
    concentration_position = findfirst(==(Symbol(:concentration)), solvent_names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    concentration_index = n_population + concentration_position
    solid_mass_concentration_threshold = _validate_solid_mass_concentration_threshold(CryProblem)

    condition = (state, time, integrator) ->
        CryProblem.crystal_density * CryProblem.volume_shape_factor * state[4] - solid_mass_concentration_threshold
    affect! = integrator -> nothing
    affect_neg! = integrator -> begin
        state = integrator.u
        # Transfer the actual residual solid mass represented by µ3.  Do not
        # clamp it: an unexpected negative crossing must remain observable as
        # a failed physical state rather than being silently hidden.
        residual_mass = CryProblem.crystal_density * CryProblem.volume_shape_factor * state[4]
        integrator.u = SVector(ntuple(Val(n_states)) do index
            if index <= n_population
                zero(state[index])
            elseif index == concentration_index
                state[index] + residual_mass
            else
                state[index]
            end
        end)
        nothing
    end
    return ContinuousCallback(condition, affect!, affect_neg!;
                              save_positions = (false, true))
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
            hasproperty(rates, name) ? getproperty(rates, name) : _rate_zero(growth)
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
    if (CryProblem.kinetics_growthfunction isa AbstractFPLengthGrowthFunction ||
        CryProblem.kinetics_growthfunction isa AbstractFPLengthDissolutionFunction ||
        CryProblem.kinetics_dissolutionfunction isa AbstractFPLengthDissolutionFunction)
        throw(ArgumentError("Length-dependent kinetics are not supported by MoM; " *
                            "use FiniteVol or WENO."))
    end
    if ((!(CryProblem.kinetics_dissolutionfunction isa nodissolution) &&
         CryProblem.kinetics_dissolutionfunction isa AbstractFPScalarDissolutionFunction) ||
        CryProblem.kinetics_growthfunction isa AbstractFPScalarDissolutionFunction) && n_mom < 3
        throw(ArgumentError("MoM dissolution requires nmoments >= 3 to track crystal volume."))
    end

    function MoM_model(u, p, t)
        scalar_growth_rate = net_growth_rate(CryProblem.kinetics_growthfunction,
                                             p.gr,
                                             CryProblem.kinetics_dissolutionfunction,
                                             p.diss,
                                             CryProblem,
                                             u,
                                             t)
        B = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                           CryProblem, u, t)
        solvent_rates = _solvent_derivatives(CryProblem, u, t, scalar_growth_rate)
        return _mom_rhs(CryProblem, scalar_growth_rate, B, solvent_rates, u)
    end

    θ = ComponentArray(;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       diss = CryProblem.parameterset_dissolution)

    ET = eltype(CryProblem.parameterset_nucleation)
    n_mom = CryProblem.solver.nmoments
    u0_vec = ET.(_get_initial_state(CryProblem))
    n_states = n_mom + 1 + length(propertynames(CryProblem.initial_solvent_state))
    u0_typed = SVector(ntuple(k -> u0_vec[k], Val(n_states)))
    ODEprob = ODEProblem(MoM_model,
                         u0_typed,
                         (saveat[1], saveat[end]),
                         θ;
                         callback = _mom_extinction_callback(CryProblem))

    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5)
    return (ODEprob, tstep_solver)
end

function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                                       noaggregation, MoM,
                                                                       NuP, GrP, BrP, AggP,
                                                                       TP},
                                    saveat) where {NuclF <: AbstractNucleationFunction,
                                                   GrF <: AbstractFPLengthGrowthFunction,
                                                   NuP <: AbstractVector{<:Real},
                                                   GrP <: AbstractVector{<:Real},
                                                   BrP <: AbstractVector{<:Real},
                                                   AggP <: AbstractVector{<:Real},
                                                   TP <: AbstractTemperature}
    throw(ArgumentError("$(GrF) is length-dependent and is not supported by MoM; " *
                        "use FiniteVol or WENO."))
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
    d32 = n_mom >= 3 ?
          [_safe_moment_size_ratio(sol[4, index], sol[3, index], sol[1, index])
           for index in eachindex(sol.t)] :
          fill(NaN, length(sol.t))
    d43 = n_mom >= 4 ?
          [_safe_moment_size_ratio(sol[5, index], sol[4, index], sol[1, index])
           for index in eachindex(sol.t)] :
          fill(NaN, length(sol.t))
    moment2 = n_mom >= 2 ? sol[3, :] : fill(NaN, length(sol.t))
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

    return CrystallisationMoMSolution(sol.t,
                                      solvent_solution_state.concentration,
                                      [_safe_moment_size_ratio(sol[2, index],
                                                               sol[1, index],
                                                               sol[1, index])
                                       for index in eachindex(sol.t)],
                                      d32,
                                      d43,
                                      moment2,
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
    abstol_tol, auto_tol_cb = _auto_abstol_opts(CryProblem.solver, ODEprob.u0,
                                                CryProblem.solver.abstol)
    sol = solve(ODEprob,
                tstep_solver;
                callback = auto_tol_cb,
                saveat = saveat,
                reltol = CryProblem.solver.reltol,
                abstol = abstol_tol,)
    return _wrap_solution(CryProblem, sol)
end
