"""Method-of-moments RHS for scalar nucleation and growth."""
@inline function _mom_rhs(CryProblem, scalar_growth_rate, B, solvent_rates,
                          u::SVector{N}) where {N}
    n_population = _population_state_count(CryProblem.solver)
    reactor_count = _operation_state_count(CryProblem.operation)
    return SVector(ntuple(Val(N)) do k
        k <= n_population ?
            (k == 1 ? B : (k - 1) * scalar_growth_rate * u[k - 1]) :
            (k <= n_population + reactor_count ? zero(u[k]) :
             solvent_rates[k - n_population - reactor_count])
    end)
end

function _solvent_derivatives(problem::CrystallisationProblem, state, time, growth)
    return _solvent_derivatives(problem, state, time, growth,
                                problem.solvent_dynamics)
end

_solvent_derivatives(problem::CrystallisationProblem, state, time, growth, solvent_dynamics) =
    _ordered_solvent_rates(solvent_dynamics(problem, state, time, growth), problem, growth)

"""Solvent rates in `initial_solvent_state` order; a name a NamedTuple omits has zero rate."""
function _ordered_solvent_rates(rates::NamedTuple, problem::CrystallisationProblem, growth)
    names = propertynames(problem.initial_solvent_state)
    return ntuple(index -> begin
        name = names[index]
        hasproperty(rates, name) ? getproperty(rates, name) : _rate_zero(growth)
    end, Val(length(names)))
end

function _ordered_solvent_rates(rates, problem::CrystallisationProblem, growth)
    length(rates) == length(propertynames(problem.initial_solvent_state)) ||
        throw(ArgumentError("solvent_dynamics must return one rate per initial solvent variable."))
    return rates
end

function _write_solvent_derivatives!(dst, problem::CrystallisationProblem, rates)
    n_solvent = length(propertynames(problem.initial_solvent_state))
    first_solvent = first(_solvent_state_range(problem))
    @inbounds for index in 1:n_solvent
        dst[first_solvent + index - 1] = rates[index]
    end
    return nothing
end

function _solvent_solution_state(problem::CrystallisationProblem, solution)
    names = propertynames(problem.initial_solvent_state)
    n_solvent = length(names)
    first_solvent = first(_solvent_state_range(problem))
    values = ntuple(index -> solution[first_solvent + index - 1, :], Val(n_solvent))
    return NamedTuple{names}(values)
end

"""
    _validate_solver_kinetics(solver, problem)

Solver-specific kinetic contract, checked once by `_validate_crystallisation_problem`
and again when the solver's ODE problem is built. Only `MoM` restricts the
kinetic families here; the quadrature solvers validate their closures when
the ODE problem is built.
"""
_validate_solver_kinetics(solver::AbstractSolver, problem::CrystallisationProblem) = nothing
function _validate_solver_kinetics(solver::MoM, problem::CrystallisationProblem)
    solver.nmoments >= 2 ||
        throw(ArgumentError("MoM requires nmoments >= 2 (the concentration closure uses µ2)."))
    _uses_length_rates(problem) &&
        throw(ArgumentError("MoM requires scalar kinetics; choose DQMOM for length growth or a mesh solver."))
    _has_binary_sources(problem) &&
        throw(ArgumentError("MoM does not support aggregation or breakage; choose a quadrature or discretised solver."))
    _dissolution_enabled(problem) && solver.nmoments < 3 &&
        throw(ArgumentError("MoM dissolution requires nmoments >= 3 to track crystal volume."))
    return nothing
end

function crystallisation_odeproblem(solver::MoM, CryProblem::CrystallisationProblem, saveat, kinetic_values)
    _validate_solver_kinetics(solver, CryProblem)
    feed_population = _operation_feed_population(CryProblem)
    function MoM_model(u, p, t)
        scalar_growth_rate = net_growth_rate(CryProblem.model.growth.law,
                                             p.gr,
                                             CryProblem.model.dissolution.law,
                                             p.diss,
                                             CryProblem,
                                             u,
                                             t)
        B = nucleationrate(CryProblem.model.nucleation.law, p.nucl,
                           CryProblem, u, t)
        solvent_rates = _solvent_derivatives(CryProblem, u, t, scalar_growth_rate)
        internal_rates = _mom_rhs(CryProblem, scalar_growth_rate, B, solvent_rates, u)
        return _compose_operation_rhs(CryProblem, u, t, internal_rates, feed_population)
    end

    ET = eltype(kinetic_values)
    u0_vec = ET.(_get_initial_state(CryProblem))
    n_states = last(_solvent_state_range(CryProblem))
    u0_typed = SVector(ntuple(k -> u0_vec[k], Val(n_states)))
    ODEprob = ODEProblem{false}(MoM_model,
                                u0_typed,
                                (saveat[1], saveat[end]),
                                kinetic_values)
    return (ODEprob, _timestepping_algorithm(solver, Tsit5()))
end

_solver_callbacks(solver::MoM, problem::CrystallisationProblem, odeproblem) =
    _solid_mass_extinction_callback(problem)

function _wrap_solution(solver::MoM, problem::CrystallisationProblem, ode_solution,
                        boundary_flow_state::NamedTuple)
    # State k holds µ_{k-1}; untracked ratios (nmoments < 4) are NaN.
    moment_matrix = Array(ode_solution[1:moment_count(solver), :])
    return _crystallisation_solution(problem, ode_solution, MomentPopulation(moment_matrix),
                                     _raw_moment_size_metrics(moment_matrix),
                                     collect(ode_solution[:, end]),
                                     merge(_reactor_solution_state(problem, ode_solution),
                                           boundary_flow_state))
end
