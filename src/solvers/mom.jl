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
    n_states = n_population + _operation_state_count(CryProblem.operation) + n_solvent
    solvent_names = propertynames(CryProblem.initial_solvent_state)
    concentration_position = findfirst(==(Symbol(:concentration)), solvent_names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    concentration_index = _solvent_state_index(CryProblem, :concentration)
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

_solvent_solution_index(problem::CrystallisationProblem, solution, name::Symbol) =
    _solvent_state_index(problem, name)

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
    if problem.kinetics_growthfunction isa AbstractFPLengthGrowthFunction ||
       problem.kinetics_growthfunction isa AbstractFPLengthDissolutionFunction ||
       problem.kinetics_dissolutionfunction isa AbstractFPLengthDissolutionFunction
        throw(ArgumentError("MoM requires scalar kinetics; choose DQMOM for length growth or a mesh solver."))
    end
    problem.kinetics_aggregationfunction isa noaggregation &&
        problem.kinetics_breakagefunction isa nobreakage ||
        throw(ArgumentError("MoM does not support aggregation or breakage; choose a quadrature or discretised solver."))
    signed_dissolution = !(problem.kinetics_dissolutionfunction isa nodissolution) ||
                         problem.kinetics_growthfunction isa AbstractFPScalarDissolutionFunction
    signed_dissolution && solver.nmoments < 3 &&
        throw(ArgumentError("MoM dissolution requires nmoments >= 3 to track crystal volume."))
    return nothing
end

function crystallisation_odeproblem(solver::MoM, CryProblem::CrystallisationProblem, saveat)
    _validate_solver_kinetics(solver, CryProblem)
    feed_population = _operation_feed_population(CryProblem)
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
        internal_rates = _mom_rhs(CryProblem, scalar_growth_rate, B, solvent_rates, u)
        return _compose_operation_rhs(CryProblem, u, t, internal_rates, feed_population)
    end

    θ = ComponentArray(;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       diss = CryProblem.parameterset_dissolution)

    ET = eltype(CryProblem.parameterset_nucleation)
    u0_vec = ET.(_get_initial_state(CryProblem))
    n_states = moment_count(solver) + _operation_state_count(CryProblem.operation) +
               length(propertynames(CryProblem.initial_solvent_state))
    u0_typed = SVector(ntuple(k -> u0_vec[k], Val(n_states)))
    ODEprob = ODEProblem(MoM_model,
                         u0_typed,
                         (saveat[1], saveat[end]),
                         θ;
                         callback = _mom_extinction_callback(CryProblem))
    return (ODEprob, _timestepping_algorithm(solver, Tsit5()))
end

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
