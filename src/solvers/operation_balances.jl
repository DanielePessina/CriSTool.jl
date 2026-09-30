"""Population, reactor and solvent ranges of the public/internal numerical state."""
_reactor_state_range(problem::CrystallisationProblem) =
    (_population_state_count(problem.solver) + 1):(_population_state_count(problem.solver) + _operation_state_count(problem.operation))

"""
    reactor_volume(problem, state)

Current reactor volume (m³). Fed-batch volume is a reactor state, not a solvent
variable. A default batch operation uses a unit-volume reference.
"""
reactor_volume(problem::CrystallisationProblem, state) =
    _reactor_volume(problem.operation, problem, state)
_reactor_volume(operation::BatchOperation, problem, state) = operation.volume
_reactor_volume(operation::MSMPROperation, problem, state) = operation.volume
_reactor_volume(operation::FedBatchOperation, problem, state) =
    state[first(_reactor_state_range(problem))]

"""Named inlet/outlet volumetric flows (m³/s) for an operation at time (s)."""
operation_flows(problem::CrystallisationProblem, simulation_time) =
    operation_flows(problem.operation, simulation_time)
operation_flows(operation::BatchOperation, simulation_time) =
    (; inflow = zero(operation.volume), outflow = zero(operation.volume))
function operation_flows(operation::MSMPROperation, simulation_time)
    evaluated_flow = _validate_operation_inflow(_operation_profile_value(operation.inflow, simulation_time))
    return (; inflow = evaluated_flow, outflow = evaluated_flow)
end
function operation_flows(operation::FedBatchOperation, simulation_time)
    evaluated_flow = _validate_operation_inflow(_operation_profile_value(operation.inflow, simulation_time))
    return (; inflow = evaluated_flow, outflow = zero(evaluated_flow))
end

function _operation_feed_values(feed::CrystallisationFeed, simulation_time)
    feed_names = propertynames(feed.solvent_state)
    evaluated_values = map(profile -> _operation_profile_value(profile, simulation_time),
                           values(feed.solvent_state))
    named_values = NamedTuple{feed_names}(evaluated_values)
    _validate_feed_concentration(named_values.concentration)
    all(value -> value isa Real && isfinite(value), evaluated_values) ||
        throw(ArgumentError("Feed solvent values must be finite real numbers."))
    return named_values
end

_validate_operation(problem::CrystallisationProblem, ::BatchOperation) = nothing
function _validate_operation(problem::CrystallisationProblem,
                             operation::Union{MSMPROperation, FedBatchOperation})
    operation_flows(operation, 0.0)
    feed_values = _operation_feed_values(operation.feed, 0.0)
    solvent_names = propertynames(problem.initial_solvent_state)
    :volume in solvent_names && throw(ArgumentError("Volume is a reactor state; remove :volume from initial_solvent_state."))
    all(feed_name -> feed_name in solvent_names, propertynames(feed_values)) ||
        throw(ArgumentError("Feed solvent names must belong to initial_solvent_state."))
    if length(solvent_names) > 1 && isnothing(operation.feed.transport)
        throw(ArgumentError("Extra solvent variables require an explicit feed transport hook; " *
                            "derived variables such as pH cannot be mixed as concentrations."))
    end
    if !isnothing(problem.initial_state) && operation isa FedBatchOperation
        _validate_operation_volume(reactor_volume(problem, problem.initial_state))
    end
    return nothing
end

function _operation_feed_population(problem::CrystallisationProblem)
    operation = problem.operation
    _validate_operation(problem, operation)
    operation isa BatchOperation && return nothing
    population_count = _population_state_count(problem.solver)
    feed_crystals = operation.feed.crystals
    if isnothing(feed_crystals) || feed_crystals.d43 == 0
        return zeros(population_count)
    end
    _validate_initial_crystal_domain(problem, feed_crystals.d43)
    if problem.solver isa DQMOM
        # The direct-quadrature adapter requires physical feed moments, not
        # the weights/nodes representation used for public initial states.
        return _initial_moment_population(problem, feed_crystals, 2 * problem.solver.nquadrature)
    elseif problem.solver isa AbstractMomentSolver
        return _initial_moment_population(problem, feed_crystals, population_count)
    else
        return _initial_mesh_population(problem, feed_crystals)
    end
end

function _operation_transport_context(problem, numerical_state, simulation_time)
    operation = problem.operation
    volume_value = _validate_operation_volume(reactor_volume(problem, numerical_state))
    flow_values = operation_flows(operation, simulation_time)
    feed_values = _operation_feed_values(operation.feed, simulation_time)
    extra_rates = if isnothing(operation.feed.transport)
        (;)
    else
        operation.feed.transport(problem, numerical_state, simulation_time,
                                 feed_values, flow_values.inflow, volume_value)
    end
    extra_rates isa NamedTuple ||
        throw(ArgumentError("Feed transport hook must return named extra solvent rates."))
    hasproperty(extra_rates, :concentration) &&
        throw(ArgumentError("Feed transport hook must not replace built-in concentration transport."))
    solvent_names = propertynames(problem.initial_solvent_state)
    all(rate_name -> rate_name in solvent_names, propertynames(extra_rates)) ||
        throw(ArgumentError("Unknown solvent variable returned by feed transport hook."))
    for solvent_name in solvent_names
        solvent_name === :concentration || hasproperty(extra_rates, solvent_name) ||
            throw(ArgumentError("Missing transport rate for solvent variable :$solvent_name."))
    end
    return (; exchange_rate = flow_values.inflow / volume_value,
              volume_rate = flow_values.inflow - flow_values.outflow,
              feed_values, extra_rates)
end

_compose_operation_rhs(problem, numerical_state, simulation_time, internal_rates,
                       feed_population, ::BatchOperation) = internal_rates
function _compose_operation_rhs(problem, numerical_state, simulation_time, internal_rates,
                                 feed_population, operation::AbstractCrystallisationOperation)
    operation_context = _operation_transport_context(problem, numerical_state, simulation_time)
    population_count = _population_state_count(problem.solver)
    reactor_count = _operation_state_count(operation)
    solvent_names = propertynames(problem.initial_solvent_state)
    return SVector(ntuple(Val(length(numerical_state))) do state_index
        if state_index <= population_count
            internal_rates[state_index] + operation_context.exchange_rate *
                (feed_population[state_index] - numerical_state[state_index])
        elseif state_index <= population_count + reactor_count
            operation_context.volume_rate
        else
            solvent_name = solvent_names[state_index - population_count - reactor_count]
            transport_rate = solvent_name === :concentration ?
                operation_context.exchange_rate *
                    (operation_context.feed_values.concentration - numerical_state[state_index]) :
                getproperty(operation_context.extra_rates, solvent_name)
            internal_rates[state_index] + transport_rate
        end
    end)
end
_compose_operation_rhs(problem, numerical_state, simulation_time, internal_rates, feed_population) =
    _compose_operation_rhs(problem, numerical_state, simulation_time, internal_rates,
                           feed_population, problem.operation)

"""Hydraulic trajectory variables, separate from solvent-phase variables."""
reactor_vars(solution::AbstractSolution) = hasproperty(solution, :reactor_state) ?
    solution.reactor_state : (;)
_reactor_solution_state(problem, ode_solution, ::BatchOperation) = (;)
_reactor_solution_state(problem, ode_solution, operation::MSMPROperation) =
    (; volume = fill(operation.volume, length(ode_solution.t)))
_reactor_solution_state(problem, ode_solution, operation::FedBatchOperation) =
    (; volume = collect(ode_solution[first(_reactor_state_range(problem)), :]))
_reactor_solution_state(problem, ode_solution) =
    _reactor_solution_state(problem, ode_solution, problem.operation)
