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
    evaluated_flow = _operation_inflow_value(operation.inflow, simulation_time)
    return (; inflow = evaluated_flow, outflow = evaluated_flow)
end
function operation_flows(operation::FedBatchOperation, simulation_time)
    evaluated_flow = _operation_inflow_value(operation.inflow, simulation_time)
    return (; inflow = evaluated_flow, outflow = zero(evaluated_flow))
end

# Constant profiles are validated once by the operation constructors and
# `_validate_operation`; only time-varying profiles are re-checked per call.
_operation_inflow_value(inflow::Real, simulation_time) = inflow
_operation_inflow_value(inflow, simulation_time) =
    _validate_operation_inflow(inflow(simulation_time))
_feed_profile_value(profile::Real, simulation_time) = profile
function _feed_profile_value(profile, simulation_time)
    profile_value = profile(simulation_time)
    profile_value isa Real && isfinite(profile_value) ||
        throw(ArgumentError("Feed solvent values must be finite real numbers."))
    return profile_value
end

_feed_concentration(feed::CrystallisationFeed, simulation_time) =
    feed.concentration isa Real ? feed.concentration :
    _validate_feed_concentration(_feed_profile_value(feed.concentration, simulation_time))

"""Feed `concentration` and extra solvent species evaluated at `simulation_time`."""
_operation_feed_values(feed::CrystallisationFeed, simulation_time) =
    merge((; concentration = _feed_concentration(feed, simulation_time)),
          map(profile -> _feed_profile_value(profile, simulation_time), feed.solvent_state))

_validate_operation(problem::CrystallisationProblem, ::BatchOperation) = nothing
function _validate_operation(problem::CrystallisationProblem,
                             operation::Union{MSMPROperation, FedBatchOperation})
    _validate_operation_inflow(_operation_profile_value(operation.inflow, 0.0))
    feed_values = _operation_feed_values(operation.feed, 0.0)
    _validate_feed_concentration(feed_values.concentration)
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

"""
    _validate_feed_transport_hook(problem)

Check once, at ODE construction, that the feed transport hook returns exactly
the named non-concentration solvent rates. The RHS then trusts its schema.
"""
_validate_feed_transport_hook(problem, ::BatchOperation) = nothing
function _validate_feed_transport_hook(problem, operation::Union{MSMPROperation, FedBatchOperation})
    isnothing(operation.feed.transport) && return nothing
    initial_state = _get_initial_state(problem)
    extra_rates = operation.feed.transport(problem, initial_state, 0.0,
                                           _operation_feed_values(operation.feed, 0.0),
                                           operation_flows(operation, 0.0).inflow,
                                           reactor_volume(problem, initial_state))
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
    return nothing
end

"""
    _operation_feed_population(problem)

Validate the operation once for ODE construction and return the inlet
population in the solver's population coordinates (`nothing` for batch).
"""
function _operation_feed_population(problem::CrystallisationProblem)
    operation = problem.operation
    _validate_operation(problem, operation)
    _validate_feed_transport_hook(problem, operation)
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

"""
    _operation_transport_context(problem, numerical_state, simulation_time)

Exchange rate (1/s), volume rate (m³/s), feed concentration and hook rates at
one RHS evaluation, or `nothing` for a closed batch. Schema checks happen once
in `_operation_feed_population`; only the state volume and time profiles are
checked here.
"""
_operation_transport_context(problem, numerical_state, simulation_time) =
    _operation_transport_context(problem.operation, problem, numerical_state, simulation_time)
_operation_transport_context(::BatchOperation, problem, numerical_state, simulation_time) = nothing
function _operation_transport_context(operation::Union{MSMPROperation, FedBatchOperation},
                                      problem, numerical_state, simulation_time)
    volume_value = _validate_operation_volume(reactor_volume(problem, numerical_state))
    flow_values = operation_flows(operation, simulation_time)
    extra_rates = isnothing(operation.feed.transport) ? (;) :
        operation.feed.transport(problem, numerical_state, simulation_time,
                                 _operation_feed_values(operation.feed, simulation_time),
                                 flow_values.inflow, volume_value)
    return (; exchange_rate = flow_values.inflow / volume_value,
              volume_rate = flow_values.inflow - flow_values.outflow,
              feed_concentration = _feed_concentration(operation.feed, simulation_time),
              extra_rates)
end

"""
    _operation_transported_rate(internal_rate, numerical_state, state_index,
                                problem, operation_context, feed_population)

Derivative of one numerical state entry after adding open-reactor transport to
its internal (crystallisation) rate. Population entries gain
`exchange_rate * (feed - state)`, the reactor-volume entry is the volume rate
(it has no internal rate), concentration gains its built-in exchange term and
any other solvent variable gains its hook rate. This is the single transport
kernel used by every solver family.
"""
@inline function _operation_transported_rate(internal_rate, numerical_state, state_index,
                                             problem, operation_context, feed_population)
    population_count = _population_state_count(problem.solver)
    if state_index <= population_count
        return internal_rate + operation_context.exchange_rate *
               (feed_population[state_index] - numerical_state[state_index])
    end
    reactor_count = _operation_state_count(problem.operation)
    state_index <= population_count + reactor_count && return operation_context.volume_rate
    solvent_name = propertynames(problem.initial_solvent_state)[
        state_index - population_count - reactor_count]
    solvent_name === :concentration &&
        return internal_rate + operation_context.exchange_rate *
               (operation_context.feed_concentration - numerical_state[state_index])
    return internal_rate + getproperty(operation_context.extra_rates, solvent_name)
end

"""Out-of-place (static-state) RHS with operation transport; batch is unchanged."""
_compose_operation_rhs(problem, numerical_state, simulation_time, internal_rates,
                       feed_population) =
    _compose_operation_rhs(problem.operation, problem, numerical_state, simulation_time,
                           internal_rates, feed_population)
_compose_operation_rhs(::BatchOperation, problem, numerical_state, simulation_time,
                       internal_rates, feed_population) = internal_rates
function _compose_operation_rhs(operation, problem, numerical_state, simulation_time,
                                internal_rates, feed_population)
    operation_context = _operation_transport_context(operation, problem, numerical_state,
                                                     simulation_time)
    return SVector(ntuple(Val(length(numerical_state))) do state_index
        _operation_transported_rate(internal_rates[state_index], numerical_state,
                                    state_index, problem, operation_context, feed_population)
    end)
end

"""Add inlet, outlet and reactor-volume terms to an in-place mesh RHS.

The batch specialisation is an empty inline method, so its RHS keeps the
allocation-free path.
"""
@inline _add_operation_transport!(derivative, numerical_state, problem,
                                  simulation_time, feed_population) =
    _add_operation_transport!(problem.operation, derivative, numerical_state, problem,
                              simulation_time, feed_population)
@inline _add_operation_transport!(::BatchOperation, derivative, numerical_state, problem,
                                  simulation_time, feed_population) = nothing
function _add_operation_transport!(operation, derivative, numerical_state, problem,
                                   simulation_time, feed_population)
    operation_context = _operation_transport_context(operation, problem, numerical_state,
                                                     simulation_time)
    @inbounds for state_index in eachindex(derivative)
        derivative[state_index] = _operation_transported_rate(
            derivative[state_index], numerical_state, state_index, problem,
            operation_context, feed_population)
    end
    return nothing
end

"""Reactor and mesh-boundary trajectories, separate from solvent variables.

Finite-volume and WENO results include signed lower/upper size-boundary
solid-mass flow in kg/s. A positive flow is outward from the represented size
domain.
"""
reactor_vars(solution::AbstractSolution) = hasproperty(solution, :reactor_state) ?
    solution.reactor_state : (;)
_reactor_solution_state(problem, ode_solution, ::BatchOperation) = (;)
_reactor_solution_state(problem, ode_solution, operation::MSMPROperation) =
    (; volume = fill(operation.volume, length(ode_solution.t)))
_reactor_solution_state(problem, ode_solution, operation::FedBatchOperation) =
    (; volume = collect(ode_solution[first(_reactor_state_range(problem)), :]))
_reactor_solution_state(problem, ode_solution) =
    _reactor_solution_state(problem, ode_solution, problem.operation)
