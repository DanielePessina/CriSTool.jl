"""Physical raw moments represented by a normalized DQMOM state."""
function _dqmom_physical_moments(numerical_state, dqmom_solver::DQMOM)
    nquadrature = dqmom_solver.nquadrature
    maximum_order = 2 * nquadrature - 1
    element_type = promote_type(eltype(numerical_state),
                                typeof(dqmom_solver.coordinate_scale),
                                typeof(dqmom_solver.weight_scale))
    physical_moments = zeros(element_type, maximum_order + 1)
    @inbounds for moment_index in 0:maximum_order
        scaled_moment = zero(element_type)
        for node_index in 1:nquadrature
            scaled_weight = numerical_state[node_index]
            scaled_node = numerical_state[nquadrature + node_index] / scaled_weight
            scaled_moment += scaled_weight * scaled_node^moment_index
        end
        physical_moments[moment_index + 1] = dqmom_solver.weight_scale *
                                             dqmom_solver.coordinate_scale^moment_index *
                                             scaled_moment
    end
    return physical_moments
end

"""Return operation transport inputs for DQMOM; batch keeps its direct fast path."""
_dqmom_operation_transport_context(dqmom_problem, numerical_state, simulation_time) =
    dqmom_problem.operation isa BatchOperation ? nothing :
    _operation_transport_context(dqmom_problem, numerical_state, simulation_time)

"""Convert physical feed exchange into DQMOM's scaled moment coordinates."""
function _dqmom_add_operation_moment_source!(scaled_source,
                                             numerical_state,
                                             dqmom_problem::CrystallisationProblem,
                                             operation_context,
                                             feed_population)
    isnothing(feed_population) &&
        throw(ArgumentError("DQMOM operation transport requires physical feed moments."))
    dqmom_solver = dqmom_problem.solver
    tank_moments = _dqmom_physical_moments(numerical_state, dqmom_solver)
    @inbounds for moment_index in 0:(2 * dqmom_solver.nquadrature - 1)
        physical_exchange_rate = operation_context.exchange_rate *
                                (feed_population[moment_index + 1] -
                                  tank_moments[moment_index + 1])
        scaled_exchange_rate = physical_exchange_rate /
                               (dqmom_solver.weight_scale *
                                dqmom_solver.coordinate_scale^moment_index)
        scaled_source[moment_index + 1] += scaled_exchange_rate
    end
    return nothing
end

"""Write reactor volume and named solvent transport into a DQMOM RHS."""
function _dqmom_add_operation_state_rates!(native_rates,
                                           numerical_state,
                                           dqmom_problem::CrystallisationProblem,
                                           operation_context)
    isnothing(operation_context) && return nothing

    reactor_range = _reactor_state_range(dqmom_problem)
    @inbounds for reactor_index in 1:_operation_state_count(dqmom_problem.operation)
        native_rates[first(reactor_range) + reactor_index - 1] =
            operation_context.volume_rate
    end

    concentration_index = _solvent_state_index(dqmom_problem, :concentration)
    native_rates[concentration_index] += operation_context.exchange_rate *
        (operation_context.feed_values.concentration - numerical_state[concentration_index])

    for solvent_name in propertynames(dqmom_problem.initial_solvent_state)
        solvent_name === :concentration && continue
        solvent_index = _solvent_state_index(dqmom_problem, solvent_name)
        native_rates[solvent_index] += getproperty(operation_context.extra_rates,
                                                   solvent_name)
    end
    return nothing
end
