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
    # Population transport enters through the projected moment source.
    first_non_population = _population_state_count(dqmom_problem.solver) + 1
    @inbounds for state_index in first_non_population:length(native_rates)
        native_rates[state_index] = _operation_transported_rate(
            native_rates[state_index], numerical_state, state_index, dqmom_problem,
            operation_context, nothing)
    end
    return nothing
end
