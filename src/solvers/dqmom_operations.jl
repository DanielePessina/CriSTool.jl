"""
    _dqmom_operation_moment_source(scaled_source, numerical_state, problem,
                                   operation_context, feed_population)

Add physical feed exchange, converted into DQMOM's scaled moment coordinates,
to the kinetic moment source. A closed batch (`nothing` context) is unchanged.
"""
_dqmom_operation_moment_source(scaled_source::SVector, numerical_state,
                               dqmom_problem::CrystallisationProblem,
                               operation_context::Nothing, feed_population) = scaled_source

function _dqmom_operation_moment_source(scaled_source::SVector{K},
                                        numerical_state,
                                        dqmom_problem::CrystallisationProblem,
                                        operation_context,
                                        feed_population) where {K}
    isnothing(feed_population) &&
        throw(ArgumentError("DQMOM operation transport requires physical feed moments."))
    dqmom_solver = dqmom_problem.solver
    return SVector(ntuple(Val(K)) do moment_position
        moment_index = moment_position - 1
        tank_moment = _dqmom_physical_moment(dqmom_solver, numerical_state, moment_index)
        physical_exchange_rate = operation_context.exchange_rate *
                                 (feed_population[moment_position] - tank_moment)
        scaled_exchange_rate = physical_exchange_rate /
                               (dqmom_solver.weight_scale *
                                dqmom_solver.coordinate_scale^moment_index)
        scaled_source[moment_position] + scaled_exchange_rate
    end)
end

"""Reactor volume and named solvent transport of a DQMOM RHS; batch is unchanged."""
_dqmom_operation_state_rates(native_rates::SVector, numerical_state,
                             dqmom_problem::CrystallisationProblem,
                             operation_context::Nothing) = native_rates

function _dqmom_operation_state_rates(native_rates::SVector{S}, numerical_state,
                                      dqmom_problem::CrystallisationProblem,
                                      operation_context) where {S}
    # Population transport enters through the projected moment source.
    population_count = _population_state_count(dqmom_problem.solver)
    rate_type = promote_type(eltype(native_rates), eltype(numerical_state))
    return SVector(ntuple(Val(S)) do state_index
        state_index <= population_count ? convert(rate_type, native_rates[state_index]) :
        convert(rate_type,
                _operation_transported_rate(native_rates[state_index], numerical_state,
                                            state_index, dqmom_problem, operation_context,
                                            nothing))
    end)
end
