"""
Direct Quadrature Method of Moments (DQMOM) for seeded, one-dimensional
crystallisation population balances.

The public initial-state population block is `[weights; physical_nodes]`.
The ODE uses normalized weights and weighted, dimensionless nodes so the
projection matrix does not contain powers of lengths measured in metres.
"""

@inline function _validate_dqmom_solver(solver::DQMOM)
    solver.nquadrature >= 2 ||
        throw(ArgumentError("DQMOM requires nquadrature >= 2."))
    isfinite(solver.coordinate_scale) && solver.coordinate_scale > 0.0 ||
        throw(ArgumentError("DQMOM coordinate_scale must be finite and strictly positive."))
    isfinite(solver.weight_scale) && solver.weight_scale > 0.0 ||
        throw(ArgumentError("DQMOM weight_scale must be finite and strictly positive."))
    isfinite(solver.realizability_tolerance) && solver.realizability_tolerance >= 0.0 ||
        throw(ArgumentError("DQMOM realizability_tolerance must be finite and nonnegative."))
    isfinite(solver.node_coalescence_tolerance) &&
        solver.node_coalescence_tolerance >= 0.0 ||
        throw(ArgumentError("DQMOM node_coalescence_tolerance must be finite and nonnegative."))
    isfinite(solver.minimum_size) && solver.minimum_size >= 0.0 ||
        throw(ArgumentError("DQMOM minimum_size must be finite and nonnegative."))
    return solver
end

@inline _dqmom_population_count(solver::DQMOM) = 2 * solver.nquadrature

function _validate_dqmom_public_state(problem::CrystallisationProblem)
    solver = problem.solver
    _validate_dqmom_solver(solver)
    problem.initial_state === nothing &&
        throw(ArgumentError("DQMOM requires a nonempty seeded initial population. " *
                            "Provide initial_crystals or a direct [weights; nodes] initial_state."))

    direct_state = _get_initial_state(problem)
    nquadrature = solver.nquadrature
    n_solvent = length(propertynames(problem.initial_solvent_state))
    expected_length = 2 * nquadrature + n_solvent
    length(direct_state) == expected_length ||
        throw(ArgumentError("DQMOM initial_state must contain $nquadrature weights, " *
                            "$nquadrature physical nodes, and $n_solvent solvent values."))

    weights = @view direct_state[1:nquadrature]
    nodes = @view direct_state[(nquadrature + 1):(2 * nquadrature)]
    all(weight -> isfinite(weight) && weight > 0.0, weights) ||
        throw(ArgumentError("DQMOM requires strictly positive initial weights."))
    all(node -> isfinite(node) && node > solver.minimum_size, nodes) ||
        throw(ArgumentError("DQMOM requires initial nodes strictly above minimum_size."))

    separation_tolerance = solver.node_coalescence_tolerance * solver.coordinate_scale
    @inbounds for first_index in 1:(nquadrature - 1)
        for second_index in (first_index + 1):nquadrature
            abs(nodes[first_index] - nodes[second_index]) > separation_tolerance ||
                throw(ArgumentError("DQMOM initial nodes must be pairwise distinct and " *
                                    "separated by more than $separation_tolerance m."))
        end
    end
    return problem
end

function _validate_dqmom_problem(problem::CrystallisationProblem)
    _validate_dqmom_public_state(problem)

    growthfunction = problem.kinetics_growthfunction
    supported_growth = growthfunction isa AbstractFPScalarGrowthFunction ||
                       growthfunction isa AbstractFPScalarDissolutionFunction ||
                       growthfunction isa AbstractFPLengthGrowthFunction
    supported_growth ||
        throw(ArgumentError("DQMOM requires a scalar or length-dependent first-principles " *
                            "growth function."))
    growthfunction isa AbstractFPLengthDissolutionFunction &&
        throw(ArgumentError("Length-dependent dissolution is not supported by DQMOM; " *
                            "use FiniteVol or WENO."))

    dissolutionfunction = problem.kinetics_dissolutionfunction
    dissolutionfunction isa AbstractFPLengthDissolutionFunction &&
        throw(ArgumentError("Length-dependent dissolution is not supported by DQMOM; " *
                            "use FiniteVol or WENO."))
    dissolutionfunction isa AbstractFPScalarDissolutionFunction ||
        throw(ArgumentError("DQMOM requires a scalar dissolution function or nodissolution()."))

    aggregationfunction = problem.kinetics_aggregationfunction
    supported_aggregation = aggregationfunction isa noaggregation ||
                            aggregationfunction isa aggr_scalar ||
                            aggregationfunction isa aggr_linear ||
                            aggregationfunction isa aggr_linearvol ||
                            aggregationfunction isa aggr_avg
    supported_aggregation ||
        throw(ArgumentError("Aggregation function $(typeof(aggregationfunction)) " *
                            "has no validated DQMOM moment closure."))

    breakagefunction = problem.kinetics_breakagefunction
    supported_breakage = breakagefunction isa nobreakage ||
                        breakagefunction isa breakage_empirical ||
                        breakagefunction isa breakage_uniform
    supported_breakage ||
        throw(ArgumentError("Breakage function $(typeof(breakagefunction)) " *
                            "has no validated DQMOM moment closure."))
    return problem
end

"""Allocation-free scalar DQMOM3 RHS for the default concentration path.

When there are no binary sources, growth is evaluated on the three physical
nodes and the 6x6 projection is solved statically; the solvent slot receives
the exact volume-rate coupling.  The default-solvent contract is enforced by
the caller's gate, so no custom solvent-dynamics branch or aggregation /
breakage sources appear here.
"""
@inline function _dqmom_scalar_model_3(problem::CrystallisationProblem,
                                       state, parameters, time)
    solver = problem.solver
    coordinate_scale = solver.coordinate_scale
    weight_scale = solver.weight_scale
    scaled_weights = SVector(state[1], state[2], state[3])
    scaled_nodes = SVector(state[4] / state[1],
                           state[5] / state[2],
                           state[6] / state[3])
    physical_nodes = coordinate_scale .* scaled_nodes
    growth_rates = SVector(
        net_growth_rate_at_length(problem.kinetics_growthfunction,
                                  parameters.gr,
                                  problem.kinetics_dissolutionfunction,
                                  parameters.diss,
                                  problem,
                                  state,
                                  time,
                                  physical_nodes[1]),
        net_growth_rate_at_length(problem.kinetics_growthfunction,
                                  parameters.gr,
                                  problem.kinetics_dissolutionfunction,
                                  parameters.diss,
                                  problem,
                                  state,
                                  time,
                                  physical_nodes[2]),
        net_growth_rate_at_length(problem.kinetics_growthfunction,
                                  parameters.gr,
                                  problem.kinetics_dissolutionfunction,
                                  parameters.diss,
                                  problem,
                                  state,
                                  time,
                                  physical_nodes[3]))
    nucleation_rate_value = nucleationrate(problem.kinetics_nucleationfunction,
                                           parameters.nucl,
                                           problem,
                                           state,
                                           time)
    source = _dqmom_source_n3(scaled_weights,
                              scaled_nodes,
                              growth_rates,
                              nucleation_rate_value,
                              coordinate_scale,
                              weight_scale)
    direct_rates = _dqmom_projection_matrix_n3(scaled_nodes) \ source

    scaled_volume_rate = -2 * scaled_nodes[1]^3 * direct_rates[1] +
                         3 * scaled_nodes[1]^2 * direct_rates[4] -
                         2 * scaled_nodes[2]^3 * direct_rates[2] +
                         3 * scaled_nodes[2]^2 * direct_rates[5] -
                         2 * scaled_nodes[3]^3 * direct_rates[3] +
                         3 * scaled_nodes[3]^2 * direct_rates[6]
    physical_volume_rate = weight_scale * coordinate_scale^3 * scaled_volume_rate
    concentration_rate = -problem.crystal_density *
                         problem.volume_shape_factor * physical_volume_rate
    return SVector(direct_rates[1], direct_rates[2], direct_rates[3],
                   direct_rates[4], direct_rates[5], direct_rates[6],
                   concentration_rate)
end

"""Static three-node DQMOM model exposing both ODE calling conventions.

The solver runs the out-of-place form on an `SVector` state (fully static
Tsit5 stages); the in-place form exists so callers that treat the RHS as a
mutating function (e.g. tests probing `ode_problem.f(du, u, p, t)`) keep
working.  Both methods compute exactly the same derivative vector.
"""
struct DQMOMFastModel{P <: CrystallisationProblem}
    problem::P
end

@inline (model::DQMOMFastModel)(state, parameters, time) =
    _dqmom_scalar_model_3(model.problem, state, parameters, time)

@inline function (model::DQMOMFastModel)(destination, state, parameters, time)
    rates = _dqmom_scalar_model_3(model.problem, state, parameters, time)
    @inbounds for index in eachindex(destination)
        destination[index] = rates[index]
    end
    return nothing
end

"""Build the scaled DQMOM projection matrix for dimensionless nodes."""
function _dqmom_projection_matrix(scaled_nodes::AbstractVector)
    nquadrature = length(scaled_nodes)
    element_type = eltype(scaled_nodes)
    matrix = Matrix{element_type}(undef, 2 * nquadrature, 2 * nquadrature)
    @inbounds for moment_index in 0:(2 * nquadrature - 1)
        row_index = moment_index + 1
        for node_index in 1:nquadrature
            scaled_node = scaled_nodes[node_index]
            matrix[row_index, node_index] = moment_index == 0 ?
                one(element_type) :
                (one(element_type) - moment_index) * scaled_node^moment_index
            matrix[row_index, nquadrature + node_index] = moment_index == 0 ?
                zero(element_type) :
                moment_index * scaled_node^(moment_index - 1)
        end
    end
    return matrix
end

"""Scaled node coordinates at a saved time index, static for the default 3-node path."""
@inline function _dqmom_scaled_nodes_at(node_matrix, solver::DQMOM, time_index)
    if solver.nquadrature == 3
        coordinate_scale = solver.coordinate_scale
        return SVector(node_matrix[1, time_index] / coordinate_scale,
                       node_matrix[2, time_index] / coordinate_scale,
                       node_matrix[3, time_index] / coordinate_scale)
    end
    return node_matrix[:, time_index] ./ solver.coordinate_scale
end

"""Projection-matrix condition estimate without a full SVD.

For the default 3-node path the 6x6 matrix is inverted statically (zero
heap allocations); the generic dense fallback uses the LU-based 1-norm
condition number.
"""
@inline function _dqmom_projection_condition_estimate(scaled_nodes::SVector{3})
    static_matrix = _dqmom_projection_matrix_n3(scaled_nodes)
    return opnorm(static_matrix, 1) * opnorm(inv(static_matrix), 1)
end

@inline function _dqmom_projection_condition_estimate(scaled_nodes::AbstractVector)
    matrix = _dqmom_projection_matrix(scaled_nodes)
    return cond(matrix, 1)
end

"""Static projection matrix for the default three-node DQMOM path."""
@inline function _dqmom_projection_matrix_n3(scaled_nodes)
    projection_values = ntuple(Val(36)) do linear_index
        row_index = ((linear_index - 1) % 6) + 1
        column_index = ((linear_index - 1) ÷ 6) + 1
        moment_index = row_index - 1
        node_index = column_index <= 3 ? column_index : column_index - 3
        scaled_node = scaled_nodes[node_index]
        if column_index <= 3
            moment_index == 0 ? one(scaled_node) :
                (one(scaled_node) - moment_index) * scaled_node^moment_index
        else
            moment_index == 0 ? zero(scaled_node) :
                moment_index * scaled_node^(moment_index - 1)
        end
    end
    return SMatrix{6, 6}(projection_values)
end

"""DQMOM source vector without binary aggregation or breakage sources."""
@inline function _dqmom_source_n3(scaled_weights,
                                  scaled_nodes,
                                  growth_rates,
                                  nucleation_rate_value,
                                  coordinate_scale,
                                  weight_scale)
    source_type = promote_type(eltype(scaled_weights),
                               eltype(scaled_nodes),
                               eltype(growth_rates),
                               typeof(nucleation_rate_value),
                               typeof(coordinate_scale),
                               typeof(weight_scale))
    growth_rate_one = growth_rates[1] / coordinate_scale
    growth_rate_two = growth_rates[2] / coordinate_scale
    growth_rate_three = growth_rates[3] / coordinate_scale
    source_zero = nucleation_rate_value / weight_scale
    source_one = scaled_weights[1] * growth_rate_one +
                 scaled_weights[2] * growth_rate_two +
                 scaled_weights[3] * growth_rate_three
    source_two = 2 * (scaled_weights[1] * scaled_nodes[1] * growth_rate_one +
                      scaled_weights[2] * scaled_nodes[2] * growth_rate_two +
                      scaled_weights[3] * scaled_nodes[3] * growth_rate_three)
    source_three = 3 * (scaled_weights[1] * scaled_nodes[1]^2 * growth_rate_one +
                        scaled_weights[2] * scaled_nodes[2]^2 * growth_rate_two +
                        scaled_weights[3] * scaled_nodes[3]^2 * growth_rate_three)
    source_four = 4 * (scaled_weights[1] * scaled_nodes[1]^3 * growth_rate_one +
                       scaled_weights[2] * scaled_nodes[2]^3 * growth_rate_two +
                       scaled_weights[3] * scaled_nodes[3]^3 * growth_rate_three)
    source_five = 5 * (scaled_weights[1] * scaled_nodes[1]^4 * growth_rate_one +
                       scaled_weights[2] * scaled_nodes[2]^4 * growth_rate_two +
                       scaled_weights[3] * scaled_nodes[3]^4 * growth_rate_three)
    return SVector{6, source_type}(source_zero, source_one, source_two,
                                   source_three, source_four, source_five)
end

"""Allocation-reduced N=3 RHS for the common no-binary-source workload."""
function _dqmom_rhs_no_binary_n3!(destination,
                                   state,
                                   parameters,
                                   time,
                                   problem::CrystallisationProblem)
    solver = problem.solver
    coordinate_scale = solver.coordinate_scale
    weight_scale = solver.weight_scale
    scaled_weights = SVector(state[1], state[2], state[3])
    scaled_nodes = SVector(state[4] / state[1],
                           state[5] / state[2],
                           state[6] / state[3])
    physical_nodes = coordinate_scale .* scaled_nodes
    growth_rates = SVector(
        net_growth_rate_at_length(problem.kinetics_growthfunction,
                                  parameters.gr,
                                  problem.kinetics_dissolutionfunction,
                                  parameters.diss,
                                  problem,
                                  state,
                                  time,
                                  physical_nodes[1]),
        net_growth_rate_at_length(problem.kinetics_growthfunction,
                                  parameters.gr,
                                  problem.kinetics_dissolutionfunction,
                                  parameters.diss,
                                  problem,
                                  state,
                                  time,
                                  physical_nodes[2]),
        net_growth_rate_at_length(problem.kinetics_growthfunction,
                                  parameters.gr,
                                  problem.kinetics_dissolutionfunction,
                                  parameters.diss,
                                  problem,
                                  state,
                                  time,
                                  physical_nodes[3]))
    nucleation_rate_value = nucleationrate(problem.kinetics_nucleationfunction,
                                           parameters.nucl,
                                           problem,
                                           state,
                                           time)
    source = _dqmom_source_n3(scaled_weights,
                              scaled_nodes,
                              growth_rates,
                              nucleation_rate_value,
                              coordinate_scale,
                              weight_scale)
    direct_rates = _dqmom_projection_matrix_n3(scaled_nodes) \ source

    @inbounds for node_index in 1:3
        destination[node_index] = direct_rates[node_index]
        destination[3 + node_index] = direct_rates[3 + node_index]
    end

    scaled_volume_rate = -2 * scaled_nodes[1]^3 * direct_rates[1] +
                         3 * scaled_nodes[1]^2 * direct_rates[4] -
                         2 * scaled_nodes[2]^3 * direct_rates[2] +
                         3 * scaled_nodes[2]^2 * direct_rates[5] -
                         2 * scaled_nodes[3]^3 * direct_rates[3] +
                         3 * scaled_nodes[3]^2 * direct_rates[6]
    physical_volume_rate = weight_scale * coordinate_scale^3 * scaled_volume_rate
    first_solvent = 7
    n_solvent = length(propertynames(problem.initial_solvent_state))
    if n_solvent == 1 && problem.solvent_dynamics isa DefaultSolventDynamics
        destination[first_solvent] = -problem.crystal_density *
                                     problem.volume_shape_factor *
                                     physical_volume_rate
    else
        growth_context = problem.kinetics_growthfunction isa AbstractFPLengthGrowthFunction ?
                         growth_rates :
                         net_growth_rate(problem.kinetics_growthfunction,
                                         parameters.gr,
                                         problem.kinetics_dissolutionfunction,
                                         parameters.diss,
                                         problem,
                                         state,
                                         time)
        solvent_rates = _dqmom_solvent_derivatives(problem,
                                                   state,
                                                   time,
                                                   growth_context,
                                                   physical_volume_rate)
        _write_solvent_derivatives!(destination, problem, solvent_rates)
    end
    return nothing
end

function _dqmom_direct_state(problem::CrystallisationProblem)
    solver = problem.solver
    public_state = _get_initial_state(problem)
    nquadrature = solver.nquadrature
    n_solvent = length(propertynames(problem.initial_solvent_state))
    element_type = promote_type(eltype(public_state),
                                typeof(solver.coordinate_scale),
                                typeof(solver.weight_scale))
    internal_state = Vector{element_type}(undef, 2 * nquadrature + n_solvent)
    @inbounds for node_index in 1:nquadrature
        weight = public_state[node_index]
        physical_node = public_state[nquadrature + node_index]
        scaled_weight = weight / solver.weight_scale
        scaled_node = physical_node / solver.coordinate_scale
        internal_state[node_index] = scaled_weight
        internal_state[nquadrature + node_index] = scaled_weight * scaled_node
    end
    first_solvent = 2 * nquadrature + 1
    @inbounds for solvent_index in 1:n_solvent
        internal_state[first_solvent + solvent_index - 1] =
            public_state[first_solvent + solvent_index - 1]
    end
    return internal_state
end

@inline function _dqmom_length_and_weight(state,
                                          solver::DQMOM,
                                          node_index::Integer)
    nquadrature = solver.nquadrature
    scaled_weight = state[node_index]
    scaled_node = state[nquadrature + node_index] / scaled_weight
    physical_weight = solver.weight_scale * scaled_weight
    physical_node = solver.coordinate_scale * scaled_node
    return physical_node, physical_weight, scaled_node
end

function _dqmom_source_vector(problem::CrystallisationProblem,
                              state,
                              parameters,
                              time,
                              physical_nodes::AbstractVector,
                              scaled_weights::AbstractVector,
                              scaled_nodes::AbstractVector,
                              growth_rates::AbstractVector,
                              nucleation_rate_value)
    solver = problem.solver
    nquadrature = solver.nquadrature
    maximum_order = 2 * nquadrature - 1
    coordinate_scale = solver.coordinate_scale
    weight_scale = solver.weight_scale
    element_type = promote_type(eltype(state), eltype(parameters),
                                eltype(physical_nodes), typeof(time))
    source = zeros(element_type, maximum_order + 1)

    # Boundary-at-zero nucleation changes the number moment only.  A finite
    # nucleus would instead add J * L_nucleus^k to every moment here.
    source[1] = nucleation_rate_value / weight_scale

    @inbounds for moment_index in 1:maximum_order
        growth_integral = zero(element_type)
        for node_index in 1:nquadrature
            growth_integral += scaled_weights[node_index] *
                               scaled_nodes[node_index]^(moment_index - 1) *
                               (growth_rates[node_index] / coordinate_scale)
        end
        source[moment_index + 1] = moment_index * growth_integral
    end

    aggregation_source = _aggregation_moment_source(
        problem.kinetics_aggregationfunction,
        parameters.agg,
        physical_nodes,
        scaled_weights,
        maximum_order;
        shape_factor = problem.volume_shape_factor,
        pair_weight_scale = weight_scale)
    breakage_source = _breakage_moment_source(
        problem.kinetics_breakagefunction,
        parameters.br,
        physical_nodes,
        scaled_weights,
        maximum_order)

    @inbounds for moment_index in 0:maximum_order
        physical_scale = coordinate_scale^moment_index
        source[moment_index + 1] += (aggregation_source[moment_index + 1] +
                                     breakage_source[moment_index + 1]) / physical_scale
    end
    return source
end

function _dqmom_solvent_derivatives(problem::CrystallisationProblem,
                                    state,
                                    time,
                                    growth_context,
                                    volume_moment_rate)
    names = propertynames(problem.initial_solvent_state)
    n_solvent = length(names)
    concentration_position = findfirst(==(Symbol(:concentration)), names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))

    element_type = promote_type(eltype(state), typeof(volume_moment_rate))
    rates = zeros(element_type, n_solvent)
    concentration_rate = -problem.crystal_density * problem.volume_shape_factor *
                         volume_moment_rate
    rates[concentration_position] = concentration_rate

    if !(problem.solvent_dynamics isa DefaultSolventDynamics)
        base_rates = _solvent_derivatives(problem, state, time, growth_context)
        @inbounds for solvent_index in 1:n_solvent
            solvent_index == concentration_position ||
                (rates[solvent_index] = base_rates[solvent_index])
        end
    end
    return rates
end

"""DQMOM RHS in the normalized weighted-node state."""
function _dqmom_rhs!(destination,
                     state,
                     parameters,
                     time,
                     problem::CrystallisationProblem)
    solver = problem.solver
    nquadrature = solver.nquadrature
    n_solvent = length(propertynames(problem.initial_solvent_state))
    n_states = 2 * nquadrature + n_solvent
    length(destination) == n_states ||
        throw(ArgumentError("DQMOM RHS received an incompatible state length."))

    if nquadrature == 3 &&
       problem.kinetics_aggregationfunction isa noaggregation &&
       problem.kinetics_breakagefunction isa nobreakage
        return _dqmom_rhs_no_binary_n3!(destination, state, parameters, time, problem)
    end

    physical_nodes = Vector{promote_type(eltype(state), eltype(parameters))}(undef,
                                                                              nquadrature)
    scaled_nodes = Vector{promote_type(eltype(state), eltype(parameters))}(undef,
                                                                            nquadrature)
    scaled_weights = @view state[1:nquadrature]
    @inbounds for node_index in 1:nquadrature
        physical_node, _, scaled_node =
            _dqmom_length_and_weight(state, solver, node_index)
        physical_nodes[node_index] = physical_node
        scaled_nodes[node_index] = scaled_node
    end

    growth_rates = Vector{promote_type(eltype(state), eltype(parameters))}(undef,
                                                                             nquadrature)
    @inbounds for node_index in 1:nquadrature
        growth_rates[node_index] = net_growth_rate_at_length(
            problem.kinetics_growthfunction,
            parameters.gr,
            problem.kinetics_dissolutionfunction,
            parameters.diss,
            problem,
            state,
            time,
            physical_nodes[node_index])
    end
    nucleation_rate_value = nucleationrate(problem.kinetics_nucleationfunction,
                                           parameters.nucl,
                                           problem,
                                           state,
                                           time)

    source = _dqmom_source_vector(problem,
                                  state,
                                  parameters,
                                  time,
                                  physical_nodes,
                                  scaled_weights,
                                  scaled_nodes,
                                  growth_rates,
                                  nucleation_rate_value)
    projection_matrix = _dqmom_projection_matrix(scaled_nodes)
    direct_rates = projection_matrix \ source

    @inbounds for node_index in 1:nquadrature
        destination[node_index] = direct_rates[node_index]
        destination[nquadrature + node_index] = direct_rates[nquadrature + node_index]
    end

    # Differentiate m₃ = Σ qᵢ xᵢ³ in the direct variables.  This keeps the
    # solvent coupling tied to the actual state derivative rather than to a
    # second, potentially inconsistent approximation.
    scaled_volume_rate = zero(eltype(direct_rates))
    @inbounds for node_index in 1:nquadrature
        scaled_node = scaled_nodes[node_index]
        scaled_volume_rate += -2 * scaled_node^3 * direct_rates[node_index] +
                              3 * scaled_node^2 * direct_rates[nquadrature + node_index]
    end
    physical_volume_rate = solver.weight_scale * solver.coordinate_scale^3 *
                            scaled_volume_rate

    growth_context = if problem.kinetics_growthfunction isa AbstractFPLengthGrowthFunction
        growth_rates
    else
        net_growth_rate(problem.kinetics_growthfunction,
                        parameters.gr,
                        problem.kinetics_dissolutionfunction,
                        parameters.diss,
                        problem,
                        state,
                        time)
    end
    solvent_rates = _dqmom_solvent_derivatives(problem,
                                               state,
                                               time,
                                               growth_context,
                                               physical_volume_rate)
    _write_solvent_derivatives!(destination, problem, solvent_rates)
    return nothing
end

"""Construct the DQMOM ODE problem and selected time-stepping algorithm."""
function crystallisation_odeproblem(problem::CrystallisationProblem{NuclF, GrF, BrF,
                                                                    AggF, DQMOM,
                                                                    NuP, GrP, BrP,
                                                                    AggP, TP},
                                    saveat) where {NuclF <: AbstractNucleationFunction,
                                                   GrF <: AbstractGrowthFunction,
                                                   BrF <: AbstractBreakageFunction,
                                                   AggF <: AbstractAggregationFunction,
                                                   NuP <: AbstractVector{<:Real},
                                                   GrP <: AbstractVector{<:Real},
                                                   BrP <: AbstractVector{<:Real},
                                                   AggP <: AbstractVector{<:Real},
                                                   TP <: AbstractTemperature}
    _validate_dqmom_problem(problem)
    length(saveat) >= 2 ||
        throw(ArgumentError("saveat must contain at least two time points."))

    parameters = ComponentArray(; nucl = problem.parameterset_nucleation,
                                gr = problem.parameterset_growth,
                                diss = problem.parameterset_dissolution,
                                agg = problem.parameterset_aggregation,
                                br = problem.parameterset_breakage)
    internal_values = _dqmom_direct_state(problem)
    element_type = promote_type(eltype(internal_values),
                                eltype(problem.parameterset_nucleation),
                                eltype(problem.parameterset_growth),
                                eltype(problem.parameterset_dissolution),
                                eltype(problem.parameterset_aggregation),
                                eltype(problem.parameterset_breakage),
                                Float64)
    initial_values = element_type.(internal_values)

    n_solvent = length(propertynames(problem.initial_solvent_state))
    no_binary_sources = problem.kinetics_aggregationfunction isa noaggregation &&
                        problem.kinetics_breakagefunction isa nobreakage
    # Static three-node path: SVector state + out-of-place RHS (like the QMOM
    # fast path), so Tsit5 runs fully unrolled static stages instead of
    # heap-backed Vector stages.  Restricted to the default concentration-only
    # solvent coupling, which the RHS writes directly.
    fast_scalar_path = no_binary_sources && problem.solver.nquadrature == 3 &&
                       n_solvent == 1 &&
                       propertynames(problem.initial_solvent_state) == (:concentration,) &&
                       problem.solvent_dynamics isa DefaultSolventDynamics

    # Static three-node path: run the ODE on an SVector state so
    # OrdinaryDiffEq uses fully static Tsit5 stages instead of heap-backed
    # Vector stages (the QMOM fast-path trick).  The fast model supports both
    # calling conventions; the solver uses the out-of-place form because an
    # immutable SVector state cannot be mutated by an in-place integrator.
    # The gate is a compile-time constant for a concrete problem type.
    if fast_scalar_path
        model = DQMOMFastModel(problem)
        initial_state = SVector(ntuple(index -> initial_values[index], Val(7)))
        ode_problem = ODEProblem{false}(model,
                                        initial_state,
                                        (saveat[1], saveat[end]),
                                        parameters)
    else
        DQMOM_model! = (destination, state, p, time) ->
            _dqmom_rhs!(destination, state, p, time, problem)
        ode_problem = ODEProblem(DQMOM_model!,
                                 initial_values,
                                 (saveat[1], saveat[end]),
                                 parameters)
    end
    time_step_solver = _resolve_timestepping_algorithm(problem.solver, :tsit5)
    return ode_problem, time_step_solver
end

function _wrap_solution(problem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                          DQMOM, NuP, GrP, BrP,
                                                          AggP, TP},
                        solution) where {NuclF <: AbstractNucleationFunction,
                                         GrF <: AbstractGrowthFunction,
                                         BrF <: AbstractBreakageFunction,
                                         AggF <: AbstractAggregationFunction,
                                         NuP <: AbstractVector{<:Real},
                                         GrP <: AbstractVector{<:Real},
                                         BrP <: AbstractVector{<:Real},
                                         AggP <: AbstractVector{<:Real},
                                         TP <: AbstractTemperature}
    solver = problem.solver
    nquadrature = solver.nquadrature
    n_solvent = length(propertynames(problem.initial_solvent_state))
    internal_matrix = Array(solution)
    time_points = collect(solution.t)
    n_time_points = length(time_points)
    element_type = eltype(internal_matrix)
    weight_matrix = zeros(element_type, nquadrature, n_time_points)
    node_matrix = zeros(element_type, nquadrature, n_time_points)
    moment_matrix = zeros(element_type, 2 * nquadrature, n_time_points)

    for time_index in 1:n_time_points
        @inbounds for node_index in 1:nquadrature
            scaled_weight = internal_matrix[node_index, time_index]
            scaled_node = internal_matrix[nquadrature + node_index, time_index] /
                          scaled_weight
            weight_matrix[node_index, time_index] = solver.weight_scale * scaled_weight
            node_matrix[node_index, time_index] = solver.coordinate_scale * scaled_node
        end

        @inbounds for moment_index in 0:(2 * nquadrature - 1)
            moment_value = zero(element_type)
            for node_index in 1:nquadrature
                moment_value += weight_matrix[node_index, time_index] *
                                node_matrix[node_index, time_index]^moment_index
            end
            moment_matrix[moment_index + 1, time_index] = moment_value
        end
    end

    if element_type <: AbstractFloat
        projection_diagnostics = Vector{DQMOMProjectionDiagnostics{element_type}}(
            undef, n_time_points)
        @inbounds for time_index in 1:n_time_points
            scaled_nodes = _dqmom_scaled_nodes_at(node_matrix, solver, time_index)
            minimum_gap = typemax(element_type)
            for first_index in 1:(nquadrature - 1)
                for second_index in (first_index + 1):nquadrature
                    minimum_gap = min(minimum_gap,
                                      abs(scaled_nodes[first_index] -
                                          scaled_nodes[second_index]))
                end
            end
            projection_diagnostics[time_index] = DQMOMProjectionDiagnostics(
                :ok,
                minimum(@view weight_matrix[:, time_index]),
                minimum(@view node_matrix[:, time_index]),
                solver.coordinate_scale * minimum_gap,
                # The 3-node default uses a static 6x6 inverse (zero heap
                # allocations).  The dense fallback uses `cond(A, 1)`: one LU
                # + norms instead of a full SVD (gesdd), which dominated the
                # wrap-solution allocations (~65% of a DQMOM solve).  Both
                # return a legitimate condition estimate for the projection
                # matrix diagnostics.
                _dqmom_projection_condition_estimate(scaled_nodes))
        end
    else
        # SVD-based diagnostics are intentionally omitted for Dual-valued
        # trajectories; they are not part of the differentiated model.
        projection_diagnostics = nothing
    end

    d10 = [_safe_moment_size_ratio(moment_matrix[2, index],
                                   moment_matrix[1, index],
                                   moment_matrix[1, index])
           for index in 1:n_time_points]
    d32 = [_safe_moment_size_ratio(moment_matrix[4, index],
                                   moment_matrix[3, index],
                                   moment_matrix[1, index])
           for index in 1:n_time_points]
    d43 = 2 * nquadrature >= 5 ?
          [_safe_moment_size_ratio(moment_matrix[5, index],
                                   moment_matrix[4, index],
                                   moment_matrix[1, index])
           for index in 1:n_time_points] :
          zeros(element_type, n_time_points)
    moment2 = collect(@view moment_matrix[3, :])
    solvent_solution_state = _solvent_solution_state(problem, solution)
    solvent_final = Vector{element_type}(undef, n_solvent)
    first_solvent = 2 * nquadrature + 1
    @inbounds for solvent_index in 1:n_solvent
        solvent_final[solvent_index] = internal_matrix[
            first_solvent + solvent_index - 1, n_time_points]
    end
    final_state = vcat(collect(view(weight_matrix, :, n_time_points)),
                       collect(view(node_matrix, :, n_time_points)),
                       solvent_final)
    successful = OrdinaryDiffEq.SciMLBase.successful_retcode(solution.retcode)

    if element_type <: AbstractFloat
        all(node -> node > solver.minimum_size, node_matrix) ||
            throw(DomainError(minimum(node_matrix),
                              "DQMOM trajectory reached the lower size boundary."))
        all(weight -> weight > 0.0, weight_matrix) ||
            throw(DomainError(minimum(weight_matrix),
                              "DQMOM trajectory produced a nonpositive weight."))
    end

    return CrystallisationDQMOMSolution(time_points,
                                        solvent_solution_state.concentration,
                                        weight_matrix,
                                        node_matrix,
                                        moment_matrix,
                                        d10,
                                        d32,
                                        d43,
                                        moment2,
                                        solvent_solution_state,
                                        final_state,
                                        solution.stats,
                                        successful,
                                        projection_diagnostics)
end

function _simulatecrystallisation(problem::CrystallisationProblem{NuclF, GrF, BrF,
                                                                   AggF, DQMOM,
                                                                   NuP, GrP, BrP,
                                                                   AggP, TP},
                                  saveat)::CrystallisationDQMOMSolution where {
                                      NuclF <: AbstractNucleationFunction,
                                      GrF <: AbstractGrowthFunction,
                                      BrF <: AbstractBreakageFunction,
                                      AggF <: AbstractAggregationFunction,
                                      NuP <: AbstractVector{<:Real},
                                      GrP <: AbstractVector{<:Real},
                                      BrP <: AbstractVector{<:Real},
                                      AggP <: AbstractVector{<:Real},
                                      TP <: AbstractTemperature}
    ode_problem, time_step_solver = crystallisation_odeproblem(problem, saveat)
    abstol_tol, auto_tol_cb = _auto_abstol_opts(problem.solver, ode_problem.u0,
                                                problem.solver.abstol)
    ode_solution = solve(ode_problem,
                         time_step_solver;
                         callback = auto_tol_cb,
                         saveat = saveat,
                         reltol = problem.solver.reltol,
                         abstol = abstol_tol,
                         dense = false,
                         alg_hints = [:stiff],
                         maxiters = CRISTOOL_MAX_SOLVER_ITERS)
    return _wrap_solution(problem, ode_solution)
end

"""Reconstruct a physical quadrature rule at one DQMOM output index."""
function quadrature(solution::CrystallisationDQMOMSolution, time_index::Integer)
    1 <= time_index <= length(solution.time) ||
        throw(BoundsError(solution.time, time_index))
    diagnostics = solution.projection_diagnostics === nothing ? nothing :
                  solution.projection_diagnostics[time_index]
    nquadrature = size(solution.nodes, 1)
    return QMOMQuadrature(collect(view(solution.nodes, :, time_index)),
                          collect(view(solution.weights, :, time_index)),
                          nquadrature,
                          diagnostics)
end

quadrature(solution::CrystallisationDQMOMSolution) =
    [quadrature(solution, index) for index in eachindex(solution.time)]
dqmom_quadrature(solution::CrystallisationDQMOMSolution, time_index::Integer) =
    quadrature(solution, time_index)

time(solution::CrystallisationDQMOMSolution) = solution.time
state_vars(solution::CrystallisationDQMOMSolution) =
    merge(solution.solvent_state,
          (; weights = solution.weights,
             nodes = solution.nodes,
             moments = solution.moments,
             quadrature_weights = solution.weights,
             quadrature_nodes = solution.nodes))
size_metrics(solution::CrystallisationDQMOMSolution) =
    (; d10 = solution.d10,
       d32 = solution.d32,
       d43 = solution.d43,
       moment2 = solution.moment2)
get_characteristic_size(solution::CrystallisationDQMOMSolution) = solution.d43[end]
_size_trajectory(solution::CrystallisationDQMOMSolution) = solution.d43

function getmomentsizes(problem::CrystallisationProblem,
                        solution::CrystallisationDQMOMSolution)
    return size_metrics(solution)
end

"""Third physical length moment for secondary nucleation closures."""
function _secondary_third_moment(solver::DQMOM,
                                 problem::CrystallisationProblem,
                                 direct_population)
    nquadrature = solver.nquadrature
    volume_moment = zero(promote_type(eltype(direct_population),
                                      typeof(solver.coordinate_scale),
                                      typeof(solver.weight_scale)))
    @inbounds for node_index in 1:nquadrature
        scaled_weight = direct_population[node_index]
        scaled_node = direct_population[nquadrature + node_index] / scaled_weight
        volume_moment += scaled_weight * scaled_node^3
    end
    return solver.weight_scale * solver.coordinate_scale^3 * volume_moment
end
