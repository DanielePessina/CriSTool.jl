"""
Direct Quadrature Method of Moments (DQMOM) for seeded, one-dimensional
crystallisation population balances.

The public initial-state population block is `[weights; physical_nodes]`.
The ODE uses normalized weights and weighted, dimensionless nodes so the
projection matrix does not contain powers of lengths measured in metres.
"""

include("dqmom_operations.jl")

@inline function _validate_dqmom_solver(solver::DQMOM)
    solver.nquadrature >= 2 ||
        throw(ArgumentError("DQMOM requires nquadrature >= 2."))
    isfinite(solver.weight_scale) && solver.weight_scale > 0.0 ||
        throw(ArgumentError("DQMOM weight_scale must be finite and strictly positive."))
    return _validate_quadrature_solver(solver)
end

"""Smallest pairwise distance between the entries of `nodes` (`typemax` for fewer than two)."""
function _dqmom_minimum_node_gap(nodes::AbstractVector{T}) where {T}
    minimum_gap = typemax(T)
    @inbounds for first_index in 1:(length(nodes) - 1)
        for second_index in (first_index + 1):length(nodes)
            minimum_gap = min(minimum_gap, abs(nodes[first_index] - nodes[second_index]))
        end
    end
    return minimum_gap
end

function _validate_dqmom_public_state(problem::CrystallisationProblem)
    solver = problem.solver
    _validate_dqmom_solver(solver)
    problem.initial_population === nothing &&
        throw(ArgumentError("DQMOM requires a nonempty seeded initial population. " *
                            "Provide seed crystals in the run conditions."))

    direct_state = _get_initial_state(problem)
    nquadrature = solver.nquadrature
    n_solvent = length(propertynames(problem.initial_solvent_state))
    n_reactor = _operation_state_count(problem.operation)
    expected_length = last(_solvent_state_range(problem))
    length(direct_state) == expected_length ||
        throw(ArgumentError("DQMOM initial_state must contain $nquadrature weights, " *
                            "$nquadrature physical nodes, $n_reactor reactor values, " *
                            "and $n_solvent solvent values."))

    weights = @view direct_state[1:nquadrature]
    nodes = @view direct_state[(nquadrature + 1):(2 * nquadrature)]
    all(weight -> isfinite(weight) && weight > 0.0, weights) ||
        throw(ArgumentError("DQMOM requires strictly positive initial weights."))
    all(node -> isfinite(node) && node > solver.minimum_size, nodes) ||
        throw(ArgumentError("DQMOM requires initial nodes strictly above minimum_size."))

    separation_tolerance = solver.node_coalescence_tolerance * solver.coordinate_scale
    _dqmom_minimum_node_gap(nodes) > separation_tolerance ||
        throw(ArgumentError("DQMOM initial nodes must be pairwise distinct and " *
                            "separated by more than $separation_tolerance m."))
    return problem
end

function _validate_dqmom_problem(problem::CrystallisationProblem)
    _validate_dqmom_public_state(problem)

    _is_length_law(problem.model.dissolution.law) &&
        throw(ArgumentError("Length-dependent dissolution is not supported by DQMOM; " *
                            "use FiniteVol or WENO."))
    for binary_law in (problem.model.aggregation.law, problem.model.breakage.law)
        _has_moment_closure(binary_law) ||
            throw(ArgumentError("$(typeof(binary_law)) has no validated DQMOM moment closure."))
    end
    return problem
end

"""Static scaled DQMOM projection matrix `[(1-k) xᵢᵏ | k xᵢᵏ⁻¹]` for dimensionless nodes."""
@inline function _dqmom_projection_matrix(scaled_nodes::SVector{N, T}) where {N, T}
    node_powers = map(scaled_node -> _ascending_powers(scaled_node, Val(2 * N)), scaled_nodes)
    projection_values = ntuple(Val(4 * N * N)) do linear_index
        row_index = ((linear_index - 1) % (2 * N)) + 1
        column_index = ((linear_index - 1) ÷ (2 * N)) + 1
        moment_index = row_index - 1
        node_index = column_index <= N ? column_index : column_index - N
        powers = node_powers[node_index]
        if column_index <= N
            moment_index == 0 ? one(T) :
                (one(T) - moment_index) * powers[moment_index + 1]
        else
            moment_index == 0 ? zero(T) :
                moment_index * powers[moment_index]
        end
    end
    return SMatrix{2 * N, 2 * N, T}(projection_values)
end

"""Projection-matrix condition estimate in the 1-norm, without a full SVD."""
@inline function _dqmom_projection_condition_estimate(scaled_nodes::SVector)
    projection_matrix = _dqmom_projection_matrix(scaled_nodes)
    return opnorm(projection_matrix, 1) * opnorm(inv(projection_matrix), 1)
end

function _dqmom_direct_state(problem::CrystallisationProblem)
    solver = problem.solver
    public_state = _get_initial_state(problem)
    nquadrature = solver.nquadrature
    element_type = promote_type(eltype(public_state),
                                typeof(solver.coordinate_scale),
                                typeof(solver.weight_scale))
    internal_state = Vector{element_type}(undef, last(_solvent_state_range(problem)))
    @inbounds for node_index in 1:nquadrature
        weight = public_state[node_index]
        physical_node = public_state[nquadrature + node_index]
        scaled_weight = weight / solver.weight_scale
        scaled_node = physical_node / solver.coordinate_scale
        internal_state[node_index] = scaled_weight
        internal_state[nquadrature + node_index] = scaled_weight * scaled_node
    end
    # The reactor and solvent blocks follow the population contiguously and are
    # not rescaled.
    tail_range = (2 * nquadrature + 1):length(internal_state)
    internal_state[tail_range] = @view public_state[tail_range]
    return internal_state
end

"""Scaled weights, scaled nodes and physical nodes of a normalized DQMOM state."""
@inline function _dqmom_static_nodes(state, solver::DQMOM, ::Val{N}) where {N}
    scaled_weights = SVector(ntuple(node_index -> state[node_index], Val(N)))
    scaled_nodes = SVector(ntuple(node_index -> state[N + node_index] / state[node_index],
                                  Val(N)))
    physical_nodes = solver.coordinate_scale .* scaled_nodes
    return scaled_weights, scaled_nodes, physical_nodes
end

"""Scaled moment sources `S₀:S₂N₋₁` of crystallisation kinetics on the DQMOM nodes."""
function _dqmom_moment_source(problem::CrystallisationProblem,
                              parameters,
                              physical_nodes::SVector{N},
                              scaled_weights::SVector{N},
                              scaled_nodes::SVector{N},
                              growth_rates::SVector{N},
                              nucleation_rate_value) where {N}
    solver = problem.solver
    coordinate_scale = solver.coordinate_scale
    weight_scale = solver.weight_scale
    element_type = promote_type(eltype(scaled_weights), eltype(scaled_nodes),
                                eltype(growth_rates), typeof(nucleation_rate_value),
                                eltype(parameters), typeof(coordinate_scale))

    aggregation_source = _aggregation_moment_source(
        problem.model.aggregation.law,
        parameters.agg,
        physical_nodes,
        scaled_weights,
        Val(2 * N);
        shape_factor = problem.model.system.volume_shape_factor,
        pair_weight_scale = weight_scale)
    breakage_source = _breakage_moment_source(
        problem.model.breakage.law,
        parameters.br,
        physical_nodes,
        scaled_weights,
        Val(2 * N))

    # growth_integrals[k] = Σᵢ wᵢ xᵢᵏ⁻¹ Gᵢ / coordinate_scale feeds moment k.
    growth_integrals = _weighted_power_sums(scaled_weights, scaled_nodes,
                                            growth_rates ./ coordinate_scale, N, Val(2 * N))
    physical_scales = _ascending_powers(coordinate_scale, Val(2 * N))
    return SVector{2 * N, element_type}(ntuple(Val(2 * N)) do moment_position
        # Boundary-at-zero nucleation changes the number moment only.  A finite
        # nucleus would instead add J * L_nucleus^k to every moment here.
        kinetic_source = moment_position == 1 ? nucleation_rate_value / weight_scale :
                         (moment_position - 1) * growth_integrals[moment_position - 1]
        kinetic_source + (aggregation_source[moment_position] +
                          breakage_source[moment_position]) / physical_scales[moment_position]
    end)
end

"""
Growth argument handed to custom solvent dynamics: length-dependent growth
hands them the node rates, scalar growth the scalar net rate.  The built-in
dynamics need none, so nothing is computed for them.
"""
_dqmom_growth_context(::DefaultSolventDynamics, problem::CrystallisationProblem,
                      state, parameters, time, growth_rates) = nothing

function _dqmom_growth_context(solvent_dynamics, problem::CrystallisationProblem,
                               state, parameters, time, growth_rates)
    return _is_length_law(problem.model.growth.law) ? growth_rates :
           net_growth_rate(problem.model.growth.law,
                           parameters.gr,
                           problem.model.dissolution.law,
                           parameters.diss,
                           problem,
                           state,
                           time)
end

"""DQMOM RHS on the static normalized weighted-node state; `Val(N)` is the node count."""
function _dqmom_rhs(problem::CrystallisationProblem, state::SVector{S}, parameters, time,
                    feed_population, quadrature_count_value::Val{N}) where {S, N}
    solver = problem.solver
    scaled_weights, scaled_nodes, physical_nodes =
        _dqmom_static_nodes(state, solver, quadrature_count_value)
    growth_rates = map(crystal_length -> net_growth_rate_at_length(problem.model.growth.law,
                                                                   parameters.gr,
                                                                   problem.model.dissolution.law,
                                                                   parameters.diss,
                                                                   problem,
                                                                   state,
                                                                   time,
                                                                   crystal_length),
                       physical_nodes)
    nucleation_rate_value = nucleationrate(problem.model.nucleation.law,
                                           parameters.nucl,
                                           problem,
                                           state,
                                           time)

    kinetic_source = _dqmom_moment_source(problem, parameters, physical_nodes,
                                          scaled_weights, scaled_nodes, growth_rates,
                                          nucleation_rate_value)
    operation_context = _operation_transport_context(problem, state, time)
    projected_source = _dqmom_operation_moment_source(kinetic_source, state, problem,
                                                      operation_context, feed_population)
    direct_rates = _dqmom_projection_matrix(scaled_nodes) \ projected_source

    # Solvent exchange is internal to crystallisation. Inlet and washout
    # moments are already included in the projected population derivative but
    # must not be interpreted as crystal growth or dissolution.
    physical_volume_rate = solver.weight_scale * solver.coordinate_scale^3 *
                           kinetic_source[4]
    growth_context = _dqmom_growth_context(problem.solvent_dynamics, problem, state,
                                           parameters, time, growth_rates)
    solvent_rates = _quadrature_solvent_derivatives(problem, problem.solvent_dynamics, state,
                                                    time, growth_context,
                                                    physical_volume_rate)
    reactor_count = _operation_state_count(problem.operation)
    internal_rates = SVector(ntuple(Val(S)) do state_index
        state_index <= 2 * N ? direct_rates[state_index] :
        (state_index <= 2 * N + reactor_count ? zero(direct_rates[1]) :
         solvent_rates[state_index - 2 * N - reactor_count])
    end)
    return _dqmom_operation_state_rates(internal_rates, state, problem, operation_context)
end

"""
    _dqmom_boundary_callback(problem)

Stop a DQMOM solve when dissolution carries a node down to `minimum_size`.

The DQMOM state stores each node's weight and position, so a node cannot be
removed mid-solve the way QMOM drops one from its moment inversion.  Instead of
integrating past the boundary into negative crystal sizes, the solve throws at
the crossing and points to the solvers that model complete dissolution.
Partial dissolution that stays above the boundary is unaffected.
"""
function _dqmom_boundary_callback(problem::CrystallisationProblem)
    _dissolution_enabled(problem) || return nothing
    solver = problem.solver
    nquadrature = solver.nquadrature
    scaled_minimum_size = solver.minimum_size / solver.coordinate_scale
    condition = (state, time, integrator) ->
        minimum(node_index -> state[nquadrature + node_index] / state[node_index],
                1:nquadrature) - scaled_minimum_size
    affect! = integrator -> nothing
    affect_neg! = integrator -> throw(DomainError(integrator.t,
        "A DQMOM node dissolved down to minimum_size at this time; DQMOM cannot remove " *
        "a node mid-solve. Use QMOM, FiniteVol or WENO for complete dissolution."))
    return ContinuousCallback(condition, affect!, affect_neg!; save_positions = (false, false))
end

"""Construct the DQMOM ODE problem and selected time-stepping algorithm."""
function crystallisation_odeproblem(solver::DQMOM, problem::CrystallisationProblem, saveat, kinetic_values)
    _validate_dqmom_problem(problem)
    length(saveat) >= 2 ||
        throw(ArgumentError("saveat must contain at least two time points."))

    parameters = kinetic_values
    internal_values = _dqmom_direct_state(problem)
    element_type = promote_type(eltype(internal_values), eltype(kinetic_values), Float64)
    initial_values = element_type.(internal_values)

    feed_population = _operation_feed_population(problem)
    # The node count is a type parameter of the closure: the ODE runs on an
    # SVector state with static Tsit5 stages, and the projection solve is a
    # static `SMatrix` LU for every node count, operation and binary source.
    quadrature_count_value = Val(problem.solver.nquadrature)
    DQMOM_model = (state, p, time) ->
        _dqmom_rhs(problem, state, p, time, feed_population, quadrature_count_value)
    initial_state = SVector(ntuple(index -> initial_values[index], Val(length(initial_values))))
    ode_problem = ODEProblem{false}(DQMOM_model,
                                    initial_state,
                                    (saveat[1], saveat[end]),
                                    parameters)
    return ode_problem, _timestepping_algorithm(solver, Tsit5())
end

_solver_callbacks(solver::DQMOM, problem::CrystallisationProblem, odeproblem) =
    _dqmom_boundary_callback(problem)

function _wrap_solution(solver::DQMOM, problem::CrystallisationProblem, solution,
                        boundary_flow_state::NamedTuple)
    nquadrature = solver.nquadrature
    internal_matrix = Array(solution)
    n_time_points = length(solution.t)
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

    projection_status = _dqmom_projection_diagnostics(node_matrix, weight_matrix, solver,
                                                      Val(nquadrature))
    trajectory_status = _dqmom_trajectory_failure(node_matrix, weight_matrix, solver)

    reactor_solution_state = _reactor_solution_state(problem, solution)
    # Reactor and solvent values follow the population and are not rescaled.
    final_state = vcat(collect(view(weight_matrix, :, n_time_points)),
                       collect(view(node_matrix, :, n_time_points)),
                       internal_matrix[(2 * nquadrature + 1):end, n_time_points])

    return _reconstruct_from(projection_status, trajectory_status) do projection_diagnostics, _
        _crystallisation_solution(problem, solution,
                                  QuadraturePopulation(moment_matrix, node_matrix, weight_matrix,
                                                       projection_diagnostics),
                                  _raw_moment_size_metrics(moment_matrix), final_state,
                                  merge(reactor_solution_state, boundary_flow_state))
    end
end

"""
    _dqmom_trajectory_failure(node_matrix, weight_matrix, solver)

`ReconstructionFailure` when a saved node reached `minimum_size` or a weight is
nonpositive, else `nothing`. Dual-valued trajectories are not checked.
"""
_dqmom_trajectory_failure(node_matrix::AbstractMatrix, weight_matrix, solver::DQMOM) = nothing

function _dqmom_trajectory_failure(node_matrix::AbstractMatrix{<:AbstractFloat}, weight_matrix,
                                   solver::DQMOM)
    all(node -> node > solver.minimum_size, node_matrix) ||
        return ReconstructionFailure(minimum(node_matrix),
                                     "DQMOM trajectory reached the lower size boundary.")
    all(weight -> weight > 0.0, weight_matrix) ||
        return ReconstructionFailure(minimum(weight_matrix),
                                     "DQMOM trajectory produced a nonpositive weight.")
    return nothing
end

"""
    _dqmom_projection_diagnostics(node_matrix, weight_matrix, solver, Val(N))

Projection diagnostics of every saved DQMOM state.  The static projection
matrix gives a 1-norm condition estimate without heap allocation.  A saved
node pair closer than the coalescence tolerance returns a `ReconstructionFailure`.
Diagnostics are omitted (`nothing`) for Dual-valued trajectories; they are
not part of the differentiated model.
"""
_dqmom_projection_diagnostics(node_matrix::AbstractMatrix, weight_matrix, solver::DQMOM,
                              quadrature_count_value::Val) = nothing

function _dqmom_projection_diagnostics(node_matrix::AbstractMatrix{T}, weight_matrix,
                                       solver::DQMOM, ::Val{N}) where {T <: AbstractFloat, N}
    n_time_points = size(node_matrix, 2)
    projection_diagnostics = Vector{DQMOMProjectionDiagnostics{T}}(undef, n_time_points)
    @inbounds for time_index in 1:n_time_points
        scaled_nodes = SVector(ntuple(node_index -> node_matrix[node_index, time_index] /
                                                    solver.coordinate_scale, Val(N)))
        minimum_gap = _dqmom_minimum_node_gap(scaled_nodes)
        physical_minimum_gap = solver.coordinate_scale * minimum_gap
        physical_minimum_gap > solver.node_coalescence_tolerance * solver.coordinate_scale ||
            return ReconstructionFailure(physical_minimum_gap,
                                         "DQMOM trajectory reached node coalescence; " *
                                         "active quadrature nodes cannot merge or be reborn.")
        projection_diagnostics[time_index] = DQMOMProjectionDiagnostics(
            :ok,
            minimum(@view weight_matrix[:, time_index]),
            minimum(@view node_matrix[:, time_index]),
            physical_minimum_gap,
            _dqmom_projection_condition_estimate(scaled_nodes))
    end
    return projection_diagnostics
end

"""Physical raw moment `µ_order` of a normalized DQMOM state."""
function _dqmom_physical_moment(solver::DQMOM, direct_population, order::Integer)
    nquadrature = solver.nquadrature
    scaled_moment = zero(promote_type(eltype(direct_population),
                                      typeof(solver.coordinate_scale),
                                      typeof(solver.weight_scale)))
    @inbounds for node_index in 1:nquadrature
        scaled_weight = direct_population[node_index]
        scaled_node = direct_population[nquadrature + node_index] / scaled_weight
        scaled_moment += scaled_weight * scaled_node^order
    end
    return solver.weight_scale * solver.coordinate_scale^order * scaled_moment
end

"""Third physical length moment for secondary nucleation closures."""
_secondary_third_moment(solver::DQMOM, problem::CrystallisationProblem, direct_population) =
    _dqmom_physical_moment(solver, direct_population, 3)
