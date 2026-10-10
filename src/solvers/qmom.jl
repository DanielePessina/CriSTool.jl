"""
Numerical inversion utilities for the classical one-dimensional Quadrature
Method of Moments (QMOM).

The solver integration is intentionally kept separate from this first layer:
`invert_moments` is a pure moment-to-quadrature operation and can therefore be
tested against known atomic measures before it is used in an ODE right-hand
side.
"""

@inline function _validate_qmom_solver(solver::QMOM)
    solver.nquadrature >= 2 ||
        throw(ArgumentError("QMOM requires nquadrature >= 2 for the M₂ concentration closure."))
    isfinite(solver.coordinate_scale) && solver.coordinate_scale > 0.0 ||
        throw(ArgumentError("QMOM coordinate_scale must be finite and strictly positive."))
    isfinite(solver.realizability_tolerance) && solver.realizability_tolerance >= 0.0 ||
        throw(ArgumentError("QMOM realizability_tolerance must be finite and nonnegative."))
    isfinite(solver.empty_population_tolerance) && solver.empty_population_tolerance >= 0.0 ||
        throw(ArgumentError("QMOM empty_population_tolerance must be finite and nonnegative."))
    isfinite(solver.node_coalescence_tolerance) && solver.node_coalescence_tolerance >= 0.0 ||
        throw(ArgumentError("QMOM node_coalescence_tolerance must be finite and nonnegative."))
    isfinite(solver.minimum_size) && solver.minimum_size >= 0.0 ||
        throw(ArgumentError("QMOM minimum_size must be finite and nonnegative."))
    return solver
end

"""Scalar type of a rule reconstructed from `moment_values` (at least `Float64`)."""
@inline function _qmom_rule_eltype(moment_values::AbstractArray)
    promoted_type = promote_type(eltype(moment_values), Float64)
    return promoted_type <: Number ? promoted_type : Float64
end

"""
What a QMOM inversion does with a non-realizable moment set.  The strict
inversion (public `invert_moments`, saved output, node tracking) throws the
`ArgumentError`.  An explicit-RK stage of the ODE instead falls back to a
rank-one rule; see `_qmom_stage_quadrature`.  Keeping the policy a type
parameter keeps exceptions out of the right-hand side and gives the rule
construction one concrete return type.
"""
struct StrictMomentInversion end
struct StageMomentInversion end

_qmom_inversion_failure(::StrictMomentInversion, moment_values, solver::QMOM,
                        failure::ArgumentError) = throw(failure)

"""Moment vector as a static `SVector` after the public length contract."""
function _qmom_static_moments(moment_values::AbstractVector, solver::QMOM)
    _validate_qmom_solver(solver)
    expected_moments = moment_count(solver)
    length(moment_values) == expected_moments ||
        throw(ArgumentError("QMOM expects $expected_moments raw moments " *
                            "(M₀:M$(moment_order(solver))), got $(length(moment_values))."))
    return SVector{expected_moments, _qmom_rule_eltype(moment_values)}(moment_values)
end

"""Static rule of an empty population: every node is inactive."""
@inline function _qmom_empty_rule(::Type{T}, ::Val{N}) where {T, N}
    zero_value = zero(T)
    diagnostics = QMOMInversionDiagnostics{T}(:empty, 0, zero_value, zero_value,
                                              zero_value, zero_value)
    return QMOMQuadrature(zero(SVector{N, T}), zero(SVector{N, T}), 0, diagnostics)
end

"""Active nodes and weights of a rule; static rules pad inactive nodes."""
@inline _qmom_active_nodes(quadrature::QMOMQuadrature) =
    view(quadrature.nodes, 1:quadrature.active_nodes)
@inline _qmom_active_weights(quadrature::QMOMQuadrature) =
    view(quadrature.weights, 1:quadrature.active_nodes)

"""Heap-backed public rule holding only the active nodes of a static rule."""
_qmom_public_rule(quadrature::QMOMQuadrature) =
    QMOMQuadrature(collect(_qmom_active_nodes(quadrature)),
                   collect(_qmom_active_weights(quadrature)),
                   quadrature.active_nodes, quadrature.diagnostics)

"""
    _qmom_wheeler_coefficients(normalized_moments, solver)

Compute the diagonal and subdiagonal coefficients of the Jacobi matrix from
the normalized moments.  This is Wheeler's recurrence-table form of the
long quotient-modified difference algorithm.  `normalized_moments` contains
`r₀:r₂N₋₁`, with `r₀ = 1`.  The subdiagonal has `N` slots; the last one is
unused.  Returns an `ArgumentError` for a non-realizable sequence.
"""
function _qmom_wheeler_coefficients(normalized_moments::SVector{M, T},
                                    solver::QMOM) where {M, T}
    nquadrature = M ÷ 2

    # s_{-1,j} and s_{0,j} in the Wheeler recurrence.  The table is kept as
    # two rows because only the previous row is needed for the next update.
    previous_row = zero(SVector{M, T})
    current_row = normalized_moments
    diagonal = zero(SVector{M ÷ 2, T})
    subdiagonal = zero(SVector{M ÷ 2, T})

    # Raw moments are integrated quantities, accurate only to the ODE relative
    # tolerance.  ρᵢ of a rank-deficient measure is a cancellation of terms of
    # size `cancellation_scale`, so it is zero only up to `reltol` times that
    # scale; below that floor the absolute coalescence tolerance applies.
    noise_tolerance = solver.reltol
    minimum_recurrence = zero(T)
    active_nodes = nquadrature
    was_deflated = false

    for recurrence_index in 0:(nquadrature - 1)
        # σᵢ = sᵢ,ᵢ₊₁ − sᵢ₋₁,ᵢ is the Jacobi diagonal coefficient.
        sigma = current_row[recurrence_index + 2] - previous_row[recurrence_index + 1]
        isfinite(sigma) ||
            return ArgumentError("QMOM Wheeler inversion produced a non-finite diagonal coefficient " *
                                 "at recurrence index $recurrence_index.")
        diagonal = setindex(diagonal, sigma, recurrence_index + 1)

        recurrence_index == nquadrature - 1 && break

        # ρᵢ = −σᵢ sᵢ,ᵢ₊₁ + sᵢ,ᵢ₊₂ − sᵢ₋₁,ᵢ₊₁.  Its square root is the
        # off-diagonal Jacobi coefficient.
        rho = -sigma * current_row[recurrence_index + 2] +
              current_row[recurrence_index + 3] - previous_row[recurrence_index + 2]
        isfinite(rho) ||
            return ArgumentError("QMOM Wheeler inversion produced a non-finite recurrence " *
                                 "coefficient at recurrence index $recurrence_index.")
        cancellation_scale = max(abs(sigma * current_row[recurrence_index + 2]),
                                 abs(current_row[recurrence_index + 3]),
                                 abs(previous_row[recurrence_index + 2]))
        recurrence_tolerance = max(solver.node_coalescence_tolerance,
                                   noise_tolerance * cancellation_scale)

        minimum_recurrence = recurrence_index == 0 ? rho : min(minimum_recurrence, rho)
        if rho < -recurrence_tolerance
            return ArgumentError("QMOM moments are not realizable: Wheeler recurrence " *
                                 "coefficient $rho is negative at index $recurrence_index.")
        elseif rho <= recurrence_tolerance
            # A vanishing recurrence coefficient means that the represented
            # measure has lower rank (e.g. a monodisperse population).  Stop
            # before dividing by rho and diagonalize only the active Jacobi
            # block.
            active_nodes = recurrence_index + 1
            was_deflated = true
            break
        end

        subdiagonal = setindex(subdiagonal, sqrt(rho), recurrence_index + 1)

        # sᵢ₊₁,j = [−σᵢsᵢ,j + sᵢ,j₊₁ − sᵢ₋₁,j] / ρᵢ,
        # j = i+2, ..., 2N−2−i.
        next_row = _qmom_wheeler_next_row(previous_row, current_row, sigma, rho,
                                          recurrence_index + 2,
                                          2 * nquadrature - 2 - recurrence_index)
        previous_row, current_row = current_row, next_row
    end

    return (; diagonal, subdiagonal, active_nodes, was_deflated, minimum_recurrence)
end

"""Wheeler row sᵢ₊₁,j = [−σᵢsᵢ,j + sᵢ,j₊₁ − sᵢ₋₁,j] / ρᵢ on `lower_column:upper_column`, zero elsewhere."""
@inline function _qmom_wheeler_next_row(previous_row::SVector{M, T}, current_row::SVector{M, T},
                                        sigma, rho, lower_column::Int,
                                        upper_column::Int) where {M, T}
    return SVector{M, T}(ntuple(Val(M)) do column_position
        column = column_position - 1
        lower_column <= column <= upper_column ?
            (-sigma * current_row[column + 1] + current_row[column + 2] -
             previous_row[column + 1]) / rho :
            zero(T)
    end)
end

"""
    _givens_radius(first_value, second_value)

`hypot` for the QL rotations: the plain `sqrt(x² + y²)` when the sum of
squares is safely inside the normal range (then neither square can have lost
precision to underflow), the overflow- and underflow-safe `hypot` otherwise.
`hypot` was 40% of the eigen-solve.
"""
@inline function _givens_radius(first_value, second_value)
    sum_of_squares = first_value * first_value + second_value * second_value
    primal_sum = _qmom_primal(sum_of_squares)
    return floatmin(primal_sum) / eps(primal_sum) < primal_sum < floatmax(primal_sum) ?
           sqrt(sum_of_squares) : hypot(first_value, second_value)
end

"""
    _qmom_golub_welsch(diagonal, subdiagonal, active_nodes)

Eigenvalues and first eigenvector components of the leading `active_nodes`
block of a symmetric tridiagonal Jacobi matrix, sorted by eigenvalue.  This is
the Golub-Welsch computation by implicit QL iteration with Wilkinson shifts:
only the first row of the eigenvector matrix is rotated, which is all a Gauss
rule needs, and every value stays on the stack.  Entries past `active_nodes`
are zero.  Returns an `ArgumentError` if the iteration does not converge.
"""
function _qmom_golub_welsch(diagonal::SVector{N, T}, subdiagonal::SVector{N, T},
                            active_nodes::Int) where {N, T}
    eigenvalues = SVector{N, T}(ntuple(index -> index <= active_nodes ? diagonal[index] :
                                                zero(T), Val(N)))
    # offdiagonal[i] couples rows i and i + 1; the last active slot is zero.
    offdiagonal = SVector{N, T}(ntuple(index -> index < active_nodes ? subdiagonal[index] :
                                                zero(T), Val(N)))
    first_row = SVector{N, T}(ntuple(index -> index == 1 ? one(T) : zero(T), Val(N)))
    iteration_limit = 30 * N

    for lower_index in 1:active_nodes
        iteration_count = 0
        while true
            # Find the first negligible off-diagonal entry at or after lower_index.
            upper_index = lower_index
            while upper_index < active_nodes
                # Primal values: Dual `==` also compares partials.
                diagonal_scale = _qmom_primal(abs(eigenvalues[upper_index]) +
                                              abs(eigenvalues[upper_index + 1]))
                _qmom_primal(abs(offdiagonal[upper_index])) + diagonal_scale == diagonal_scale &&
                    break
                upper_index += 1
            end
            upper_index == lower_index && break

            iteration_count += 1
            iteration_count > iteration_limit &&
                return ArgumentError("QMOM Golub-Welsch eigen-solve did not converge.")

            shift_ratio = (eigenvalues[lower_index + 1] - eigenvalues[lower_index]) /
                          (2 * offdiagonal[lower_index])
            shift_radius = _givens_radius(shift_ratio, one(T))
            signed_radius = shift_ratio >= zero(T) ? shift_radius : -shift_radius
            shifted_value = eigenvalues[upper_index] - eigenvalues[lower_index] +
                            offdiagonal[lower_index] / (shift_ratio + signed_radius)
            sine_value = one(T)
            cosine_value = one(T)
            pending_shift = zero(T)
            rotation_underflow = false
            row_index = upper_index - 1
            while row_index >= lower_index
                rotation_input = sine_value * offdiagonal[row_index]
                rotation_coupling = cosine_value * offdiagonal[row_index]
                rotation_radius = _givens_radius(rotation_input, shifted_value)
                offdiagonal = setindex(offdiagonal, rotation_radius, row_index + 1)
                if iszero(_qmom_primal(rotation_radius))
                    # Recover from underflow: the block splits here.
                    eigenvalues = setindex(eigenvalues,
                                           eigenvalues[row_index + 1] - pending_shift,
                                           row_index + 1)
                    offdiagonal = setindex(offdiagonal, zero(T), upper_index)
                    rotation_underflow = true
                    break
                end
                sine_value = rotation_input / rotation_radius
                cosine_value = shifted_value / rotation_radius
                shifted_value = eigenvalues[row_index + 1] - pending_shift
                rotation_radius = (eigenvalues[row_index] - shifted_value) * sine_value +
                                  2 * cosine_value * rotation_coupling
                pending_shift = sine_value * rotation_radius
                eigenvalues = setindex(eigenvalues, shifted_value + pending_shift,
                                       row_index + 1)
                shifted_value = cosine_value * rotation_radius - rotation_coupling
                trailing_component = first_row[row_index + 1]
                first_row = setindex(first_row,
                                     sine_value * first_row[row_index] +
                                     cosine_value * trailing_component,
                                     row_index + 1)
                first_row = setindex(first_row,
                                     cosine_value * first_row[row_index] -
                                     sine_value * trailing_component,
                                     row_index)
                row_index -= 1
            end
            rotation_underflow && continue
            eigenvalues = setindex(eigenvalues, eigenvalues[lower_index] - pending_shift,
                                   lower_index)
            offdiagonal = setindex(offdiagonal, shifted_value, lower_index)
            offdiagonal = setindex(offdiagonal, zero(T), upper_index)
        end
    end

    # Insertion sort of the active block by eigenvalue (node position).
    for sorted_end in 2:active_nodes
        insert_index = sorted_end
        while insert_index > 1 && eigenvalues[insert_index] < eigenvalues[insert_index - 1]
            eigenvalues = setindex(setindex(eigenvalues, eigenvalues[insert_index],
                                            insert_index - 1),
                                   eigenvalues[insert_index - 1], insert_index)
            first_row = setindex(setindex(first_row, first_row[insert_index],
                                          insert_index - 1),
                                 first_row[insert_index - 1], insert_index)
            insert_index -= 1
        end
    end
    return eigenvalues, first_row
end

"""Gaussian rule of `M₀:M₂N₋₁` on the whole real line.

This is the inversion without the `minimum_size` support check.  Under
dissolution the moments of a translated point measure stay realizable on the
real line while a node is carried through the support boundary; the node
tracking callback needs that node position to locate the crossing.  The rule
is static (`SVector` nodes padded past `active_nodes`); a non-realizable
sequence is handed to `_qmom_inversion_failure` for `inversion_policy`.
"""
@inline function _qmom_real_line_rule(moment_values::SVector{M}, solver::QMOM,
                                      inversion_policy) where {M}
    T = _qmom_rule_eltype(moment_values)
    first_moment = moment_values[1]
    isfinite(first_moment) ||
        return _qmom_inversion_failure(inversion_policy, moment_values, solver,
            ArgumentError("QMOM M₀ must be finite."))
    first_moment < zero(T) &&
        return _qmom_inversion_failure(inversion_policy, moment_values, solver,
            ArgumentError("QMOM M₀ must be nonnegative; got $first_moment."))

    first_moment <= solver.empty_population_tolerance &&
        return _qmom_empty_rule(T, Val(M ÷ 2))

    coordinate_scale = solver.coordinate_scale
    for moment_index in 1:(M - 1)
        isfinite(moment_values[moment_index + 1]) ||
            return _qmom_inversion_failure(inversion_policy, moment_values, solver,
                ArgumentError("QMOM moment M$moment_index must be finite."))
    end
    scaled_moments = SVector{M, T}(ntuple(Val(M)) do moment_position
        moment_position == 1 ? one(first_moment) :
        moment_values[moment_position] /
        (first_moment * coordinate_scale^(moment_position - 1))
    end)
    for moment_index in 1:(M - 1)
        isfinite(scaled_moments[moment_index + 1]) ||
            return _qmom_inversion_failure(inversion_policy, moment_values, solver,
                ArgumentError("QMOM scaled moment r$moment_index is non-finite."))
    end

    coefficients = _qmom_wheeler_coefficients(scaled_moments, solver)
    coefficients isa ArgumentError &&
        return _qmom_inversion_failure(inversion_policy, moment_values, solver, coefficients)
    active_nodes = coefficients.active_nodes

    eigensystem = _qmom_golub_welsch(coefficients.diagonal, coefficients.subdiagonal,
                                     active_nodes)
    eigensystem isa ArgumentError &&
        return _qmom_inversion_failure(inversion_policy, moment_values, solver, eigensystem)
    eigenvalues, first_eigenvector_components = eigensystem

    nodes = coordinate_scale .* eigenvalues
    weights = first_moment .* (first_eigenvector_components .^ 2)

    weight_tolerance = solver.realizability_tolerance * max(abs(first_moment), one(first_moment))

    for node_index in 1:active_nodes
        isfinite(nodes[node_index]) ||
            return _qmom_inversion_failure(inversion_policy, moment_values, solver,
                ArgumentError("QMOM inversion produced a non-finite node."))
    end
    for node_index in 1:active_nodes
        weight = weights[node_index]
        isfinite(weight) || return _qmom_inversion_failure(inversion_policy, moment_values, solver,
            ArgumentError("QMOM inversion produced a non-finite weight."))
        weight < -weight_tolerance &&
            return _qmom_inversion_failure(inversion_policy, moment_values, solver,
                ArgumentError("QMOM moments are not realizable: quadrature weight $weight " *
                              "is negative."))
    end

    # Check the reconstructed normalized moments independently of the
    # recurrence table.  This catches indexing and scaling mistakes while
    # keeping the public rule in physical units.  An m-node Gauss rule
    # reproduces r₀:r₂ₘ₋₁; the higher moments of a deflated measure were
    # already judged consistent by the deflation criterion.
    reconstruction_error = zero(T)
    reconstructed_moments = zero(SVector{M, T})
    for node_index in 1:active_nodes
        reconstructed_moments += (weights[node_index] / first_moment) .*
                                 _ascending_powers(nodes[node_index] / coordinate_scale, Val(M))
    end
    for moment_index in 0:(2 * active_nodes - 1)
        reconstructed = reconstructed_moments[moment_index + 1]
        # Relative to rₖ: scaled moments grow like (L/coordinate_scale)ᵏ, so an
        # absolute bound would be below roundoff for the top moments.
        reconstruction_error = max(reconstruction_error,
                                   abs(reconstructed - scaled_moments[moment_index + 1]) /
                                   max(one(T), abs(scaled_moments[moment_index + 1])))
    end
    reconstruction_error <= 100 * solver.realizability_tolerance ||
        return _qmom_inversion_failure(inversion_policy, moment_values, solver,
            ArgumentError("QMOM inversion reconstruction error $reconstruction_error " *
                          "exceeds the configured tolerance."))

    minimum_node = nodes[1]
    minimum_weight = weights[1]
    for node_index in 2:active_nodes
        minimum_node = min(minimum_node, nodes[node_index])
        minimum_weight = min(minimum_weight, weights[node_index])
    end
    status = coefficients.was_deflated ? :deflated : :ok
    diagnostics = QMOMInversionDiagnostics{T}(status, active_nodes, minimum_node,
                                              minimum_weight,
                                              coefficients.minimum_recurrence,
                                              reconstruction_error)
    return QMOMQuadrature(nodes, weights, active_nodes, diagnostics)
end

"""Reject a rule with an active node below `solver.minimum_size`."""
@inline function _qmom_supported_rule(rule::QMOMQuadrature, moment_values, solver::QMOM,
                                      inversion_policy)
    support_tolerance = solver.realizability_tolerance * solver.coordinate_scale
    for node in _qmom_active_nodes(rule)
        node < solver.minimum_size - support_tolerance &&
            return _qmom_inversion_failure(inversion_policy, moment_values, solver,
                ArgumentError("QMOM moments are not realizable on the configured " *
                              "support: node $node is below minimum_size $(solver.minimum_size)."))
    end
    return rule
end

"""
    invert_moments(moment_values, solver::QMOM) -> QMOMQuadrature

Invert the raw physical moments `M₀:M₂N₋₁` into an `N`-node Gaussian rule.
The implementation first scales the coordinate by `solver.coordinate_scale`,
computes Wheeler recurrence coefficients, and then applies Golub-Welsch to
the resulting symmetric tridiagonal Jacobi matrix.

Empty populations return a typed empty rule.  Rank-deficient populations are
represented by a rule with fewer active nodes.  A negative or otherwise
non-realizable moment sequence raises `ArgumentError`; no moment clipping or
silent projection is performed.
"""
function invert_moments(moment_values::AbstractVector, solver::QMOM)
    static_moments = _qmom_static_moments(moment_values, solver)
    static_rule = _qmom_supported_rule(_qmom_real_line_rule(static_moments, solver,
                                                            StrictMomentInversion()),
                                       static_moments, solver, StrictMomentInversion())
    return _qmom_public_rule(static_rule)
end

"""Construct a fully active QMOM rule from nodes and particle weights.

This convenience constructor is useful for independently testing the moment
source adapters.  Solver-generated rules additionally carry rank-deflation
diagnostics and use the four-argument constructor internally.
"""
function QMOMQuadrature(nodes::AbstractVector, weights::AbstractVector)
    length(nodes) == length(weights) ||
        throw(ArgumentError("QMOM nodes and weights must have the same length."))
    promoted_type_candidate = promote_type(eltype(nodes), eltype(weights), Float64)
    promoted_type = promoted_type_candidate <: Number ? promoted_type_candidate : Float64
    promoted_nodes = Vector{promoted_type}(nodes)
    promoted_weights = Vector{promoted_type}(weights)
    diagnostic = QMOMInversionDiagnostics(:ok,
                                          length(promoted_nodes),
                                          isempty(promoted_nodes) ? zero(promoted_type) :
                                          minimum(promoted_nodes),
                                          isempty(promoted_weights) ? zero(promoted_type) :
                                          minimum(promoted_weights),
                                          zero(promoted_type),
                                          zero(promoted_type))
    return QMOMQuadrature(promoted_nodes,
                          promoted_weights,
                          length(promoted_nodes),
                          diagnostic)
end

## QMOM population-balance integration

"""
    _ascending_powers(base, Val(K)) -> SVector{K}

`base⁰, base¹, …, base^(K-1)` by repeated multiplication. The moment sources
need every power of each node; `^` with an integer exponent costs a
compensated power evaluation per entry, which dominated the quadrature
right-hand sides.
"""
@inline _ascending_powers(base, ::Val{K}) where {K} =
    accumulate(*, SVector{K}(ntuple(power_position -> power_position == 1 ? one(base) : base,
                                    Val(K))))

"""`Σᵢ wᵢ xᵢᵏ gᵢ` for `k = 0:K-1` over the first `node_count` nodes."""
@inline function _weighted_power_sums(weights, nodes, node_rates, node_count::Int,
                                      ::Val{K}) where {K}
    sum_type = promote_type(eltype(weights), eltype(nodes), eltype(node_rates))
    power_sums = zero(SVector{K, sum_type})
    for node_index in 1:node_count
        power_sums += (weights[node_index] .* _ascending_powers(nodes[node_index], Val(K))) .*
                      node_rates[node_index]
    end
    return power_sums
end

"""
    _aggregation_moment_source(aggregationfunction, parameters,
                               nodes, weights, maximum_order;
                               shape_factor=1, pair_weight_scale=1)

Evaluate the selected binary aggregation kernels directly on a quadrature
rule. The result contains sources for `M₀:M_max`: a `Vector` for an integer
`maximum_order`, an `SVector{K}` of `M₀:M_{K-1}` for `Val(K)`. Only the kernels whose
volume-additive closure is implemented here are accepted; silently applying a
discretised mesh operator to moments would violate the QMOM contract.
"""
function _aggregation_moment_source(aggregationfunction::AbstractAggregationFunction,
                                    parameters,
                                    nodes::AbstractVector,
                                    weights::AbstractVector,
                                    maximum_order::Integer;
                                    shape_factor = 1.0,
                                    pair_weight_scale = 1.0)
    maximum_order >= 0 || throw(ArgumentError("maximum_order must be nonnegative."))
    return Vector(_aggregation_moment_source(aggregationfunction, parameters, nodes, weights,
                                             Val(maximum_order + 1); shape_factor,
                                             pair_weight_scale))
end

function _aggregation_moment_source(aggregationfunction::AbstractAggregationFunction,
                                    parameters,
                                    nodes::AbstractVector,
                                    weights::AbstractVector,
                                    ::Val{K};
                                    shape_factor = 1.0,
                                    pair_weight_scale = 1.0) where {K}
    length(nodes) == length(weights) ||
        throw(ArgumentError("Aggregation nodes and weights must have the same length."))
    _has_moment_closure(aggregationfunction) ||
        throw(ArgumentError("Aggregation function $(typeof(aggregationfunction)) " *
                            "has no validated QMOM moment closure."))

    _is_null_kinetic(aggregationfunction) &&
        return zero(SVector{K, promote_type(eltype(nodes), eltype(weights), typeof(shape_factor),
                                            typeof(pair_weight_scale))})

    named_parameters = _named_params(aggregationfunction, parameters)
    kernel_scale = exp10(named_parameters.log10_aggregation_coefficient)
    source_type = promote_type(eltype(nodes),
                               eltype(weights),
                               typeof(shape_factor),
                               typeof(pair_weight_scale),
                               typeof(kernel_scale))
    source = zero(SVector{K, source_type})

    # The factor 1/2 accounts for the ordered (i,j) double sum.  The
    # volume-additive product is evaluated in volume coordinates so that
    # M₃ is preserved to roundoff for every supported kernel.
    @inbounds for first_node in eachindex(nodes, weights)
        first_length = nodes[first_node]
        first_weight = weights[first_node]
        first_powers = _ascending_powers(first_length, Val(K))
        for second_node in eachindex(nodes, weights)
            second_length = nodes[second_node]
            second_weight = weights[second_node]
            event_rate = 0.5 * pair_weight_scale * first_weight * second_weight *
                         _aggregation_kernel(aggregationfunction,
                                             kernel_scale,
                                             shape_factor,
                                             first_length,
                                             second_length)
            product_length = (first_length^3 + second_length^3)^(one(source_type) / 3)
            source += event_rate .* (_ascending_powers(product_length, Val(K)) .-
                                     first_powers .-
                                     _ascending_powers(second_length, Val(K)))
        end
    end
    return source
end

"""
    aggregation_moment_source(aggregationfunction, parameters,
                              quadrature, maximum_order; shape_factor=1)

Evaluate a validated binary aggregation closure on a QMOM quadrature rule.
The returned vector contains sources for the physical moments `M₀:M_max`.
"""
function aggregation_moment_source(aggregationfunction::AbstractAggregationFunction,
                                   parameters,
                                   quadrature::QMOMQuadrature,
                                   maximum_order::Integer;
                                   shape_factor = 1.0)
    return _aggregation_moment_source(aggregationfunction, parameters,
                                      _qmom_active_nodes(quadrature),
                                      _qmom_active_weights(quadrature),
                                      maximum_order; shape_factor = shape_factor)
end

aggregation_moment_source(aggregationfunction::AbstractAggregationFunction,
                          parameters,
                          quadrature::QMOMQuadrature,
                          maximum_order::Integer,
                          problem::CrystallisationProblem) =
    aggregation_moment_source(aggregationfunction, parameters, quadrature,
                              maximum_order; shape_factor = problem.model.system.volume_shape_factor)

"""
    _breakage_moment_source(breakagefunction, parameters,
                            nodes, weights, maximum_order; weight_scale=1)

Evaluate the exact raw-moment source of the existing uniform-in-volume
binary daughter law on a QMOM rule.  A parent of length `L` contributes
`β(L) [6/(k+3)-1] L^k` to moment `k`; consequently the source preserves
`M₃` and adds one net particle to `M₀` per breakage event.  As for
aggregation, an integer `maximum_order` returns a `Vector` and `Val(K)` an
`SVector{K}`.
"""
function _breakage_moment_source(breakagefunction::AbstractBreakageFunction,
                                 parameters,
                                 nodes::AbstractVector,
                                 weights::AbstractVector,
                                 maximum_order::Integer;
                                 weight_scale = 1.0)
    maximum_order >= 0 || throw(ArgumentError("maximum_order must be nonnegative."))
    return Vector(_breakage_moment_source(breakagefunction, parameters, nodes, weights,
                                          Val(maximum_order + 1); weight_scale))
end

function _breakage_moment_source(breakagefunction::AbstractBreakageFunction,
                                 parameters,
                                 nodes::AbstractVector,
                                 weights::AbstractVector,
                                 ::Val{K};
                                 weight_scale = 1.0) where {K}
    length(nodes) == length(weights) ||
        throw(ArgumentError("Breakage nodes and weights must have the same length."))
    _has_moment_closure(breakagefunction) ||
        throw(ArgumentError("Breakage function $(typeof(breakagefunction)) " *
                            "has no validated QMOM moment closure."))

    source_type = promote_type(eltype(nodes), eltype(weights), eltype(parameters),
                               typeof(weight_scale))
    source = zero(SVector{K, source_type})
    _is_null_kinetic(breakagefunction) && return source

    named_parameters = _named_params(breakagefunction, parameters)
    @inbounds for node_index in eachindex(nodes, weights)
        crystal_length = nodes[node_index]
        event_rate = weight_scale * weights[node_index] *
                     _breakage_frequency(breakagefunction,
                                         named_parameters,
                                         crystal_length)
        source += event_rate .* (_breakage_daughter_factors(Val(K)) .*
                                 _ascending_powers(crystal_length, Val(K)))
    end
    return source
end

"""Uniform-in-volume daughter factors `6/(k+3) - 1` for `k = 0:K-1`."""
@inline _breakage_daughter_factors(::Val{K}) where {K} =
    SVector{K}(ntuple(moment_position -> 6 / (moment_position + 2) - 1, Val(K)))

"""
    breakage_moment_source(breakagefunction, parameters, quadrature,
                           maximum_order)

Evaluate the validated uniform-in-volume daughter closure on a QMOM
quadrature rule. The returned vector contains physical moment sources.
"""
function breakage_moment_source(breakagefunction::AbstractBreakageFunction,
                                parameters,
                                quadrature::QMOMQuadrature,
                                maximum_order::Integer)
    return _breakage_moment_source(breakagefunction, parameters,
                                   _qmom_active_nodes(quadrature),
                                   _qmom_active_weights(quadrature),
                                   maximum_order)
end

"""Net growth rate at every node of a static rule (inactive nodes included)."""
@inline function _qmom_node_growth_rates(quadrature::QMOMQuadrature,
                                         growthfunction::AbstractGrowthFunction,
                                         growth_parameters,
                                         dissolutionfunction::AbstractDissolutionFunction,
                                         dissolution_parameters,
                                         problem::CrystallisationProblem,
                                         state,
                                         time)
    return map(crystal_length -> net_growth_rate_at_length(growthfunction,
                                                           growth_parameters,
                                                           dissolutionfunction,
                                                           dissolution_parameters,
                                                           problem,
                                                           state,
                                                           time,
                                                           crystal_length),
               quadrature.nodes)
end

"""Growth contribution to `M₀:M₂N₋₁` of a static `N`-node rule."""
function _qmom_growth_moment_source(quadrature::QMOMQuadrature{<:SVector{N}},
                                    node_growth_rates::SVector{N},
                                    nucleation_rate_value) where {N}
    source_type = promote_type(eltype(quadrature.nodes),
                               eltype(quadrature.weights),
                               typeof(nucleation_rate_value),
                               eltype(node_growth_rates))
    # growth_integrals[k] = Σᵢ wᵢ Lᵢᵏ⁻¹ Gᵢ feeds moment k.
    growth_integrals = _weighted_power_sums(quadrature.weights, quadrature.nodes,
                                            node_growth_rates, quadrature.active_nodes,
                                            Val(2 * N))
    return SVector{2 * N, source_type}(ntuple(Val(2 * N)) do moment_position
        moment_position == 1 ? nucleation_rate_value :
        (moment_position - 1) * growth_integrals[moment_position - 1]
    end)
end

"""Exact scalar-growth-rate moment source without a moment inversion.

For a size-independent growth rate, `Iₖ = G Mₖ`; retaining this direct path is
important for both speed and realizability because an RK stage of the linear
moment equations need not itself be a positive point measure even though the
accepted solution is.  A quadrature is still reconstructed for output and for
the size-dependent source adapters.
"""
function _qmom_scalar_growth_moment_source(moment_values::SVector{K},
                                           scalar_growth_rate,
                                           nucleation_rate_value) where {K}
    source_type = promote_type(eltype(moment_values), typeof(scalar_growth_rate),
                               typeof(nucleation_rate_value))
    return SVector{K, source_type}(ntuple(Val(K)) do moment_position
        moment_position == 1 ? nucleation_rate_value :
        (moment_position - 1) * scalar_growth_rate * moment_values[moment_position - 1]
    end)
end

"""Crystal volume rate `3 Σ wᵢ Lᵢ² Gᵢ` of a static rule."""
@inline function _qmom_volume_rate(quadrature::QMOMQuadrature{<:SVector{N}},
                                   node_growth_rates::SVector{N}) where {N}
    volume_rate = zero(promote_type(eltype(quadrature.nodes),
                                    eltype(quadrature.weights),
                                    eltype(node_growth_rates)))
    for node_index in 1:quadrature.active_nodes
        volume_rate += quadrature.weights[node_index] * quadrature.nodes[node_index]^2 *
                       node_growth_rates[node_index]
    end
    return 3 * volume_rate
end

"""Solvent coupling for a QMOM volume-rate closure.

Custom solvent dynamics continue to receive the scalar net growth rate so
existing user-defined auxiliary-state laws remain source-compatible.  The
concentration component is replaced with the exact quadrature volume rate;
the built-in dynamics use the same expression directly.
"""
function _qmom_solvent_derivatives(problem::CrystallisationProblem,
                                   solvent_dynamics::DefaultSolventDynamics,
                                   state,
                                   time,
                                   scalar_growth_rate,
                                   quadrature_volume_rate)
    concentration_rate = _quadrature_concentration_rate(problem, quadrature_volume_rate)
    concentration_position = _quadrature_concentration_position(problem)
    return ntuple(index -> index == concentration_position ? concentration_rate :
                  zero(concentration_rate),
                  Val(length(propertynames(problem.initial_solvent_state))))
end

function _qmom_solvent_derivatives(problem::CrystallisationProblem,
                                   solvent_dynamics,
                                   state,
                                   time,
                                   scalar_growth_rate,
                                   quadrature_volume_rate)
    concentration_rate = _quadrature_concentration_rate(problem, quadrature_volume_rate)
    concentration_position = _quadrature_concentration_position(problem)
    base_rates = _solvent_derivatives(problem, state, time, scalar_growth_rate,
                                      solvent_dynamics)
    return ntuple(index -> index == concentration_position ? concentration_rate :
                  base_rates[index],
                  Val(length(propertynames(problem.initial_solvent_state))))
end

@inline _quadrature_concentration_rate(problem::CrystallisationProblem, quadrature_volume_rate) =
    -problem.model.system.crystal_density * problem.model.system.volume_shape_factor *
    quadrature_volume_rate

@inline function _quadrature_concentration_position(problem::CrystallisationProblem)
    concentration_position = findfirst(==(Symbol(:concentration)),
                                       propertynames(problem.initial_solvent_state))
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    return concentration_position
end

function _qmom_extinction_callback(problem::CrystallisationProblem)
    _dissolution_enabled(problem) || return nothing
    n_moments = moment_count(problem.solver)
    n_moments >= 4 || return nothing

    names = propertynames(problem.initial_solvent_state)
    concentration_position = findfirst(==(Symbol(:concentration)), names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    concentration_index = _solvent_state_index(problem, :concentration)
    state_count = n_moments + _operation_state_count(problem.operation) + length(names)
    threshold = _validate_solid_mass_concentration_threshold(problem)
    condition = (state, time, integrator) -> problem.model.system.crystal_density * problem.model.system.volume_shape_factor * state[4] - threshold
    affect! = integrator -> nothing
    affect_neg! = integrator -> begin
        state = integrator.u
        residual_mass = problem.model.system.crystal_density * problem.model.system.volume_shape_factor * state[4]
        integrator.u = SVector(ntuple(Val(state_count)) do state_index
            if state_index <= n_moments
                zero(state[state_index])
            elseif state_index == concentration_index
                state[state_index] + residual_mass
            else
                state[state_index]
            end
        end)
        nothing
    end
    return ContinuousCallback(condition, affect!, affect_neg!;
                              save_positions = (false, true))
end

# Event location only needs primal values; the node positions are tracked on
# the primal trajectory.
_qmom_primal(number::AbstractFloat) = number
_qmom_primal(number::Integer) = float(number)
_qmom_primal(number::Real) = _qmom_primal_unwrapped(number, OrdinaryDiffEq.SciMLBase.value(number))
_qmom_primal_unwrapped(number::T, unwrapped::T) where {T} =
    throw(ArgumentError("QMOM node tracking cannot extract a primal value from $T."))
_qmom_primal_unwrapped(number, unwrapped) = _qmom_primal(unwrapped)

_qmom_primal_moments(state, ::Val{M}) where {M} =
    SVector(ntuple(moment_index -> _qmom_primal(state[moment_index]), Val(M)))

"""Position and weight of the smallest active node of a static rule."""
function _qmom_smallest_node(rule::QMOMQuadrature)
    smallest_index = 1
    for node_index in 2:rule.active_nodes
        rule.nodes[node_index] < rule.nodes[smallest_index] && (smallest_index = node_index)
    end
    return rule.nodes[smallest_index], rule.weights[smallest_index]
end

"""Scaled distance of the smallest quadrature node above `minimum_size`.

An empty population has no node to cross the boundary; it reports one scaled
unit so that the event function stays positive.
"""
function _qmom_smallest_node_clearance(state, solver::QMOM, moment_count_value::Val)
    primal_moments = _qmom_primal_moments(state, moment_count_value)
    rule = _qmom_real_line_rule(primal_moments, solver, StrictMomentInversion())
    isempty(rule) && return one(eltype(primal_moments))
    smallest_length, _ = _qmom_smallest_node(rule)
    return (smallest_length - solver.minimum_size) / solver.coordinate_scale
end

"""
    _qmom_vanishing_node_callback(problem, moment_count_value)

Remove a quadrature node when dissolution carries it to `minimum_size`.

For a size-independent signed rate `G`, the population balance
`∂n/∂t + G ∂n/∂L = 0` on `L > minimum_size` gives

    dMₖ/dt = k G Mₖ₋₁ + G Lₘᵢₙᵏ n(Lₘᵢₙ, t).

The first term is what the QMOM right-hand side integrates: it translates
every node, `dLᵢ/dt = G`, with constant weights.  For `G < 0` the boundary term
is the outflux of crystals that have dissolved completely.  On a quadrature
measure `n = Σ wᵢ δ(L - Lᵢ)` that outflux is the impulse `wᵢ Lₘᵢₙᵏ` at the time
node `i` reaches `Lₘᵢₙ`; this callback applies it.  The removed volume
`wᵢ Lₘᵢₙ³` (zero for the default `minimum_size = 0`) is released as solute, so
`C + ρ kᵥ M₃` is unchanged by the event.  A crossing while `G ≥ 0` has no
boundary outflux and is ignored.

When the removed node is the last one, the population is set exactly to zero
and the remaining `ρ kᵥ M₃` is released, matching the solid-mass extinction
callback.
"""
function _qmom_vanishing_node_callback(problem::CrystallisationProblem,
                                       moment_count_value::Val{M}) where {M}
    _dissolution_enabled(problem) || return nothing
    solver = problem.solver
    names = propertynames(problem.initial_solvent_state)
    concentration_index = _solvent_state_index(problem, :concentration)
    state_count = M + _operation_state_count(problem.operation) + length(names)
    solid_mass_factor = problem.model.system.crystal_density * problem.model.system.volume_shape_factor

    condition = (state, time, integrator) ->
        _qmom_smallest_node_clearance(state, solver, moment_count_value)
    affect! = integrator -> nothing
    affect_neg! = integrator -> begin
        state = integrator.u
        # Only a dissolving rate carries crystals out through the boundary; a
        # down-crossing while G ≥ 0 (e.g. fresh nuclei at L = 0) is roundoff.
        boundary_rate = net_growth_rate(problem.model.growth.law, integrator.p.gr,
                                        problem.model.dissolution.law, integrator.p.diss,
                                        problem, state, integrator.t)
        _qmom_primal(boundary_rate) < 0 || return nothing
        rule = _qmom_real_line_rule(_qmom_primal_moments(state, moment_count_value), solver,
                                    StrictMomentInversion())
        isempty(rule) && return nothing
        vanished_length, vanished_weight = _qmom_smallest_node(rule)
        last_node = rule.active_nodes == 1
        released_mass = last_node ? solid_mass_factor * state[4] :
                        solid_mass_factor * vanished_weight * vanished_length^3
        integrator.u = SVector(ntuple(Val(state_count)) do state_index
            if state_index <= M
                last_node ? zero(state[state_index]) :
                state[state_index] - vanished_weight * vanished_length^(state_index - 1)
            elseif state_index == concentration_index
                state[state_index] + released_mass
            else
                state[state_index]
            end
        end)
        nothing
    end
    # A removed node is a partial dissolution step, not a reset of the
    # population: keep the saved trajectory on the requested save times.
    return ContinuousCallback(condition, affect!, affect_neg!;
                              save_positions = (false, false))
end

function _validate_qmom_problem(problem::CrystallisationProblem)
    _validate_qmom_solver(problem.solver)
    for rate_law in (problem.model.growth.law, problem.model.dissolution.law)
        _is_length_law(rate_law) &&
            throw(ArgumentError("$(typeof(rate_law)) is length-dependent and is not supported by QMOM; " *
                                "use FiniteVol or WENO."))
    end
    for binary_law in (problem.model.aggregation.law, problem.model.breakage.law)
        _has_moment_closure(binary_law) ||
            throw(ArgumentError("$(typeof(binary_law)) has no validated QMOM moment closure."))
    end
    return problem
end

"""Best-effort rule for an intermediate explicit-RK stage.

The moment equations with binary sources are nonlinear.  Consequently an
explicit Runge--Kutta stage can lie just outside the realizable cone even when
the accepted state is realizable.  Public `invert_moments` remains strict; the
ODE stage fallback keeps integration alive by using the positive rank-one
measure defined by `(M₀,M₁)` and is only reached after a strict inversion has
failed.  Initial states are validated before the solve so malformed user data
are still rejected.
"""
@inline _qmom_stage_quadrature(moment_values::SVector, solver::QMOM) =
    _qmom_supported_rule(_qmom_real_line_rule(moment_values, solver, StageMomentInversion()),
                         moment_values, solver, StageMomentInversion())

function _qmom_inversion_failure(::StageMomentInversion, moment_values::SVector{M},
                                 solver::QMOM, failure::ArgumentError) where {M}
    T = _qmom_rule_eltype(moment_values)
    first_moment = moment_values[1]
    first_moment > solver.empty_population_tolerance || throw(failure)
    second_moment = moment_values[2]
    isfinite(second_moment) || throw(failure)
    representative_length = second_moment / first_moment
    isfinite(representative_length) &&
        representative_length >= solver.minimum_size || throw(failure)
    diagnostic = QMOMInversionDiagnostics{T}(:stage_deflated,
                                             1,
                                             representative_length,
                                             first_moment,
                                             zero(first_moment),
                                             zero(first_moment))
    single_node(value) = SVector{M ÷ 2, T}(ntuple(index -> index == 1 ? value : zero(T),
                                                  Val(M ÷ 2)))
    return QMOMQuadrature(single_node(representative_length), single_node(first_moment),
                          1, diagnostic)
end

"""
Moment derivatives `dM₀:M₂N₋₁/dt` and crystal volume rate.  Without
aggregation or breakage the moment equations close exactly on the scalar
growth rate and need no inversion; binary sources are evaluated on the
quadrature reconstructed from the current stage.
"""
@inline _qmom_moment_rates(::NoAggregation, ::NoBreakage, problem::CrystallisationProblem,
                           moment_values::SVector, state, parameters, time, scalar_growth_rate,
                           nucleation_rate_value) =
    (_qmom_scalar_growth_moment_source(moment_values, scalar_growth_rate,
                                       nucleation_rate_value),
     3 * scalar_growth_rate * moment_values[3])

function _qmom_moment_rates(aggregationfunction::AbstractAggregationFunction,
                            breakagefunction::AbstractBreakageFunction,
                            problem::CrystallisationProblem,
                            moment_values::SVector{M}, state, parameters, time,
                            scalar_growth_rate, nucleation_rate_value) where {M}
    reconstructed_rule = _qmom_stage_quadrature(moment_values, problem.solver)
    node_growth_rates = _qmom_node_growth_rates(reconstructed_rule,
                                                problem.model.growth.law,
                                                parameters.gr,
                                                problem.model.dissolution.law,
                                                parameters.diss,
                                                problem,
                                                state,
                                                time)
    active_nodes = _qmom_active_nodes(reconstructed_rule)
    active_weights = _qmom_active_weights(reconstructed_rule)
    moment_derivatives =
        _qmom_growth_moment_source(reconstructed_rule, node_growth_rates,
                                   nucleation_rate_value) +
        _aggregation_moment_source(aggregationfunction, parameters.agg, active_nodes,
                                   active_weights, Val(M);
                                   shape_factor = problem.model.system.volume_shape_factor) +
        _breakage_moment_source(breakagefunction, parameters.br, active_nodes,
                                active_weights, Val(M))
    return moment_derivatives, _qmom_volume_rate(reconstructed_rule, node_growth_rates)
end

"""QMOM right-hand side on the static state; `Val(M)` is the raw-moment count."""
function _qmom_rhs(problem::CrystallisationProblem, state::SVector{S}, parameters, time,
                   feed_population, moment_count_value::Val{M}) where {S, M}
    moment_values = SVector(ntuple(moment_index -> state[moment_index], moment_count_value))
    scalar_growth_rate = net_growth_rate(problem.model.growth.law,
                                         parameters.gr,
                                         problem.model.dissolution.law,
                                         parameters.diss,
                                         problem,
                                         state,
                                         time)
    nucleation_rate_value = nucleationrate(problem.model.nucleation.law,
                                           parameters.nucl,
                                           problem,
                                           state,
                                           time)
    moment_derivatives, volume_rate = _qmom_moment_rates(problem.model.aggregation.law,
                                                         problem.model.breakage.law,
                                                         problem, moment_values, state,
                                                         parameters, time,
                                                         scalar_growth_rate,
                                                         nucleation_rate_value)
    solvent_derivatives = _qmom_solvent_derivatives(problem,
                                                    problem.solvent_dynamics,
                                                    state,
                                                    time,
                                                    scalar_growth_rate,
                                                    volume_rate)
    reactor_count = _operation_state_count(problem.operation)
    internal_rates = SVector(ntuple(Val(S)) do state_index
        state_index <= M ? moment_derivatives[state_index] :
        (state_index <= M + reactor_count ? zero(state[state_index]) :
         solvent_derivatives[state_index - M - reactor_count])
    end)
    return _compose_operation_rhs(problem, state, time, internal_rates, feed_population)
end

"""Construct the QMOM ODE problem and selected time-stepping algorithm."""
function crystallisation_odeproblem(solver::QMOM, problem::CrystallisationProblem, saveat, kinetic_values)
    _validate_qmom_problem(problem)
    n_moments = moment_count(problem.solver)
    n_solvent = length(propertynames(problem.initial_solvent_state))
    n_states = n_moments + _operation_state_count(problem.operation) + n_solvent
    n_states == length(_get_initial_state(problem)) ||
        throw(ArgumentError("QMOM initial_state must contain $n_moments raw moments " *
                            "followed by $n_solvent solvent-state values."))
    # Keep user-provided non-realizable data distinguishable from transient
    # explicit-RK stages.  Empty states are valid and need no inversion.
    initial_moments = @view _get_initial_state(problem)[1:n_moments]
    initial_moments[1] > problem.solver.empty_population_tolerance &&
        invert_moments(initial_moments, problem.solver)

    feed_population = _operation_feed_population(problem)
    # The moment count is a type parameter of the closure, so every stage
    # array (moments, quadrature, sources) is a static vector.
    moment_count_value = Val(n_moments)
    QMOM_model = (state, parameters, time) ->
        _qmom_rhs(problem, state, parameters, time, feed_population, moment_count_value)

    parameters = kinetic_values
    element_type = promote_type(eltype(kinetic_values), Float64)
    initial_values = element_type.(_get_initial_state(problem))
    initial_state = SVector(ntuple(index -> initial_values[index], Val(n_states)))
    ode_problem = ODEProblem{false}(QMOM_model,
                                    initial_state,
                                    (saveat[1], saveat[end]),
                                    parameters;
                                    callback = CallbackSet(
                                        _qmom_extinction_callback(problem),
                                        _qmom_vanishing_node_callback(problem,
                                                                      moment_count_value)))
    return ode_problem, _timestepping_algorithm(solver, Tsit5())
end

"""Saved quadrature rules of a moment trajectory; inactive nodes stay zero."""
function _qmom_saved_quadratures(moment_matrix::AbstractMatrix, solver::QMOM, ::Val{M}) where {M}
    saved_count = size(moment_matrix, 2)
    node_matrix = zeros(eltype(moment_matrix), solver.nquadrature, saved_count)
    weight_matrix = zeros(eltype(moment_matrix), solver.nquadrature, saved_count)
    diagnostics_type = QMOMInversionDiagnostics{_qmom_rule_eltype(moment_matrix)}
    diagnostics = Vector{diagnostics_type}(undef, saved_count)
    for time_index in 1:saved_count
        saved_moments = SVector{M}(ntuple(moment_index -> moment_matrix[moment_index, time_index],
                                          Val(M)))
        rule = _qmom_supported_rule(_qmom_real_line_rule(saved_moments, solver,
                                                         StrictMomentInversion()),
                                    saved_moments, solver, StrictMomentInversion())
        for node_index in 1:rule.active_nodes
            node_matrix[node_index, time_index] = rule.nodes[node_index]
            weight_matrix[node_index, time_index] = rule.weights[node_index]
        end
        diagnostics[time_index] = rule.diagnostics
    end
    return node_matrix, weight_matrix, diagnostics
end

function _wrap_solution(solver::QMOM, problem::CrystallisationProblem, solution,
                        boundary_flow_state::NamedTuple)
    moment_matrix = Array(solution[1:moment_count(solver), :])
    _validate_qmom_solver(solver)
    node_matrix, weight_matrix, diagnostics =
        _qmom_saved_quadratures(moment_matrix, solver, Val(moment_count(solver)))
    population = QuadraturePopulation(moment_matrix, node_matrix, weight_matrix, diagnostics)
    return _crystallisation_solution(problem, solution, population,
                                     _raw_moment_size_metrics(moment_matrix),
                                     collect(solution[:, end]),
                                     merge(_reactor_solution_state(problem, solution),
                                           boundary_flow_state))
end

"""
    quadrature(solution, time_index) -> QMOMQuadrature
    quadrature(solution) -> Vector{QMOMQuadrature}

Quadrature rule of a QMOM or DQMOM solution at one saved time (or at every
saved time). Inactive QMOM nodes are dropped; DQMOM rules keep every node.
"""
quadrature(solution::CrystallisationSolution{<:QuadraturePopulation}, time_index::Integer) =
    _saved_quadrature(solution.population, time_index)
quadrature(solution::CrystallisationSolution{<:QuadraturePopulation}) =
    [quadrature(solution, time_index) for time_index in eachindex(solution.time)]

function _saved_quadrature(population::QuadraturePopulation, time_index::Integer)
    checkbounds(population.nodes, :, time_index)
    diagnostic = _saved_diagnostic(population.diagnostics, time_index)
    active_nodes = _active_node_count(diagnostic, size(population.nodes, 1))
    return QMOMQuadrature(population.nodes[1:active_nodes, time_index],
                          population.weights[1:active_nodes, time_index],
                          active_nodes, diagnostic)
end

_saved_diagnostic(diagnostics::AbstractVector, time_index) = diagnostics[time_index]
_saved_diagnostic(diagnostics::Nothing, time_index) = nothing
_active_node_count(diagnostic::QMOMInversionDiagnostics, node_count) = diagnostic.active_nodes
_active_node_count(diagnostic, node_count) = node_count
