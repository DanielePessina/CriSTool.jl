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
    solver.inversion_algorithm === :wheeler ||
        throw(ArgumentError("Unsupported QMOM inversion algorithm: $(solver.inversion_algorithm). " *
                            "Only :wheeler is currently implemented."))
    return solver
end

@inline function _qmom_empty_quadrature(moment_values::AbstractVector,
                                        status::Symbol,
                                        solver::QMOM)
    T = promote_type(eltype(moment_values), typeof(solver.coordinate_scale))
    zero_value = zero(T)
    diagnostics = QMOMInversionDiagnostics(status, 0, zero_value, zero_value,
                                           zero_value, zero_value)
    return QMOMQuadrature(Vector{T}(undef, 0), Vector{T}(undef, 0), 0,
                          diagnostics)
end

"""
    _qmom_wheeler_coefficients(normalized_moments, solver)

Compute the diagonal and subdiagonal coefficients of the Jacobi matrix from
the normalized moments.  This is Wheeler's recurrence-table form of the
long quotient-modified difference algorithm.  `normalized_moments` contains
`r₀:r₂N₋₁`, with `r₀ = 1`.
"""
function _qmom_wheeler_coefficients(normalized_moments::AbstractVector,
                                    solver::QMOM)
    nquadrature = solver.nquadrature
    moment_count_value = 2 * nquadrature
    T = eltype(normalized_moments)

    # s_{-1,j} and s_{0,j} in the Wheeler recurrence.  The table is kept as
    # two rows because only the previous row is needed for the next update.
    previous_row = zeros(T, moment_count_value)
    current_row = collect(normalized_moments)
    diagonal = Vector{T}(undef, nquadrature)
    subdiagonal = Vector{T}(undef, nquadrature - 1)

    recurrence_tolerance = solver.node_coalescence_tolerance
    minimum_recurrence = zero(T)
    active_nodes = nquadrature
    was_deflated = false

    for recurrence_index in 0:(nquadrature - 1)
        # σᵢ = sᵢ,ᵢ₊₁ − sᵢ₋₁,ᵢ is the Jacobi diagonal coefficient.
        sigma = current_row[recurrence_index + 2] - previous_row[recurrence_index + 1]
        isfinite(sigma) ||
            throw(ArgumentError("QMOM Wheeler inversion produced a non-finite diagonal coefficient " *
                                "at recurrence index $recurrence_index."))
        diagonal[recurrence_index + 1] = sigma

        recurrence_index == nquadrature - 1 && break

        # ρᵢ = −σᵢ sᵢ,ᵢ₊₁ + sᵢ,ᵢ₊₂ − sᵢ₋₁,ᵢ₊₁.  Its square root is the
        # off-diagonal Jacobi coefficient.
        rho = -sigma * current_row[recurrence_index + 2] +
              current_row[recurrence_index + 3] - previous_row[recurrence_index + 2]
        isfinite(rho) ||
            throw(ArgumentError("QMOM Wheeler inversion produced a non-finite recurrence " *
                                "coefficient at recurrence index $recurrence_index."))

        minimum_recurrence = recurrence_index == 0 ? rho : min(minimum_recurrence, rho)
        if rho < -recurrence_tolerance
            throw(ArgumentError("QMOM moments are not realizable: Wheeler recurrence " *
                                "coefficient $rho is negative at index $recurrence_index."))
        elseif rho <= recurrence_tolerance
            # A vanishing recurrence coefficient means that the represented
            # measure has lower rank (e.g. a monodisperse population).  Stop
            # before dividing by rho and let the caller diagonalize only the
            # active Jacobi block.
            active_nodes = recurrence_index + 1
            was_deflated = true
            break
        end

        subdiagonal[recurrence_index + 1] = sqrt(rho)

        # sᵢ₊₁,j = [−σᵢsᵢ,j + sᵢ,j₊₁ − sᵢ₋₁,j] / ρᵢ,
        # j = i+2, ..., 2N−2−i.
        next_row = zeros(T, moment_count_value)
        lower_column = recurrence_index + 2
        upper_column = 2 * nquadrature - 2 - recurrence_index
        if lower_column <= upper_column
            @inbounds for column in lower_column:upper_column
                next_row[column + 1] =
                    (-sigma * current_row[column + 1] + current_row[column + 2] -
                     previous_row[column + 1]) / rho
            end
        end
        previous_row, current_row = current_row, next_row
    end

    return diagonal, subdiagonal, active_nodes, was_deflated, minimum_recurrence
end

@inline function _qmom_rule_diagnostic(status::Symbol,
                                       active_nodes::Int,
                                       nodes::AbstractVector,
                                       weights::AbstractVector,
                                       minimum_recurrence,
                                       reconstruction_error)
    T = promote_type(eltype(nodes), eltype(weights), typeof(minimum_recurrence),
                     typeof(reconstruction_error))
    minimum_node = isempty(nodes) ? zero(T) : minimum(nodes)
    minimum_weight = isempty(weights) ? zero(T) : minimum(weights)
    return QMOMInversionDiagnostics(status, active_nodes, minimum_node,
                                    minimum_weight, minimum_recurrence,
                                    reconstruction_error)
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
    _validate_qmom_solver(solver)
    expected_moments = moment_count(solver)
    length(moment_values) == expected_moments ||
        throw(ArgumentError("QMOM expects $expected_moments raw moments " *
                            "(M₀:M$(moment_order(solver))), got $(length(moment_values))."))

    promoted_type = promote_type(eltype(moment_values),
                                 typeof(solver.coordinate_scale),
                                 Float64)
    T = promoted_type <: Number ? promoted_type : Float64
    zero_value = zero(T)
    first_moment = moment_values[1]
    isfinite(first_moment) ||
        throw(ArgumentError("QMOM M₀ must be finite."))
    first_moment < zero_value &&
        throw(ArgumentError("QMOM M₀ must be nonnegative; got $first_moment."))

    first_moment <= solver.empty_population_tolerance &&
        return _qmom_empty_quadrature(moment_values, :empty, solver)

    scaled_moments = Vector{T}(undef, expected_moments)
    scaled_moments[1] = one(first_moment)
    coordinate_scale = solver.coordinate_scale
    @inbounds for moment_index in 1:(expected_moments - 1)
        raw_moment = moment_values[moment_index + 1]
        isfinite(raw_moment) ||
            throw(ArgumentError("QMOM moment M$moment_index must be finite."))
        scaled_moments[moment_index + 1] =
            raw_moment / (first_moment * coordinate_scale^moment_index)
        isfinite(scaled_moments[moment_index + 1]) ||
            throw(ArgumentError("QMOM scaled moment r$moment_index is non-finite."))
    end

    diagonal, subdiagonal, active_nodes, was_deflated, minimum_recurrence =
        _qmom_wheeler_coefficients(scaled_moments, solver)

    active_diagonal = diagonal[1:active_nodes]
    active_subdiagonal = active_nodes > 1 ? subdiagonal[1:(active_nodes - 1)] : T[]
    jacobi = SymTridiagonal(copy(active_diagonal), copy(active_subdiagonal))
    eigendecomposition = eigen(jacobi)

    nodes = coordinate_scale .* collect(eigendecomposition.values)
    first_eigenvector_components = vec(eigendecomposition.vectors[1, :])
    weights = first_moment .* (first_eigenvector_components .^ 2)

    support_tolerance = solver.realizability_tolerance * coordinate_scale
    weight_tolerance = solver.realizability_tolerance * max(abs(first_moment), one(first_moment))

    @inbounds for node in nodes
        isfinite(node) || throw(ArgumentError("QMOM inversion produced a non-finite node."))
        node < solver.minimum_size - support_tolerance &&
            throw(ArgumentError("QMOM moments are not realizable on the configured " *
                                "support: node $node is below minimum_size $(solver.minimum_size)."))
    end
    @inbounds for weight in weights
        isfinite(weight) || throw(ArgumentError("QMOM inversion produced a non-finite weight."))
        weight < -weight_tolerance &&
            throw(ArgumentError("QMOM moments are not realizable: quadrature weight $weight " *
                                "is negative."))
    end

    # Check the reconstructed normalized moments independently of the
    # recurrence table.  This catches indexing and scaling mistakes while
    # keeping the public rule in physical units.
    reconstruction_error = zero(T)
    @inbounds for moment_index in 0:(expected_moments - 1)
        reconstructed = zero(T)
        for node_index in eachindex(nodes, weights)
            reconstructed += (weights[node_index] / first_moment) *
                             (nodes[node_index] / coordinate_scale)^moment_index
        end
        reconstruction_error = max(reconstruction_error,
                                   abs(reconstructed - scaled_moments[moment_index + 1]))
    end
    reconstruction_error <= 100 * solver.realizability_tolerance ||
        throw(ArgumentError("QMOM inversion reconstruction error $reconstruction_error " *
                            "exceeds the configured tolerance."))

    status = was_deflated ? :deflated : :ok
    diagnostics = _qmom_rule_diagnostic(status, active_nodes, nodes, weights,
                                        minimum_recurrence, reconstruction_error)
    return QMOMQuadrature(nodes, weights, active_nodes, diagnostics)
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
    aggregation_moment_source(aggregationfunction, parameters,
                              quadrature, maximum_order; shape_factor=1)

Evaluate the selected binary aggregation kernels directly on a QMOM rule.
The returned vector contains sources for `M₀:M_max`.  Only the kernels whose
volume-additive closure is implemented here are accepted; silently applying a
discretised mesh operator to moments would violate the QMOM contract.
"""
function aggregation_moment_source(aggregationfunction::AbstractAggregationFunction,
                                   parameters,
                                   quadrature::QMOMQuadrature,
                                   maximum_order::Integer;
                                   shape_factor = 1.0)
    maximum_order >= 0 || throw(ArgumentError("maximum_order must be nonnegative."))
    supported = aggregationfunction isa noaggregation ||
                aggregationfunction isa aggr_scalar ||
                aggregationfunction isa aggr_linear ||
                aggregationfunction isa aggr_linearvol ||
                aggregationfunction isa aggr_avg
    supported || throw(ArgumentError("Aggregation function $(typeof(aggregationfunction)) " *
                                    "has no validated QMOM moment closure."))

    aggregationfunction isa noaggregation &&
        return zeros(promote_type(eltype(quadrature.nodes),
                                  eltype(quadrature.weights),
                                  typeof(shape_factor)), maximum_order + 1)
    isempty(quadrature) &&
        return zeros(promote_type(eltype(quadrature.nodes),
                                  eltype(quadrature.weights),
                                  typeof(shape_factor)), maximum_order + 1)

    named_parameters = _named_params(aggregationfunction, parameters)
    kernel_scale = exp10(named_parameters.log10_aggregation_coefficient)
    source_type = promote_type(eltype(quadrature.nodes),
                               eltype(quadrature.weights),
                               typeof(shape_factor),
                               typeof(kernel_scale))
    source = zeros(source_type, maximum_order + 1)
    active_nodes = quadrature.active_nodes
    nodes = quadrature.nodes
    weights = quadrature.weights

    # The factor 1/2 accounts for the ordered (i,j) double sum.  The
    # volume-additive product is evaluated in volume coordinates so that
    # M₃ is preserved to roundoff for every supported kernel.
    @inbounds for first_node in 1:active_nodes
        first_length = nodes[first_node]
        first_weight = weights[first_node]
        for second_node in 1:active_nodes
            second_length = nodes[second_node]
            second_weight = weights[second_node]
            event_rate = 0.5 * first_weight * second_weight *
                         _aggregation_kernel(aggregationfunction,
                                             kernel_scale,
                                             shape_factor,
                                             first_length,
                                             second_length)
            product_length = (first_length^3 + second_length^3)^(one(source_type) / 3)
            for moment_index in 0:maximum_order
                source[moment_index + 1] += event_rate *
                                            (product_length^moment_index -
                                             first_length^moment_index -
                                             second_length^moment_index)
            end
        end
    end
    return source
end

aggregation_moment_source(aggregationfunction::AbstractAggregationFunction,
                          parameters,
                          quadrature::QMOMQuadrature,
                          maximum_order::Integer,
                          problem::CrystallisationProblem) =
    aggregation_moment_source(aggregationfunction, parameters, quadrature,
                              maximum_order; shape_factor = problem.volume_shape_factor)

"""
    breakage_moment_source(breakagefunction, parameters, quadrature,
                           maximum_order)

Evaluate the exact raw-moment source of the existing uniform-in-volume
binary daughter law on a QMOM rule.  A parent of length `L` contributes
`β(L) [6/(k+3)-1] L^k` to moment `k`; consequently the source preserves
`M₃` and adds one net particle to `M₀` per breakage event.
"""
function breakage_moment_source(breakagefunction::AbstractBreakageFunction,
                                parameters,
                                quadrature::QMOMQuadrature,
                                maximum_order::Integer)
    maximum_order >= 0 || throw(ArgumentError("maximum_order must be nonnegative."))
    supported = breakagefunction isa nobreakage ||
                breakagefunction isa breakage_empirical ||
                breakagefunction isa breakage_uniform
    supported || throw(ArgumentError("Breakage function $(typeof(breakagefunction)) " *
                                    "has no validated QMOM moment closure."))

    breakagefunction isa nobreakage &&
        return zeros(promote_type(eltype(quadrature.nodes),
                                  eltype(quadrature.weights),
                                  eltype(parameters)), maximum_order + 1)
    isempty(quadrature) &&
        return zeros(promote_type(eltype(quadrature.nodes),
                                  eltype(quadrature.weights),
                                  eltype(parameters)), maximum_order + 1)

    named_parameters = _named_params(breakagefunction, parameters)
    source_type = promote_type(eltype(quadrature.nodes),
                               eltype(quadrature.weights),
                               eltype(parameters))
    source = zeros(source_type, maximum_order + 1)
    active_nodes = quadrature.active_nodes
    @inbounds for node_index in 1:active_nodes
        crystal_length = quadrature.nodes[node_index]
        event_rate = quadrature.weights[node_index] *
                     _breakage_frequency(breakagefunction,
                                         named_parameters,
                                         crystal_length)
        for moment_index in 0:maximum_order
            source[moment_index + 1] += event_rate *
                                        ((6 / (moment_index + 3) - 1) *
                                         crystal_length^moment_index)
        end
    end
    return source
end

"""Return the growth contribution to `M₀:M_max` for a QMOM rule."""
function _qmom_growth_moment_source(quadrature::QMOMQuadrature,
                                    growthfunction::AbstractGrowthFunction,
                                    growth_parameters,
                                    dissolutionfunction::AbstractDissolutionFunction,
                                    dissolution_parameters,
                                    problem::CrystallisationProblem,
                                    state,
                                    time,
                                    nucleation_rate_value,
                                    maximum_order::Integer)
    source_type = promote_type(eltype(quadrature.nodes),
                               eltype(quadrature.weights),
                               typeof(nucleation_rate_value),
                               eltype(growth_parameters),
                               eltype(dissolution_parameters))
    source = zeros(source_type, maximum_order + 1)
    source[1] = nucleation_rate_value
    isempty(quadrature) && return source

    @inbounds for moment_index in 1:maximum_order
        growth_integral = zero(source_type)
        for node_index in 1:quadrature.active_nodes
            crystal_length = quadrature.nodes[node_index]
            growth_rate_value = net_growth_rate_at_length(
                growthfunction,
                growth_parameters,
                dissolutionfunction,
                dissolution_parameters,
                problem,
                state,
                time,
                crystal_length)
            growth_integral += quadrature.weights[node_index] *
                               crystal_length^(moment_index - 1) * growth_rate_value
        end
        source[moment_index + 1] = moment_index * growth_integral
    end
    return source
end

"""Exact scalar-growth-rate moment source without a moment inversion.

For a size-independent growth rate, `Iₖ = G Mₖ`; retaining this direct path is
important for both speed and realizability because an RK stage of the linear
moment equations need not itself be a positive point measure even though the
accepted solution is.  A quadrature is still reconstructed for output and for
the size-dependent source adapters.
"""
function _qmom_scalar_growth_moment_source(moment_values::AbstractVector,
                                           scalar_growth_rate,
                                           nucleation_rate_value,
                                           maximum_order::Integer)
    source_type = promote_type(eltype(moment_values), typeof(scalar_growth_rate),
                               typeof(nucleation_rate_value))
    source = zeros(source_type, maximum_order + 1)
    source[1] = nucleation_rate_value
    @inbounds for moment_index in 1:maximum_order
        source[moment_index + 1] = moment_index * scalar_growth_rate *
                                   moment_values[moment_index]
    end
    return source
end

@inline function _qmom_volume_rate(quadrature::QMOMQuadrature,
                                   growthfunction::AbstractGrowthFunction,
                                   growth_parameters,
                                   dissolutionfunction::AbstractDissolutionFunction,
                                   dissolution_parameters,
                                   problem::CrystallisationProblem,
                                   state,
                                   time)
    volume_rate = zero(promote_type(eltype(quadrature.nodes),
                                    eltype(quadrature.weights),
                                    eltype(growth_parameters),
                                    eltype(dissolution_parameters)))
    @inbounds for node_index in 1:quadrature.active_nodes
        crystal_length = quadrature.nodes[node_index]
        volume_rate += quadrature.weights[node_index] * crystal_length^2 *
                       net_growth_rate_at_length(
                           growthfunction,
                           growth_parameters,
                           dissolutionfunction,
                           dissolution_parameters,
                           problem,
                           state,
                           time,
                           crystal_length)
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
                                   state,
                                   time,
                                   scalar_growth_rate,
                                   quadrature_volume_rate)
    names = propertynames(problem.initial_solvent_state)
    concentration_position = findfirst(==(Symbol(:concentration)), names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    solvent_count = length(names)
    concentration_rate = -problem.crystal_density * problem.volume_shape_factor * quadrature_volume_rate

    if problem.solvent_dynamics isa DefaultSolventDynamics
        return ntuple(index -> index == concentration_position ? concentration_rate :
                      zero(concentration_rate), Val(solvent_count))
    end

    base_rates = _solvent_derivatives(problem, state, time, scalar_growth_rate)
    return ntuple(index -> index == concentration_position ? concentration_rate :
                  base_rates[index], Val(solvent_count))
end

function _qmom_extinction_callback(problem::CrystallisationProblem)
    dissolution_enabled = !(problem.kinetics_dissolutionfunction isa nodissolution) ||
                          (problem.kinetics_growthfunction isa
                           AbstractFPScalarDissolutionFunction)
    dissolution_enabled ||
        return nothing
    n_moments = moment_count(problem.solver)
    n_moments >= 4 || return nothing

    names = propertynames(problem.initial_solvent_state)
    concentration_position = findfirst(==(Symbol(:concentration)), names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    concentration_index = n_moments + concentration_position
    state_count = n_moments + length(names)
    threshold = _validate_solid_mass_concentration_threshold(problem)
    condition = (state, time, integrator) -> problem.crystal_density * problem.volume_shape_factor * state[4] - threshold
    affect! = integrator -> nothing
    affect_neg! = integrator -> begin
        state = integrator.u
        residual_mass = problem.crystal_density * problem.volume_shape_factor * state[4]
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

function _validate_qmom_problem(problem::CrystallisationProblem)
    _validate_qmom_solver(problem.solver)
    growthfunction = problem.kinetics_growthfunction
    (growthfunction isa AbstractFPScalarGrowthFunction ||
     growthfunction isa AbstractFPScalarDissolutionFunction) ||
        throw(ArgumentError("QMOM requires a scalar growth function; " *
                            "length-dependent kinetics are supported by FiniteVol/WENO only."))
    (growthfunction isa AbstractFPLengthGrowthFunction ||
     growthfunction isa AbstractFPLengthDissolutionFunction) &&
        throw(ArgumentError("$(typeof(growthfunction)) is length-dependent and is not supported by QMOM; " *
                            "use FiniteVol or WENO."))

    dissolutionfunction = problem.kinetics_dissolutionfunction
    dissolutionfunction isa AbstractFPLengthDissolutionFunction &&
        throw(ArgumentError("$(typeof(dissolutionfunction)) is length-dependent and is not " *
                            "supported by QMOM; use FiniteVol or WENO."))
    dissolutionfunction isa AbstractFPScalarDissolutionFunction ||
        throw(ArgumentError("QMOM requires an AbstractFPScalarDissolutionFunction or " *
                            "nodissolution()."))

    aggregationfunction = problem.kinetics_aggregationfunction
    (aggregationfunction isa noaggregation ||
     aggregationfunction isa aggr_scalar ||
     aggregationfunction isa aggr_linear ||
     aggregationfunction isa aggr_linearvol ||
     aggregationfunction isa aggr_avg) ||
        throw(ArgumentError("Aggregation function $(typeof(aggregationfunction)) " *
                            "has no validated QMOM moment closure."))

    breakagefunction = problem.kinetics_breakagefunction
    (breakagefunction isa nobreakage ||
     breakagefunction isa breakage_empirical ||
     breakagefunction isa breakage_uniform) ||
        throw(ArgumentError("Breakage function $(typeof(breakagefunction)) " *
                            "has no validated QMOM moment closure."))
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
function _qmom_stage_quadrature(moment_values::AbstractVector, solver::QMOM)
    try
        return invert_moments(moment_values, solver)
    catch error
        first_moment = moment_values[1]
        first_moment > solver.empty_population_tolerance || rethrow()
        second_moment = moment_values[2]
        isfinite(second_moment) || rethrow()
        representative_length = second_moment / first_moment
        isfinite(representative_length) &&
            representative_length >= solver.minimum_size || rethrow()
        diagnostic = QMOMInversionDiagnostics(:stage_deflated,
                                              1,
                                              representative_length,
                                              first_moment,
                                              zero(first_moment),
                                              zero(first_moment))
        return QMOMQuadrature([representative_length],
                              [first_moment],
                              1,
                              diagnostic)
    end
end

"""Allocation-free scalar QMOM3 RHS for the ordinary concentration path.

When growth is size independent and there are no binary sources, the moment
equations are closed analytically and do not need a quadrature reconstruction.
This specialized six-moment/seven-state form also keeps the derivative in an
`SVector`, matching the optimized MoM path.
"""
@inline function _qmom_scalar_model_3(problem::CrystallisationProblem,
                                      state,
                                      parameters,
                                      time)
    scalar_growth_rate = net_growth_rate(problem.kinetics_growthfunction,
                                          parameters.gr,
                                          problem.kinetics_dissolutionfunction,
                                          parameters.diss,
                                          problem,
                                          state,
                                          time)
    nucleation_rate_value = nucleationrate(problem.kinetics_nucleationfunction,
                                           parameters.nucl,
                                           problem,
                                           state,
                                           time)
    concentration_rate = -3 * problem.volume_shape_factor * problem.crystal_density * scalar_growth_rate * state[3]
    return SVector(nucleation_rate_value,
                   scalar_growth_rate * state[1],
                   2 * scalar_growth_rate * state[2],
                   3 * scalar_growth_rate * state[3],
                   4 * scalar_growth_rate * state[4],
                   5 * scalar_growth_rate * state[5],
                   concentration_rate)
end

"""Construct the QMOM ODE problem and selected time-stepping algorithm."""
function crystallisation_odeproblem(problem::CrystallisationProblem{NuclF, GrF, BrF,
                                                                    AggF, QMOM,
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
    _validate_qmom_problem(problem)
    length(saveat) >= 2 ||
        throw(ArgumentError("saveat must contain at least two time points."))
    n_moments = moment_count(problem.solver)
    n_solvent = length(propertynames(problem.initial_solvent_state))
    n_states = n_moments + n_solvent
    n_states == length(_get_initial_state(problem)) ||
        throw(ArgumentError("QMOM initial_state must contain $n_moments raw moments " *
                            "followed by $n_solvent solvent-state values."))
    # Keep user-provided non-realizable data distinguishable from transient
    # explicit-RK stages.  Empty states are valid and need no inversion.
    initial_moments = @view _get_initial_state(problem)[1:n_moments]
    initial_moments[1] > problem.solver.empty_population_tolerance &&
        invert_moments(initial_moments, problem.solver)

    no_binary_sources = problem.kinetics_aggregationfunction isa noaggregation &&
                        problem.kinetics_breakagefunction isa nobreakage
    fast_scalar_path = no_binary_sources && n_moments == 6 && n_solvent == 1 &&
                       propertynames(problem.initial_solvent_state) == (:concentration,) &&
                       problem.solvent_dynamics isa DefaultSolventDynamics

    if fast_scalar_path
        QMOM_model = (state, parameters, time) ->
            _qmom_scalar_model_3(problem, state, parameters, time)
    else
        function QMOM_model(state, parameters, time)
            moment_values = @view state[1:n_moments]
            scalar_growth_rate = net_growth_rate(problem.kinetics_growthfunction,
                                                 parameters.gr,
                                                 problem.kinetics_dissolutionfunction,
                                                 parameters.diss,
                                                 problem,
                                                 state,
                                                 time)
            nucleation_rate_value = nucleationrate(problem.kinetics_nucleationfunction,
                                                   parameters.nucl,
                                                   problem,
                                                   state,
                                                   time)
            if no_binary_sources
                moment_derivatives = _qmom_scalar_growth_moment_source(
                    moment_values,
                    scalar_growth_rate,
                    nucleation_rate_value,
                    moment_order(problem.solver))
                volume_rate = 3 * scalar_growth_rate * moment_values[3]
            else
                reconstructed_rule = _qmom_stage_quadrature(moment_values,
                                                             problem.solver)
                moment_derivatives = _qmom_growth_moment_source(
                    reconstructed_rule,
                    problem.kinetics_growthfunction,
                    parameters.gr,
                    problem.kinetics_dissolutionfunction,
                    parameters.diss,
                    problem,
                    state,
                    time,
                    nucleation_rate_value,
                    moment_order(problem.solver))
                moment_derivatives .+= aggregation_moment_source(
                    problem.kinetics_aggregationfunction,
                    parameters.agg,
                    reconstructed_rule,
                    moment_order(problem.solver);
                    shape_factor = problem.volume_shape_factor)
                moment_derivatives .+= breakage_moment_source(
                    problem.kinetics_breakagefunction,
                    parameters.br,
                    reconstructed_rule,
                    moment_order(problem.solver))
                volume_rate = _qmom_volume_rate(reconstructed_rule,
                                                problem.kinetics_growthfunction,
                                                parameters.gr,
                                                problem.kinetics_dissolutionfunction,
                                                parameters.diss,
                                                problem,
                                                state,
                                                time)
            end

            solvent_derivatives = _qmom_solvent_derivatives(problem,
                                                            state,
                                                            time,
                                                            scalar_growth_rate,
                                                            volume_rate)
            return SVector(ntuple(Val(n_states)) do state_index
                state_index <= n_moments ? moment_derivatives[state_index] :
                solvent_derivatives[state_index - n_moments]
            end)
        end
    end

    parameters = ComponentArray(; nucl = problem.parameterset_nucleation,
                                gr = problem.parameterset_growth,
                                diss = problem.parameterset_dissolution,
                                agg = problem.parameterset_aggregation,
                                br = problem.parameterset_breakage)
    element_type = promote_type(eltype(problem.parameterset_nucleation),
                                eltype(problem.parameterset_growth),
                                eltype(problem.parameterset_dissolution),
                                eltype(problem.parameterset_aggregation),
                                eltype(problem.parameterset_breakage),
                                Float64)
    initial_values = element_type.(_get_initial_state(problem))
    initial_state = SVector(ntuple(index -> initial_values[index], Val(n_states)))
    ode_problem = ODEProblem(QMOM_model,
                             initial_state,
                             (saveat[1], saveat[end]),
                             parameters;
                             callback = _qmom_extinction_callback(problem))
    time_step_solver = _resolve_timestepping_algorithm(problem.solver, :tsit5)
    return ode_problem, time_step_solver
end

function _wrap_solution(problem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                          QMOM, NuP, GrP, BrP,
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
    n_moments = moment_count(problem.solver)
    n_solvent = length(propertynames(problem.initial_solvent_state))
    moment_matrix = Array(solution[1:n_moments, :])
    time_points = collect(solution.t)
    n_time_points = length(time_points)
    n_nodes = problem.solver.nquadrature
    element_type = eltype(moment_matrix)
    node_matrix = zeros(element_type, n_nodes, n_time_points)
    weight_matrix = zeros(element_type, n_nodes, n_time_points)
    diagnostics = Vector{Any}(undef, n_time_points)

    for time_index in 1:n_time_points
        reconstructed_rule = invert_moments(view(moment_matrix, :, time_index),
                                             problem.solver)
        active_nodes = reconstructed_rule.active_nodes
        diagnostics[time_index] = reconstructed_rule.diagnostics
        if active_nodes > 0
            node_matrix[1:active_nodes, time_index] .= reconstructed_rule.nodes
            weight_matrix[1:active_nodes, time_index] .= reconstructed_rule.weights
        end
    end

    d10 = [_safe_moment_size_ratio(moment_matrix[2, index],
                                   moment_matrix[1, index], moment_matrix[1, index])
           for index in 1:n_time_points]
    d32 = [_safe_moment_size_ratio(moment_matrix[4, index],
                                   moment_matrix[3, index], moment_matrix[1, index])
           for index in 1:n_time_points]
    d43 = n_moments >= 5 ?
          [_safe_moment_size_ratio(moment_matrix[5, index],
                                   moment_matrix[4, index], moment_matrix[1, index])
           for index in 1:n_time_points] :
          zeros(element_type, n_time_points)
    moment2 = collect(@view moment_matrix[3, :])
    solvent_solution_state = _solvent_solution_state(problem, solution)
    final_state = collect(solution[:, end])
    successful = OrdinaryDiffEq.SciMLBase.successful_retcode(solution.retcode)

    return CrystallisationQMOMSolution(time_points,
                                       solvent_solution_state.concentration,
                                       moment_matrix,
                                       node_matrix,
                                       weight_matrix,
                                       d10,
                                       d32,
                                       d43,
                                       moment2,
                                       solvent_solution_state,
                                       final_state,
                                       solution.stats,
                                       successful,
                                       diagnostics)
end

function _simulatecrystallisation(problem::CrystallisationProblem{NuclF, GrF, BrF,
                                                                   AggF, QMOM,
                                                                   NuP, GrP, BrP,
                                                                   AggP, TP},
                                  saveat)::CrystallisationQMOMSolution where {
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
    # Raw moments in metres have dimensions spanning 30 or more orders of
    # magnitude.  A scalar absolute tolerance would effectively freeze the
    # high moments (and immediately drive an otherwise valid rule outside the
    # realizable cone).  Scale each moment tolerance by M₀ L_scale^k while
    # retaining the configured absolute tolerance for the solvent variables.
    initial_state = _get_initial_state(problem)
    n_moments = moment_count(problem.solver)
    moment_zero = abs(initial_state[1])
    moment_tolerances = [max(problem.solver.abstol *
                             max(moment_zero * problem.solver.coordinate_scale^index,
                                 eps(Float64) * problem.solver.coordinate_scale^index),
                             eps(Float64) * problem.solver.coordinate_scale^index)
                         for index in 0:(n_moments - 1)]
    solvent_tolerances = fill(problem.solver.abstol,
                              length(propertynames(problem.initial_solvent_state)))
    ode_solution = solve(ode_problem,
                         time_step_solver;
                         saveat = saveat,
                         reltol = problem.solver.reltol,
                         abstol = vcat(moment_tolerances, solvent_tolerances),
                         dense = false,
                         alg_hints = [:stiff],
                         maxiters = CRISTOOL_MAX_SOLVER_ITERS)
    return _wrap_solution(problem, ode_solution)
end

"""Reconstruct a saved QMOM rule at one trajectory index."""
function quadrature(solution::CrystallisationQMOMSolution, time_index::Integer)
    1 <= time_index <= length(solution.time) ||
        throw(BoundsError(solution.time, time_index))
    diagnostic = solution.inversion_diagnostics[time_index]
    active_nodes = diagnostic.active_nodes
    return QMOMQuadrature(collect(view(solution.quadrature_nodes, 1:active_nodes,
                                    time_index)),
                          collect(view(solution.quadrature_weights, 1:active_nodes,
                                       time_index)),
                          active_nodes,
                          diagnostic)
end

quadrature(solution::CrystallisationQMOMSolution) =
    [quadrature(solution, index) for index in eachindex(solution.time)]
qmom_quadrature(solution::CrystallisationQMOMSolution, time_index::Integer) =
    quadrature(solution, time_index)

time(solution::CrystallisationQMOMSolution) = solution.time
state_vars(solution::CrystallisationQMOMSolution) =
    merge(solution.solvent_state,
          (; moments = solution.moments,
             quadrature_nodes = solution.quadrature_nodes,
             quadrature_weights = solution.quadrature_weights))
size_metrics(solution::CrystallisationQMOMSolution) =
    (; d10 = solution.d10, d32 = solution.d32, d43 = solution.d43,
       moment2 = solution.moment2)
get_characteristic_size(solution::CrystallisationQMOMSolution) = solution.d43[end]
_size_trajectory(solution::CrystallisationQMOMSolution) = solution.d43

function getmomentsizes(problem::CrystallisationProblem,
                        solution::CrystallisationQMOMSolution)
    return size_metrics(solution)
end
