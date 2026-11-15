"""
Shared parameter-axis and population-state interfaces for the kinetic and
population-balance implementations.
"""

#### Flux limiters for high-resolution finite volume schemes
"""
    fluxlimiter_ospre(r)

Ospre flux limiter for high-resolution schemes.
Computes a smoothness-based limiter value for numerical flux reconstruction.
"""
fluxlimiter_ospre(gradient_ratio) = gradient_ratio <= zero(gradient_ratio) ?
    zero(gradient_ratio) :
    1.5 * (gradient_ratio^2 + gradient_ratio) / (gradient_ratio^2 + gradient_ratio + 1)

"""
    _inactive_rate(parameters, rate_inputs...)

Zero rate of a kinetic law below its activation threshold, typed like the
active branch. A literal `0.0` would make the law return
`Union{Float64, Dual}` under ForwardDiff parameters, which type-destabilises
every solver right-hand side that calls it.
"""
@inline _inactive_rate(parameters, rate_inputs...) =
    zero(promote_type(eltype(parameters), map(_rate_input_type, rate_inputs)...))
_rate_input_type(rate_input::Type) = rate_input
_rate_input_type(rate_input) = typeof(rate_input)

@inline _named_params(model, p::AbstractVector) = ComponentArray(p, paramaxis(model))
@inline _named_params(_, p::ComponentArrays.ComponentArray) = p

"""`named_parameters(law, values)` provides supported named access for custom dispatch-based kinetics."""
@inline named_parameters(kinetic_law, parameter_values::AbstractVector) = _named_params(kinetic_law, parameter_values)

"""
    paramaxis(nucleation, growth, dissolution, aggregation, breakage)

Composite parameter axis with one nested block per kinetic slot, in the order
`nucl`, `gr`, `diss`, `agg`, `br`.
"""
function paramaxis(nucl::AbstractNucleationFunction,
                   gr::AbstractGrowthFunction,
                   diss::AbstractDissolutionFunction,
                   agg::AbstractAggregationFunction,
                   br::AbstractBreakageFunction)
    nν, ng, nd, na, nb = map(_parameter_count, (nucl, gr, diss, agg, br))
    return ComponentArrays.Axis(
        nucl = ViewAxis(1:nν, paramaxis(nucl)),
        gr = ViewAxis((nν + 1):(nν + ng), paramaxis(gr)),
        diss = ViewAxis((nν + ng + 1):(nν + ng + nd), paramaxis(diss)),
        agg = ViewAxis((nν + ng + nd + 1):(nν + ng + nd + na), paramaxis(agg)),
        br = ViewAxis((nν + ng + nd + na + 1):(nν + ng + nd + na + nb),
                      paramaxis(br)))
end

"""
    paramaxis(problem::CrystallisationProblem) -> Axis

Composite kinetic parameter axis of the problem's model. It does not depend on
the solver, so `problem.p` is laid out on it before any solve.
"""
paramaxis(prob::Union{CrystallisationProblem, DiscretisedProblem}) =
    paramaxis(prob.model.nucleation.law, prob.model.growth.law, prob.model.dissolution.law,
              prob.model.aggregation.law, prob.model.breakage.law)


"""
    crystal_state(problem, state) -> AbstractVector

Crystal-population part of a solver state: the named `n` component for
ComponentArray states (MoM moments), or the solver population range for flat
states, excluding optional reactor and solvent blocks.
"""
crystal_state(state::ComponentArrays.ComponentVector) = state.n
crystal_state(state::AbstractVector) = @view state[1:(end - 1)]
crystal_state(problem::DiscretisedProblem, state) =
    @view state[_population_state_range(problem)]
