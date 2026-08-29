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
fluxlimiter_ospre(r) = (1.5 * (r^2) + r) / (r^2 + r + 1)

@inline _named_params(model, p::AbstractVector) = ComponentArray(p, paramaxis(model))
@inline _named_params(_, p::ComponentArrays.ComponentArray) = p

# Generic fallback for kinetics that haven't declared a custom paramaxis.
# Generates `θ1, θ2, ...` from `model.nparams`. Specific paramaxis methods
# (e.g. `paramaxis(::nucl_CNT) = ComponentArrays.Axis(Aj=1, γ=2)`) take precedence.
function paramaxis(model::Union{AbstractNucleationFunction, AbstractGrowthFunction,
                                AbstractAggregationFunction, AbstractBreakageFunction})
    n = model.nparams
    n == 0 && return ComponentArrays.Axis()
    return ComponentArrays.Axis(NamedTuple{Tuple(Symbol("θ", i) for i in 1:n)}(Tuple(1:n)))
end

# Composite axis spanning the four kinetic families. Slot order (nucl, gr,
# agg, br) matches the legacy positional convention.
function paramaxis(nucl::AbstractNucleationFunction,
                   gr::AbstractGrowthFunction,
                   agg::AbstractAggregationFunction,
                   br::AbstractBreakageFunction)
    nν, ng, na, nb = nucl.nparams, gr.nparams, agg.nparams, br.nparams
    return ComponentArrays.Axis(nucl = ViewAxis(1:nν, paramaxis(nucl)),
                gr = ViewAxis((nν + 1):(nν + ng), paramaxis(gr)),
                agg = ViewAxis((nν + ng + 1):(nν + ng + na), paramaxis(agg)),
                br = ViewAxis((nν + ng + na + 1):(nν + ng + na + nb), paramaxis(br)))
end

paramaxis(prob::CrystallisationProblem) = paramaxis(prob.kinetics_nucleationfunction,
                                                    prob.kinetics_growthfunction,
                                                    prob.kinetics_aggregationfunction,
                                                    prob.kinetics_breakagefunction)


"""
    crystal_state(problem, state) -> AbstractVector

Crystal-population part of a solver state: the named `n` component for
ComponentArray states (MoM moments), or the state prefix before the named
solvent-state variables for flat discretised states (mesh densities).
"""
crystal_state(state::ComponentArrays.ComponentVector) = state.n
crystal_state(state::AbstractVector) = @view state[1:(end - 1)]
crystal_state(problem::CrystallisationProblem, state) =
    @view state[1:(end - length(propertynames(problem.initial_solvent_state)))]
