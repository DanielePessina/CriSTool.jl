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

@inline _named_params(model, p::AbstractVector) = ComponentArray(p, paramaxis(model))
@inline _named_params(_, p::ComponentArrays.ComponentArray) = p

"""`named_parameters(law, values)` provides supported named access for custom dispatch-based kinetics."""
@inline named_parameters(kinetic_law, parameter_values::AbstractVector) = _named_params(kinetic_law, parameter_values)

# Generic fallback for kinetics that haven't declared a custom paramaxis.
# Generates `θ1, θ2, ...` from `model.nparams`. Specific paramaxis methods
# (e.g. `paramaxis(::nucl_CNT) = ComponentArrays.Axis(ln_nucleation_prefactor=1,
# surface_energy=2)`) take precedence.
function paramaxis(model::Union{AbstractNucleationFunction, AbstractGrowthFunction,
                                AbstractDissolutionFunction,
                                AbstractAggregationFunction, AbstractBreakageFunction})
    n = model.nparams
    n == 0 && return ComponentArrays.Axis()
    return ComponentArrays.Axis(NamedTuple{Tuple(Symbol("θ", i) for i in 1:n)}(Tuple(1:n)))
end

"""Composite parameter axis for nucleation, growth, and dissolution models.

The independent dissolution block is inserted between growth and the binary
population-balance terms.  This is the canonical axis for new callers.
"""
function paramaxis(nucl::AbstractNucleationFunction,
                   gr::AbstractGrowthFunction,
                   diss::AbstractDissolutionFunction,
                   agg::AbstractAggregationFunction,
                   br::AbstractBreakageFunction)
    nν, ng, nd, na, nb = nucl.nparams, gr.nparams, diss.nparams,
                         agg.nparams, br.nparams
    return ComponentArrays.Axis(
        nucl = ViewAxis(1:nν, paramaxis(nucl)),
        gr = ViewAxis((nν + 1):(nν + ng), paramaxis(gr)),
        diss = ViewAxis((nν + ng + 1):(nν + ng + nd), paramaxis(diss)),
        agg = ViewAxis((nν + ng + nd + 1):(nν + ng + nd + na), paramaxis(agg)),
        br = ViewAxis((nν + ng + nd + na + 1):(nν + ng + nd + na + nb),
                      paramaxis(br)))
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
                                                    prob.kinetics_dissolutionfunction,
                                                    prob.kinetics_aggregationfunction,
                                                    prob.kinetics_breakagefunction)


"""
    crystal_state(problem, state) -> AbstractVector

Crystal-population part of a solver state: the named `n` component for
ComponentArray states (MoM moments), or the solver population range for flat
states, excluding optional reactor and solvent blocks.
"""
crystal_state(state::ComponentArrays.ComponentVector) = state.n
crystal_state(state::AbstractVector) = @view state[1:(end - 1)]
crystal_state(problem::CrystallisationProblem, state) =
    @view state[_population_state_range(problem)]
