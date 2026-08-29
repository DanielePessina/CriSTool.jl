### One-pass Runner functions
"""
    runsimulation(parameters, nucleationfunction, growthfunction,
                  aggregationfunction, breakagefunction, initialconc;
                  save_idx = 0:5.0:480.0,
                  solver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                  initial_state = nothing)

Simulate batch crystallisation for a given set of kinetic models.

# Arguments
- `parameters::AbstractArray{<:Real}`: concatenated parameter vector
  `[p_ν; p_g; p_agg; p_br]` whose length must equal
  `nucleationfunction.nparams + growthfunction.nparams +
  aggregationfunction.nparams + breakagefunction.nparams`.
- `nucleationfunction`, `growthfunction`, `aggregationfunction`,
  `breakagefunction`: kinetic models describing nucleation, growth,
  aggregation and breakage respectively.
- `initialconc::Float64`: initial solute concentration.
- `save_idx::AbstractVector{<:Real}`: times (in minutes) at which the
  solution is saved. The number of time steps is `length(save_idx)`.
- `solver::AbstractSolver`: numerical solver (finite volume or method of
  moments).
- `initial_state::AbstractVector`: optional initial number density of
  length equal to the solver mesh size when using a discretised solver.

# Returns
A tuple `(problem, solution)` where `problem` is a
`CrystallisationProblem` and `solution` is either
`CrystallisationFVSolution` or `CrystallisationMoMSolution` depending on
`solver`.
"""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractNucleationFunction,
                       growthfunction::AbstractGrowthFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction,
                       initialconc::Float64;
                       save_idx = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractVector} = nothing) where {TPara <: Real}
    flat_parameters = vec(parameters)
    expected_nparams = nucleationfunction.nparams + growthfunction.nparams +
                       aggregationfunction.nparams + breakagefunction.nparams
    if length(flat_parameters) != expected_nparams
        throw(ArgumentError("Parameter vector has length $(length(flat_parameters)) but expected $expected_nparams " *
                            "(nucleation: $(nucleationfunction.nparams), growth: $(growthfunction.nparams), " *
                            "aggregation: $(aggregationfunction.nparams), breakage: $(breakagefunction.nparams))"))
    end

    structured_parameters = ComponentArray(flat_parameters,
                                           paramaxis(nucleationfunction,
                                                     growthfunction,
                                                     aggregationfunction,
                                                     breakagefunction))
    return runsimulation(structured_parameters;
                         nucl = nucleationfunction,
                         gr = growthfunction,
                         agg = aggregationfunction,
                         br = breakagefunction,
                         solver = solver,
                         initial_concentration = initialconc,
                         initial_state = initial_state,
                         save_idx = save_idx)
end
"""
    runsimulation(nucleationfunction::AbstractDDNucleationFunction,
                  growthfunction::AbstractDDGrowthFunction, noaggregation, nobreakage,
                  initialconc; ...) -> (problem, solution)

Simulate with data-driven kinetics and no aggregation/breakage (no parameters needed).

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(nucleationfunction::AbstractDDNucleationFunction,
                       growthfunction::AbstractDDGrowthFunction,
                       aggregationfunction::noaggregation,
                       breakagefunction::nobreakage,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing) where {S <:
                                                                                              AbstractArray{<:Real}}

    return runsimulation(Float64[],
                         nucleationfunction,
                         growthfunction,
                         aggregationfunction,
                         breakagefunction,
                         initialconc;
                         save_idx = save_idx,
                         solver = solver,
                         initial_state = initial_state)
end
"""
    runsimulation(parameters, nucleationfunction::AbstractFPNucleationFunction,
                  growthfunction::AbstractFPGrowthFunction, initialconc; ...) -> (problem, solution)

Simplified interface for nucleation and growth only (no aggregation/breakage).

Calls the full runsimulation with noaggregation() and nobreakage().

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(parameters::AbstractVector{TPara},
                       nucleationfunction::AbstractFPNucleationFunction,
                       growthfunction::AbstractFPGrowthFunction,
                       initialconc::Float64;
                       save_idx = 0:5.0:480.0,
                       solver::AbstractSolver = MoM(),
                       initial_state::Union{Nothing, AbstractVector} = nothing) where {TPara <: Real}
    return runsimulation(parameters,
                         nucleationfunction,
                         growthfunction,
                         noaggregation(),
                         nobreakage(),
                         initialconc;
                         save_idx = save_idx,
                         solver = solver,
                         initial_state = initial_state)
end

"""
    runsimulation(parameters::ComponentArray; nucl, gr, agg, br, solver,
                  initial_concentration = 18.0, initial_state = nothing,
                  save_idx = 0:5.0:480.0, cry_kwargs...)

Core kwarg-form entry. Expects `parameters` to be a ComponentArray laid out
with the top-level axis built by `paramaxis(nucl, gr, agg, br)` — that is, with
named slices `parameters.nucl`, `parameters.gr`, `parameters.agg`, `parameters.br`.

Use this when you already hold a structured parameter vector (e.g. from an
optimiser that supports ComponentArrays). For a flat `Vector{Float64}`, call
the AbstractVector overload — it wraps the input here.
"""
function runsimulation(parameters::ComponentArrays.ComponentArray;
                       nucl::AbstractNucleationFunction = noaggregation(),
                       gr::AbstractGrowthFunction = noaggregation(),
                       agg::AbstractAggregationFunction = noaggregation(),
                       br::AbstractBreakageFunction = nobreakage(),
                       solver::AbstractSolver = FiniteVol(meshsize = 500, lmax = 50e-6),
                       initial_concentration = 18.0,
                       initial_state::Union{Nothing, AbstractVector} = nothing,
                       save_idx = 0:5.0:480.0,
                       cry_kwargs...)
    cry = CrystallisationProblem(; kinetics_nucleationfunction = nucl,
                                 kinetics_growthfunction = gr,
                                 kinetics_aggregationfunction = agg,
                                 kinetics_breakagefunction = br,
                                 parameterset_nucleation = parameters.nucl,
                                 parameterset_growth = parameters.gr,
                                 parameterset_aggregation = parameters.agg,
                                 parameterset_breakage = parameters.br,
                                 solver = solver,
                                 initial_concentration,
                                 initial_state,
                                 cry_kwargs...)

    return cry, _simulatecrystallisation(cry, save_idx)
end

"""
    runsimulation(parameters::AbstractVector; nucl, gr, agg, br, ...)

Backward-compatible wrapper for callers (optimisers, scripts) that hand in a
flat parameter vector. Validates length, builds a ComponentArray view with the
composite axis from the kinetic models, and forwards to the ComponentArray
core. AD types (e.g. `Vector{Dual}`) flow through unchanged.
"""
function runsimulation(parameters::AbstractVector;
                       nucl::AbstractNucleationFunction = noaggregation(),
                       gr::AbstractGrowthFunction = noaggregation(),
                       agg::AbstractAggregationFunction = noaggregation(),
                       br::AbstractBreakageFunction = nobreakage(),
                       kwargs...)
    nν, ng, na, nb = nucl.nparams, gr.nparams, agg.nparams, br.nparams
    expected_nparams = nν + ng + na + nb
    if length(parameters) != expected_nparams
        throw(ArgumentError("Parameter vector has length $(length(parameters)) but expected $expected_nparams " *
                            "(nucleation: $nν, growth: $ng, aggregation: $na, breakage: $nb)"))
    end
    p = ComponentArray(parameters, paramaxis(nucl, gr, agg, br))
    return runsimulation(p; nucl = nucl, gr = gr, agg = agg, br = br, kwargs...)
end
