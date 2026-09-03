### One-pass Runner functions

function _validate_save_times(save_times)
    length(save_times) >= 1 ||
        throw(ArgumentError("save_idx must contain at least one time."))
    all(time -> time isa Real && isfinite(time) && time >= 0, save_times) ||
        throw(ArgumentError("save_idx must contain finite nonnegative SI times in seconds."))
    length(save_times) == 1 || all(diff(collect(save_times)) .> 0) ||
        throw(ArgumentError("save_idx must be strictly increasing."))
    return save_times
end

"""
    runsimulation(parameters, nucleationfunction, growthfunction,
                  aggregationfunction, breakagefunction, initialconc;
                  save_idx = 0:300.0:28_800.0,
                  solver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                  initial_state = nothing,
                  initial_crystals = nothing)

Simulate batch crystallisation for a given set of kinetic models.

# Arguments
- `parameters::AbstractArray{<:Real}`: concatenated parameter vector
  `[p_ν; p_g; p_diss; p_agg; p_br]` whose length must equal the sum of the
  selected model parameter counts. With the default `nodissolution()`, the
  dissolution block is empty.
- `nucleationfunction`, `growthfunction`, `aggregationfunction`,
  `breakagefunction`: kinetic models describing nucleation, growth,
  aggregation and breakage respectively.
- `initialconc::Float64`: initial solute concentration.
- `save_idx::AbstractVector{<:Real}`: times (in seconds) at which the
  solution is saved. The number of time steps is `length(save_idx)`.
- `solver::AbstractSolver`: numerical solver (finite volume or method of
  moments).
- `initial_state::AbstractVector`: optional complete solver state. The
  population variables are followed by the named solvent-state values.
- `initial_crystals::Union{Nothing,AbstractInitialCrystals}`: optional initial seed
  characteristics passed to `initial_state_from_characteristics`.

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
                       diss::AbstractDissolutionFunction = nodissolution(),
                       save_idx = 0:300.0:28_800.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractVector} = nothing,
                       initial_crystals::Union{Nothing, AbstractInitialCrystals} = nothing) where {TPara <: Real}
    flat_parameters = vec(parameters)
    expected_nparams = nucleationfunction.nparams + growthfunction.nparams +
                       diss.nparams + aggregationfunction.nparams + breakagefunction.nparams
    if length(flat_parameters) != expected_nparams
        throw(ArgumentError("Parameter vector has length $(length(flat_parameters)) but expected $expected_nparams " *
                            "(nucleation: $(nucleationfunction.nparams), growth: $(growthfunction.nparams), " *
                            "dissolution: $(diss.nparams), " *
                            "aggregation: $(aggregationfunction.nparams), breakage: $(breakagefunction.nparams))"))
    end

    structured_parameters = ComponentArray(flat_parameters,
                                           paramaxis(nucleationfunction,
                                                     growthfunction,
                                                     diss,
                                                     aggregationfunction,
                                                     breakagefunction))
    return runsimulation(structured_parameters;
                         nucl = nucleationfunction,
                         gr = growthfunction,
                         diss = diss,
                         agg = aggregationfunction,
                         br = breakagefunction,
                         solver = solver,
                         initial_concentration = initialconc,
                         initial_state = initial_state,
                         initial_crystals = initial_crystals,
                         save_idx = save_idx)
end

"""Canonical positional form including the independent dissolution model."""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractNucleationFunction,
                       growthfunction::AbstractGrowthFunction,
                       dissolutionfunction::AbstractDissolutionFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction,
                       initialconc::Float64;
                       kwargs...) where {TPara <: Real}
    return runsimulation(parameters, nucleationfunction, growthfunction,
                         aggregationfunction, breakagefunction, initialconc;
                         diss = dissolutionfunction, kwargs...)
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
                       diss::AbstractDissolutionFunction = nodissolution(),
                       save_idx::S = 0:300.0:28_800.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing,
                       initial_crystals::Union{Nothing, AbstractInitialCrystals} = nothing) where {S <:
                                                                                              AbstractArray{<:Real}}

    return runsimulation(Float64[],
                         nucleationfunction,
                         growthfunction,
                         aggregationfunction,
                         breakagefunction,
                         initialconc;
                         diss = diss,
                         save_idx = save_idx,
                         solver = solver,
                         initial_state = initial_state,
                         initial_crystals = initial_crystals)
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
                       diss::AbstractDissolutionFunction = nodissolution(),
                       save_idx = 0:300.0:28_800.0,
                       solver::AbstractSolver = MoM(),
                       initial_state::Union{Nothing, AbstractVector} = nothing,
                       initial_crystals::Union{Nothing, AbstractInitialCrystals} = nothing) where {TPara <: Real}
    return runsimulation(parameters,
                         nucleationfunction,
                         growthfunction,
                         noaggregation(),
                         nobreakage(),
                         initialconc;
                         diss = diss,
                         save_idx = save_idx,
                         solver = solver,
                         initial_state = initial_state,
                         initial_crystals = initial_crystals)
end

"""
    runsimulation(parameters::ComponentArray; nucl, gr, diss, agg, br, solver,
                  initial_concentration = 18.0, initial_state = nothing,
                  initial_crystals = nothing,
                  save_idx = 0:300.0:28_800.0, cry_kwargs...)

Core kwarg-form entry. Expects `parameters` to be a ComponentArray laid out
with the top-level axis built by `paramaxis(nucl, gr, diss, agg, br)` — that is,
with named slices `parameters.nucl`, `parameters.gr`, `parameters.diss`,
`parameters.agg`, and `parameters.br`.

Use this when you already hold a structured parameter vector (e.g. from an
optimiser that supports ComponentArrays). For a flat `Vector{Float64}`, call
the AbstractVector overload — it wraps the input here.
"""
function runsimulation(parameters::ComponentArrays.ComponentArray;
                       nucl::AbstractNucleationFunction = nucl_CNT(),
                       gr::AbstractGrowthFunction = growth_empirical(),
                       diss::AbstractDissolutionFunction = nodissolution(),
                       agg::AbstractAggregationFunction = noaggregation(),
                       br::AbstractBreakageFunction = nobreakage(),
                       solver::AbstractSolver = FiniteVol(meshsize = 500, lmax = 50e-6),
                       initial_concentration = 18.0,
                       initial_state::Union{Nothing, AbstractVector} = nothing,
                       initial_crystals::Union{Nothing, AbstractInitialCrystals} = nothing,
                       save_idx = 0:300.0:28_800.0,
                       cry_kwargs...)
    isnothing(initial_state) || isnothing(initial_crystals) ||
        throw(ArgumentError("Provide either initial_state or initial_crystals, not both."))
    cry = CrystallisationProblem(; kinetics_nucleationfunction = nucl,
                                 kinetics_growthfunction = gr,
                                 kinetics_dissolutionfunction = diss,
                                 kinetics_aggregationfunction = agg,
                                 kinetics_breakagefunction = br,
                                 parameterset_nucleation = parameters.nucl,
                                 parameterset_growth = parameters.gr,
                                 parameterset_dissolution = hasproperty(parameters, :diss) ?
                                     parameters.diss : Float64[],
                                 parameterset_aggregation = parameters.agg,
                                 parameterset_breakage = parameters.br,
                                 solver = solver,
                                 initial_concentration,
                                 initial_state,
                                 cry_kwargs...)

    if !isnothing(initial_crystals)
        generated_state = initial_state_from_characteristics(cry, initial_crystals)
        cry = _problem_with_initial_state(cry, generated_state)
    end

    _validate_crystallisation_problem(cry)

    return cry, _simulatecrystallisation(cry, _validate_save_times(save_idx))
end

"""
    runsimulation(parameters::AbstractVector; nucl, gr, agg, br, ...)

Flat-vector wrapper for callers (optimisers, scripts) that hand in numeric
parameters. Validates length, builds a ComponentArray view with the composite
axis from the kinetic models, and forwards to the ComponentArray core. AD types
(e.g. `Vector{Dual}`) flow through unchanged.
"""
function runsimulation(parameters::AbstractVector;
                       nucl::AbstractNucleationFunction = nucl_CNT(),
                       gr::AbstractGrowthFunction = growth_empirical(),
                       diss = nothing,
                       agg::AbstractAggregationFunction = noaggregation(),
                       br::AbstractBreakageFunction = nobreakage(),
                       kwargs...)
    if isnothing(diss)
        nν, ng, na, nb = nucl.nparams, gr.nparams, agg.nparams, br.nparams
        expected_nparams = nν + ng + na + nb
        length(parameters) == expected_nparams ||
            throw(ArgumentError("Parameter vector has length $(length(parameters)) but expected $expected_nparams " *
                                "(nucleation: $nν, growth: $ng, aggregation: $na, breakage: $nb)"))
        legacy_parameters = ComponentArray(parameters, paramaxis(nucl, gr, agg, br))
        return runsimulation(legacy_parameters; nucl = nucl, gr = gr,
                             diss = nodissolution(), agg = agg, br = br, kwargs...)
    end
    diss isa AbstractDissolutionFunction ||
        throw(ArgumentError("diss must be an AbstractDissolutionFunction or nothing."))
    nν, ng, nd, na, nb = nucl.nparams, gr.nparams, diss.nparams,
                         agg.nparams, br.nparams
    expected_nparams = nν + ng + nd + na + nb
    if length(parameters) != expected_nparams
        throw(ArgumentError("Parameter vector has length $(length(parameters)) but expected $expected_nparams " *
                            "(nucleation: $nν, growth: $ng, dissolution: $nd, " *
                            "aggregation: $na, breakage: $nb)"))
    end
    p = ComponentArray(parameters, paramaxis(nucl, gr, diss, agg, br))
    return runsimulation(p; nucl = nucl, gr = gr, diss = diss, agg = agg, br = br, kwargs...)
end
