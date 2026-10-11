## Problem struct

struct DefaultSolventDynamics end

function (::DefaultSolventDynamics)(problem, state, time, growth)
    solvent_names = propertynames(problem.initial_solvent_state)
    solvent_count = length(solvent_names)
    growth_type = growth isa AbstractVector ? eltype(growth) : typeof(growth)
    derivative_type = promote_type(eltype(state), growth_type)
    derivatives = MVector{solvent_count, derivative_type}(undef)
    fill!(derivatives, zero(derivative_type))
    concentration_position = findfirst(==(Symbol(:concentration)), solvent_names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))

    material_system = problem.model.system
    derivatives[concentration_position] = -3 * material_system.volume_shape_factor * material_system.crystal_density *
                                           _growth_surface_rate(problem.solver, state, growth)
    return SVector(derivatives)
end

"""
    _growth_surface_rate(solver, state, growth)

`∫ G(L) L² n(L) dL` for the solver's population state: `G µ2` for a scalar
growth rate on a moment or quadrature state, a mesh quadrature otherwise.

Mesh RHS evaluations do not use the mesh quadrature: with the default
dynamics they couple the solute to the exact discrete µ3 change of their face
fluxes (`_write_mesh_solvent_derivatives!`). The quadrature is reached only
when a custom `solvent_dynamics` calls `default_solvent_dynamics` on a mesh
state, where the fluxes are not available; it is first-order consistent but
not discretely conservative.
"""
_growth_surface_rate(solver::Union{MoM, QMOM}, state, growth::Number) = growth * state[3]
_growth_surface_rate(solver::DQMOM, state, growth::Number) =
    growth * _dqmom_physical_moment(solver, state, 2)
_growth_surface_rate(solver::AbstractMomentSolver, state, growth) =
    throw(ArgumentError("Moment-solver solvent coupling requires a scalar growth rate."))
_growth_surface_rate(solver::AbstractDiscretisedSolver, state, growth) =
    _concentration_depletion(solver.cell_dL, view(state, 1:solver.meshsize), growth,
                             solver.cell_centre)

const default_solvent_dynamics = DefaultSolventDynamics()

"""
    CrystallisationProblem(model, conditions; solver=MoM(), solvent_dynamics=default_solvent_dynamics,
                           solid_mass_concentration_threshold=1e-12)

One configured run: a `CrystallisationModel` under resolved `RunConditions`,
integrated by `solver`. Every value lives in exactly one place:

- `model::CrystallisationModel`: material system and bound kinetic laws, whose
  parameter values are the default kinetic parameters (`kinetic_parameter_vector(problem)`).
- `temp_profile::AbstractTemperature`: temperature profile of the run.
- `initial_solvent_state::NamedTuple`: initial solvent-phase values, including
  `:concentration` (read with `initial_concentration(problem)`).
- `initial_population`: initial solver population block, or `nothing` for an
  empty population. Reactor and solvent initial values are not duplicated here.
- `operation::AbstractCrystallisationOperation`: reactor operation.
- `solver::AbstractSolver`: numerical solver and its time-stepping options.
- `solvent_dynamics`: callable `(problem, state, time, growth) -> rates`
  returning one derivative per named solvent variable.
- `solid_mass_concentration_threshold::Float64`: positive solid mass-concentration
  threshold (kg/m³) used by scalar signed-dissolution moment solvers to reset an
  extinguished population.

Universal constants are the module constants `CRISTOOL_GAS_CONSTANT` and
`CRISTOOL_BOLTZMANN_CONSTANT`.
"""
struct CrystallisationProblem{Model, Temperature <: AbstractTemperature, Solvent <: NamedTuple,
                              Population <: Union{Nothing, Vector{Float64}},
                              Operation <: AbstractCrystallisationOperation,
                              Solver <: AbstractSolver, SolventDynamics}
    model::Model
    temp_profile::Temperature
    initial_solvent_state::Solvent
    initial_population::Population
    operation::Operation
    solver::Solver
    solvent_dynamics::SolventDynamics
    solid_mass_concentration_threshold::Float64
end

"""Copy a problem with a new initial population block and solvent initial values."""
_problem_with_initial_values(problem::CrystallisationProblem, population, solvent_values::NamedTuple) =
    CrystallisationProblem(problem.model, problem.temp_profile, solvent_values, population,
        problem.operation, problem.solver, problem.solvent_dynamics,
        problem.solid_mass_concentration_threshold)

"""
    _problem_with_initial_state(problem, state)

Copy a problem whose complete initial solver `state` is split into its single
sources: the population block and the named solvent values. Reactor values
belong to the operation and must match it.
"""
function _problem_with_initial_state(problem::CrystallisationProblem, state::AbstractVector)
    population_count = _population_state_count(problem.solver)
    solvent_range = _solvent_state_range(problem)
    length(state) == last(solvent_range) ||
        throw(ArgumentError("initial_state has length $(length(state)); " *
                            "expected $(last(solvent_range)) for $(typeof(problem.solver))."))
    reactor_values = state[(population_count + 1):(first(solvent_range) - 1)]
    reactor_values == _operation_initial_state(problem.operation) ||
        throw(ArgumentError("Reactor initial values belong to the operation; they cannot be replaced by a state vector."))
    solvent_names = propertynames(problem.initial_solvent_state)
    solvent_values = NamedTuple{solvent_names}(Tuple(state[solvent_range]))
    return _problem_with_initial_values(problem, Vector{Float64}(state[1:population_count]), solvent_values)
end

"""Indices of the solver's population block (raw moments, mesh or direct quadrature)."""
_population_state_range(problem::CrystallisationProblem) =
    1:_population_state_count(problem.solver)

"""Indices of the named solvent block following the solver population."""
function _solvent_state_range(problem::CrystallisationProblem)
    population_count = _population_state_count(problem.solver)
    reactor_count = _operation_state_count(problem.operation)
    return (population_count + reactor_count + 1):(population_count + reactor_count + length(problem.initial_solvent_state))
end

function _solvent_state_index(problem::CrystallisationProblem, name::Symbol)
    position = findfirst(==(name), propertynames(problem.initial_solvent_state))
    position === nothing && throw(ArgumentError("Unknown solvent-state variable :$name."))
    return first(_solvent_state_range(problem)) + position - 1
end

"""
    initial_concentration(problem::CrystallisationProblem) -> Real

Initial solute concentration (kg/m³) supplied to the solver.
"""
initial_concentration(problem::CrystallisationProblem) = problem.initial_solvent_state.concentration

"""
    kinetic_parameter_vector(problem::CrystallisationProblem) -> ComponentVector

The model's bound kinetic values as one vector on the composite
`paramaxis(problem)` (blocks `nucl`, `gr`, `diss`, `agg`, `br`). These are the
values `runsimulation(problem)` uses; `runsimulation(parameters, problem)`
replaces them.
"""
function kinetic_parameter_vector(problem::CrystallisationProblem)
    crystal_model = problem.model
    flat_kinetic_values = Float64[]
    for slot_name in _MODEL_KINETIC_SLOTS
        _append_bound_parameter_values!(flat_kinetic_values,
                                        getfield(crystal_model, slot_name).parameters)
    end
    return ComponentArray(flat_kinetic_values, paramaxis(problem))
end

"""
    _append_bound_parameter_values!(flat_kinetic_values, bound_parameters)

Append bound kinetic values depth-first in field order, the layout
`ComponentArray(; bound_parameters...)` gives: nested tuples recurse and
vector leaves contribute their elements.
"""
function _append_bound_parameter_values!(flat_kinetic_values::Vector{Float64},
                                         bound_parameters::NamedTuple)
    for bound_value in values(bound_parameters)
        _append_bound_parameter_values!(flat_kinetic_values, bound_value)
    end
    return flat_kinetic_values
end
_append_bound_parameter_values!(flat_kinetic_values::Vector{Float64}, bound_value::Real) =
    push!(flat_kinetic_values, bound_value)
_append_bound_parameter_values!(flat_kinetic_values::Vector{Float64}, bound_values::AbstractVector) =
    append!(flat_kinetic_values, bound_values)

function _validate_solid_mass_concentration_threshold(problem::CrystallisationProblem)
    threshold = problem.solid_mass_concentration_threshold
    isfinite(threshold) && threshold > 0.0 ||
        throw(ArgumentError("solid_mass_concentration_threshold must be finite and strictly positive."))
    return threshold
end

"""Whether the dissolution slot holds an active law, making crystal growth signed."""
_dissolution_enabled(problem::CrystallisationProblem) = !_is_null_kinetic(problem.model.dissolution.law)

"""Whether growth or dissolution returns a rate per crystal length."""
_uses_length_rates(problem::CrystallisationProblem) =
    _is_length_law(problem.model.growth.law) || _is_length_law(problem.model.dissolution.law)

"""Whether aggregation or breakage contributes a population source."""
_has_binary_sources(problem::CrystallisationProblem) =
    !(_is_null_kinetic(problem.model.aggregation.law) && _is_null_kinetic(problem.model.breakage.law))

"""Hook for model-specific growth-parameter validation."""
_validate_growth_parameters(model, parameters, problem) = nothing

"""Hook for model-specific dissolution-parameter validation."""
_validate_dissolution_parameters(model, parameters, problem) = nothing

"""
    _validate_kinetic_values(problem, kinetic_values)

Check a composite kinetic parameter vector (on `paramaxis(problem)`) against
the law-specific growth and dissolution domains.
"""
function _validate_kinetic_values(problem::CrystallisationProblem, kinetic_values)
    _validate_growth_parameters(problem.model.growth.law, kinetic_values.gr, problem)
    _validate_dissolution_parameters(problem.model.dissolution.law, kinetic_values.diss, problem)
    return kinetic_values
end

"""
    _validate_crystallisation_problem(problem)

Validate the structural contract shared by all runners.  This is deliberately
called once at the public simulation boundary; the hot RHS only evaluates the
already validated rate laws.
"""
function _validate_crystallisation_problem(problem::CrystallisationProblem)
    _validate_material_model(problem.model)
    _validate_solver_kinetics(problem.solver, problem)
    solvent_names = propertynames(problem.initial_solvent_state)
    :concentration in solvent_names ||
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    _validate_solid_mass_concentration_threshold(problem)

    for solvent_value in values(problem.initial_solvent_state)
        solvent_value isa Real && isfinite(solvent_value) ||
            throw(ArgumentError("initial solvent-state values must be finite real numbers."))
    end
    initial_concentration(problem) >= 0.0 ||
        throw(ArgumentError("initial solvent concentration must be nonnegative."))

    _validate_kinetic_values(problem, kinetic_parameter_vector(problem))

    if !isnothing(problem.initial_population)
        population_state_count = _population_state_count(problem.solver)
        length(problem.initial_population) == population_state_count ||
            throw(ArgumentError("initial population has length $(length(problem.initial_population)); " *
                                "expected $population_state_count for $(typeof(problem.solver))."))
        all(isfinite, problem.initial_population) ||
            throw(ArgumentError("initial population must contain only finite values."))
        _validate_initial_population(problem.solver, problem.initial_population)
    end

    _validate_operation(problem, problem.operation)

    saturation_value = saturation_concentration(problem, 0.0)
    isfinite(saturation_value) && saturation_value > 0.0 ||
        throw(ArgumentError("saturation concentration must be finite and strictly positive at t=0."))
    return problem
end

"""Solver-specific check of an explicit initial population block."""
_validate_initial_population(solver::AbstractSolver, population) = nothing
function _validate_initial_population(solver::AbstractDiscretisedSolver, population)
    any(value -> value < 0.0, population) &&
        throw(ArgumentError("initial number density must be nonnegative."))
    return nothing
end

"""
    saturation_concentration(prob::CrystallisationProblem, t) -> Real

Solubility (kg/m³) of the problem at time `t` under its temperature profile.
"""
saturation_concentration(prob::CrystallisationProblem, t) =
    saturation_concentration(prob.model.system.saturation_model, prob.temp_profile, t)

"""
    solvent_state(prob, state) -> NamedTuple

Read the named solvent-phase variables from the tail of a numerical solver
state. `initial_solvent_state` defines both their names and their ordering.
"""
function solvent_state(prob::CrystallisationProblem, state)
    names = propertynames(prob.initial_solvent_state)
    n_solvent = length(names)
    values = ntuple(index -> state[first(_solvent_state_range(prob)) + index - 1], Val(n_solvent))
    return NamedTuple{names}(values)
end

function _solvent_state_index(prob::CrystallisationProblem, state, name::Symbol)
    return _solvent_state_index(prob, name)
end

"""
    supersaturation(prob::CrystallisationProblem, state, t) -> Real

Supersaturation ratio from the named `:concentration` solvent-state variable.
"""
supersaturation(prob::CrystallisationProblem, state, t) =
    state[_solvent_state_index(prob, state, :concentration)] /
    saturation_concentration(prob, t)
