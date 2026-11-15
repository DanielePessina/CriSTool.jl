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
    CrystallisationProblem(model, conditions, tspan, p = <model values>;
                           solvent_dynamics=default_solvent_dynamics,
                           solid_mass_concentration_threshold=1e-12)

One configured run, independent of the numerical solver: a
`CrystallisationModel` under numeric `RunConditions` over the time span
`tspan = (t0, tf)` in seconds. Solve it with `solve(problem, solver; saveat)`;
the solver (`MoM()`, `QMOM()`, `FiniteVol()`, ...) is chosen at solve time, as
the algorithm is for an `ODEProblem`. Every value lives in exactly one place:

- `model::CrystallisationModel`: material system and bound kinetic laws.
- `conditions::RunConditions`: temperature, initial solvent values, seed and
  reactor operation. The seed is turned into a solver population at solve time.
- `tspan::Tuple{Float64, Float64}`: integration span (s); `t0 >= 0`, `tf >= t0`.
- `p`: kinetic parameters as one `ComponentVector` on `paramaxis(problem)`
  (blocks `nucl`, `gr`, `diss`, `agg`, `br`). Defaults to the model's bound
  values; `remake(problem; p = values)` replaces them, and its element type may
  be a dual number, so this is the differentiable entry point.
- `solvent_dynamics`: callable `(problem, state, time, growth) -> rates`
  returning one derivative per named solvent variable. It receives the
  solve-time problem, whose `state` layout depends on the solver.
- `solid_mass_concentration_threshold::Float64`: positive solid mass-concentration
  threshold (kg/m³) used by scalar signed-dissolution moment solvers to reset an
  extinguished population.

Universal constants are the module constants `CRISTOOL_GAS_CONSTANT` and
`CRISTOOL_BOLTZMANN_CONSTANT`.
"""
struct CrystallisationProblem{Model, Conditions <: RunConditions, Parameters <: AbstractVector,
                              SolventDynamics} <: SciMLBase.AbstractSciMLProblem
    model::Model
    conditions::Conditions
    tspan::Tuple{Float64, Float64}
    p::Parameters
    solvent_dynamics::SolventDynamics
    solid_mass_concentration_threshold::Float64
end

"""
    DiscretisedProblem(problem::CrystallisationProblem, solver::AbstractSolver)

Internal solve-time form of a `CrystallisationProblem` under one solver: the
run conditions resolved into the solver's state layout (the seed becomes the
`initial_population` block). Every solver-dependent function (state ranges,
right-hand sides, rate laws, solution reconstruction) takes this type; it is
built once per `solve`, or once per prepared run in inference.

- `temp_profile`, `initial_solvent_state`, `operation`: from the conditions.
- `initial_population`: initial solver population block, or `nothing` for an
  empty population. Reactor and solvent initial values are not duplicated here.
- `solver::AbstractSolver`: numerical solver and its time-stepping options.

Kinetic laws and `solvent_dynamics` receive this problem; read it through
`supersaturation(problem, state, t)`, `temperature(problem.temp_profile, t)`,
`solvent_state(problem, state)` and `problem.model`.
"""
struct DiscretisedProblem{Model, Temperature <: AbstractTemperature, Solvent <: NamedTuple,
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
_problem_with_initial_values(problem::DiscretisedProblem, population, solvent_values::NamedTuple) =
    DiscretisedProblem(problem.model, problem.temp_profile, solvent_values, population,
        problem.operation, problem.solver, problem.solvent_dynamics,
        problem.solid_mass_concentration_threshold)

"""
    _problem_with_initial_state(problem, state)

Copy a problem whose complete initial solver `state` is split into its single
sources: the population block and the named solvent values. Reactor values
belong to the operation and must match it.
"""
function _problem_with_initial_state(problem::DiscretisedProblem, state::AbstractVector)
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
_population_state_range(problem::DiscretisedProblem) =
    1:_population_state_count(problem.solver)

"""Indices of the named solvent block following the solver population."""
function _solvent_state_range(problem::DiscretisedProblem)
    population_count = _population_state_count(problem.solver)
    reactor_count = _operation_state_count(problem.operation)
    return (population_count + reactor_count + 1):(population_count + reactor_count + length(problem.initial_solvent_state))
end

function _solvent_state_index(problem::DiscretisedProblem, name::Symbol)
    position = findfirst(==(name), propertynames(problem.initial_solvent_state))
    position === nothing && throw(ArgumentError("Unknown solvent-state variable :$name."))
    return first(_solvent_state_range(problem)) + position - 1
end

"""Crystal solid density times the volume shape factor (kg/m³), the mass per unit third moment."""
_solid_mass_factor(problem::DiscretisedProblem) =
    problem.model.system.crystal_density * problem.model.system.volume_shape_factor

"""
    initial_concentration(problem) -> Real

Initial solute concentration (kg/m³) of the run.
"""
initial_concentration(problem::CrystallisationProblem) = problem.conditions.solvent.concentration
initial_concentration(problem::DiscretisedProblem) = problem.initial_solvent_state.concentration

"""
    kinetic_parameter_vector(problem::CrystallisationProblem) -> ComponentVector

The kinetic values `problem.p` the problem is solved with, on the composite
`paramaxis(problem)` (blocks `nucl`, `gr`, `diss`, `agg`, `br`). They are the
model's bound values unless replaced with `remake(problem; p = values)`.
"""
kinetic_parameter_vector(problem::CrystallisationProblem) = problem.p

"""
    _checked_tspan(tspan) -> Tuple{Float64, Float64}

A finite `(t0, tf)` span in seconds with `0 <= t0 <= tf`.
"""
function _checked_tspan(tspan::Tuple{Real, Real})
    start_time, end_time = Float64(first(tspan)), Float64(last(tspan))
    isfinite(start_time) && isfinite(end_time) && 0.0 <= start_time <= end_time ||
        throw(ArgumentError("tspan must be finite SI times (t0, tf) with 0 <= t0 <= tf; got $tspan."))
    return (start_time, end_time)
end
_checked_tspan(tspan) =
    throw(ArgumentError("tspan must be a (t0, tf) tuple of times in seconds; got $(typeof(tspan))."))

function _validate_solid_mass_concentration_threshold(threshold::Float64)
    isfinite(threshold) && threshold > 0.0 ||
        throw(ArgumentError("solid_mass_concentration_threshold must be finite and strictly positive."))
    return threshold
end

"""Whether the dissolution slot holds an active law, making crystal growth signed."""
_dissolution_enabled(problem::DiscretisedProblem) = !_is_null_kinetic(problem.model.dissolution.law)

"""Whether growth or dissolution returns a rate per crystal length."""
_uses_length_rates(problem::DiscretisedProblem) =
    _is_length_law(problem.model.growth.law) || _is_length_law(problem.model.dissolution.law)

"""Whether aggregation or breakage contributes a population source."""
_has_binary_sources(problem::DiscretisedProblem) =
    !(_is_null_kinetic(problem.model.aggregation.law) && _is_null_kinetic(problem.model.breakage.law))

"""Hook for model-specific growth-parameter validation."""
_validate_growth_parameters(model, parameters, problem) = nothing

"""Hook for model-specific dissolution-parameter validation."""
_validate_dissolution_parameters(model, parameters, problem) = nothing

"""
    _validate_kinetic_values(problem, kinetic_values)

Check a composite kinetic parameter vector (on `paramaxis(problem)`) against
the law-specific growth and dissolution domains. Some domains depend on the
solver mesh, so this runs on the solve-time problem.
"""
function _validate_kinetic_values(problem::DiscretisedProblem, kinetic_values)
    _validate_growth_parameters(problem.model.growth.law, kinetic_values.gr, problem)
    _validate_dissolution_parameters(problem.model.dissolution.law, kinetic_values.diss, problem)
    return kinetic_values
end

"""
    _validate_crystallisation_problem(problem::CrystallisationProblem)

Validate the solver-independent contract once, when the problem is built:
material model, numeric initial solvent values, the operation and the
solubility at `t0`. Solver-specific checks run in `DiscretisedProblem`.
"""
function _validate_crystallisation_problem(problem::CrystallisationProblem)
    _validate_material_model(problem.model)
    initial_solvent_state = problem.conditions.solvent
    :concentration in propertynames(initial_solvent_state) ||
        throw(ArgumentError("initial_solvent_state must define :concentration."))
    _validate_solid_mass_concentration_threshold(problem.solid_mass_concentration_threshold)

    for solvent_value in values(initial_solvent_state)
        solvent_value isa Real && isfinite(solvent_value) ||
            throw(ArgumentError("initial solvent-state values must be finite real numbers."))
    end
    initial_concentration(problem) >= 0.0 ||
        throw(ArgumentError("initial solvent concentration must be nonnegative."))
    _validate_operation(initial_solvent_state, problem.conditions.operation)

    saturation_value = saturation_concentration(problem, first(problem.tspan))
    isfinite(saturation_value) && saturation_value > 0.0 ||
        throw(ArgumentError("saturation concentration must be finite and strictly positive at t0."))
    return problem
end

"""
    _validate_discretised_problem(problem::DiscretisedProblem)

Solver-specific contract shared by all runners: the solver's kinetic
restrictions and any explicit initial population block. Called once when a
solve or prepared run is built; the hot RHS only evaluates the already
validated rate laws.
"""
function _validate_discretised_problem(problem::DiscretisedProblem)
    _validate_solver_kinetics(problem.solver, problem)
    _validate_initial_population_block(problem, problem.initial_population)
    return problem
end

_validate_initial_population_block(problem::DiscretisedProblem, ::Nothing) = nothing
function _validate_initial_population_block(problem::DiscretisedProblem, initial_population::Vector{Float64})
    population_state_count = _population_state_count(problem.solver)
    length(initial_population) == population_state_count ||
        throw(ArgumentError("initial population has length $(length(initial_population)); " *
                            "expected $population_state_count for $(typeof(problem.solver))."))
    all(isfinite, initial_population) ||
        throw(ArgumentError("initial population must contain only finite values."))
    _validate_initial_population(problem.solver, initial_population)
    return nothing
end

"""Solver-specific check of an explicit initial population block."""
_validate_initial_population(solver::AbstractSolver, population) = nothing
function _validate_initial_population(solver::AbstractDiscretisedSolver, population)
    any(value -> value < 0.0, population) &&
        throw(ArgumentError("initial number density must be nonnegative."))
    return nothing
end

"""
    saturation_concentration(prob, t) -> Real

Solubility (kg/m³) of the problem at time `t` under its temperature profile.
"""
saturation_concentration(prob::CrystallisationProblem, t) =
    saturation_concentration(prob.model.system.saturation_model, prob.conditions.temperature, t)
saturation_concentration(prob::DiscretisedProblem, t) =
    saturation_concentration(prob.model.system.saturation_model, prob.temp_profile, t)

"""
    solvent_state(prob, state) -> NamedTuple

Read the named solvent-phase variables from the tail of a numerical solver
state. `initial_solvent_state` defines both their names and their ordering.
"""
function solvent_state(prob::DiscretisedProblem, state)
    names = propertynames(prob.initial_solvent_state)
    n_solvent = length(names)
    values = ntuple(index -> state[first(_solvent_state_range(prob)) + index - 1], Val(n_solvent))
    return NamedTuple{names}(values)
end

"""
    supersaturation(prob, state, t) -> Real

Supersaturation ratio from the named `:concentration` solvent-state variable.
"""
supersaturation(prob::DiscretisedProblem, state, t) =
    state[_solvent_state_index(prob, :concentration)] /
    saturation_concentration(prob, t)
