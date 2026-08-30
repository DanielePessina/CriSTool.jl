## Problem struct

"""
    AbstractCrystallisationProblem

Abstract supertype for crystallization problem definitions.
"""
abstract type AbstractCrystallisationProblem end

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

    population_count = length(state) - solvent_count
    if problem.solver isa AbstractMomentSolver
        moment_order(problem.solver) >= 2 ||
            throw(ArgumentError("Moment solver must track at least the second raw moment."))
        growth isa Number ||
            throw(ArgumentError("Moment-solver solvent coupling requires a scalar growth rate."))
        depletion = 3 * problem.kv * problem.ρ * growth * state[3]
    else
        population = @view state[1:population_count]
        depletion = 3 * problem.kv * problem.ρ *
                    _concentration_depletion(problem.solver.cell_dL,
                                              population,
                                              growth,
                                              problem.solver.cell_centre)
    end
    derivatives[concentration_position] = -depletion
    return SVector(derivatives)
end

const default_solvent_dynamics = DefaultSolventDynamics()

"""
    CrystallisationProblem{NuF,GrF,BrF,AggF,solmethod,NuP,GrP,BrP,AggP} <: AbstractCrystallisationProblem

Main structure defining a crystallization problem with kinetics and solver specifications.

Type Parameters:
- `NuF<:AbstractNucleationFunction`: Type of nucleation function
- `GrF<:AbstractGrowthFunction`: Type of growth function
- `BrF<:AbstractBreakageFunction`: Type of breakage function
- `AggF<:AbstractAggregationFunction`: Type of aggregation function
- `solmethod<:AbstractSolver`: Type of solver
- `NuP,GrP,BrP,AggP<:AbstractVector{<:Real}`: Parameter vector types

Fields:
- `temp_profile::TP`: Temperature profile (see `AbstractTemperature`)
- `ρ::Float64`: Crystal density (kg/m³)
- `initial_concentration::Float64`: Initial solute concentration (kg/m³)
- `initial_solvent_state::SS`: Named initial values for solvent-phase variables;
  `:concentration` is required
- `solvent_dynamics::SD`: Callable `(problem, state, time, growth) -> rates`
  returning one derivative per named solvent variable
- `saturation_model::AbstractSolubilityModel`: Solubility model (default `lysozyme_solubility()`)
- `kv::Float64`: Volume shape factor
- `solid_volume_threshold::Float64`: Positive solid mass-concentration threshold
  used by scalar signed-dissolution moment solvers to reset an extinguished
  population (kg/m³)
- `molecular_volume::Float64`: Molecular volume (m³)
- `kinetics_nucleationfunction::NuF`: Nucleation function
- `kinetics_growthfunction::GrF`: Growth function
- `kinetics_dissolutionfunction::DissF`: Independent dissolution function
- `parameterset_nucleation::NuP`: Nucleation parameters
- `parameterset_growth::GrP`: Growth parameters
- `parameterset_dissolution::DissP`: Dissolution parameters
- `kinetics_breakagefunction::BrF`: Breakage function (default: nobreakage())
- `parameterset_breakage::BrP`: Breakage parameters (default: [0.0])
- `kinetics_aggregationfunction::AggF`: Aggregation function (default: noaggregation())
- `parameterset_aggregation::AggP`: Aggregation parameters (default: [0.0])
- `R::Float64`: Gas constant (J/mol/K)
- `kb::Float64`: Boltzmann constant (J/K)
- `solver::solmethod`: Numerical solver (holds time-stepping configuration).
"""
Base.@kwdef @concrete struct CrystallisationProblem{NuF <: AbstractNucleationFunction,
                                                    GrF <: AbstractGrowthFunction,
                                                    BrF <: AbstractBreakageFunction,
                                                    AggF <: AbstractAggregationFunction,
                                                    solmethod <: AbstractSolver,
                                                    NuP <: AbstractVector{<:Real},
                                                    GrP <: AbstractVector{<:Real},
                                                    BrP <: AbstractVector{<:Real},
                                                    AggP <: AbstractVector{<:Real},
                                                    TP <: AbstractTemperature,
                                                    SM <: AbstractSolubilityModel,
                                                    SS <: NamedTuple,
                                                    SD,
                                                    DissF <: AbstractDissolutionFunction,
                                                    DissP <: AbstractVector{<:Real}} <:
                             AbstractCrystallisationProblem

    # Operation
    temp_profile::TP = ConstantTemperature(273.15 + 20.0) # Default to 25°C

    # Loading
    loading::Float64 = 0.0

    # Solute
    ρ::Float64 = 1370.0
    initial_concentration::Float64 = 20.0
    saturation_model::SM = lysozyme_solubility()
    kv::Float64 = 0.81 #0.55
    solid_volume_threshold::Float64 = 1e-12
    molecular_volume::Float64 = 2.97e-26
    initial_solvent_state::SS = (; concentration = initial_concentration)
    solvent_dynamics::SD = default_solvent_dynamics

    # Chosen Kinetics
    kinetics_nucleationfunction::NuF = nucl_CNT()
    kinetics_growthfunction::GrF = growth_empirical()
    parameterset_nucleation::NuP = [1.0, 1.0]
    parameterset_growth::GrP = [1.0, 1.0]

    kinetics_breakagefunction::BrF = nobreakage()
    parameterset_breakage::BrP = [0.0]

    kinetics_aggregationfunction::AggF = noaggregation()
    parameterset_aggregation::AggP = [0.0]

    initial_state::Union{Nothing, AbstractVector{<:Real}} = nothing

    # Constants
    R::Float64 = 8.314
    kb::Float64 = 1.380649e-23

    solver::solmethod = MoM()

    # Dissolution is an independent signed crystal-growth-rate contribution. These
    # fields are deliberately appended after the legacy fields so existing
    # positional type signatures remain source-compatible (they wildcard
    # trailing type parameters).
    kinetics_dissolutionfunction::DissF = nodissolution()
    parameterset_dissolution::DissP = Float64[]

end

function _validate_solid_volume_threshold(problem::CrystallisationProblem)
    threshold = problem.solid_volume_threshold
    isfinite(threshold) && threshold > 0.0 ||
        throw(ArgumentError("solid_volume_threshold must be finite and strictly positive."))
    return threshold
end

"""Validate a kinetic parameter block against its model declaration."""
function _validate_parameter_block(model, parameters, label::Symbol)
    expected = model.nparams
    actual = length(parameters)
    actual == expected ||
        throw(ArgumentError("$label parameter block has length $actual, expected $expected " *
                            "for $(typeof(model))."))
    return parameters
end

"""Hook for model-specific construction-time growth validation."""
_validate_growth_parameters(model, parameters, problem) = nothing

"""Hook for model-specific construction-time dissolution validation."""
_validate_dissolution_parameters(model, parameters, problem) = nothing

"""
    _validate_crystallisation_problem(problem)

Validate the structural contract shared by all runners.  This is deliberately
called once at the public simulation boundary; the hot RHS only evaluates the
already validated rate laws.
"""
function _validate_crystallisation_problem(problem::CrystallisationProblem)
    solvent_names = propertynames(problem.initial_solvent_state)
    concentration_position = findfirst(==(Symbol(:concentration)), solvent_names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))

    isfinite(problem.initial_concentration) && problem.initial_concentration >= 0.0 ||
        throw(ArgumentError("initial_concentration must be finite and nonnegative."))
    isfinite(problem.ρ) && problem.ρ > 0.0 ||
        throw(ArgumentError("ρ must be finite and strictly positive."))
    isfinite(problem.kv) && problem.kv > 0.0 ||
        throw(ArgumentError("kv must be finite and strictly positive."))
    isfinite(problem.R) && problem.R > 0.0 ||
        throw(ArgumentError("R must be finite and strictly positive."))
    isfinite(problem.kb) && problem.kb > 0.0 ||
        throw(ArgumentError("kb must be finite and strictly positive."))
    _validate_solid_volume_threshold(problem)

    solvent_values = values(problem.initial_solvent_state)
    @inbounds for solvent_value in solvent_values
        solvent_value isa Real && isfinite(solvent_value) ||
            throw(ArgumentError("initial solvent-state values must be finite real numbers."))
    end
    concentration_value = solvent_values[concentration_position]
    concentration_value >= 0.0 ||
        throw(ArgumentError("initial solvent concentration must be nonnegative."))

    _validate_parameter_block(problem.kinetics_nucleationfunction,
                              problem.parameterset_nucleation, :nucleation)
    _validate_parameter_block(problem.kinetics_growthfunction,
                              problem.parameterset_growth, :growth)
    _validate_parameter_block(problem.kinetics_dissolutionfunction,
                              problem.parameterset_dissolution, :dissolution)
    _validate_parameter_block(problem.kinetics_aggregationfunction,
                              problem.parameterset_aggregation, :aggregation)
    _validate_parameter_block(problem.kinetics_breakagefunction,
                              problem.parameterset_breakage, :breakage)
    _validate_growth_parameters(problem.kinetics_growthfunction,
                                problem.parameterset_growth, problem)
    _validate_dissolution_parameters(problem.kinetics_dissolutionfunction,
                                     problem.parameterset_dissolution, problem)

    population_state_count = _population_state_count(problem.solver)
    expected_state_count = population_state_count + length(solvent_names)
    if !isnothing(problem.initial_state)
        length(problem.initial_state) == expected_state_count ||
            throw(ArgumentError("initial_state has length $(length(problem.initial_state)); " *
                                "expected $expected_state_count for $(typeof(problem.solver))."))
        all(isfinite, problem.initial_state) ||
            throw(ArgumentError("initial_state must contain only finite values."))
        if problem.solver isa AbstractDiscretisedSolver
            any(value -> value < 0.0, @view problem.initial_state[1:population_state_count]) &&
                throw(ArgumentError("initial number density must be nonnegative."))
        end
    end

    saturation_value = saturation_concentration(problem, 0.0)
    isfinite(saturation_value) && saturation_value > 0.0 ||
        throw(ArgumentError("saturation concentration must be finite and strictly positive at t=0."))
    return problem
end

"""
    saturation_concentration(prob::CrystallisationProblem, t) -> Real

Solubility (kg/m³) of the problem at time `t` under its temperature profile.
"""
saturation_concentration(prob::CrystallisationProblem, t) =
    saturation_concentration(prob.saturation_model, prob.temp_profile, t)

"""
    solvent_state(prob, state) -> NamedTuple

Read the named solvent-phase variables from the tail of a numerical solver
state. `initial_solvent_state` defines both their names and their ordering.
"""
function solvent_state(prob::CrystallisationProblem, state)
    names = propertynames(prob.initial_solvent_state)
    n_solvent = length(names)
    values = ntuple(index -> state[length(state) - n_solvent + index], Val(n_solvent))
    return NamedTuple{names}(values)
end

function _solvent_state_index(prob::CrystallisationProblem, state, name::Symbol)
    position = findfirst(==(name), propertynames(prob.initial_solvent_state))
    position === nothing && throw(ArgumentError("Unknown solvent-state variable :$name."))
    return length(state) - length(propertynames(prob.initial_solvent_state)) + position
end

"""
    supersaturation(prob::CrystallisationProblem, state, t) -> Real

Supersaturation ratio from the named `:concentration` solvent-state variable.
"""
supersaturation(prob::CrystallisationProblem, state, t) =
    state[_solvent_state_index(prob, state, :concentration)] /
    saturation_concentration(prob, t)
