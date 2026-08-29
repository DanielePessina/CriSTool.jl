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
    derivative_type = promote_type(eltype(state), typeof(growth))
    derivatives = MVector{solvent_count, derivative_type}(undef)
    fill!(derivatives, zero(growth))
    concentration_position = findfirst(==(Symbol(:concentration)), solvent_names)
    concentration_position === nothing &&
        throw(ArgumentError("initial_solvent_state must define :concentration."))

    population_count = length(state) - solvent_count
    depletion = zero(growth)
    if hasproperty(problem.solver, :nmoments)
        problem.solver.nmoments >= 2 ||
            throw(ArgumentError("MoM solver requires nmoments >= 2 for concentration closure."))
        depletion = 3 * problem.kv * problem.ρ * growth * state[3]
    else
        @inbounds for index in 1:population_count
            crystal_length = problem.solver.cell_centre[index]
            depletion += problem.solver.cell_dL * state[index] * growth * crystal_length^2
        end
        depletion *= 3 * problem.kv * problem.ρ
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
- `saturation_model::AbstractSaturationModel`: Solubility model (default `lysozyme_saturation()`)
- `kv::Float64`: Volume shape factor
- `molecular_volume::Float64`: Molecular volume (m³)
- `kinetics_nucleationfunction::NuF`: Nucleation function
- `kinetics_growthfunction::GrF`: Growth function
- `parameterset_nucleation::NuP`: Nucleation parameters
- `parameterset_growth::GrP`: Growth parameters
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
                                                    SM <: AbstractSaturationModel,
                                                    SS <: NamedTuple,
                                                    SD} <:
                             AbstractCrystallisationProblem

    # Operation
    temp_profile::TP = ConstantTemperature(273.15 + 20.0) # Default to 25°C

    # Loading
    loading::Float64 = 0.0

    # Solute
    ρ::Float64 = 1370.0
    initial_concentration::Float64 = 20.0
    saturation_model::SM = lysozyme_saturation()
    kv::Float64 = 0.81 #0.55
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
