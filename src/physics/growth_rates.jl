## Growth

"""
    growthrate!(destination, gf::AbstractFPScalarGrowthFunction, parameters,
                problem, state, time, mesh)

Fill a mesh-sized destination with a scalar signed growth rate.  Scalar models
use the same value at every mesh point; keeping this mutating seam alongside
the length-dependent implementation lets discretised RHSs reuse their
`DiffCache` storage.
"""
function growthrate!(destination::AbstractVector,
                     gf::AbstractFPScalarGrowthFunction,
                     parameters,
                     problem::CrystallisationProblem,
                     state,
                     time,
                     mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growthrate! destination and mesh must have the same length."))
    scalar_rate = growthrate(gf, parameters, problem, state, time)
    fill!(destination, scalar_rate)
    return destination
end

"""Compatibility mesh adapter for scalar dissolution laws used as a legacy
growth model.  New code should put the law in the independent `diss` slot and
call `dissolutionrate!`/`net_growth_rate!` instead."""
function growthrate!(destination::AbstractVector,
                     gf::AbstractFPScalarDissolutionFunction,
                     parameters,
                     problem::CrystallisationProblem,
                     state,
                     time,
                     mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growthrate! destination and mesh must have the same length."))
    scalar_rate = growthrate(gf, parameters, problem, state, time)
    fill!(destination, scalar_rate)
    return destination
end

"""Compatibility mesh adapter for legacy length dissolution models supplied
through the growth slot."""
function growthrate!(destination::AbstractVector,
                     gf::AbstractFPLengthDissolutionFunction,
                     parameters,
                     problem::CrystallisationProblem,
                     state,
                     time,
                     mesh::AbstractVector)
    return dissolutionrate!(destination, gf, parameters, problem, state, time,
                            mesh)
end

"""
    growthrate_at_length(gf::AbstractFPScalarGrowthFunction, parameters,
                         problem, state, time, crystal_length)

Scalar growth/dissolution laws are independent of crystal length, so this
adapter returns the scalar rate without allocating a mesh-sized vector.
"""
growthrate_at_length(gf::AbstractFPScalarGrowthFunction,
                     parameters,
                     problem::CrystallisationProblem,
                     state,
                     time,
                     crystal_length::Real) = growthrate(gf, parameters, problem, state, time)

growthrate_at_length(gf::AbstractFPScalarDissolutionFunction,
                     parameters,
                     problem::CrystallisationProblem,
                     state,
                     time,
                     crystal_length::Real) = growthrate(gf, parameters, problem, state, time)

growthrate_at_length(gf::AbstractFPLengthDissolutionFunction,
                     parameters,
                     problem::CrystallisationProblem,
                     state,
                     time,
                     crystal_length::Real) = dissolutionrate_at_length(
                         gf, parameters, problem, state, time, crystal_length)

@inline _growth_activation_rate(Ag, activation_energy, temperature, gas_constant) =
    exp10(Ag) * exp(-activation_energy / (gas_constant * temperature))

@inline function _dissolution_drive(supersaturation_ratio)
    tolerance = CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    supersaturation_ratio < one(supersaturation_ratio) - tolerance ?
        one(supersaturation_ratio) - supersaturation_ratio :
        zero(supersaturation_ratio)
end

@inline function _dissolution_rate(Ad, Ead, exponent, supersaturation_ratio,
                                   temperature, gas_constant)
    driving_force = _dissolution_drive(supersaturation_ratio)
    return driving_force == zero(driving_force) ?
           zero(driving_force) :
           -(Ad * 1e-9) * exp(-(Ead * 1e3) / (gas_constant * temperature)) *
           driving_force^exponent
end

@inline _dissolution_zero(parameters, state) =
    zero(promote_type(eltype(parameters), eltype(state)))

"""Return the scalar dissolution contribution of a dissolution model."""
@inline dissolutionrate(::nodissolution, parameters, problem, state, time) =
    _dissolution_zero(parameters, state)

"""Fill a mesh-sized destination with scalar dissolution."""
function dissolutionrate!(destination::AbstractVector,
                          dissolutionfunction::AbstractFPScalarDissolutionFunction,
                          parameters,
                          problem::CrystallisationProblem,
                          state,
                          time,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("dissolutionrate! destination and mesh must have the same length."))
    fill!(destination, dissolutionrate(dissolutionfunction, parameters,
                                       problem, state, time))
    return destination
end

"""Evaluate length-dependent dissolution at one mesh/node location."""
@inline dissolutionrate_at_length(
    dissolutionfunction::AbstractFPScalarDissolutionFunction,
    parameters,
    problem::CrystallisationProblem,
    state,
    time,
    crystal_length::Real) = dissolutionrate(dissolutionfunction, parameters,
                                             problem, state, time)

function dissolutionrate_at_length(
    dissolutionfunction::AbstractFPLengthDissolutionFunction,
    parameters,
    problem::CrystallisationProblem,
    state,
    time,
    crystal_length::Real)
    throw(MethodError(dissolutionrate_at_length,
                      (dissolutionfunction, parameters, problem, state, time,
                       crystal_length)))
end

"""Fill a mesh-sized destination with length-dependent dissolution."""
function dissolutionrate!(destination::AbstractVector,
                          dissolutionfunction::AbstractFPLengthDissolutionFunction,
                          parameters,
                          problem::CrystallisationProblem,
                          state,
                          time,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("dissolutionrate! destination and mesh must have the same length."))
    @inbounds for mesh_index in eachindex(destination, mesh)
        destination[mesh_index] = dissolutionrate_at_length(
            dissolutionfunction, parameters, problem, state, time,
            mesh[mesh_index])
    end
    return destination
end

"""Allocate a mesh-sized dissolution rate for convenience callers."""
function dissolutionrate(dissolutionfunction::AbstractFPLengthDissolutionFunction,
                         parameters,
                         problem::CrystallisationProblem,
                         state,
                         time)
    mesh = hasproperty(problem.solver, :cell_centre) ? problem.solver.cell_centre :
           throw(ArgumentError("$(typeof(dissolutionfunction)) requires a discretised solver mesh."))
    rate_type = promote_type(eltype(parameters), eltype(state))
    destination = Vector{rate_type}(undef, length(mesh))
    dissolutionrate!(destination, dissolutionfunction, parameters, problem, state,
                     time, mesh)
    return destination
end

@inline function _growth_energy_rate(Ag, exponent, activation_energy,
                                     supersaturation_ratio, temperature,
                                     gas_constant)
    threshold = one(supersaturation_ratio) +
                CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    return supersaturation_ratio > threshold ?
           _growth_activation_rate(Ag, activation_energy, temperature, gas_constant) *
           (supersaturation_ratio - one(supersaturation_ratio))^exponent :
           zero(supersaturation_ratio)
end

"""Return the signed net crystal growth rate from independent growth and
dissolution laws."""
@inline function net_growth_rate(growthfunction::AbstractGrowthFunction,
                                 growth_parameters,
                                 dissolutionfunction::AbstractDissolutionFunction,
                                 dissolution_parameters,
                                 problem::CrystallisationProblem,
                                 state,
                                 time)
    return growthrate(growthfunction, growth_parameters, problem, state, time) +
           dissolutionrate(dissolutionfunction, dissolution_parameters,
                           problem, state, time)
end

"""Evaluate the net crystal growth rate at one physical crystal length."""
@inline function net_growth_rate_at_length(
    growthfunction::AbstractGrowthFunction,
    growth_parameters,
    dissolutionfunction::AbstractDissolutionFunction,
    dissolution_parameters,
    problem::CrystallisationProblem,
    state,
    time,
    crystal_length::Real)
    return growthrate_at_length(growthfunction, growth_parameters, problem,
                                state, time, crystal_length) +
           dissolutionrate_at_length(dissolutionfunction, dissolution_parameters,
                                     problem, state, time, crystal_length)
end

"""Fill a mesh-aligned signed growth-rate field with growth plus dissolution.

The helper keeps scalar models on a single-value fast path and evaluates
length-dependent dissolution directly at each mesh point, avoiding a second
temporary vector in FV/WENO RHS calls.
"""
function net_growth_rate!(destination::AbstractVector,
                          growthfunction::AbstractGrowthFunction,
                          growth_parameters,
                          dissolutionfunction::AbstractDissolutionFunction,
                          dissolution_parameters,
                          problem::CrystallisationProblem,
                          state,
                          time,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growth-rate destination and mesh must have the same length."))

    if growthfunction isa Union{AbstractFPLengthGrowthFunction,
                                AbstractFPLengthDissolutionFunction}
        growthrate!(destination, growthfunction, growth_parameters, problem,
                    state, time, mesh)
    else
        growth_value = growthrate(growthfunction, growth_parameters, problem,
                                  state, time)
        fill!(destination, growth_value)
    end

    if dissolutionfunction isa AbstractFPLengthDissolutionFunction
        @inbounds for mesh_index in eachindex(destination, mesh)
            destination[mesh_index] += dissolutionrate_at_length(
                dissolutionfunction, dissolution_parameters, problem, state,
                time, mesh[mesh_index])
        end
    else
        dissolution_value = dissolutionrate(dissolutionfunction,
                                            dissolution_parameters, problem,
                                            state, time)
        @inbounds for mesh_index in eachindex(destination)
            destination[mesh_index] += dissolution_value
        end
    end
    return destination
end

"""Legacy three-argument net-rate alias for callers without dissolution."""
@inline net_growth_rate(growthfunction, parameters, problem, state, time) =
    growthrate(growthfunction, parameters, problem, state, time)

"""Compatibility spelling for the legacy growth-only net-rate helper."""
@inline net_growthrate(args...) = net_growth_rate(args...)

function _validate_dissolution_scalar_parameters(model, parameters)
    all(isfinite, parameters) ||
        throw(ArgumentError("$(typeof(model)) parameters must be finite."))
    return nothing
end

function _validate_dissolution_parameters(model::AbstractDissolutionFunction,
                                          parameters,
                                          problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    return nothing
end

# Legacy `growth_dissolution` models were historically supplied through the
# growth field.  Keep their validation there while sharing the independent
# dissolution validation hook for the new `diss` field.
_validate_growth_parameters(model::AbstractDissolutionFunction,
                            parameters,
                            problem::CrystallisationProblem) =
    _validate_dissolution_parameters(model, parameters, problem)

function _validate_dissolution_parameters(model::growth_dissolution,
                                          parameters,
                                          problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    p = _named_params(model, parameters)
    p.Ad >= 0.0 || throw(ArgumentError("growth_dissolution Ad must be nonnegative."))
    p.d > 0.0 || throw(ArgumentError("growth_dissolution d must be strictly positive."))
    return nothing
end

_validate_growth_parameters(model::growth_dissolution,
                            parameters,
                            problem::CrystallisationProblem) =
    _validate_dissolution_parameters(model, parameters, problem)

function _validate_dissolution_parameters(model::growth_energy_dissolution,
                                          parameters,
                                          problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    p = _named_params(model, parameters)
    p.Ad >= 0.0 || throw(ArgumentError("growth_energy_dissolution Ad must be nonnegative."))
    p.g > 0.0 || throw(ArgumentError("growth_energy_dissolution g must be strictly positive."))
    p.d > 0.0 || throw(ArgumentError("growth_energy_dissolution d must be strictly positive."))
    return nothing
end

_validate_growth_parameters(model::growth_energy_dissolution,
                            parameters,
                            problem::CrystallisationProblem) =
    _validate_dissolution_parameters(model, parameters, problem)

function _validate_dissolution_parameters(model::growth_dissolution_length,
                                          parameters,
                                          problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    isfinite(model.Lref) && model.Lref > 0.0 ||
        throw(ArgumentError("growth_dissolution_length Lref must be finite and strictly positive."))
    p = _named_params(model, parameters)
    p.Ad >= 0.0 || throw(ArgumentError("growth_dissolution_length Ad must be nonnegative."))
    p.d > 0.0 || throw(ArgumentError("growth_dissolution_length d must be strictly positive."))
    if problem.solver isa AbstractDiscretisedSolver
        lower_base = 1.0 + p.κ * problem.solver.lmin / model.Lref
        upper_base = 1.0 + p.κ * problem.solver.lmax / model.Lref
        lower_base > 0.0 && upper_base > 0.0 ||
            throw(ArgumentError("growth_dissolution_length size factor must stay positive " *
                                "over the configured mesh."))
    end
    return nothing
end

_validate_growth_parameters(model::growth_dissolution_length,
                            parameters,
                            problem::CrystallisationProblem) =
    _validate_dissolution_parameters(model, parameters, problem)

"""
    growthrate(::growth_empirical, parameters::AbstractVector, S::Real,
               system, temperature, numberdensity)

Calculate crystal growth rate using an empirical power law model.

# Arguments
- `parameters`: Vector containing [A_g, g] where:
  - A_g: Growth rate constant
  - g: Growth rate order
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Instantaneous temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ? (p.Ag * 1e-9) * ((S - 1)^p.g) : 0.0
end
"""
    growthrate(gf::growth_energy, parameters::AbstractVector, S::Real,
               system, temperature, numberdensity) -> Real

Calculate growth rate with Arrhenius temperature dependence using fixed activation energy.

# Arguments
- `gf`: Growth function with embedded activation energy Ea
- `parameters`: Vector [Ag, g] where Ag is pre-exponential (log10 scale), g is exponent
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_energy, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _growth_energy_rate(p.Ag, p.g, gf.Ea, S, temp, prob.R)
    ## multiply by 1e12 to convert to typical units
end

"""
    growthrate(gf::growth_energy_est, parameters::AbstractVector, S::Real,
               system, temperature, numberdensity) -> Real

Calculate growth rate with Arrhenius temperature dependence and estimated activation energy.

# Arguments
- `gf`: Growth function struct
- `parameters`: Vector [Ag, Eag, g] where Ag is pre-exponential (log10), Eag is activation energy (kJ/mol), g is exponent
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_energy_est, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _growth_energy_rate(p.Ag, p.g, p.Eag * 1e3, S, temp, prob.R)
    ## multiply by 1e12 to convert to typical units
end

"""
    growthrate(::growth_BCF, parameters::AbstractVector, S::Real,
               system::CrystallisationProblem, temperature, numberdensity)

Calculate crystal growth rate using Burton-Cabrera-Frank (BCF) surface diffusion model.

# Arguments
- `parameters`: Vector containing [C3, C4] where:
  - C3: Growth rate constant (model parameter based on physical properties)
  - C4: Surface energy barrier parameter (model parameter based on physical properties)
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Instantaneous temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0

# Notes
- Developed by Burton, Cabrera, and Frank (1951) for growth via screw dislocation geometry
- Assumes surface diffusion of adsorbed species is rate-limiting
- Alternative to Birth and Spread model for low supersaturation conditions
"""
function growthrate(gf::growth_BCF, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (1e-9) * p.C3 * temp / p.C4 *
        (S - 1) *
        tanh(p.C4 / (temp * log(S)))
    else
        0.0
    end
end

"""
    growthrate(::growth_BpS, parameters::AbstractVector, S::Real,
               system::CrystallisationProblem, temperature, numberdensity)

Calculate crystal growth rate using Birth and Spread (B+S) model based on nucleation theory.

# Arguments
- `parameters`: Vector containing [C1, C2] where:
  - C1: Growth rate constant (model parameter based on physical properties fitted to experimental data)
  - C2: Energy barrier parameter (model parameter based on physical properties fitted to experimental data)
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Instantaneous temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0

# Notes
- Based on O'Hara (1973) nucleation theory expression
- Critical nuclei form on surface and spread at constant rates
- Assumes smooth crystal surface at high supersaturations
- May deviate from measured values at low supersaturations
"""
function growthrate(gf::growth_BpS, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (1e-9) *
        p.C1 *
        ((S - 1)^(2 / 3)) *
        (log(S))^(1 / 6) *
        exp(-p.C2 / (temp^2 * log(S)))
    else
        0.0
    end
end

### Dissolution

"""
    growthrate(gf::growth_dissolution_length, parameters::AbstractVector, S::Real,
               mesh::AbstractVector, temperature, numberdensity) -> Vector

Calculate length-dependent dissolution rate (negative growth).

# Arguments
- `gf`: Dissolution growth function
- `parameters`: Vector [Ad, Ead, d, κ, p] for dissolution kinetics
- `S`: Supersaturation ratio
- `mesh`: Cell-centre coordinates (length-dependence is evaluated on each mesh cell)
- `temperature`: Temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Vector of dissolution rates (m/s) at each mesh point, zero vector if supersaturated
"""
function dissolutionrate(gf::growth_dissolution_length, parameters::T,
                         prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    mesh = hasproperty(prob.solver, :cell_centre) ? prob.solver.cell_centre :
           throw(ArgumentError("growth_dissolution_length requires a discretised solver mesh."))
    rate_type = promote_type(eltype(parameters), eltype(state))
    destination = Vector{rate_type}(undef, length(mesh))
    dissolutionrate!(destination, gf, parameters, prob, state, t, mesh)
    return destination
end

# Source-compatible name for callers that supplied this law through `gr`.
growthrate(gf::growth_dissolution_length, parameters::T,
           prob::CrystallisationProblem, state, t) where {T <: AbstractVector} =
    dissolutionrate(gf, parameters, prob, state, t)

"""
    growthrate!(destination, gf::growth_dissolution_length, parameters,
                prob, state, t[, mesh])

Fill `destination` with the signed, length-dependent dissolution rate.
The optional mesh defaults to `prob.solver.cell_centre`.  The destination is
kept separate from the public allocating `growthrate` convenience method so
finite-volume and WENO RHS evaluations can reuse a typed scratch buffer.
"""
function growthrate!(destination::AbstractVector,
                     gf::growth_dissolution_length,
                     parameters,
                     prob::CrystallisationProblem,
                     state,
                     t)
    mesh = hasproperty(prob.solver, :cell_centre) ? prob.solver.cell_centre :
           throw(ArgumentError("growth_dissolution_length requires a discretised solver mesh."))
    return dissolutionrate!(destination, gf, parameters, prob, state, t, mesh)
end

function dissolutionrate!(destination::AbstractVector,
                          gf::growth_dissolution_length,
                          parameters,
                          prob::CrystallisationProblem,
                          state,
                          t,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growthrate! destination and mesh must have the same length."))

    pn = _named_params(gf, parameters)
    supersaturation_ratio = supersaturation(prob, state, t)
    driving_force = _dissolution_drive(supersaturation_ratio)
    if driving_force == zero(driving_force)
        fill!(destination, zero(driving_force))
        return destination
    end

    temp = temperature(prob.temp_profile, t)
    dissolution_prefactor = -(pn.Ad * 1e-9) *
                            exp(-(pn.Ead * 1e3) / (prob.R * temp)) *
                            driving_force^pn.d
    @inbounds for index in eachindex(destination, mesh)
        size_factor_base = one(mesh[index]) + pn.κ * (mesh[index] / gf.Lref)
        size_factor_base > zero(size_factor_base) ||
            throw(DomainError(size_factor_base,
                              "growth_dissolution_length size factor must be positive."))
        destination[index] = dissolution_prefactor * size_factor_base^pn.p
    end
    return destination
end

# Source-compatible mutating name for the old `gr` slot.
growthrate!(destination::AbstractVector,
            gf::growth_dissolution_length,
            parameters,
            prob::CrystallisationProblem,
            state,
            t,
            mesh::AbstractVector) =
    dissolutionrate!(destination, gf, parameters, prob, state, t, mesh)

"""
    growthrate_at_length(gf::growth_dissolution_length, parameters,
                         prob, state, t, length)

Evaluate a length-dependent signed dissolution rate at one crystal length
without constructing a mesh-sized result.
"""
function dissolutionrate_at_length(gf::growth_dissolution_length,
                                   parameters,
                                   prob::CrystallisationProblem,
                                   state,
                                   t,
                                   crystal_length::Real)
    pn = _named_params(gf, parameters)
    supersaturation_ratio = supersaturation(prob, state, t)
    driving_force = _dissolution_drive(supersaturation_ratio)
    driving_force == zero(driving_force) && return zero(driving_force)

    temp = temperature(prob.temp_profile, t)
    size_factor_base = one(crystal_length) + pn.κ * (crystal_length / gf.Lref)
    size_factor_base > zero(size_factor_base) ||
        throw(DomainError(size_factor_base,
                          "growth_dissolution_length size factor must be positive."))
    return -(pn.Ad * 1e-9) * exp(-(pn.Ead * 1e3) / (prob.R * temp)) *
           driving_force^pn.d * size_factor_base^pn.p
end

growthrate_at_length(gf::growth_dissolution_length,
                     parameters,
                     prob::CrystallisationProblem,
                     state,
                     t,
                     crystal_length::Real) =
    dissolutionrate_at_length(gf, parameters, prob, state, t, crystal_length)

"""
    growthrate(gf::growth_dissolution, parameters::AbstractVector, S::Real,
               system, temperature, numberdensity) -> Real

Calculate scalar dissolution rate (negative growth) with Arrhenius temperature dependence.

# Arguments
- `gf`: Dissolution growth function
- `parameters`: Vector [Ad, Ead, d] for dissolution kinetics
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Dissolution rate (negative m/s) if undersaturated, 0 otherwise
"""
function growthrate(gf::growth_dissolution, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _dissolution_rate(p.Ad, p.Ead, p.d, S, temp, prob.R)
end

"""Independent dissolution dispatch for the legacy empirical law."""
dissolutionrate(gf::growth_dissolution, parameters::T,
                prob::CrystallisationProblem, state, t) where {T <: AbstractVector} =
    growthrate(gf, parameters, prob, state, t)

"""
    growthrate(gf::growth_energy_dissolution, parameters::AbstractVector, S::Real,
               system, temperature, numberdensity) -> Real

Calculate growth or dissolution rate depending on supersaturation.

Uses `growth_energy` for supersaturated conditions (S > 1.001) and
`growth_dissolution` for undersaturated conditions.

# Arguments
- `gf`: Combined growth/dissolution function
- `parameters`: Vector [Ag, g, Ad, Ead, d] for combined kinetics
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if supersaturated, dissolution rate if undersaturated
"""
function growthrate(gf::growth_energy_dissolution, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    growth_threshold = one(S) + CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    dissolution_threshold = one(S) - CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    if S > growth_threshold
        return _growth_energy_rate(p.Ag, p.g, 53e3, S, temp, prob.R)
    elseif S < dissolution_threshold
        return _dissolution_rate(p.Ad, p.Ead, p.d, S, temp, prob.R)
    end
    return zero(S)
end

"""Independent dissolution contribution of the legacy combined law."""
function dissolutionrate(gf::growth_energy_dissolution, parameters::T,
                         prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _dissolution_rate(p.Ad, p.Ead, p.d, S, temp, prob.R)
end

###

"""
    growthrate(grf::growth_empirical_fixed, parameters, S::Real,
               system::CrystallisationProblem, temperature, numberdensity) -> Real

Calculate empirical growth rate using pre-fixed parameters embedded in the struct.

# Arguments
- `grf`: Fixed empirical growth function with embedded Ag and g parameters
- `parameters`: Ignored (parameters taken from grf)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(grf::growth_empirical_fixed, parameters, prob::CrystallisationProblem, state, t)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? (grf.Ag * 1e-9) * ((S - 1)^grf.g) : 0.0
end
