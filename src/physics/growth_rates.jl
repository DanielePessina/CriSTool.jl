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

@inline _growth_activation_rate(log10_growth_coefficient, activation_energy,
                                temperature, gas_constant) =
    exp10(log10_growth_coefficient) *
    exp(-activation_energy / (gas_constant * temperature))

@inline function _dissolution_drive(supersaturation_ratio)
    tolerance = CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    supersaturation_ratio < one(supersaturation_ratio) - tolerance ?
        one(supersaturation_ratio) - supersaturation_ratio :
        zero(supersaturation_ratio)
end

@inline function _length_dissolution_context(
    dissolutionfunction::growth_dissolution_length,
    parameters,
    problem::CrystallisationProblem,
    state,
    time)
    named_parameters = _named_params(dissolutionfunction, parameters)
    supersaturation_ratio = supersaturation(problem, state, time)
    driving_force = _dissolution_drive(supersaturation_ratio)
    return named_parameters, driving_force
end

@inline function _length_dissolution_prefactor(named_parameters,
                                               driving_force,
                                               problem::CrystallisationProblem,
                                               time)
    temperature_value = temperature(problem.temp_profile, time)
    return -named_parameters.dissolution_coefficient *
           exp(-named_parameters.activation_energy /
               (problem.R_gas_constant * temperature_value)) *
           driving_force^named_parameters.dissolution_order
end

@inline function _length_dissolution_size_factor_base(
    dissolutionfunction::growth_dissolution_length,
    named_parameters,
    crystal_length::Real)
    size_factor_base = one(crystal_length) +
                       named_parameters.size_dependence_coefficient *
                       (crystal_length / dissolutionfunction.Lref)
    size_factor_base > zero(size_factor_base) ||
        throw(DomainError(size_factor_base,
                          "growth_dissolution_length size factor must be positive."))
    return size_factor_base
end

@inline function _length_dissolution_rate_from_context(
    dissolutionfunction::growth_dissolution_length,
    named_parameters,
    dissolution_prefactor,
    crystal_length::Real)
    size_factor_base = _length_dissolution_size_factor_base(
        dissolutionfunction, named_parameters, crystal_length)
    return dissolution_prefactor *
           size_factor_base^named_parameters.size_dependence_exponent
end

function _add_length_dissolution_rates!(destination::AbstractVector,
                                        dissolutionfunction::growth_dissolution_length,
                                        parameters,
                                        problem::CrystallisationProblem,
                                        state,
                                        time,
                                        mesh::AbstractVector)
    named_parameters, driving_force = _length_dissolution_context(
        dissolutionfunction, parameters, problem, state, time)
    driving_force == zero(driving_force) && return destination

    dissolution_prefactor = _length_dissolution_prefactor(
        named_parameters, driving_force, problem, time)
    @inbounds for mesh_index in eachindex(destination, mesh)
        destination[mesh_index] += _length_dissolution_rate_from_context(
            dissolutionfunction, named_parameters, dissolution_prefactor,
            mesh[mesh_index])
    end
    return destination
end

function _add_length_dissolution_rates!(destination::AbstractVector,
                                        dissolutionfunction::AbstractFPLengthDissolutionFunction,
                                        parameters,
                                        problem::CrystallisationProblem,
                                        state,
                                        time,
                                        mesh::AbstractVector)
    @inbounds for mesh_index in eachindex(destination, mesh)
        destination[mesh_index] += dissolutionrate_at_length(
            dissolutionfunction, parameters, problem, state, time,
            mesh[mesh_index])
    end
    return destination
end

@inline function _dissolution_rate(dissolution_coefficient, activation_energy,
                                   dissolution_order, supersaturation_ratio,
                                   temperature, gas_constant)
    driving_force = _dissolution_drive(supersaturation_ratio)
    return driving_force == zero(driving_force) ?
           zero(driving_force) :
           -dissolution_coefficient *
           exp(-activation_energy / (gas_constant * temperature)) *
           driving_force^dissolution_order
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
    mesh = _length_mesh(problem.solver)
    rate_type = promote_type(eltype(parameters), eltype(state))
    destination = Vector{rate_type}(undef, length(mesh))
    dissolutionrate!(destination, dissolutionfunction, parameters, problem, state,
                     time, mesh)
    return destination
end

@inline function _growth_energy_rate(log10_growth_coefficient, growth_order,
                                     activation_energy,
                                     supersaturation_ratio, temperature,
                                     gas_constant)
    threshold = one(supersaturation_ratio) +
                CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    return supersaturation_ratio > threshold ?
           _growth_activation_rate(log10_growth_coefficient, activation_energy,
                                   temperature, gas_constant) *
           (supersaturation_ratio - one(supersaturation_ratio))^growth_order :
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

    if _is_length_law(growthfunction)
        growthrate!(destination, growthfunction, growth_parameters, problem,
                    state, time, mesh)
    else
        growth_value = growthrate(growthfunction, growth_parameters, problem,
                                  state, time)
        fill!(destination, growth_value)
    end

    if _is_length_law(dissolutionfunction)
        _add_length_dissolution_rates!(destination, dissolutionfunction,
                                       dissolution_parameters, problem, state,
                                       time, mesh)
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

function _validate_dissolution_parameters(model::growth_dissolution,
                                          parameters,
                                          problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    p = _named_params(model, parameters)
    p.dissolution_coefficient >= 0.0 ||
        throw(ArgumentError("dissolution_coefficient must be nonnegative."))
    p.dissolution_order > 0.0 ||
        throw(ArgumentError("dissolution_order must be strictly positive."))
    return nothing
end

function _validate_dissolution_parameters(model::growth_dissolution_length,
                                          parameters,
                                          problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    isfinite(model.Lref) && model.Lref > 0.0 ||
        throw(ArgumentError("growth_dissolution_length Lref must be finite and strictly positive."))
    p = _named_params(model, parameters)
    p.dissolution_coefficient >= 0.0 ||
        throw(ArgumentError("dissolution_coefficient must be nonnegative."))
    p.dissolution_order > 0.0 ||
        throw(ArgumentError("dissolution_order must be strictly positive."))
    if problem.solver isa AbstractDiscretisedSolver
        lower_base = 1.0 + p.size_dependence_coefficient * problem.solver.lmin / model.Lref
        upper_base = 1.0 + p.size_dependence_coefficient * problem.solver.lmax / model.Lref
        lower_base > 0.0 && upper_base > 0.0 ||
            throw(ArgumentError("growth_dissolution_length size factor must stay positive " *
                                "over the configured mesh."))
    end
    return nothing
end

function _validate_growth_parameters(model::growth_empirical_length,
                                     parameters,
                                     problem::CrystallisationProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    isfinite(model.Lref) && model.Lref > 0.0 ||
        throw(ArgumentError("growth_empirical_length Lref must be finite and strictly positive."))
    p = _named_params(model, parameters)
    p.growth_coefficient >= 0.0 ||
        throw(ArgumentError("growth_coefficient must be nonnegative."))
    p.growth_order > 0.0 ||
        throw(ArgumentError("growth_order must be strictly positive."))
    return nothing
end

"""
    growthrate(::growth_empirical, parameters::AbstractVector,
               problem::CrystallisationProblem, state, t) -> Real

Calculate crystal growth rate using an empirical power law model.

# Arguments
- `parameters`: Vector containing [growth_coefficient, growth_order], where
  the coefficient is in m/s.
- `problem`: Crystallisation problem providing the saturation model.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ? p.growth_coefficient * (S - 1)^p.growth_order : 0.0
end

@inline function _growth_empirical_length_size_factor_base(
    growthfunction::growth_empirical_length,
    named_parameters,
    crystal_length::Real)
    size_factor_base = one(crystal_length) +
                       named_parameters.size_dependence_coefficient *
                       (crystal_length / growthfunction.Lref)
    size_factor_base > zero(size_factor_base) ||
        throw(DomainError(size_factor_base,
                          "growth_empirical_length size factor must be positive."))
    return size_factor_base
end

"""
    growthrate_at_length(gf::growth_empirical_length, parameters,
                         problem, state, time, crystal_length)

Evaluate the empirical growth law with the length-dependent multiplier

`(1 + size_dependence_coefficient * crystal_length / Lref)^size_dependence_exponent`.
The returned rate is in metres per second.  The multiplier base must remain
positive so non-integer exponents remain real-valued.
"""
@inline function growthrate_at_length(gf::growth_empirical_length,
                                      parameters,
                                      prob::CrystallisationProblem,
                                      state,
                                      time,
                                      crystal_length::Real)
    named_parameters = _named_params(gf, parameters)
    supersaturation_ratio = supersaturation(prob, state, time)
    zero_type = promote_type(typeof(supersaturation_ratio),
                              eltype(parameters),
                              typeof(crystal_length))
    growth_threshold = one(supersaturation_ratio) +
                       CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    supersaturation_ratio > growth_threshold || return zero(zero_type)

    size_factor_base = _growth_empirical_length_size_factor_base(
        gf, named_parameters, crystal_length)
    return named_parameters.growth_coefficient *
           (supersaturation_ratio - one(supersaturation_ratio)) ^
           named_parameters.growth_order *
           size_factor_base ^ named_parameters.size_dependence_exponent
end

"""Fill a mesh-sized destination with length-dependent empirical growth."""
function growthrate!(destination::AbstractVector,
                     gf::growth_empirical_length,
                     parameters,
                     prob::CrystallisationProblem,
                     state,
                     time,
                     mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growthrate! destination and mesh must have the same length."))
    isempty(destination) && return destination

    named_parameters = _named_params(gf, parameters)
    supersaturation_ratio = supersaturation(prob, state, time)
    growth_threshold = one(supersaturation_ratio) +
                       CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE
    if !(supersaturation_ratio > growth_threshold)
        @inbounds for mesh_index in eachindex(destination, mesh)
            zero_type = promote_type(typeof(supersaturation_ratio),
                                     eltype(parameters),
                                     typeof(mesh[mesh_index]))
            destination[mesh_index] = zero(zero_type)
        end
        return destination
    end

    growth_prefactor = named_parameters.growth_coefficient *
                       (supersaturation_ratio - one(supersaturation_ratio)) ^
                       named_parameters.growth_order
    @inbounds for mesh_index in eachindex(destination, mesh)
        size_factor_base = _growth_empirical_length_size_factor_base(
            gf, named_parameters, mesh[mesh_index])
        destination[mesh_index] = growth_prefactor *
                                  size_factor_base ^
                                  named_parameters.size_dependence_exponent
    end
    return destination
end

"""Allocate a mesh-sized length-dependent empirical growth field."""
function growthrate(gf::growth_empirical_length,
                    parameters::T,
                    prob::CrystallisationProblem,
                    state,
                    time) where {T <: AbstractVector}
    mesh = _length_mesh(prob.solver)
    rate_type = promote_type(eltype(parameters), eltype(state), eltype(mesh))
    destination = Vector{rate_type}(undef, length(mesh))
    return growthrate!(destination, gf, parameters, prob, state, time, mesh)
end

"""
    growthrate(gf::growth_energy, parameters::AbstractVector,
               problem::CrystallisationProblem, state, t) -> Real

Power-law growth with an Arrhenius temperature dependence.

# Arguments
- `parameters`: Vector [log10_growth_coefficient, activation_energy,
  growth_order], with activation energy in J/mol.
- `problem`: Crystallisation problem providing the saturation model and gas
  constant.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_energy, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _growth_energy_rate(p.log10_growth_coefficient, p.growth_order,
                               p.activation_energy, S, temp, prob.R_gas_constant)
end

"""
    growthrate(::growth_BCF, parameters::AbstractVector,
               problem::CrystallisationProblem, state, t) -> Real

Calculate crystal growth rate using Burton-Cabrera-Frank (BCF) surface diffusion model.

# Arguments
- `parameters`: Vector containing [growth_coefficient, activation_temperature]
  where the first entry is in m/s and the second is in K.
- `problem`: Crystallisation problem providing the saturation model.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

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
        p.growth_coefficient * temp / p.activation_temperature *
        (S - 1) *
        tanh(p.activation_temperature / (temp * log(S)))
    else
        0.0
    end
end

"""
    growthrate(::growth_BpS, parameters::AbstractVector,
               problem::CrystallisationProblem, state, t) -> Real

Calculate crystal growth rate using Birth and Spread (B+S) model based on nucleation theory.

# Arguments
- `parameters`: Vector containing [growth_coefficient,
  energy_barrier_temperature_squared], with the first entry in m/s and the
  second in K².
- `problem`: Crystallisation problem providing the saturation model.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

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
        p.growth_coefficient *
        ((S - 1)^(2 / 3)) *
        (log(S))^(1 / 6) *
        exp(-p.energy_barrier_temperature_squared / (temp^2 * log(S)))
    else
        0.0
    end
end

### Dissolution

function dissolutionrate!(destination::AbstractVector,
                          gf::growth_dissolution_length,
                          parameters,
                          prob::CrystallisationProblem,
                          state,
                          t,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growthrate! destination and mesh must have the same length."))

    named_parameters, driving_force = _length_dissolution_context(
        gf, parameters, prob, state, t)
    if driving_force == zero(driving_force)
        fill!(destination, zero(driving_force))
        return destination
    end

    dissolution_prefactor = _length_dissolution_prefactor(
        named_parameters, driving_force, prob, t)
    @inbounds for index in eachindex(destination, mesh)
        destination[index] = _length_dissolution_rate_from_context(
            gf, named_parameters, dissolution_prefactor, mesh[index])
    end
    return destination
end

"""
    dissolutionrate_at_length(gf::growth_dissolution_length, parameters,
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
    named_parameters, driving_force = _length_dissolution_context(
        gf, parameters, prob, state, t)
    driving_force == zero(driving_force) && return zero(driving_force)

    dissolution_prefactor = _length_dissolution_prefactor(
        named_parameters, driving_force, prob, t)
    return _length_dissolution_rate_from_context(
        gf, named_parameters, dissolution_prefactor, crystal_length)
end

"""
    dissolutionrate(gf::growth_dissolution, parameters::AbstractVector,
                    problem::CrystallisationProblem, state, t) -> Real

Scalar dissolution rate (negative growth) with Arrhenius temperature dependence.

# Arguments
- `parameters`: Vector [dissolution_coefficient, activation_energy,
  dissolution_order]; the coefficient is in m/s and activation energy in J/mol.
- `problem`: Crystallisation problem providing the saturation model, gas
  constant, and temperature profile.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Dissolution rate (negative m/s) if undersaturated, 0 otherwise
"""
function dissolutionrate(gf::growth_dissolution, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _dissolution_rate(p.dissolution_coefficient, p.activation_energy,
                             p.dissolution_order, S, temp, prob.R_gas_constant)
end
