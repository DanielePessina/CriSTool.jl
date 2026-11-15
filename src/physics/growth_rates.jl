## Growth

"""
    growthrate!(destination, gf::AbstractScalarGrowthFunction, parameters,
                problem, state, time, mesh)

Fill a mesh-sized destination with a scalar signed growth rate.  Scalar models
use the same value at every mesh point; keeping this mutating seam alongside
the length-dependent implementation lets discretised RHSs reuse their
`DiffCache` storage.
"""
function growthrate!(destination::AbstractVector,
                     gf::AbstractScalarGrowthFunction,
                     parameters,
                     problem::DiscretisedProblem,
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
    growthrate_at_length(gf::AbstractScalarGrowthFunction, parameters,
                         problem, state, time, crystal_length)

Scalar growth/dissolution laws are independent of crystal length, so this
adapter returns the scalar rate without allocating a mesh-sized vector.
"""
growthrate_at_length(gf::AbstractScalarGrowthFunction,
                     parameters,
                     problem::DiscretisedProblem,
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
    dissolutionfunction::ArrheniusLengthDissolution,
    parameters,
    problem::DiscretisedProblem,
    state,
    time)
    named_parameters = _named_params(dissolutionfunction, parameters)
    supersaturation_ratio = supersaturation(problem, state, time)
    driving_force = _dissolution_drive(supersaturation_ratio)
    return named_parameters, driving_force
end

@inline _length_dissolution_prefactor(named_parameters, driving_force,
                                      problem::DiscretisedProblem, time) =
    _dissolution_prefactor(named_parameters.dissolution_coefficient,
                           named_parameters.activation_energy,
                           named_parameters.dissolution_order, driving_force,
                           temperature(problem.temp_profile, time),
                           CRISTOOL_GAS_CONSTANT)

"""
    _length_size_factor_base(law, named_parameters, crystal_length)

Base `1 + size_dependence_coefficient * crystal_length / Lref` of the
length-dependent multiplier shared by `EmpiricalLengthGrowth` and
`ArrheniusLengthDissolution`. It must stay positive so non-integer exponents
remain real-valued.
"""
@inline function _length_size_factor_base(
    law::Union{EmpiricalLengthGrowth, ArrheniusLengthDissolution},
    named_parameters,
    crystal_length::Real)
    size_factor_base = one(crystal_length) +
                       named_parameters.size_dependence_coefficient *
                       (crystal_length / law.Lref)
    size_factor_base > zero(size_factor_base) ||
        throw(DomainError(size_factor_base,
                          "$(nameof(typeof(law))) size factor must be positive."))
    return size_factor_base
end

@inline function _length_dissolution_rate_from_context(
    dissolutionfunction::ArrheniusLengthDissolution,
    named_parameters,
    dissolution_prefactor,
    crystal_length::Real)
    size_factor_base = _length_size_factor_base(
        dissolutionfunction, named_parameters, crystal_length)
    return dissolution_prefactor *
           size_factor_base^named_parameters.size_dependence_exponent
end

function _add_dissolution_rates!(destination::AbstractVector,
                                 dissolutionfunction::ArrheniusLengthDissolution,
                                 parameters,
                                 problem::DiscretisedProblem,
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

function _add_dissolution_rates!(destination::AbstractVector,
                                 dissolutionfunction::AbstractLengthDissolutionFunction,
                                 parameters,
                                 problem::DiscretisedProblem,
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

# Scalar dissolution adds one value to every mesh point.
function _add_dissolution_rates!(destination::AbstractVector,
                                 dissolutionfunction::AbstractDissolutionFunction,
                                 parameters,
                                 problem::DiscretisedProblem,
                                 state,
                                 time,
                                 mesh::AbstractVector)
    dissolution_value = dissolutionrate(dissolutionfunction, parameters, problem,
                                        state, time)
    @inbounds for mesh_index in eachindex(destination)
        destination[mesh_index] += dissolution_value
    end
    return destination
end

@inline _dissolution_prefactor(dissolution_coefficient, activation_energy,
                               dissolution_order, driving_force, temperature,
                               gas_constant) =
    -dissolution_coefficient *
    exp(-activation_energy / (gas_constant * temperature)) *
    driving_force^dissolution_order

@inline function _dissolution_rate(dissolution_coefficient, activation_energy,
                                   dissolution_order, supersaturation_ratio,
                                   temperature, gas_constant)
    driving_force = _dissolution_drive(supersaturation_ratio)
    return driving_force == zero(driving_force) ?
           zero(driving_force) :
           _dissolution_prefactor(dissolution_coefficient, activation_energy,
                                  dissolution_order, driving_force, temperature,
                                  gas_constant)
end

@inline _dissolution_zero(parameters, state) =
    zero(promote_type(eltype(parameters), eltype(state)))

"""Return the scalar dissolution contribution of a dissolution model."""
@inline dissolutionrate(::NoDissolution, parameters, problem, state, time) =
    _dissolution_zero(parameters, state)

"""Fill a mesh-sized destination with scalar dissolution."""
function dissolutionrate!(destination::AbstractVector,
                          dissolutionfunction::AbstractScalarDissolutionFunction,
                          parameters,
                          problem::DiscretisedProblem,
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
    dissolutionfunction::AbstractScalarDissolutionFunction,
    parameters,
    problem::DiscretisedProblem,
    state,
    time,
    crystal_length::Real) = dissolutionrate(dissolutionfunction, parameters,
                                             problem, state, time)

"""Fill a mesh-sized destination with length-dependent dissolution."""
function dissolutionrate!(destination::AbstractVector,
                          dissolutionfunction::AbstractLengthDissolutionFunction,
                          parameters,
                          problem::DiscretisedProblem,
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
function dissolutionrate(dissolutionfunction::AbstractLengthDissolutionFunction,
                         parameters,
                         problem::DiscretisedProblem,
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
           zero(promote_type(typeof(log10_growth_coefficient), typeof(growth_order),
                             typeof(activation_energy), typeof(supersaturation_ratio),
                             typeof(temperature)))
end

"""Return the signed net crystal growth rate from independent growth and
dissolution laws."""
@inline function net_growth_rate(growthfunction::AbstractGrowthFunction,
                                 growth_parameters,
                                 dissolutionfunction::AbstractDissolutionFunction,
                                 dissolution_parameters,
                                 problem::DiscretisedProblem,
                                 state,
                                 time)
    return growthrate(growthfunction, growth_parameters, problem, state, time) +
           dissolutionrate(dissolutionfunction, dissolution_parameters,
                           problem, state, time)
end

"""Without a dissolution law the net rate is the growth rate itself."""
@inline net_growth_rate(growthfunction::AbstractGrowthFunction,
                        growth_parameters,
                        ::NoDissolution,
                        dissolution_parameters,
                        problem::DiscretisedProblem,
                        state,
                        time) =
    growthrate(growthfunction, growth_parameters, problem, state, time)

"""Evaluate the net crystal growth rate at one physical crystal length."""
@inline function net_growth_rate_at_length(
    growthfunction::AbstractGrowthFunction,
    growth_parameters,
    dissolutionfunction::AbstractDissolutionFunction,
    dissolution_parameters,
    problem::DiscretisedProblem,
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
                          problem::DiscretisedProblem,
                          state,
                          time,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growth-rate destination and mesh must have the same length."))

    growthrate!(destination, growthfunction, growth_parameters, problem,
                state, time, mesh)
    _add_dissolution_rates!(destination, dissolutionfunction,
                            dissolution_parameters, problem, state, time, mesh)
    return destination
end

function _validate_dissolution_scalar_parameters(model, parameters)
    all(isfinite, parameters) ||
        throw(ArgumentError("$(typeof(model)) parameters must be finite."))
    return nothing
end

function _validate_dissolution_parameters(model::AbstractDissolutionFunction,
                                          parameters,
                                          problem::DiscretisedProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    return nothing
end

function _validate_dissolution_parameters(model::ArrheniusDissolution,
                                          parameters,
                                          problem::DiscretisedProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    _validate_arrhenius_dissolution_terms(_named_params(model, parameters))
    return nothing
end

function _validate_dissolution_parameters(model::ArrheniusLengthDissolution,
                                          parameters,
                                          problem::DiscretisedProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    isfinite(model.Lref) && model.Lref > 0.0 ||
        throw(ArgumentError("ArrheniusLengthDissolution Lref must be finite and strictly positive."))
    named_parameters = _named_params(model, parameters)
    _validate_arrhenius_dissolution_terms(named_parameters)
    _validate_mesh_size_factor(problem.solver, model, named_parameters)
    return nothing
end

"""Coefficient and order checks shared by the Arrhenius dissolution laws."""
function _validate_arrhenius_dissolution_terms(named_parameters)
    named_parameters.dissolution_coefficient >= 0.0 ||
        throw(ArgumentError("dissolution_coefficient must be nonnegative."))
    named_parameters.dissolution_order > 0.0 ||
        throw(ArgumentError("dissolution_order must be strictly positive."))
    return nothing
end

"""The length-dependent size factor must stay positive over a discretised mesh."""
function _validate_mesh_size_factor(solver::AbstractDiscretisedSolver,
                                    model::ArrheniusLengthDissolution,
                                    named_parameters)
    lower_base = 1.0 + named_parameters.size_dependence_coefficient * solver.lmin / model.Lref
    upper_base = 1.0 + named_parameters.size_dependence_coefficient * solver.lmax / model.Lref
    lower_base > 0.0 && upper_base > 0.0 ||
        throw(ArgumentError("ArrheniusLengthDissolution size factor must stay positive " *
                            "over the configured mesh."))
    return nothing
end
_validate_mesh_size_factor(solver::AbstractSolver, model, named_parameters) = nothing

function _validate_growth_parameters(model::EmpiricalLengthGrowth,
                                     parameters,
                                     problem::DiscretisedProblem)
    _validate_dissolution_scalar_parameters(model, parameters)
    isfinite(model.Lref) && model.Lref > 0.0 ||
        throw(ArgumentError("EmpiricalLengthGrowth Lref must be finite and strictly positive."))
    p = _named_params(model, parameters)
    p.growth_coefficient >= 0.0 ||
        throw(ArgumentError("growth_coefficient must be nonnegative."))
    p.growth_order > 0.0 ||
        throw(ArgumentError("growth_order must be strictly positive."))
    return nothing
end

"""
    growthrate(::EmpiricalGrowth, parameters::AbstractVector,
               problem::DiscretisedProblem, state, t) -> Real

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
function growthrate(gf::EmpiricalGrowth, parameters::T, prob::DiscretisedProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ? p.growth_coefficient * (S - 1)^p.growth_order : _inactive_rate(parameters, S, temp)
end

"""
    growthrate_at_length(gf::EmpiricalLengthGrowth, parameters,
                         problem, state, time, crystal_length)

Evaluate the empirical growth law with the length-dependent multiplier

`(1 + size_dependence_coefficient * crystal_length / Lref)^size_dependence_exponent`.
The returned rate is in metres per second.  The multiplier base must remain
positive so non-integer exponents remain real-valued.
"""
@inline function growthrate_at_length(gf::EmpiricalLengthGrowth,
                                      parameters,
                                      prob::DiscretisedProblem,
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

    size_factor_base = _length_size_factor_base(
        gf, named_parameters, crystal_length)
    return named_parameters.growth_coefficient *
           (supersaturation_ratio - one(supersaturation_ratio)) ^
           named_parameters.growth_order *
           size_factor_base ^ named_parameters.size_dependence_exponent
end

"""Fill a mesh-sized destination with length-dependent empirical growth."""
function growthrate!(destination::AbstractVector,
                     gf::EmpiricalLengthGrowth,
                     parameters,
                     prob::DiscretisedProblem,
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
        size_factor_base = _length_size_factor_base(
            gf, named_parameters, mesh[mesh_index])
        destination[mesh_index] = growth_prefactor *
                                  size_factor_base ^
                                  named_parameters.size_dependence_exponent
    end
    return destination
end

"""Allocate a mesh-sized length-dependent empirical growth field."""
function growthrate(gf::EmpiricalLengthGrowth,
                    parameters::T,
                    prob::DiscretisedProblem,
                    state,
                    time) where {T <: AbstractVector}
    mesh = _length_mesh(prob.solver)
    rate_type = promote_type(eltype(parameters), eltype(state), eltype(mesh))
    destination = Vector{rate_type}(undef, length(mesh))
    return growthrate!(destination, gf, parameters, prob, state, time, mesh)
end

"""
    growthrate(gf::ArrheniusGrowth, parameters::AbstractVector,
               problem::DiscretisedProblem, state, t) -> Real

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
function growthrate(gf::ArrheniusGrowth, parameters::T, prob::DiscretisedProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _growth_energy_rate(p.log10_growth_coefficient, p.growth_order,
                               p.activation_energy, S, temp, CRISTOOL_GAS_CONSTANT)
end

"""
    growthrate(::BCFGrowth, parameters::AbstractVector,
               problem::DiscretisedProblem, state, t) -> Real

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
function growthrate(gf::BCFGrowth, parameters::T, prob::DiscretisedProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        p.growth_coefficient * temp / p.activation_temperature *
        (S - 1) *
        tanh(p.activation_temperature / (temp * log(S)))
    else
        _inactive_rate(parameters, S, temp)
    end
end

"""
    growthrate(::BirthSpreadGrowth, parameters::AbstractVector,
               problem::DiscretisedProblem, state, t) -> Real

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
function growthrate(gf::BirthSpreadGrowth, parameters::T, prob::DiscretisedProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        p.growth_coefficient *
        ((S - 1)^(2 / 3)) *
        (log(S))^(1 / 6) *
        exp(-p.energy_barrier_temperature_squared / (temp^2 * log(S)))
    else
        _inactive_rate(parameters, S, temp)
    end
end

### Dissolution

function dissolutionrate!(destination::AbstractVector,
                          gf::ArrheniusLengthDissolution,
                          parameters,
                          prob::DiscretisedProblem,
                          state,
                          t,
                          mesh::AbstractVector)
    length(destination) == length(mesh) ||
        throw(ArgumentError("growthrate! destination and mesh must have the same length."))

    fill!(destination, 0)
    return _add_dissolution_rates!(destination, gf, parameters, prob, state, t, mesh)
end

"""
    dissolutionrate_at_length(gf::ArrheniusLengthDissolution, parameters,
                              prob, state, t, length)

Evaluate a length-dependent signed dissolution rate at one crystal length
without constructing a mesh-sized result.
"""
function dissolutionrate_at_length(gf::ArrheniusLengthDissolution,
                                   parameters,
                                   prob::DiscretisedProblem,
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
    dissolutionrate(gf::ArrheniusDissolution, parameters::AbstractVector,
                    problem::DiscretisedProblem, state, t) -> Real

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
function dissolutionrate(gf::ArrheniusDissolution, parameters::T, prob::DiscretisedProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return _dissolution_rate(p.dissolution_coefficient, p.activation_energy,
                             p.dissolution_order, S, temp, CRISTOOL_GAS_CONSTANT)
end
