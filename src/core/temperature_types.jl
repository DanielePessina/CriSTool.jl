############### Temperature strategy ########################

"""
    AbstractTemperature

Abstract supertype for temperature profile strategies in crystallization simulations.
"""
abstract type AbstractTemperature end

"""
    ConstantTemperature{T<:Real} <: AbstractTemperature

Constant temperature profile.

Fields:
- `value::T`: Temperature value (K)
"""
struct ConstantTemperature{T <: Real} <: AbstractTemperature
    value::T
end

"""
    temperature(ct::ConstantTemperature, t) -> Real

Return the constant temperature value (ignores time argument).

# Arguments
- `ct::ConstantTemperature`: Temperature profile
- `t`: Time (unused)

# Returns
- Temperature value
"""
@inline temperature(ct::ConstantTemperature, t) = ct.value

"""
    temperature(ct::ConstantTemperature) -> Real

Return the constant temperature value.

# Arguments
- `ct::ConstantTemperature`: Temperature profile

# Returns
- Temperature value
"""
@inline temperature(ct::ConstantTemperature) = temperature(ct, 0.0)

"""
    LinearTemperature{T<:Real} <: AbstractTemperature

Linear temperature profile (cooling or heating ramp).

Fields:
- `T0::T`: Initial temperature (K)
- `slope::T`: Temperature change rate (K/s), negative for cooling
"""
struct LinearTemperature{T <: Real} <: AbstractTemperature
    T0::T
    slope::T  # K s⁻¹
end

"""
    temperature(lp::LinearTemperature, t) -> Real

Compute temperature at time t for a linear profile.

# Arguments
- `lp::LinearTemperature`: Temperature profile
- `t`: Time (s)

# Returns
- Temperature at time t: T0 + slope × t
"""
temperature(lp::LinearTemperature, t) = lp.T0 + lp.slope * t

"""
    temperature(lp::LinearTemperature) -> Real

Return the initial temperature (at t=0).

# Arguments
- `lp::LinearTemperature`: Temperature profile

# Returns
- Initial temperature T0
"""
@inline temperature(lp::LinearTemperature) = temperature(lp, 0.0)

"""
    RampTemperature{T<:Real} <: AbstractTemperature

Ramp temperature profile (cooling or heating ramp) with defined start and end times.

Fields:
- `T0::T`: Initial temperature (K)
- `t0::T`: Time when ramp starts (s)
- `t1::T`: Time when ramp ends and temperature holds constant (s)
- `slope::T`: Temperature change rate (K/s), negative for cooling
"""
struct RampTemperature{T <: Real} <: AbstractTemperature
    T0::T
    t0::T  # s
    t1::T  # s - time when cooling/heating stops
    slope::T  # K s⁻¹
end

"""
    temperature(rp::RampTemperature, t) -> Real

Compute temperature at time t for a ramp profile with start and end times.

# Arguments
- `rp::RampTemperature`: Temperature profile
- `t`: Time (s)

# Returns
- `T0` if `t < t0` (before ramp starts)
- `T0 + slope × (t - t0)` if `t0 <= t < t1` (during ramp)
- `T0 + slope × (t1 - t0)` if `t >= t1` (after ramp ends, temperature holds constant)
"""
function temperature(rp::RampTemperature, t)
    if t < rp.t0
        return rp.T0
    elseif t < rp.t1
        return rp.T0 + rp.slope * (t - rp.t0)
    else
        return rp.T0 + rp.slope * (rp.t1 - rp.t0)
    end
end


"""
    CallableTemperature{F} <: AbstractTemperature

Generic temperature profile from a user-provided callable.

Fields:
- `f::F`: Callable that takes time and returns temperature
"""
struct CallableTemperature{F} <: AbstractTemperature
    f::F
end

"""
    temperature(c::CallableTemperature, t) -> Real

Evaluate the callable temperature function at time t.

# Arguments
- `c::CallableTemperature`: Temperature profile with callable
- `t`: Time (s)

# Returns
- Temperature at time t from c.f(t)
"""
temperature(c::CallableTemperature, t) = c.f(t)

"""
    temperature(c::CallableTemperature) -> Real

Evaluate the callable temperature function at t=0.

# Arguments
- `c::CallableTemperature`: Temperature profile with callable

# Returns
- Temperature at t=0
"""
@inline temperature(c::CallableTemperature) = temperature(c, 0.0)
##############################################################

##############################################################
