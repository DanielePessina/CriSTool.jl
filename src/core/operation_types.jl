"""Abstract hydraulic operation, independent of kinetics and population solver."""
abstract type AbstractCrystallisationOperation end

"""
    BatchOperation(; volume=1.0)

Closed, fixed-volume reactor. `volume` is in m³; the default is a unit-volume
reference and does not change the existing concentration-based batch equations.
"""
struct BatchOperation{TV <: Real} <: AbstractCrystallisationOperation
    volume::TV
    function BatchOperation(; volume::Real = 1.0)
        _validate_operation_volume(volume)
        new{typeof(volume)}(volume)
    end
end

"""
    CrystallisationFeed(; concentration, crystals=nothing,
                         solvent_state=(;), transport=nothing)

One well-mixed slurry inlet. Concentration (kg/m³) may be constant or a callable
of time in seconds. `crystals` describes the fixed inlet population per m³ of
feed, independently of the initial reactor seed. Extra solvent variables require
an explicit `transport(problem, state, time, feed_values, inflow, volume)` hook
returning their named transport rates. Concentration transport is always built in;
the hook must not return a concentration rate. This avoids treating pH as a
linearly mixed concentration. Solvent dynamics describes internal rates only.
"""
struct CrystallisationFeed{TS <: NamedTuple, TC, TH}
    solvent_state::TS
    crystals::TC
    transport::TH
end
function CrystallisationFeed(; concentration,
                              crystals::Union{Nothing, AbstractInitialCrystals} = nothing,
                              solvent_state::NamedTuple = (;), transport = nothing)
    hasproperty(solvent_state, :concentration) &&
        throw(ArgumentError("Supply feed concentration once, using concentration=."))
    feed_values = merge((; concentration), solvent_state)
    concentration isa Real && _validate_feed_concentration(concentration)
    isnothing(crystals) || _validate_initial_crystals(crystals)
    return CrystallisationFeed(feed_values, crystals, transport)
end

"""
    MSMPROperation(; volume, inflow, feed)

Ideal fixed-volume mixed-suspension, mixed-product-removal operation. Inflow in
m³/s may be constant or a callable of time; the mixed slurry outlet equals the
inflow. Transient profiles are permitted; a steady-state runner requires constant
conditions. One clear or crystal-bearing feed is supported.
"""
struct MSMPROperation{TV <: Real, TQ, TF <: CrystallisationFeed} <: AbstractCrystallisationOperation
    volume::TV
    inflow::TQ
    feed::TF
    function MSMPROperation(; volume::Real, inflow, feed::CrystallisationFeed)
        _validate_operation_volume(volume)
        inflow isa Real && _validate_operation_inflow(inflow)
        new{typeof(volume), typeof(inflow), typeof(feed)}(volume, inflow, feed)
    end
end

"""
    FedBatchOperation(; initial_volume, inflow, feed)

Variable-volume operation with one inlet and no outlet: dV/dt = inflow.
`initial_volume` is in m³ and inflow is in m³/s (constant or callable of time).
Population and solute densities dilute as the volume increases.
"""
struct FedBatchOperation{TV <: Real, TQ, TF <: CrystallisationFeed} <: AbstractCrystallisationOperation
    initial_volume::TV
    inflow::TQ
    feed::TF
    function FedBatchOperation(; initial_volume::Real, inflow, feed::CrystallisationFeed)
        _validate_operation_volume(initial_volume)
        inflow isa Real && _validate_operation_inflow(inflow)
        new{typeof(initial_volume), typeof(inflow), typeof(feed)}(initial_volume, inflow, feed)
    end
end

function _validate_operation_volume(volume_value)
    isfinite(volume_value) && volume_value > 0 ||
        throw(ArgumentError("Reactor volume must be finite and strictly positive (m³)."))
    return volume_value
end
function _validate_operation_inflow(inflow_value)
    inflow_value isa Real && isfinite(inflow_value) && inflow_value >= 0 ||
        throw(ArgumentError("Inflow must be finite and nonnegative (m³/s)."))
    return inflow_value
end
function _validate_feed_concentration(concentration_value)
    concentration_value isa Real && isfinite(concentration_value) && concentration_value >= 0 ||
        throw(ArgumentError("Feed concentration must be finite and nonnegative (kg/m³)."))
    return concentration_value
end

_operation_state_count(::AbstractCrystallisationOperation) = 0
_operation_state_count(::FedBatchOperation) = 1
_operation_initial_state(::AbstractCrystallisationOperation) = Float64[]
_operation_initial_state(operation::FedBatchOperation) = [operation.initial_volume]
_operation_profile_value(profile::Real, simulation_time) = profile
_operation_profile_value(profile, simulation_time) = profile(simulation_time)
