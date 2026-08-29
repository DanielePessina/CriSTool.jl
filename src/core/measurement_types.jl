## Measurements

"""
    AbstractObservable

Abstract supertype for observable containers (see `Observable`).
"""
abstract type AbstractObservable end

"""
    Observable{T,Tt,Tσ2} <: AbstractObservable

A single measured observable (concentration, d43, pH, ...), measured once or
as a time series. The SHAPE of the fields carries the semantics: a time
series has vector-valued `time` and `mean` (plus a per-point `variance` when
replicates exist); a final-state scalar has scalar `time` and `mean` (plus a
scalar `variance`). Branch on the shape by dispatch, e.g.
`f(obs::Observable{<:AbstractVector})` — never with runtime `isa` checks.

Fields:
- `mean::T`: measured value(s); `AbstractVector` for a time series
- `time::Tt`: measurement time(s): `Real` for a scalar, `AbstractVector` for a series
- `variance::Tσ2`: variance; `nothing` when unavailable (single replicate)
"""


Base.@kwdef @concrete struct Observable{T, Tt, Tσ2} <: AbstractObservable
    mean::T
    time::Tt = zero(mean)
    variance::Tσ2 = nothing
end

"""
    AbstractExperiment

Abstract supertype for a single experimental run (see `CrystallisationExperiment`).
"""
abstract type AbstractExperiment end

"""
    CrystallisationExperiment{O<:NamedTuple,M<:NamedTuple} <: AbstractExperiment

A single crystallisation experiment: a typed `NamedTuple` of observables
plus the run conditions.

Fields:
- `observables::O`: e.g. `(concentration = Observable(...), d43 = Observable(...), d50q = Observable(...))`.
  The `concentration` observable is mandatory for loss evaluation.
- `temperature::Float64`: run temperature in Kelvin
- `loading::Float64`: loading (e.g. volumetric solids fraction)
- `exp_id::Int`: experiment identifier
- `metadata::M`: additional typed run metadata

The `NamedTuple` shape keeps the container type-stable and Tables.jl
compatible; additional observables (pH, mass, PSD, ...) are added as new
fields, not new container types.
"""
Base.@kwdef @concrete struct CrystallisationExperiment{O <: NamedTuple,
                                                       M <: NamedTuple} <:
                 AbstractExperiment
    observables::O
    temperature::Float64
    loading::Float64
    exp_id::Int
    metadata::M = NamedTuple()
end

"""
    initial_concentration(expt::CrystallisationExperiment) -> Real

Initial solute concentration of an experiment, read from the
`concentration` series observable (first timepoint).
"""
initial_concentration(expt::CrystallisationExperiment) =
    expt.observables.concentration.mean[1]
