## Kinetics
"""
    AbstractNucleationFunction

Abstract supertype for all nucleation laws.
"""
abstract type AbstractNucleationFunction end

"""
    AbstractGrowthFunction

Abstract supertype for crystal growth laws. Growth laws occupy the model's
`growth` slot and return a nonnegative rate; dissolution is a separate root,
[`AbstractDissolutionFunction`](@ref).
"""
abstract type AbstractGrowthFunction end

"""
    AbstractScalarGrowthFunction <: AbstractGrowthFunction

Growth laws whose rate is independent of crystal length.
"""
abstract type AbstractScalarGrowthFunction <: AbstractGrowthFunction end

"""
    AbstractLengthGrowthFunction <: AbstractGrowthFunction

Growth laws that return a rate per crystal length.
"""
abstract type AbstractLengthGrowthFunction <: AbstractGrowthFunction end

"""
    AbstractDissolutionFunction

Abstract supertype for dissolution laws. Dissolution laws occupy the model's
`dissolution` slot; the solvers add their (nonpositive) rate to the growth
rate to obtain the signed net crystal growth rate.
"""
abstract type AbstractDissolutionFunction end

"""
    AbstractScalarDissolutionFunction <: AbstractDissolutionFunction

Dissolution laws whose rate is independent of crystal length. The rate is
negative below saturation and zero in the equilibrium deadband.
"""
abstract type AbstractScalarDissolutionFunction <: AbstractDissolutionFunction end

"""
    AbstractLengthDissolutionFunction <: AbstractDissolutionFunction

Dissolution laws that return a rate per crystal length. They are supported by
the discretised solvers only.
"""
abstract type AbstractLengthDissolutionFunction <: AbstractDissolutionFunction end

"""
    AbstractAggregationFunction

Abstract supertype for all crystal aggregation functions.
"""
abstract type AbstractAggregationFunction end

"""
    AbstractBreakageFunction

Abstract supertype for all crystal breakage functions.
"""
abstract type AbstractBreakageFunction end

"""
    _KineticLaw

Every kinetic law family. A law is a value: its parameter values are fields,
like a `Distributions.jl` distribution, e.g.
`CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)`.
"""
const _KineticLaw = Union{AbstractNucleationFunction, AbstractGrowthFunction, AbstractDissolutionFunction,
                          AbstractAggregationFunction, AbstractBreakageFunction}

"""
    paramaxis(kinetic_law) -> ComponentArrays.Axis

Named parameter axis of a law, the single declaration of its parameters. The
default names every field, in field order, as one scalar parameter. Laws with
configuration fields (such as `Lref`) or non-scalar parameters define their
own method.
"""
paramaxis(kinetic_law::_KineticLaw) = _fieldname_axis(typeof(kinetic_law))

@generated function _fieldname_axis(::Type{KineticLawType}) where {KineticLawType}
    law_field_names = fieldnames(KineticLawType)
    return :(ComponentArrays.Axis(NamedTuple{$law_field_names}($(Tuple(1:length(law_field_names))))))
end

"""
    _parameter_count(kinetic_law) -> Int

Number of kinetic parameters declared by `paramaxis(kinetic_law)`; zero for an
empty axis. `paramaxis` is the single source of truth for a law's parameters.
"""
_parameter_count(kinetic_law) = ComponentArrays.last_index(paramaxis(kinetic_law))

"""
    _law_values(kinetic_law) -> NamedTuple

The law's parameter values, named and nested as `paramaxis(kinetic_law)`.
"""
_law_values(kinetic_law) = _axis_field_values(kinetic_law, keys(ComponentArrays.indexmap(paramaxis(kinetic_law))))
_axis_field_values(kinetic_law, law_parameter_names::Tuple{Vararg{Symbol}}) =
    NamedTuple{law_parameter_names}(map(law_parameter_name -> getfield(kinetic_law, law_parameter_name),
        law_parameter_names))

"""
    _law_with_values(kinetic_law, law_values::NamedTuple)

A copy of `kinetic_law` holding `law_values`, keeping its configuration. The
default rebuilds the law with `ConstructionBase.setproperties`.
"""
_law_with_values(kinetic_law, law_values::NamedTuple) = ConstructionBase.setproperties(kinetic_law, law_values)

"""
    _law_configuration(kinetic_law) -> NamedTuple

Configuration fields of a law that are not kinetic parameters, shown after the
parameter values.
"""
_law_configuration(kinetic_law) = (;)

"""
    NoNucleation <: AbstractNucleationFunction

Zero-rate nucleation law without parameters, for growth-only models.
"""
struct NoNucleation <: AbstractNucleationFunction end

"""
    NoDissolution <: AbstractScalarDissolutionFunction

Zero-rate dissolution law without parameters.
"""
struct NoDissolution <: AbstractScalarDissolutionFunction end

"""
    CNTNucleation(; ln_nucleation_prefactor, surface_energy) <: AbstractNucleationFunction

Classical Nucleation Theory (CNT) nucleation law.
Parameters: `ln_nucleation_prefactor`, `surface_energy` (J/m²).
"""
Base.@kwdef struct CNTNucleation{T <: Real} <: AbstractNucleationFunction
    ln_nucleation_prefactor::T
    surface_energy::T
end
CNTNucleation(ln_nucleation_prefactor::Real, surface_energy::Real) =
    CNTNucleation(promote(ln_nucleation_prefactor, surface_energy)...)

"""
    EmpiricalNucleation(; log10_nucleation_prefactor, nucleation_order) <: AbstractNucleationFunction

Empirical power-law nucleation.
Parameters: `log10_nucleation_prefactor`, `nucleation_order`.
"""
Base.@kwdef struct EmpiricalNucleation{T <: Real} <: AbstractNucleationFunction
    log10_nucleation_prefactor::T
    nucleation_order::T
end
EmpiricalNucleation(log10_nucleation_prefactor::Real, nucleation_order::Real) =
    EmpiricalNucleation(promote(log10_nucleation_prefactor, nucleation_order)...)

"""
    ArrheniusNucleation(; ln_nucleation_prefactor, activation_energy, nucleation_order) <: AbstractNucleationFunction

Empirical power-law nucleation with an Arrhenius temperature dependence.
Parameters: `ln_nucleation_prefactor`, `activation_energy` (J/mol),
`nucleation_order`.
"""
Base.@kwdef struct ArrheniusNucleation{T <: Real} <: AbstractNucleationFunction
    ln_nucleation_prefactor::T
    activation_energy::T
    nucleation_order::T
end
ArrheniusNucleation(ln_nucleation_prefactor::Real, activation_energy::Real, nucleation_order::Real) =
    ArrheniusNucleation(promote(ln_nucleation_prefactor, activation_energy, nucleation_order)...)

"""
    CNTNucleationNoSupersaturation(; ln_nucleation_prefactor, surface_energy) <: AbstractNucleationFunction

Classical Nucleation Theory without the supersaturation prefactor.
Parameters: `ln_nucleation_prefactor`, `surface_energy` (J/m²).
"""
Base.@kwdef struct CNTNucleationNoSupersaturation{T <: Real} <: AbstractNucleationFunction
    ln_nucleation_prefactor::T
    surface_energy::T
end
CNTNucleationNoSupersaturation(ln_nucleation_prefactor::Real, surface_energy::Real) =
    CNTNucleationNoSupersaturation(promote(ln_nucleation_prefactor, surface_energy)...)

"""
    SecondaryNucleation(; ln_nucleation_prefactor, activation_energy, nucleation_order) <: AbstractNucleationFunction

Secondary nucleation proportional to the crystal third moment.
Parameters: `ln_nucleation_prefactor`, `activation_energy` (J/mol),
`nucleation_order`.
"""
Base.@kwdef struct SecondaryNucleation{T <: Real} <: AbstractNucleationFunction
    ln_nucleation_prefactor::T
    activation_energy::T
    nucleation_order::T
end
SecondaryNucleation(ln_nucleation_prefactor::Real, activation_energy::Real, nucleation_order::Real) =
    SecondaryNucleation(promote(ln_nucleation_prefactor, activation_energy, nucleation_order)...)

"""
    CombinedNucleation(primary, secondary) <: AbstractNucleationFunction

Sum of two nucleation laws. Its parameters nest each component's axis under
`primary` and `secondary`, e.g.
`CombinedNucleation(CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4),
SecondaryNucleation(ln_nucleation_prefactor = 20.0, activation_energy = 0.0, nucleation_order = 2.0))`.
"""
struct CombinedNucleation{Primary <: AbstractNucleationFunction,
                          Secondary <: AbstractNucleationFunction} <: AbstractNucleationFunction
    primary::Primary
    secondary::Secondary
end

function paramaxis(combined_law::CombinedNucleation)
    primary_count = _parameter_count(combined_law.primary)
    secondary_count = _parameter_count(combined_law.secondary)
    return ComponentArrays.Axis(
        primary = ViewAxis(1:primary_count, paramaxis(combined_law.primary)),
        secondary = ViewAxis((primary_count + 1):(primary_count + secondary_count),
            paramaxis(combined_law.secondary)))
end
_law_values(combined_law::CombinedNucleation) =
    (primary = _law_values(combined_law.primary), secondary = _law_values(combined_law.secondary))
_law_with_values(combined_law::CombinedNucleation, law_values::NamedTuple) =
    CombinedNucleation(_law_with_values(combined_law.primary, law_values.primary),
        _law_with_values(combined_law.secondary, law_values.secondary))

"""
    EmpiricalGrowth(; growth_coefficient, growth_order) <: AbstractScalarGrowthFunction

Empirical power-law growth.
Parameters: `growth_coefficient` (m/s), `growth_order`.
"""
Base.@kwdef struct EmpiricalGrowth{T <: Real} <: AbstractScalarGrowthFunction
    growth_coefficient::T
    growth_order::T
end
EmpiricalGrowth(growth_coefficient::Real, growth_order::Real) =
    EmpiricalGrowth(promote(growth_coefficient, growth_order)...)

"""
    EmpiricalLengthGrowth(; growth_coefficient, growth_order, size_dependence_coefficient,
                          size_dependence_exponent, Lref = 1e-6) <: AbstractLengthGrowthFunction

Empirical growth multiplied by the length factor
`(1 + size_dependence_coefficient * L / Lref)^size_dependence_exponent`.
Parameters: `growth_coefficient`, `growth_order`, `size_dependence_coefficient`,
`size_dependence_exponent`. Configuration: `Lref` (m, default 1 µm), which is
not a kinetic parameter.
"""
Base.@kwdef struct EmpiricalLengthGrowth{T <: Real} <: AbstractLengthGrowthFunction
    growth_coefficient::T
    growth_order::T
    size_dependence_coefficient::T
    size_dependence_exponent::T
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end
EmpiricalLengthGrowth(growth_coefficient::Real, growth_order::Real, size_dependence_coefficient::Real,
        size_dependence_exponent::Real, Lref::Real = CRISTOOL_DISSOLUTION_LREF) =
    EmpiricalLengthGrowth(promote(growth_coefficient, growth_order, size_dependence_coefficient,
        size_dependence_exponent)..., Float64(Lref))

paramaxis(::EmpiricalLengthGrowth) = ComponentArrays.Axis(
    growth_coefficient = 1,
    growth_order = 2,
    size_dependence_coefficient = 3,
    size_dependence_exponent = 4)
_law_configuration(length_growth_law::EmpiricalLengthGrowth) = (; Lref = length_growth_law.Lref)

"""
    ArrheniusGrowth(; log10_growth_coefficient, activation_energy, growth_order) <: AbstractScalarGrowthFunction

Power-law growth with an Arrhenius temperature dependence.
Parameters: `log10_growth_coefficient`, `activation_energy` (J/mol),
`growth_order`. Fix the activation energy by leaving it out of the fitted
parameter selection.
"""
Base.@kwdef struct ArrheniusGrowth{T <: Real} <: AbstractScalarGrowthFunction
    log10_growth_coefficient::T
    activation_energy::T
    growth_order::T
end
ArrheniusGrowth(log10_growth_coefficient::Real, activation_energy::Real, growth_order::Real) =
    ArrheniusGrowth(promote(log10_growth_coefficient, activation_energy, growth_order)...)

"""
    BCFGrowth(; growth_coefficient, activation_temperature) <: AbstractScalarGrowthFunction

Burton-Cabrera-Frank (BCF) growth.
Parameters: `growth_coefficient` (m/s), `activation_temperature` (K).
"""
Base.@kwdef struct BCFGrowth{T <: Real} <: AbstractScalarGrowthFunction
    growth_coefficient::T
    activation_temperature::T
end
BCFGrowth(growth_coefficient::Real, activation_temperature::Real) =
    BCFGrowth(promote(growth_coefficient, activation_temperature)...)

"""
    BirthSpreadGrowth(; growth_coefficient, energy_barrier_temperature_squared) <: AbstractScalarGrowthFunction

Birth-and-spread growth.
Parameters: `growth_coefficient` (m/s), `energy_barrier_temperature_squared` (K²).
"""
Base.@kwdef struct BirthSpreadGrowth{T <: Real} <: AbstractScalarGrowthFunction
    growth_coefficient::T
    energy_barrier_temperature_squared::T
end
BirthSpreadGrowth(growth_coefficient::Real, energy_barrier_temperature_squared::Real) =
    BirthSpreadGrowth(promote(growth_coefficient, energy_barrier_temperature_squared)...)

#### Dissolution
"""
    ArrheniusDissolution(; dissolution_coefficient, activation_energy, dissolution_order) <: AbstractScalarDissolutionFunction

Scalar dissolution with an Arrhenius temperature dependence.
Parameters: `dissolution_coefficient` (m/s), `activation_energy` (J/mol),
`dissolution_order`.
"""
Base.@kwdef struct ArrheniusDissolution{T <: Real} <: AbstractScalarDissolutionFunction
    dissolution_coefficient::T
    activation_energy::T
    dissolution_order::T
end
ArrheniusDissolution(dissolution_coefficient::Real, activation_energy::Real, dissolution_order::Real) =
    ArrheniusDissolution(promote(dissolution_coefficient, activation_energy, dissolution_order)...)

"""
    ArrheniusLengthDissolution(; dissolution_coefficient, activation_energy, dissolution_order,
                               size_dependence_coefficient, size_dependence_exponent,
                               Lref = 1e-6) <: AbstractLengthDissolutionFunction

Arrhenius dissolution multiplied by the length factor
`(1 + size_dependence_coefficient * L / Lref)^size_dependence_exponent`.
Parameters: `dissolution_coefficient`, `activation_energy`,
`dissolution_order`, `size_dependence_coefficient`,
`size_dependence_exponent`. Configuration: `Lref` (m, default 1 µm), which is
not a kinetic parameter.
"""
Base.@kwdef struct ArrheniusLengthDissolution{T <: Real} <: AbstractLengthDissolutionFunction
    dissolution_coefficient::T
    activation_energy::T
    dissolution_order::T
    size_dependence_coefficient::T
    size_dependence_exponent::T
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end
ArrheniusLengthDissolution(dissolution_coefficient::Real, activation_energy::Real, dissolution_order::Real,
        size_dependence_coefficient::Real, size_dependence_exponent::Real,
        Lref::Real = CRISTOOL_DISSOLUTION_LREF) =
    ArrheniusLengthDissolution(promote(dissolution_coefficient, activation_energy, dissolution_order,
        size_dependence_coefficient, size_dependence_exponent)..., Float64(Lref))

paramaxis(::ArrheniusLengthDissolution) = ComponentArrays.Axis(
    dissolution_coefficient = 1, activation_energy = 2, dissolution_order = 3,
    size_dependence_coefficient = 4, size_dependence_exponent = 5)
_law_configuration(length_dissolution_law::ArrheniusLengthDissolution) = (; Lref = length_dissolution_law.Lref)

"""
    NoBreakage <: AbstractBreakageFunction

Zero-rate breakage law without parameters.
"""
struct NoBreakage <: AbstractBreakageFunction end

"""
    EmpiricalBreakage(; breakage_coefficient, breakage_size_exponent) <: AbstractBreakageFunction

Binary breakage with uniform-volume daughters and a power-law frequency.
Parameters: `breakage_coefficient`, `breakage_size_exponent`.
"""
Base.@kwdef struct EmpiricalBreakage{T <: Real} <: AbstractBreakageFunction
    breakage_coefficient::T
    breakage_size_exponent::T
end
EmpiricalBreakage(breakage_coefficient::Real, breakage_size_exponent::Real) =
    EmpiricalBreakage(promote(breakage_coefficient, breakage_size_exponent)...)

"""
    UniformBreakage(; ln_breakage_coefficient, breakage_size_exponent,
                    reference_length = 1e-6) <: AbstractBreakageFunction

Binary breakage with uniform-volume daughters and a log-parameterised
frequency. Parameters: `ln_breakage_coefficient`, `breakage_size_exponent`.
Configuration: `reference_length` (m, default 1 µm), which is not a kinetic
parameter.
"""
Base.@kwdef struct UniformBreakage{T <: Real} <: AbstractBreakageFunction
    ln_breakage_coefficient::T
    breakage_size_exponent::T
    reference_length::Float64 = 1e-6
end
UniformBreakage(ln_breakage_coefficient::Real, breakage_size_exponent::Real, reference_length::Real = 1e-6) =
    UniformBreakage(promote(ln_breakage_coefficient, breakage_size_exponent)..., Float64(reference_length))

paramaxis(::UniformBreakage) = ComponentArrays.Axis(ln_breakage_coefficient = 1,
                                                     breakage_size_exponent = 2)
_law_configuration(breakage_law::UniformBreakage) = (; reference_length = breakage_law.reference_length)

"""
    NoAggregation <: AbstractAggregationFunction

Zero-rate aggregation law without parameters.
"""
struct NoAggregation <: AbstractAggregationFunction end

"""
    ConstantAggregation(; log10_aggregation_coefficient) <: AbstractAggregationFunction

Size-independent aggregation kernel. Parameter: `log10_aggregation_coefficient`.
"""
Base.@kwdef struct ConstantAggregation{T <: Real} <: AbstractAggregationFunction
    log10_aggregation_coefficient::T
end

"""
    SizeSumAggregation(; log10_aggregation_coefficient) <: AbstractAggregationFunction

Aggregation kernel proportional to the sum of sizes.
Parameter: `log10_aggregation_coefficient`.
"""
Base.@kwdef struct SizeSumAggregation{T <: Real} <: AbstractAggregationFunction
    log10_aggregation_coefficient::T
end

"""
    VolumeSumAggregation(; log10_aggregation_coefficient) <: AbstractAggregationFunction

Aggregation kernel proportional to the sum of volumes.
Parameter: `log10_aggregation_coefficient`.
"""
Base.@kwdef struct VolumeSumAggregation{T <: Real} <: AbstractAggregationFunction
    log10_aggregation_coefficient::T
end

"""
    AverageAggregation(; log10_aggregation_coefficient) <: AbstractAggregationFunction

Aggregation kernel proportional to the mean of sizes.
Parameter: `log10_aggregation_coefficient`.
"""
Base.@kwdef struct AverageAggregation{T <: Real} <: AbstractAggregationFunction
    log10_aggregation_coefficient::T
end

## Kinetic traits. Each has a `false` default; custom laws opt in by adding a method.

"""
    _is_length_law(kinetic_law) -> Bool

Whether a growth or dissolution law returns a rate per crystal length rather
than one scalar rate. Subtypes of the `*Length*` families are length laws.
"""
_is_length_law(kinetic_law) = false
_is_length_law(::AbstractLengthGrowthFunction) = true
_is_length_law(::AbstractLengthDissolutionFunction) = true

"""
    _is_null_kinetic(kinetic_law) -> Bool

Whether a law is a zero-rate placeholder (`NoNucleation`, `NoDissolution`,
`NoAggregation`, `NoBreakage`), letting solvers skip the corresponding source.
"""
_is_null_kinetic(kinetic_law) = false
_is_null_kinetic(::Union{NoNucleation, NoDissolution, NoAggregation, NoBreakage}) = true

"""
    _has_moment_closure(kinetic_law) -> Bool

Whether an aggregation or breakage law has a validated QMOM/DQMOM moment
source closure.
"""
_has_moment_closure(kinetic_law) = false
_has_moment_closure(::Union{NoAggregation, ConstantAggregation, SizeSumAggregation, VolumeSumAggregation, AverageAggregation,
                            NoBreakage, EmpiricalBreakage, UniformBreakage}) = true

"""
    _is_time_invariant(model_component) -> Bool

Whether a kinetic law, temperature profile or solubility model depends on time
only through the state, so a steady MSMPR equilibrium is well defined. Opaque
callables and custom types default to `false`; steady solves then require
`autonomous = true`.
"""
_is_time_invariant(model_component) = false
_is_time_invariant(::Union{NoNucleation, CNTNucleation, EmpiricalNucleation, ArrheniusNucleation, CNTNucleationNoSupersaturation,
                           SecondaryNucleation, EmpiricalGrowth, EmpiricalLengthGrowth,
                           ArrheniusGrowth, BCFGrowth, BirthSpreadGrowth, NoDissolution,
                           ArrheniusDissolution, ArrheniusLengthDissolution,
                           NoBreakage, EmpiricalBreakage, UniformBreakage, NoAggregation,
                           ConstantAggregation, SizeSumAggregation, VolumeSumAggregation, AverageAggregation}) = true
_is_time_invariant(combined_law::CombinedNucleation) =
    _is_time_invariant(combined_law.primary) && _is_time_invariant(combined_law.secondary)
