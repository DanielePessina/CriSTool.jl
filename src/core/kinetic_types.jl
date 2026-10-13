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

# Token for the inner constructor of `@kinetic_law` structs; it keeps the
# keyword constructor below the only zero-argument method of a field-free law.
struct _KineticLawConstruction end

# Defines a kinetic law with one keyword constructor. Struct fields are
# configuration with defaults; any other keywords are kinetic parameter values
# and return a bound `KineticModel`, whose constructor reports missing and
# unknown names. `CNTNucleation()` is the unbound law, while
# `CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)` binds it.
macro kinetic_law(law_definition)
    law_definition isa Expr && law_definition.head === :struct ||
        throw(ArgumentError("@kinetic_law expects a struct definition."))
    law_name = law_definition.args[2].args[1]
    typed_fields = Any[]
    field_names = Symbol[]
    field_defaults = Any[]
    for field_line in law_definition.args[3].args
        field_line isa LineNumberNode && continue
        field_line isa Expr && field_line.head === :(=) ||
            throw(ArgumentError("@kinetic_law configuration fields need defaults."))
        push!(typed_fields, field_line.args[1])
        push!(field_names, field_line.args[1].args[1])
        push!(field_defaults, field_line.args[2])
    end
    inner_constructor = :($law_name(::_KineticLawConstruction, $(field_names...)) = new($(field_names...)))
    plain_definition = Expr(:struct, law_definition.args[1], law_definition.args[2],
        Expr(:block, typed_fields..., inner_constructor))
    field_values = map(field_names, field_defaults) do field_name, default_value
        :(get(law_keywords, $(QuoteNode(field_name)), $default_value))
    end
    return esc(quote
        Base.@__doc__ $plain_definition
        function $law_name(; law_keywords...)
            kinetic_law = $law_name(_KineticLawConstruction(), $(field_values...))
            parameter_values = Base.structdiff(values(law_keywords), NamedTuple{$(Tuple(field_names))})
            return isempty(parameter_values) ? kinetic_law : KineticModel(kinetic_law; parameters = parameter_values)
        end
    end)
end

"""
    _parameter_count(kinetic_law) -> Int

Number of kinetic parameters declared by `paramaxis(kinetic_law)`; zero for an
empty axis. `paramaxis` is the single source of truth for a law's parameters.
"""
_parameter_count(kinetic_law) = ComponentArrays.last_index(paramaxis(kinetic_law))

"""
    NoNucleation <: AbstractNucleationFunction

Zero-rate nucleation law without parameters, for growth-only models.
"""
@kinetic_law struct NoNucleation <: AbstractNucleationFunction end

paramaxis(::NoNucleation) = ComponentArrays.Axis()

"""
    NoDissolution <: AbstractScalarDissolutionFunction

Zero-rate dissolution law without parameters.
"""
@kinetic_law struct NoDissolution <: AbstractScalarDissolutionFunction end

paramaxis(::NoDissolution) = ComponentArrays.Axis()

"""
    CNTNucleation <: AbstractNucleationFunction

Classical Nucleation Theory (CNT) nucleation law.
Parameters: `ln_nucleation_prefactor`, `surface_energy` (J/m²).
"""
@kinetic_law struct CNTNucleation <: AbstractNucleationFunction end

paramaxis(::CNTNucleation) = ComponentArrays.Axis(ln_nucleation_prefactor = 1, surface_energy = 2)

"""
    EmpiricalNucleation <: AbstractNucleationFunction

Empirical power-law nucleation.
Parameters: `log10_nucleation_prefactor`, `nucleation_order`.
"""
@kinetic_law struct EmpiricalNucleation <: AbstractNucleationFunction end

paramaxis(::EmpiricalNucleation) = ComponentArrays.Axis(log10_nucleation_prefactor = 1, nucleation_order = 2)

"""
    ArrheniusNucleation <: AbstractNucleationFunction

Empirical power-law nucleation with an Arrhenius temperature dependence.
Parameters: `ln_nucleation_prefactor`, `activation_energy` (J/mol),
`nucleation_order`.
"""
@kinetic_law struct ArrheniusNucleation <: AbstractNucleationFunction end

paramaxis(::ArrheniusNucleation) = ComponentArrays.Axis(ln_nucleation_prefactor = 1,
                                                          activation_energy = 2,
                                                          nucleation_order = 3)

"""
    CNTNucleationNoSupersaturation <: AbstractNucleationFunction

Classical Nucleation Theory without the supersaturation prefactor.
Parameters: `ln_nucleation_prefactor`, `surface_energy` (J/m²).
"""
@kinetic_law struct CNTNucleationNoSupersaturation <: AbstractNucleationFunction end

paramaxis(::CNTNucleationNoSupersaturation) = paramaxis(CNTNucleation())

"""
    SecondaryNucleation <: AbstractNucleationFunction

Secondary nucleation proportional to the crystal third moment.
Parameters: `ln_nucleation_prefactor`, `activation_energy` (J/mol),
`nucleation_order`.
"""
@kinetic_law struct SecondaryNucleation <: AbstractNucleationFunction end

paramaxis(::SecondaryNucleation) = paramaxis(ArrheniusNucleation())

"""
    CombinedNucleation(primary, secondary) <: AbstractNucleationFunction

Sum of two nucleation laws. Its parameters nest each component's axis under
`primary` and `secondary`. Passing two bound `KineticModel`s returns the bound
combined model, e.g.
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

"""
    EmpiricalGrowth <: AbstractScalarGrowthFunction

Empirical power-law growth.
Parameters: `growth_coefficient` (m/s), `growth_order`.
"""
@kinetic_law struct EmpiricalGrowth <: AbstractScalarGrowthFunction end

paramaxis(::EmpiricalGrowth) = ComponentArrays.Axis(growth_coefficient = 1, growth_order = 2)

"""
    EmpiricalLengthGrowth <: AbstractLengthGrowthFunction

Empirical growth multiplied by the length factor
`(1 + size_dependence_coefficient * L / Lref)^size_dependence_exponent`.
Parameters: `growth_coefficient`, `growth_order`, `size_dependence_coefficient`,
`size_dependence_exponent`. Configuration: `Lref` (m, default 1 µm).
"""
@kinetic_law struct EmpiricalLengthGrowth <: AbstractLengthGrowthFunction
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end

paramaxis(::EmpiricalLengthGrowth) = ComponentArrays.Axis(
    growth_coefficient = 1,
    growth_order = 2,
    size_dependence_coefficient = 3,
    size_dependence_exponent = 4)

"""
    ArrheniusGrowth <: AbstractScalarGrowthFunction

Power-law growth with an Arrhenius temperature dependence.
Parameters: `log10_growth_coefficient`, `activation_energy` (J/mol),
`growth_order`. Fix the activation energy by leaving it out of the fitted
parameter selection.
"""
@kinetic_law struct ArrheniusGrowth <: AbstractScalarGrowthFunction end

paramaxis(::ArrheniusGrowth) = ComponentArrays.Axis(log10_growth_coefficient = 1,
                                                  activation_energy = 2,
                                                  growth_order = 3)

"""
    BCFGrowth <: AbstractScalarGrowthFunction

Burton-Cabrera-Frank (BCF) growth.
Parameters: `growth_coefficient` (m/s), `activation_temperature` (K).
"""
@kinetic_law struct BCFGrowth <: AbstractScalarGrowthFunction end

paramaxis(::BCFGrowth) = ComponentArrays.Axis(growth_coefficient = 1,
                                               activation_temperature = 2)

"""
    BirthSpreadGrowth <: AbstractScalarGrowthFunction

Birth-and-spread growth.
Parameters: `growth_coefficient` (m/s), `energy_barrier_temperature_squared` (K²).
"""
@kinetic_law struct BirthSpreadGrowth <: AbstractScalarGrowthFunction end

paramaxis(::BirthSpreadGrowth) = ComponentArrays.Axis(growth_coefficient = 1,
                                               energy_barrier_temperature_squared = 2)

#### Dissolution
"""
    ArrheniusDissolution <: AbstractScalarDissolutionFunction

Scalar dissolution with an Arrhenius temperature dependence.
Parameters: `dissolution_coefficient` (m/s), `activation_energy` (J/mol),
`dissolution_order`.
"""
@kinetic_law struct ArrheniusDissolution <: AbstractScalarDissolutionFunction end

paramaxis(::ArrheniusDissolution) = ComponentArrays.Axis(dissolution_coefficient = 1,
                                                       activation_energy = 2,
                                                       dissolution_order = 3)

"""
    ArrheniusLengthDissolution <: AbstractLengthDissolutionFunction

Arrhenius dissolution multiplied by the length factor
`(1 + size_dependence_coefficient * L / Lref)^size_dependence_exponent`.
Parameters: `dissolution_coefficient`, `activation_energy`,
`dissolution_order`, `size_dependence_coefficient`,
`size_dependence_exponent`. Configuration: `Lref` (m, default 1 µm).
"""
@kinetic_law struct ArrheniusLengthDissolution <: AbstractLengthDissolutionFunction
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end

paramaxis(::ArrheniusLengthDissolution) = ComponentArrays.Axis(
    dissolution_coefficient = 1, activation_energy = 2, dissolution_order = 3,
    size_dependence_coefficient = 4, size_dependence_exponent = 5)

"""
    NoBreakage <: AbstractBreakageFunction

Zero-rate breakage law without parameters.
"""
@kinetic_law struct NoBreakage <: AbstractBreakageFunction end

paramaxis(::NoBreakage) = ComponentArrays.Axis()

"""
    EmpiricalBreakage <: AbstractBreakageFunction

Binary breakage with uniform-volume daughters and a power-law frequency.
Parameters: `breakage_coefficient`, `breakage_size_exponent`.
"""
@kinetic_law struct EmpiricalBreakage <: AbstractBreakageFunction end

paramaxis(::EmpiricalBreakage) = ComponentArrays.Axis(breakage_coefficient = 1,
                                                       breakage_size_exponent = 2)

"""
    UniformBreakage <: AbstractBreakageFunction

Binary breakage with uniform-volume daughters and a log-parameterised
frequency. Parameters: `ln_breakage_coefficient`, `breakage_size_exponent`.
Configuration: `reference_length` (m, default 1 µm).
"""
@kinetic_law struct UniformBreakage <: AbstractBreakageFunction
    reference_length::Float64 = 1e-6
end

paramaxis(::UniformBreakage) = ComponentArrays.Axis(ln_breakage_coefficient = 1,
                                                     breakage_size_exponent = 2)

"""
    NoAggregation <: AbstractAggregationFunction

Zero-rate aggregation law without parameters.
"""
@kinetic_law struct NoAggregation <: AbstractAggregationFunction end

paramaxis(::NoAggregation) = ComponentArrays.Axis()

"""
    ConstantAggregation <: AbstractAggregationFunction

Size-independent aggregation kernel. Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct ConstantAggregation <: AbstractAggregationFunction end

paramaxis(::ConstantAggregation) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    SizeSumAggregation <: AbstractAggregationFunction

Aggregation kernel proportional to the sum of sizes.
Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct SizeSumAggregation <: AbstractAggregationFunction end

paramaxis(::SizeSumAggregation) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    VolumeSumAggregation <: AbstractAggregationFunction

Aggregation kernel proportional to the sum of volumes.
Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct VolumeSumAggregation <: AbstractAggregationFunction end

paramaxis(::VolumeSumAggregation) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    AverageAggregation <: AbstractAggregationFunction

Aggregation kernel proportional to the mean of sizes.
Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct AverageAggregation <: AbstractAggregationFunction end

paramaxis(::AverageAggregation) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

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
