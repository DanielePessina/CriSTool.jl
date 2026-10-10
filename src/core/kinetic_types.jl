## Kinetics
"""
    AbstractNucleationFunction

Abstract supertype for all nucleation laws.
"""
abstract type AbstractNucleationFunction end

"""
    AbstractFPNucleationFunction <: AbstractNucleationFunction

Abstract supertype for first-principles nucleation functions.
"""
abstract type AbstractFPNucleationFunction <: AbstractNucleationFunction end

"""
    AbstractGrowthFunction

Abstract supertype for crystal growth laws. Growth laws occupy the model's
`growth` slot and return a nonnegative rate; dissolution is a separate root,
[`AbstractDissolutionFunction`](@ref).
"""
abstract type AbstractGrowthFunction end

"""
    AbstractFPGrowthFunction <: AbstractGrowthFunction

Abstract supertype for first-principles growth functions.
"""
abstract type AbstractFPGrowthFunction <: AbstractGrowthFunction end

"""
    AbstractFPScalarGrowthFunction <: AbstractFPGrowthFunction

Growth laws whose rate is independent of crystal length.
"""
abstract type AbstractFPScalarGrowthFunction <: AbstractFPGrowthFunction end

"""
    AbstractFPLengthGrowthFunction <: AbstractFPGrowthFunction

Growth laws that return a rate per crystal length.
"""
abstract type AbstractFPLengthGrowthFunction <: AbstractFPGrowthFunction end

"""
    AbstractDissolutionFunction

Abstract supertype for dissolution laws. Dissolution laws occupy the model's
`dissolution` slot; the solvers add their (nonpositive) rate to the growth
rate to obtain the signed net crystal growth rate.
"""
abstract type AbstractDissolutionFunction end

"""
    AbstractFPScalarDissolutionFunction <: AbstractDissolutionFunction

Dissolution laws whose rate is independent of crystal length. The rate is
negative below saturation and zero in the equilibrium deadband.
"""
abstract type AbstractFPScalarDissolutionFunction <: AbstractDissolutionFunction end

"""
    AbstractFPLengthDissolutionFunction <: AbstractDissolutionFunction

Dissolution laws that return a rate per crystal length. They are supported by
the discretised solvers only.
"""
abstract type AbstractFPLengthDissolutionFunction <: AbstractDissolutionFunction end

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
# unknown names. `nucl_CNT()` is the unbound law, while
# `nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)` binds it.
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
    nodissolution <: AbstractFPScalarDissolutionFunction

Zero-rate dissolution law without parameters.
"""
@kinetic_law struct nodissolution <: AbstractFPScalarDissolutionFunction end

paramaxis(::nodissolution) = ComponentArrays.Axis()

"""
    nucl_CNT <: AbstractFPNucleationFunction

Classical Nucleation Theory (CNT) nucleation law.
Parameters: `ln_nucleation_prefactor`, `surface_energy` (J/m²).
"""
@kinetic_law struct nucl_CNT <: AbstractFPNucleationFunction end

paramaxis(::nucl_CNT) = ComponentArrays.Axis(ln_nucleation_prefactor = 1, surface_energy = 2)

"""
    nucl_empirical <: AbstractFPNucleationFunction

Empirical power-law nucleation.
Parameters: `log10_nucleation_prefactor`, `nucleation_order`.
"""
@kinetic_law struct nucl_empirical <: AbstractFPNucleationFunction end

paramaxis(::nucl_empirical) = ComponentArrays.Axis(log10_nucleation_prefactor = 1, nucleation_order = 2)

"""
    nucl_empirical_energy <: AbstractFPNucleationFunction

Empirical power-law nucleation with an Arrhenius temperature dependence.
Parameters: `ln_nucleation_prefactor`, `activation_energy` (J/mol),
`nucleation_order`.
"""
@kinetic_law struct nucl_empirical_energy <: AbstractFPNucleationFunction end

paramaxis(::nucl_empirical_energy) = ComponentArrays.Axis(ln_nucleation_prefactor = 1,
                                                          activation_energy = 2,
                                                          nucleation_order = 3)

"""
    nucl_CNTnoS <: AbstractFPNucleationFunction

Classical Nucleation Theory without the supersaturation prefactor.
Parameters: `ln_nucleation_prefactor`, `surface_energy` (J/m²).
"""
@kinetic_law struct nucl_CNTnoS <: AbstractFPNucleationFunction end

paramaxis(::nucl_CNTnoS) = paramaxis(nucl_CNT())

"""
    nucl_secondary <: AbstractFPNucleationFunction

Secondary nucleation proportional to the crystal third moment.
Parameters: `ln_nucleation_prefactor`, `activation_energy` (J/mol),
`nucleation_order`.
"""
@kinetic_law struct nucl_secondary <: AbstractFPNucleationFunction end

paramaxis(::nucl_secondary) = paramaxis(nucl_empirical_energy())

"""
    CombinedNucleation(primary, secondary) <: AbstractNucleationFunction

Sum of two nucleation laws. Its parameters nest each component's axis under
`primary` and `secondary`. Passing two bound `KineticModel`s returns the bound
combined model, e.g.
`CombinedNucleation(nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4),
nucl_secondary(ln_nucleation_prefactor = 20.0, activation_energy = 0.0, nucleation_order = 2.0))`.
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
    growth_empirical <: AbstractFPScalarGrowthFunction

Empirical power-law growth.
Parameters: `growth_coefficient` (m/s), `growth_order`.
"""
@kinetic_law struct growth_empirical <: AbstractFPScalarGrowthFunction end

paramaxis(::growth_empirical) = ComponentArrays.Axis(growth_coefficient = 1, growth_order = 2)

"""
    growth_empirical_length <: AbstractFPLengthGrowthFunction

Empirical growth multiplied by the length factor
`(1 + size_dependence_coefficient * L / Lref)^size_dependence_exponent`.
Parameters: `growth_coefficient`, `growth_order`, `size_dependence_coefficient`,
`size_dependence_exponent`. Configuration: `Lref` (m, default 1 µm).
"""
@kinetic_law struct growth_empirical_length <: AbstractFPLengthGrowthFunction
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end

paramaxis(::growth_empirical_length) = ComponentArrays.Axis(
    growth_coefficient = 1,
    growth_order = 2,
    size_dependence_coefficient = 3,
    size_dependence_exponent = 4)

"""
    growth_energy <: AbstractFPScalarGrowthFunction

Power-law growth with an Arrhenius temperature dependence.
Parameters: `log10_growth_coefficient`, `activation_energy` (J/mol),
`growth_order`. Fix the activation energy by leaving it out of the fitted
parameter selection.
"""
@kinetic_law struct growth_energy <: AbstractFPScalarGrowthFunction end

paramaxis(::growth_energy) = ComponentArrays.Axis(log10_growth_coefficient = 1,
                                                  activation_energy = 2,
                                                  growth_order = 3)

"""
    growth_BCF <: AbstractFPScalarGrowthFunction

Burton-Cabrera-Frank (BCF) growth.
Parameters: `growth_coefficient` (m/s), `activation_temperature` (K).
"""
@kinetic_law struct growth_BCF <: AbstractFPScalarGrowthFunction end

paramaxis(::growth_BCF) = ComponentArrays.Axis(growth_coefficient = 1,
                                               activation_temperature = 2)

"""
    growth_BpS <: AbstractFPScalarGrowthFunction

Birth-and-spread growth.
Parameters: `growth_coefficient` (m/s), `energy_barrier_temperature_squared` (K²).
"""
@kinetic_law struct growth_BpS <: AbstractFPScalarGrowthFunction end

paramaxis(::growth_BpS) = ComponentArrays.Axis(growth_coefficient = 1,
                                               energy_barrier_temperature_squared = 2)

#### Dissolution
"""
    growth_dissolution <: AbstractFPScalarDissolutionFunction

Scalar dissolution with an Arrhenius temperature dependence.
Parameters: `dissolution_coefficient` (m/s), `activation_energy` (J/mol),
`dissolution_order`.
"""
@kinetic_law struct growth_dissolution <: AbstractFPScalarDissolutionFunction end

paramaxis(::growth_dissolution) = ComponentArrays.Axis(dissolution_coefficient = 1,
                                                       activation_energy = 2,
                                                       dissolution_order = 3)

"""
    growth_dissolution_length <: AbstractFPLengthDissolutionFunction

Arrhenius dissolution multiplied by the length factor
`(1 + size_dependence_coefficient * L / Lref)^size_dependence_exponent`.
Parameters: `dissolution_coefficient`, `activation_energy`,
`dissolution_order`, `size_dependence_coefficient`,
`size_dependence_exponent`. Configuration: `Lref` (m, default 1 µm).
"""
@kinetic_law struct growth_dissolution_length <: AbstractFPLengthDissolutionFunction
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end

paramaxis(::growth_dissolution_length) = ComponentArrays.Axis(
    dissolution_coefficient = 1, activation_energy = 2, dissolution_order = 3,
    size_dependence_coefficient = 4, size_dependence_exponent = 5)

"""
    nobreakage <: AbstractBreakageFunction

Zero-rate breakage law without parameters.
"""
@kinetic_law struct nobreakage <: AbstractBreakageFunction end

paramaxis(::nobreakage) = ComponentArrays.Axis()

"""
    breakage_empirical <: AbstractBreakageFunction

Binary breakage with uniform-volume daughters and a power-law frequency.
Parameters: `breakage_coefficient`, `breakage_size_exponent`.
"""
@kinetic_law struct breakage_empirical <: AbstractBreakageFunction end

paramaxis(::breakage_empirical) = ComponentArrays.Axis(breakage_coefficient = 1,
                                                       breakage_size_exponent = 2)

"""
    breakage_uniform <: AbstractBreakageFunction

Binary breakage with uniform-volume daughters and a log-parameterised
frequency. Parameters: `ln_breakage_coefficient`, `breakage_size_exponent`.
Configuration: `reference_length` (m, default 1 µm).
"""
@kinetic_law struct breakage_uniform <: AbstractBreakageFunction
    reference_length::Float64 = 1e-6
end

paramaxis(::breakage_uniform) = ComponentArrays.Axis(ln_breakage_coefficient = 1,
                                                     breakage_size_exponent = 2)

"""
    noaggregation <: AbstractAggregationFunction

Zero-rate aggregation law without parameters.
"""
@kinetic_law struct noaggregation <: AbstractAggregationFunction end

paramaxis(::noaggregation) = ComponentArrays.Axis()

"""
    aggr_scalar <: AbstractAggregationFunction

Size-independent aggregation kernel. Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct aggr_scalar <: AbstractAggregationFunction end

paramaxis(::aggr_scalar) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    aggr_linear <: AbstractAggregationFunction

Aggregation kernel proportional to the sum of sizes.
Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct aggr_linear <: AbstractAggregationFunction end

paramaxis(::aggr_linear) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    aggr_linearvol <: AbstractAggregationFunction

Aggregation kernel proportional to the sum of volumes.
Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct aggr_linearvol <: AbstractAggregationFunction end

paramaxis(::aggr_linearvol) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    aggr_avg <: AbstractAggregationFunction

Aggregation kernel proportional to the mean of sizes.
Parameter: `log10_aggregation_coefficient`.
"""
@kinetic_law struct aggr_avg <: AbstractAggregationFunction end

paramaxis(::aggr_avg) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

## Kinetic traits. Each has a `false` default; custom laws opt in by adding a method.

"""
    _is_length_law(kinetic_law) -> Bool

Whether a growth or dissolution law returns a rate per crystal length rather
than one scalar rate. Subtypes of the `*Length*` families are length laws.
"""
_is_length_law(kinetic_law) = false
_is_length_law(::AbstractFPLengthGrowthFunction) = true
_is_length_law(::AbstractFPLengthDissolutionFunction) = true

"""
    _is_null_kinetic(kinetic_law) -> Bool

Whether a law is a zero-rate placeholder (`nodissolution`, `noaggregation`,
`nobreakage`), letting solvers skip the corresponding source.
"""
_is_null_kinetic(kinetic_law) = false
_is_null_kinetic(::Union{nodissolution, noaggregation, nobreakage}) = true

"""
    _has_moment_closure(kinetic_law) -> Bool

Whether an aggregation or breakage law has a validated QMOM/DQMOM moment
source closure.
"""
_has_moment_closure(kinetic_law) = false
_has_moment_closure(::Union{noaggregation, aggr_scalar, aggr_linear, aggr_linearvol, aggr_avg,
                            nobreakage, breakage_empirical, breakage_uniform}) = true

"""
    _is_time_invariant(model_component) -> Bool

Whether a kinetic law, temperature profile or solubility model depends on time
only through the state, so a steady MSMPR equilibrium is well defined. Opaque
callables and custom types default to `false`; steady solves then require
`autonomous = true`.
"""
_is_time_invariant(model_component) = false
_is_time_invariant(::Union{nucl_CNT, nucl_empirical, nucl_empirical_energy, nucl_CNTnoS,
                           nucl_secondary, growth_empirical, growth_empirical_length,
                           growth_energy, growth_BCF, growth_BpS, nodissolution,
                           growth_dissolution, growth_dissolution_length,
                           nobreakage, breakage_empirical, breakage_uniform, noaggregation,
                           aggr_scalar, aggr_linear, aggr_linearvol, aggr_avg}) = true
_is_time_invariant(combined_law::CombinedNucleation) =
    _is_time_invariant(combined_law.primary) && _is_time_invariant(combined_law.secondary)
