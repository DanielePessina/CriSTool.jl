## Kinetics
"""
    AbstractNucleationFunction

Abstract supertype for all nucleation functions in crystallization kinetics.
"""
abstract type AbstractNucleationFunction end

"""
    AbstractFPNucleationFunction <: AbstractNucleationFunction

Abstract supertype for first-principles nucleation functions.
"""
abstract type AbstractFPNucleationFunction <: AbstractNucleationFunction end

"""
    AbstractDDNucleationFunction <: AbstractNucleationFunction

Abstract supertype for data-driven nucleation functions.
"""
abstract type AbstractDDNucleationFunction <: AbstractNucleationFunction end

"""
    AbstractGrowthFunction

Abstract supertype for all crystal growth functions.
"""
abstract type AbstractGrowthFunction end

"""
    AbstractFPGrowthFunction <: AbstractGrowthFunction

Abstract supertype for first-principles growth functions.
"""
abstract type AbstractFPGrowthFunction <: AbstractGrowthFunction end

"""
    AbstractDDGrowthFunction <: AbstractGrowthFunction

Abstract supertype for data-driven growth functions.
"""
abstract type AbstractDDGrowthFunction <: AbstractGrowthFunction end

"""
    AbstractFPScalarGrowthFunction <: AbstractFPGrowthFunction

Abstract supertype for first-principles scalar growth functions.
"""
abstract type AbstractFPScalarGrowthFunction <: AbstractFPGrowthFunction end

"""
    AbstractDDScalarGrowthFunction <: AbstractDDGrowthFunction

Abstract supertype for data-driven scalar growth functions.
"""
abstract type AbstractDDScalarGrowthFunction <: AbstractDDGrowthFunction end

"""
    AbstractFPLengthGrowthFunction <: AbstractFPGrowthFunction

Abstract supertype for first-principles length-based growth functions.
"""
abstract type AbstractFPLengthGrowthFunction <: AbstractFPGrowthFunction end

"""
    AbstractDissolutionFunction

Abstract supertype for dissolution laws.  Dissolution is represented in a
separate parameter block from growth, but it remains below
`AbstractGrowthFunction` for source compatibility with the original
`growth_dissolution` names and runners.
"""
abstract type AbstractDissolutionFunction <: AbstractGrowthFunction end

"""
    AbstractFPScalarDissolutionFunction <: AbstractDissolutionFunction

Abstract family for first-principles scalar dissolution laws.  The returned
dissolution rate is negative below saturation and zero in the equilibrium
deadband.
"""
abstract type AbstractFPScalarDissolutionFunction <: AbstractDissolutionFunction end

"""
    AbstractFPLengthDissolutionFunction <: AbstractDissolutionFunction

Abstract family for first-principles length-dependent dissolution laws.  These
laws return a dissolution rate for each mesh length and are currently
supported by the discretised solvers only.
"""
abstract type AbstractFPLengthDissolutionFunction <: AbstractDissolutionFunction end

# Short aliases make the scalar/length distinction easy to discover while the
# FP-prefixed names remain consistent with the existing growth hierarchy.
const AbstractScalarDissolutionFunction = AbstractFPScalarDissolutionFunction
const AbstractLengthDissolutionFunction = AbstractFPLengthDissolutionFunction

"""
    nodissolution <: AbstractFPScalarDissolutionFunction

Zero-rate dissolution model.  It deliberately has no parameters, so adding
the independent dissolution block does not change the length of legacy
parameter vectors or the behaviour of existing simulations.
"""
Base.@kwdef @concrete struct nodissolution <: AbstractFPScalarDissolutionFunction
    nparams::Int64 = 0
    string::String = "No Dissolution"
    symbols::Vector{Symbol} = Symbol[]
end

paramaxis(::nodissolution) = ComponentArrays.Axis()

const nondissolution = nodissolution

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
    nucl_CNT <: AbstractFPNucleationFunction

Classical Nucleation Theory (CNT) nucleation function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("CNT")
- `symbols::Vector{Symbol}`: Parameter symbols [:ln_nucleation_prefactor, :surface_energy]
"""
Base.@kwdef @concrete struct nucl_CNT <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "CNT"
    symbols::Vector{Symbol} = [:ln_nucleation_prefactor, :surface_energy]
end

paramaxis(::nucl_CNT) = ComponentArrays.Axis(ln_nucleation_prefactor = 1, surface_energy = 2)

"""
    nucl_empirical <: AbstractFPNucleationFunction

Empirical nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:log10_nucleation_prefactor, :nucleation_order]
"""
Base.@kwdef @concrete struct nucl_empirical <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "Emp. Nu"
    symbols::Vector{Symbol} = [:log10_nucleation_prefactor, :nucleation_order]
end

paramaxis(::nucl_empirical) = ComponentArrays.Axis(log10_nucleation_prefactor = 1, nucleation_order = 2)

"""
    nucl_empirical_energy <: AbstractFPNucleationFunction

Empirical nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (3)
- `string::String`: String identifier ("Emp. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:ln_nucleation_prefactor, :activation_energy, :nucleation_order]
"""
Base.@kwdef @concrete struct nucl_empirical_energy <: AbstractFPNucleationFunction
    nparams::Int64 = 3
    string::String = "Emp. Nu"
    symbols::Vector{Symbol} = [:ln_nucleation_prefactor, :activation_energy, :nucleation_order]
end

paramaxis(::nucl_empirical_energy) = ComponentArrays.Axis(ln_nucleation_prefactor = 1,
                                                          activation_energy = 2,
                                                          nucleation_order = 3)

"""
    nucl_CNTnoS <: AbstractFPNucleationFunction

Classical Nucleation Theory without supersaturation dependency.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("CNT no S")
- `symbols::Vector{Symbol}`: Parameter symbols [:ln_nucleation_prefactor, :surface_energy]
"""
Base.@kwdef @concrete struct nucl_CNTnoS <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "CNT no S"
    symbols::Vector{Symbol} = [:ln_nucleation_prefactor, :surface_energy]
end

paramaxis(::nucl_CNTnoS) = paramaxis(nucl_CNT())

"""
    nucl_secondary <: AbstractFPNucleationFunction

Secondary nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (3)
- `string::String`: String identifier ("Sec. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:ln_nucleation_prefactor, :activation_energy, :nucleation_order]
"""
Base.@kwdef @concrete struct nucl_secondary <: AbstractFPNucleationFunction
    nparams::Int64 = 3
    string::String = "Sec. Nu"
    symbols::Vector{Symbol} = [:ln_nucleation_prefactor, :activation_energy, :nucleation_order]
end

paramaxis(::nucl_secondary) = paramaxis(nucl_empirical_energy())

"""
    nucl_prim_plus_second <: AbstractFPNucleationFunction

Primary plus secondary nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (6)
- `string::String`: String identifier ("Prim. + Sec. Nu")
- `symbols::Vector{Symbol}`: Flattened primary/secondary parameter symbols
"""
Base.@kwdef @concrete struct nucl_prim_plus_second <: AbstractFPNucleationFunction
    nparams::Int64 = 6
    string::String = "Prim. + Sec. Nu"
    symbols::Vector{Symbol} = [:ln_nucleation_prefactor_primary,
                               :activation_energy_primary,
                               :nucleation_order_primary,
                               :ln_nucleation_prefactor_secondary,
                               :activation_energy_secondary,
                               :nucleation_order_secondary]
end

paramaxis(::nucl_prim_plus_second) = ComponentArrays.Axis(prim = ViewAxis(1:3,
                                                          paramaxis(nucl_empirical_energy())),
                                          sec = ViewAxis(4:6, paramaxis(nucl_secondary())))

"""
    nucl_CNT_plus_second <: AbstractFPNucleationFunction

Classical nucleation plus secondary nucleation.

Fields:
- `nparams::Int64`: Number of parameters (5)
- `string::String`: String identifier ("CNT + Sec. Nu")
- `symbols::Vector{Symbol}`: Flattened CNT/secondary parameter symbols
"""
Base.@kwdef @concrete struct nucl_CNT_plus_second <: AbstractFPNucleationFunction
    nparams::Int64 = 5
    string::String = "CNT + Sec. Nu"
    symbols::Vector{Symbol} = [:ln_nucleation_prefactor_cnt, :surface_energy,
                               :ln_nucleation_prefactor_secondary,
                               :activation_energy_secondary,
                               :nucleation_order_secondary]
end

paramaxis(::nucl_CNT_plus_second) = ComponentArrays.Axis(cnt = ViewAxis(1:2, paramaxis(nucl_CNT())),
                                         sec = ViewAxis(3:5, paramaxis(nucl_secondary())))

"""
    nucl_CNT_fixed <: AbstractFPNucleationFunction

Classical Nucleation Theory (CNT) nucleation function with fixed (pre-set) parameters.

Fields:
- `nparams::Int64`: Number of free parameters (0, since parameters are fixed)
- `string::String`: String identifier ("CNT fixed")
- `ln_nucleation_prefactor::Float64`: Natural-log nucleation prefactor
- `surface_energy::Float64`: Interfacial energy (J/m²)
"""
Base.@kwdef @concrete struct nucl_CNT_fixed <: AbstractFPNucleationFunction
    nparams::Int64 = 0
    string::String = "CNT fixed"
    ln_nucleation_prefactor::Float64
    surface_energy::Float64
end

paramaxis(::nucl_CNT_fixed) = ComponentArrays.Axis()

"""
    _fixkinetics(NuF::nucl_CNT, params::AbstractArray{<:Real}) -> nucl_CNT_fixed

Convert a `nucl_CNT` model to a `nucl_CNT_fixed` model with embedded parameters.

# Arguments
- `NuF::nucl_CNT`: The nucleation function to fix
- `params::AbstractArray{<:Real}`: Parameter array [ln_nucleation_prefactor, surface_energy]

# Returns
- `nucl_CNT_fixed`: Fixed nucleation function with embedded parameters
"""
function _fixkinetics(NuF::nucl_CNT, params::AbstractArray{<:Real})
    nucl_CNT_fixed(ln_nucleation_prefactor = params[1], surface_energy = params[2])
end

"""
    nucl_CNT_fixed(params::AbstractArray{<:Real}) -> nucl_CNT_fixed

Construct a `nucl_CNT_fixed` from a parameter array.

# Arguments
- `params::AbstractArray{<:Real}`: Parameter array [ln_nucleation_prefactor, surface_energy]

# Returns
- `nucl_CNT_fixed`: Fixed nucleation function with embedded parameters
"""
function nucl_CNT_fixed(params::AbstractArray{<:Real})
    nucl_CNT_fixed(ln_nucleation_prefactor = params[1], surface_energy = params[2])
end
"""
    nucl_empirical_fixed <: AbstractFPNucleationFunction

Empirical nucleation function with fixed (pre-set) parameters.

Fields:
- `nparams::Int64`: Number of free parameters (0, since parameters are fixed)
- `string::String`: String identifier ("Emp. Nu Fixed")
- `log10_nucleation_prefactor::Float64`: Base-10 logarithm of the nucleation prefactor
- `nucleation_order::Float64`: Supersaturation exponent
"""
Base.@kwdef @concrete struct nucl_empirical_fixed <: AbstractFPNucleationFunction
    nparams::Int64 = 0
    string::String = "Emp. Nu Fixed"
    log10_nucleation_prefactor::Float64
    nucleation_order::Float64
end

paramaxis(::nucl_empirical_fixed) = ComponentArrays.Axis()

"""
    _fixkinetics(NuF::nucl_empirical, params::AbstractArray{<:Real}) -> nucl_empirical_fixed

Convert a `nucl_empirical` model to a `nucl_empirical_fixed` model with embedded parameters.

# Arguments
- `NuF::nucl_empirical`: The nucleation function to fix
- `params::AbstractArray{<:Real}`: Parameter array [log10_nucleation_prefactor, nucleation_order]

# Returns
- `nucl_empirical_fixed`: Fixed nucleation function with embedded parameters
"""
function _fixkinetics(NuF::nucl_empirical, params::AbstractArray{<:Real})
    nucl_empirical_fixed(log10_nucleation_prefactor = params[1], nucleation_order = params[2])
end

"""
    nucl_empirical_fixed(params::AbstractArray{<:Real}) -> nucl_empirical_fixed

Construct a `nucl_empirical_fixed` from a parameter array.

# Arguments
- `params::AbstractArray{<:Real}`: Parameter array [log10_nucleation_prefactor, nucleation_order]

# Returns
- `nucl_empirical_fixed`: Fixed nucleation function with embedded parameters
"""
function nucl_empirical_fixed(params::AbstractArray{<:Real})
    nucl_empirical_fixed(log10_nucleation_prefactor = params[1], nucleation_order = params[2])
end


"""
    growth_empirical <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:growth_coefficient, :growth_order]
"""
Base.@kwdef @concrete struct growth_empirical <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "Emp. Gr"
    symbols::Vector{Symbol} = [:growth_coefficient, :growth_order]
end

paramaxis(::growth_empirical) = ComponentArrays.Axis(growth_coefficient = 1, growth_order = 2)
"""
    growth_energy <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:log10_growth_coefficient, :growth_order]
- `activation_energy::Float64`: Activation energy (J/mol)
"""
Base.@kwdef @concrete struct growth_energy <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "GrEnergy"
    symbols::Vector{Symbol} = [:log10_growth_coefficient, :growth_order]
    activation_energy::Float64 = 53e3
end

paramaxis(::growth_energy) = ComponentArrays.Axis(log10_growth_coefficient = 1, growth_order = 2)

"""
    growth_energy_est <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:log10_growth_coefficient, :activation_energy, :growth_order]
"""
Base.@kwdef @concrete struct growth_energy_est <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 3
    string::String = "GrEnergy_Est"
    symbols::Vector{Symbol} = [:log10_growth_coefficient, :activation_energy, :growth_order]
end

paramaxis(::growth_energy_est) = ComponentArrays.Axis(log10_growth_coefficient = 1,
                                                      activation_energy = 2,
                                                      growth_order = 3)

"""
    growth_BCF <: AbstractFPScalarGrowthFunction

Burton-Cabrera-Frank (BCF) crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("BCF Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:growth_coefficient, :activation_temperature]
"""
Base.@kwdef @concrete struct growth_BCF <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "BCF Gr"
    symbols::Vector{Symbol} = [:growth_coefficient, :activation_temperature]
end

paramaxis(::growth_BCF) = ComponentArrays.Axis(growth_coefficient = 1,
                                               activation_temperature = 2)

"""
    growth_BpS <: AbstractFPScalarGrowthFunction

Birth and Spread (B+S) crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("BpS Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:growth_coefficient, :energy_barrier_temperature_squared]
"""
Base.@kwdef @concrete struct growth_BpS <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "BpS Gr"
    symbols::Vector{Symbol} = [:growth_coefficient, :energy_barrier_temperature_squared]
end

paramaxis(::growth_BpS) = ComponentArrays.Axis(growth_coefficient = 1,
                                               energy_barrier_temperature_squared = 2)

"""
    growth_empirical_fixed <: AbstractFPScalarGrowthFunction

Empirical growth function with fixed (pre-set) parameters.

Fields:
- `nparams::Int64`: Number of free parameters (0, since parameters are fixed)
- `growth_coefficient::Float64`: Growth coefficient (m/s)
- `growth_order::Float64`: Supersaturation exponent
- `string::String`: String identifier ("Emp. Gr Fixed")
"""
Base.@kwdef @concrete struct growth_empirical_fixed <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 0
    growth_coefficient::Float64
    growth_order::Float64
    string::String = "Emp. Gr Fixed"
end

paramaxis(::growth_empirical_fixed) = ComponentArrays.Axis()

"""
    _fixkinetics(GrF::growth_empirical, params::AbstractArray{<:Real}) -> growth_empirical_fixed

Convert a `growth_empirical` model to a `growth_empirical_fixed` model with embedded parameters.

# Arguments
- `GrF::growth_empirical`: The growth function to fix
- `params::AbstractArray{<:Real}`: Parameter array [growth_coefficient, growth_order]

# Returns
- `growth_empirical_fixed`: Fixed growth function with embedded parameters
"""
function _fixkinetics(GrF::growth_empirical, params::AbstractArray{<:Real})
    growth_empirical_fixed(growth_coefficient = params[1], growth_order = params[2])
end

"""
    growth_empirical_fixed(params::AbstractArray{<:Real}) -> growth_empirical_fixed

Construct a `growth_empirical_fixed` from a parameter array.

# Arguments
- `params::AbstractArray{<:Real}`: Parameter array [growth_coefficient, growth_order]

# Returns
- `growth_empirical_fixed`: Fixed growth function with embedded parameters
"""
function growth_empirical_fixed(params::AbstractArray{<:Real})
    growth_empirical_fixed(growth_coefficient = params[1], growth_order = params[2])
end


#### Dissolution
"""
    growth_dissolution <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (3)
- `string::String`: String identifier ("GrDissolution")
- `symbols::Vector{Symbol}`: Parameter symbols [:dissolution_coefficient, :activation_energy, :dissolution_order]
"""
Base.@kwdef @concrete struct growth_dissolution <: AbstractFPScalarDissolutionFunction
    nparams::Int64 = 3
    string::String = "GrDissolution"
    symbols::Vector{Symbol} = [:dissolution_coefficient, :activation_energy, :dissolution_order]
end

paramaxis(::growth_dissolution) = ComponentArrays.Axis(dissolution_coefficient = 1,
                                                       activation_energy = 2,
                                                       dissolution_order = 3)
"""
    growth_dissolution_length <: AbstractFPLengthGrowthFunction

Length-dependent dissolution growth function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (5)
- `string::String`: String identifier ("GrDissolution_length")
- `symbols::Vector{Symbol}`: Parameter symbols [:dissolution_coefficient, :activation_energy, :dissolution_order, :size_dependence_coefficient, :size_dependence_exponent]
"""
Base.@kwdef @concrete struct growth_dissolution_length <: AbstractFPLengthDissolutionFunction
    nparams::Int64 = 5
    string::String = "GrDissolution_length"
    symbols::Vector{Symbol} = [:dissolution_coefficient, :activation_energy,
                               :dissolution_order, :size_dependence_coefficient,
                               :size_dependence_exponent]
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end

paramaxis(::growth_dissolution_length) = ComponentArrays.Axis(
    dissolution_coefficient = 1, activation_energy = 2, dissolution_order = 3,
    size_dependence_coefficient = 4, size_dependence_exponent = 5)

"""
    growth_energy_dissolution <: AbstractFPScalarGrowthFunction

Combined scalar growth and dissolution rate function.

Fields:
- `nparams::Int64`: Number of parameters (5)
- `string::String`: String identifier ("GrEnergyDissolution")
- `symbols::Vector{Symbol}`: Parameter symbols [:log10_growth_coefficient, :growth_order, :dissolution_coefficient, :activation_energy, :dissolution_order]
"""
Base.@kwdef @concrete struct growth_energy_dissolution <: AbstractFPScalarDissolutionFunction
    nparams::Int64 = 5
    string::String = "GrEnergyDissolution"
    symbols::Vector{Symbol} = [:log10_growth_coefficient, :growth_order,
                               :dissolution_coefficient, :activation_energy,
                               :dissolution_order]
end

paramaxis(::growth_energy_dissolution) = ComponentArrays.Axis(
    log10_growth_coefficient = 1, growth_order = 2, dissolution_coefficient = 3,
    activation_energy = 4, dissolution_order = 5)


"""
    nobreakage <: AbstractBreakageFunction

Placeholder for simulations without crystal breakage.

Fields:
- `nparams::Int64`: Number of parameters (0)
- `string::String`: String identifier ("No Breakage")
- `symbols::Vector{Symbol}`: Empty parameter symbols
"""
Base.@kwdef @concrete struct nobreakage <: AbstractBreakageFunction
    nparams::Int64 = 0
    string::String = "No Breakage"
    symbols::Vector{Symbol} = []
end

paramaxis(::nobreakage) = ComponentArrays.Axis()

"""
    breakage_empirical <: AbstractBreakageFunction

Empirical crystal breakage function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Br")
"""
Base.@kwdef @concrete struct breakage_empirical <: AbstractBreakageFunction
    nparams::Int64 = 2
    string::String = "Emp. Br"
    symbols::Vector{Symbol} = [:breakage_coefficient, :breakage_size_exponent]
end

paramaxis(::breakage_empirical) = ComponentArrays.Axis(breakage_coefficient = 1,
                                                       breakage_size_exponent = 2)

"""
    breakage_uniform <: AbstractBreakageFunction

Uniform crystal breakage function (daughter fragments uniformly distributed).

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Uniform. Br")
"""
Base.@kwdef @concrete struct breakage_uniform <: AbstractBreakageFunction
    nparams::Int64 = 2
    string::String = "Uniform. Br"
    symbols::Vector{Symbol} = [:ln_breakage_coefficient, :breakage_size_exponent]
    reference_length::Float64 = 1e-6
end

paramaxis(::breakage_uniform) = ComponentArrays.Axis(ln_breakage_coefficient = 1,
                                                     breakage_size_exponent = 2)

"""
    noaggregation <: AbstractAggregationFunction

Placeholder for simulations without crystal aggregation.

Fields:
- `nparams::Int64`: Number of parameters (0)
- `string::String`: String identifier ("No Aggregation")
- `symbols::Vector{Symbol}`: Empty parameter symbols
"""
Base.@kwdef @concrete struct noaggregation <: AbstractAggregationFunction
    nparams::Int64 = 0
    string::String = "No Aggregation"
    symbols::Vector{Symbol} = []
end

paramaxis(::noaggregation) = ComponentArrays.Axis()

"""
    aggr_scalar <: AbstractAggregationFunction

Size-independent (scalar) aggregation kernel.

Fields:
- `nparams::Int64`: Number of parameters (1)
- `string::String`: String identifier ("Scalar Aggr")
"""
Base.@kwdef @concrete struct aggr_scalar <: AbstractAggregationFunction
    nparams::Int64 = 1
    string::String = "Scalar Aggr"
    symbols::Vector{Symbol} = [:log10_aggregation_coefficient]
end

paramaxis(::aggr_scalar) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    aggr_linear <: AbstractAggregationFunction

Linear size-dependent aggregation kernel (proportional to sum of sizes).

Fields:
- `nparams::Int64`: Number of parameters (1)
- `string::String`: String identifier ("Linear Aggr")
"""
Base.@kwdef @concrete struct aggr_linear <: AbstractAggregationFunction
    nparams::Int64 = 1
    string::String = "Linear Aggr"
    symbols::Vector{Symbol} = [:log10_aggregation_coefficient]
end

paramaxis(::aggr_linear) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    aggr_linearvol <: AbstractAggregationFunction

Linear volume-dependent aggregation kernel (proportional to sum of volumes).

Fields:
- `nparams::Int64`: Number of parameters (1)
- `string::String`: String identifier ("Linear Volume Aggr")
"""
Base.@kwdef @concrete struct aggr_linearvol <: AbstractAggregationFunction
    nparams::Int64 = 1
    string::String = "Linear Volume Aggr"
    symbols::Vector{Symbol} = [:log10_aggregation_coefficient]
end

paramaxis(::aggr_linearvol) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)

"""
    aggr_avg <: AbstractAggregationFunction

Average-based aggregation kernel.

Fields:
- `nparams::Int64`: Number of parameters (1)
- `string::String`: String identifier ("Average Aggr")
"""
Base.@kwdef @concrete struct aggr_avg <: AbstractAggregationFunction
    nparams::Int64 = 1
    string::String = "Average Aggr"
    symbols::Vector{Symbol} = [:log10_aggregation_coefficient]
end

paramaxis(::aggr_avg) = ComponentArrays.Axis(log10_aggregation_coefficient = 1)
