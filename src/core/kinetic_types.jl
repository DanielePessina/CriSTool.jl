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
- `symbols::Vector{Symbol}`: Parameter symbols [:Aⱼ, :γ]
"""
Base.@kwdef @concrete struct nucl_CNT <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "CNT"
    symbols::Vector{Symbol} = [:Aⱼ, :γ]
end

paramaxis(::nucl_CNT) = ComponentArrays.Axis(Aj = 1, γ = 2)

"""
    nucl_empirical <: AbstractFPNucleationFunction

Empirical nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:Aj, :j]
"""
Base.@kwdef @concrete struct nucl_empirical <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "Emp. Nu"
    symbols::Vector{Symbol} = [:Aj, :j]
end

paramaxis(::nucl_empirical) = ComponentArrays.Axis(Aj = 1, j = 2)

"""
    nucl_empirical_energy <: AbstractFPNucleationFunction

Empirical nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (3)
- `string::String`: String identifier ("Emp. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:Aj, :Ea, :j]
"""
Base.@kwdef @concrete struct nucl_empirical_energy <: AbstractFPNucleationFunction
    nparams::Int64 = 3
    string::String = "Emp. Nu"
    symbols::Vector{Symbol} = [:Aj, :Ea, :j]
end

paramaxis(::nucl_empirical_energy) = ComponentArrays.Axis(Aj = 1, Ea = 2, j = 3)

"""
    nucl_CNTnoS <: AbstractFPNucleationFunction

Classical Nucleation Theory without supersaturation dependency.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("CNT no S")
- `symbols::Vector{Symbol}`: Parameter symbols [:Aⱼ, :γ]
"""
Base.@kwdef @concrete struct nucl_CNTnoS <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "CNT no S"
    symbols::Vector{Symbol} = [:Aⱼ, :γ]
end

paramaxis(::nucl_CNTnoS) = ComponentArrays.Axis(Aj = 1, γ = 2)

"""
    nucl_secondary <: AbstractFPNucleationFunction

Secondary nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (3)
- `string::String`: String identifier ("Sec. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:Aⱼ, :Ea, :j]
"""
Base.@kwdef @concrete struct nucl_secondary <: AbstractFPNucleationFunction
    nparams::Int64 = 3
    string::String = "Sec. Nu"
    symbols::Vector{Symbol} = [:Aⱼ, :Ea, :j]
end

paramaxis(::nucl_secondary) = ComponentArrays.Axis(Aj = 1, Ea = 2, j = 3)

"""
    nucl_prim_plus_second <: AbstractFPNucleationFunction

Primary plus secondary nucleation rate function.

Fields:
- `nparams::Int64`: Number of parameters (6)
- `string::String`: String identifier ("Prim. + Sec. Nu")
- `symbols::Vector{Symbol}`: Parameter symbols [:Aⱼ_prim, :Ea_prim, :j_prim, :Aⱼ_sec, :Ea_sec, :j_sec]
"""
Base.@kwdef @concrete struct nucl_prim_plus_second <: AbstractFPNucleationFunction
    nparams::Int64 = 6
    string::String = "Prim. + Sec. Nu"
    symbols::Vector{Symbol} = [:Aⱼ_prim, :Ea_prim, :j_prim, :Aⱼ_sec, :Ea_sec, :j_sec]
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
- `symbols::Vector{Symbol}`: Parameter symbols [:Aⱼ_CNT, :γ, :Aⱼ_sec, :Ea, :j]
"""
Base.@kwdef @concrete struct nucl_CNT_plus_second <: AbstractFPNucleationFunction
    nparams::Int64 = 5
    string::String = "CNT + Sec. Nu"
    symbols::Vector{Symbol} = [:Aⱼ_CNT, :γ, :Aⱼ_sec, :Ea, :j]
end

paramaxis(::nucl_CNT_plus_second) = ComponentArrays.Axis(cnt = ViewAxis(1:2, paramaxis(nucl_CNT())),
                                         sec = ViewAxis(3:5, paramaxis(nucl_secondary())))

"""
    nucl_CNT_fixed <: AbstractFPNucleationFunction

Classical Nucleation Theory (CNT) nucleation function with fixed (pre-set) parameters.

Fields:
- `nparams::Int64`: Number of free parameters (0, since parameters are fixed)
- `string::String`: String identifier ("CNT fixed")
- `Aj::Float64`: Pre-exponential nucleation rate constant
- `γ::Float64`: Interfacial tension parameter
"""
Base.@kwdef @concrete struct nucl_CNT_fixed <: AbstractFPNucleationFunction
    nparams::Int64 = 0
    string::String = "CNT fixed"
    Aj::Float64
    γ::Float64
end

paramaxis(::nucl_CNT_fixed) = ComponentArrays.Axis()

"""
    _fixkinetics(NuF::nucl_CNT, params::AbstractArray{<:Real}) -> nucl_CNT_fixed

Convert a `nucl_CNT` model to a `nucl_CNT_fixed` model with embedded parameters.

# Arguments
- `NuF::nucl_CNT`: The nucleation function to fix
- `params::AbstractArray{<:Real}`: Parameter array [Aj, γ]

# Returns
- `nucl_CNT_fixed`: Fixed nucleation function with embedded parameters
"""
function _fixkinetics(NuF::nucl_CNT, params::AbstractArray{<:Real})
    nucl_CNT_fixed(Aj = params[1], γ = params[2])
end

"""
    nucl_CNT_fixed(params::AbstractArray{<:Real}) -> nucl_CNT_fixed

Construct a `nucl_CNT_fixed` from a parameter array.

# Arguments
- `params::AbstractArray{<:Real}`: Parameter array [Aj, γ]

# Returns
- `nucl_CNT_fixed`: Fixed nucleation function with embedded parameters
"""
function nucl_CNT_fixed(params::AbstractArray{<:Real})
    nucl_CNT_fixed(Aj = params[1], γ = params[2])
end
"""
    nucl_empirical_fixed <: AbstractFPNucleationFunction

Empirical nucleation function with fixed (pre-set) parameters.

Fields:
- `nparams::Int64`: Number of free parameters (0, since parameters are fixed)
- `string::String`: String identifier ("Emp. Nu Fixed")
- `Aj::Float64`: Pre-exponential nucleation rate constant
- `j::Float64`: Supersaturation exponent
"""
Base.@kwdef @concrete struct nucl_empirical_fixed <: AbstractFPNucleationFunction
    nparams::Int64 = 0
    string::String = "Emp. Nu Fixed"
    Aj::Float64
    j::Float64
end

paramaxis(::nucl_empirical_fixed) = ComponentArrays.Axis()

"""
    _fixkinetics(NuF::nucl_empirical, params::AbstractArray{<:Real}) -> nucl_empirical_fixed

Convert a `nucl_empirical` model to a `nucl_empirical_fixed` model with embedded parameters.

# Arguments
- `NuF::nucl_empirical`: The nucleation function to fix
- `params::AbstractArray{<:Real}`: Parameter array [Aj, j]

# Returns
- `nucl_empirical_fixed`: Fixed nucleation function with embedded parameters
"""
function _fixkinetics(NuF::nucl_empirical, params::AbstractArray{<:Real})
    nucl_empirical_fixed(Aj = params[1], j = params[2])
end

"""
    nucl_empirical_fixed(params::AbstractArray{<:Real}) -> nucl_empirical_fixed

Construct a `nucl_empirical_fixed` from a parameter array.

# Arguments
- `params::AbstractArray{<:Real}`: Parameter array [Aj, j]

# Returns
- `nucl_empirical_fixed`: Fixed nucleation function with embedded parameters
"""
function nucl_empirical_fixed(params::AbstractArray{<:Real})
    nucl_empirical_fixed(Aj = params[1], j = params[2])
end


"""
    growth_empirical <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ag, :g]
"""
Base.@kwdef @concrete struct growth_empirical <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "Emp. Gr"
    symbols::Vector{Symbol} = [:Ag, :g]
end

paramaxis(::growth_empirical) = ComponentArrays.Axis(Ag = 1, g = 2)
"""
    growth_energy <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ag, :g]
- `Ea::Float64`: Activation energy (default: 0.0)
"""
Base.@kwdef @concrete struct growth_energy <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "GrEnergy"
    symbols::Vector{Symbol} = [:Ag, :g]
    Ea::Float64 = 53 * 1e3  # Activation energy in J/mol - default to 50 kJ/mol, range is supposedly 50-60 kJ/mol https://doi.org/10.1016/j.jcrysgro.2016.09.049
end

paramaxis(::growth_energy) = ComponentArrays.Axis(Ag = 1, g = 2)

"""
    growth_energy_est <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("Emp. Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ag, :g]
- `Ea::Float64`: Activation energy (default: 0.0)
"""
Base.@kwdef @concrete struct growth_energy_est <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 3
    string::String = "GrEnergy_Est"
    symbols::Vector{Symbol} = [:Ag, :Eag, :g]
end

paramaxis(::growth_energy_est) = ComponentArrays.Axis(Ag = 1, Eag = 2, g = 3)

"""
    growth_BCF <: AbstractFPScalarGrowthFunction

Burton-Cabrera-Frank (BCF) crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("BCF Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:C3, :C4]
"""
Base.@kwdef @concrete struct growth_BCF <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "BCF Gr"
    symbols::Vector{Symbol} = [:C3, :C4]
end

paramaxis(::growth_BCF) = ComponentArrays.Axis(C3 = 1, C4 = 2)

"""
    growth_BpS <: AbstractFPScalarGrowthFunction

Birth and Spread (B+S) crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("BpS Gr")
- `symbols::Vector{Symbol}`: Parameter symbols [:C1, :C2]
"""
Base.@kwdef @concrete struct growth_BpS <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "BpS Gr"
    symbols::Vector{Symbol} = [:C1, :C2]
end

paramaxis(::growth_BpS) = ComponentArrays.Axis(C1 = 1, C2 = 2)

"""
    growth_empirical_fixed <: AbstractFPScalarGrowthFunction

Empirical growth function with fixed (pre-set) parameters.

Fields:
- `nparams::Int64`: Number of free parameters (0, since parameters are fixed)
- `Ag::Float64`: Growth rate constant
- `g::Float64`: Supersaturation exponent
- `string::String`: String identifier ("Emp. Gr Fixed")
"""
Base.@kwdef @concrete struct growth_empirical_fixed <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 0
    Ag::Float64
    g::Float64
    string::String = "Emp. Gr Fixed"
end

paramaxis(::growth_empirical_fixed) = ComponentArrays.Axis()

"""
    _fixkinetics(GrF::growth_empirical, params::AbstractArray{<:Real}) -> growth_empirical_fixed

Convert a `growth_empirical` model to a `growth_empirical_fixed` model with embedded parameters.

# Arguments
- `GrF::growth_empirical`: The growth function to fix
- `params::AbstractArray{<:Real}`: Parameter array [Ag, g]

# Returns
- `growth_empirical_fixed`: Fixed growth function with embedded parameters
"""
function _fixkinetics(GrF::growth_empirical, params::AbstractArray{<:Real})
    growth_empirical_fixed(Ag = params[1], g = params[2])
end

"""
    growth_empirical_fixed(params::AbstractArray{<:Real}) -> growth_empirical_fixed

Construct a `growth_empirical_fixed` from a parameter array.

# Arguments
- `params::AbstractArray{<:Real}`: Parameter array [Ag, g]

# Returns
- `growth_empirical_fixed`: Fixed growth function with embedded parameters
"""
function growth_empirical_fixed(params::AbstractArray{<:Real})
    growth_empirical_fixed(Ag = params[1], g = params[2])
end


#### Dissolution
"""
    growth_dissolution <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (3)
- `string::String`: String identifier ("GrDissolution")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ad, :Ead, :d]
"""
Base.@kwdef @concrete struct growth_dissolution <: AbstractFPScalarDissolutionFunction
    nparams::Int64 = 3
    string::String = "GrDissolution"
    symbols::Vector{Symbol} = [:Ad, :Ead, :d]
end

paramaxis(::growth_dissolution) = ComponentArrays.Axis(Ad = 1, Ead = 2, d = 3)
"""
    growth_dissolution_length <: AbstractFPLengthGrowthFunction

Length-dependent dissolution growth function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (5)
- `string::String`: String identifier ("GrDissolution_length")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ad, :Ead, :d, :κ, :p]
"""
Base.@kwdef @concrete struct growth_dissolution_length <: AbstractFPLengthDissolutionFunction
    nparams::Int64 = 5
    string::String = "GrDissolution_length"
    symbols::Vector{Symbol} = [:Ad, :Ead, :d, :κ, :p]
    Lref::Float64 = CRISTOOL_DISSOLUTION_LREF
end

paramaxis(::growth_dissolution_length) = ComponentArrays.Axis(Ad = 1, Ead = 2, d = 3, κ = 4, p = 5)

"""
    growth_energy_dissolution <: AbstractFPScalarGrowthFunction

Combined scalar growth and dissolution rate function.

Fields:
- `nparams::Int64`: Number of parameters (5)
- `string::String`: String identifier ("GrEnergyDissolution")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ag, :g, :Ad, :Ead, :d]
"""
Base.@kwdef @concrete struct growth_energy_dissolution <: AbstractFPScalarDissolutionFunction
    nparams::Int64 = 5
    string::String = "GrEnergyDissolution"
    symbols::Vector{Symbol} = [:Ag, :g, :Ad, :Ead, :d]
end

paramaxis(::growth_energy_dissolution) = ComponentArrays.Axis(Ag = 1, g = 2, Ad = 3, Ead = 4, d = 5)


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
end

paramaxis(::breakage_empirical) = ComponentArrays.Axis(b = 1, n = 2)

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
end

paramaxis(::breakage_uniform) = ComponentArrays.Axis(logb = 1, n = 2)

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
end

paramaxis(::aggr_scalar) = ComponentArrays.Axis(logβ = 1)

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
end

paramaxis(::aggr_linear) = ComponentArrays.Axis(logβ = 1)

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
end

paramaxis(::aggr_linearvol) = ComponentArrays.Axis(logβ = 1)

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
end

paramaxis(::aggr_avg) = ComponentArrays.Axis(logβ = 1)
