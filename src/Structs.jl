"""
Data structures for crystallisation problems, kinetics, solutions and measurements. Provides typed containers used throughout CriSTool.
"""

## Solution structs
"""
    AbstractSolution

Abstract supertype for all solution types in crystallization simulations.
"""
abstract type AbstractSolution end
"""
    CrystallisationFVSolution{Tt,TCo,TNn,TVd,TDq} <: AbstractSolution where {Tt<:Real,TCo<:Real,TNn<:Real,TVd<:Real,TDq<:Real}

    Stores finite volume solver results.

    Fields:
    - `time::Vector{Tt}`: Time points vector
    - `concentration::Vector{TCo}`: Concentration vector
    - `numberdensity::Array{TNn}`: Number density array
    - `voldensity::Array{TVd}`: Volume density array
    - `d50q::Vector{TDq}`: Median size vector
    - `ode_stats`: ODE solver statistics (e.g., DEStats) or `nothing`
    - `success::Bool`: Success flag

"""
@concrete struct CrystallisationFVSolution{Tt, TCo, TNn, TVd, TDq, TDm, TStats} <:
                 AbstractSolution where {Tt <: AbstractArray{<:Real},
                                         TCo <: AbstractArray{<:Real},
                                         TNn <: AbstractArray{<:Real},
                                         TVd <: AbstractArray{<:Real},
                                         TDq <: AbstractArray{<:Real},
                                         TDm <: AbstractArray{<:Real},
                                         TStats}
    time::Tt
    concentration::TCo
    numberdensity::TNn
    voldensity::TVd
    d10q::TDq
    d50q::TDq
    d90q::TDq
    d10::TDm
    d32::TDm
    d43::TDm
    mu2::TDm

    final_state::TCo
    ode_stats::TStats
    success::Bool

end

"""
    CrystallisationMoMSolution{Tt,TCo,TD1,TD3,TD4,TMu2} <: AbstractSolution where {Tt<:AbstractArray{<:Real},TCo<:AbstractArray{<:Real},TD1<:AbstractArray{<:Real},TD3<:AbstractArray{<:Real},TD4<:AbstractArray{<:Real},TMu2<:AbstractArray{<:Real}}

    Stores Method of Moments solver results.

    Fields:
    - `time::Tt`: Time points vector
    - `concentration::TCo`: Concentration vector
    - `d10::TD1`: 10th percentile of number density
    - `d32::TD3`: Sauter mean diameter
    - `d43::TD4`: Volume mean diameter
    - `mu2::TMu2`: Second moment of number density
    - `ode_stats`: ODE solver statistics (e.g., DEStats) or `nothing`
    - `success::Bool`: Success flag

"""
@concrete struct CrystallisationMoMSolution{Tt, TCo, TD1, TD3, TD4, TMu2, TStats} <:
                 AbstractSolution where {Tt <: AbstractArray{<:Real},
                                         TCo <: AbstractArray{<:Real},
                                         TD1 <: AbstractArray{<:Real},
                                         TD3 <: AbstractArray{<:Real},
                                         TD4 <: AbstractArray{<:Real},
                                         TMu2 <: AbstractArray{<:Real},
                                         TStats}
    time::Tt
    concentration::TCo
    d10::TD1
    d32::TD3
    d43::TD4

    mu2::TMu2

    final_state::TCo
    ode_stats::TStats
    success::Bool

end

"""
    EnsembleFVSolution{T<:AbstractFloat} <: AbstractSolution

    Stores ensemble simulation results and statistics from finite volume solver.

    Fields:
    - `concentration::Matrix{T}`: Matrix of concentration trajectories (samples × timepoints)
    - `time::Vector{T}`: Time points vector
    - `d43::Vector{T}`: Volume mean diameter (d43) for each sample
    - `d32::Vector{T}`: Sauter mean diameter (d32) for each sample
    - `d50q::Vector{T}`: Median size (d50) for each sample
    - `concentration_mean::Vector{T}`: Mean concentration trajectory
    - `concentration_std::Vector{T}`: Standard deviation of concentration trajectory
    - `concentration_lb::Vector{T}`: Lower bound of concentration trajectory
    - `concentration_ub::Vector{T}`: Upper bound of concentration trajectory
    - `d43_mean::T`: Mean d43 value
    - `d43_std::T`: Standard deviation of d43
    - `d32_mean::T`: Mean d32 value
    - `d32_std::T`: Standard deviation of d32
    - `d50q_mean::T`: Mean d50q value
    - `d50q_std::T`: Standard deviation of d50q
"""
@concrete struct EnsembleFVSolution{T <: AbstractArray{<:Real}, C <: AbstractArray{<:Real},
                                    D4 <: AbstractArray{<:Real},
                                    D3 <: AbstractArray{<:Real},
                                    D5 <: AbstractArray{<:Real},
                                    Cm <: AbstractArray{<:Real},
                                    Cs <: AbstractArray{<:Real},
                                    Cl <: AbstractArray{<:Real},
                                    Cu <: AbstractArray{<:Real},
                                    D4m <: AbstractArray{<:Real},
                                    D4s <: AbstractArray{<:Real},
                                    D3m <: AbstractArray{<:Real},
                                    D3s <: AbstractArray{<:Real},
                                    D5m <: AbstractArray{<:Real},
                                    D5s <: AbstractArray{<:Real}} <: AbstractSolution
    # Raw ensemble data
    time::T
    concentration::C
    d43::D4
    d32::D3
    d50q::D5

    # Statistical summaries
    concentration_mean::Cm
    concentration_std::Cs
    concentration_lb::Cl
    concentration_ub::Cu

    d43_mean::D4m
    d43_std::D4s
    d32_mean::D3m
    d32_std::D3s
    d50q_mean::D5m
    d50q_std::D5s
end

"""
    EnsembleMoMSolution{T<:AbstractFloat} <: AbstractSolution

Stores ensemble simulation results from Method of Moments solver.

Fields:
- `moments::Array{T,3}`: Array of moment trajectories (samples × moments × timepoints)
- `time::Vector{T}`: Time points vector
- `d43::Matrix{T}`: Volume mean diameter trajectories (samples × timepoints)
- `d32::Matrix{T}`: Sauter mean diameter trajectories (samples × timepoints)
- `concentration::Matrix{T}`: Concentration trajectories (samples × timepoints)
- `moments_mean::Matrix{T}`: Mean moment trajectories
- `moments_std::Matrix{T}`: Standard deviation of moment trajectories
- `d43_mean::Vector{T}`: Mean d43 trajectory
- `d43_std::Vector{T}`: Standard deviation of d43 trajectory
- `d32_mean::Vector{T}`: Mean d32 trajectory
- `d32_std::Vector{T}`: Standard deviation of d32 trajectory
- `concentration_mean::Vector{T}`: Mean concentration trajectory
- `concentration_std::Vector{T}`: Standard deviation of concentration trajectory
"""
struct EnsembleMoMSolution{T <: AbstractArray{<:Real}, D43 <: AbstractArray{<:Real},
                           D32 <: AbstractArray{<:Real}, C <: AbstractArray{<:Real},
                           Cm <: AbstractArray{<:Real}, Cs <: AbstractArray{<:Real},
                           Cl <: AbstractArray{<:Real}, Cu <: AbstractArray{<:Real},
                           D43m <: AbstractArray{<:Real}, D43s <: AbstractArray{<:Real},
                           D32m <: AbstractArray{<:Real}, D32s <: AbstractArray{<:Real}} <:
       AbstractSolution
    # Raw ensemble data
    time::T
    concentration::C
    d43::D43
    d32::D32

    # Statistical summaries
    concentration_mean::Cm
    concentration_std::Cs
    concentration_lb::Cl
    concentration_ub::Cu
    d43_mean::D43m
    d43_std::D43s
    d32_mean::D32m
    d32_std::D32s

end
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
    nucl_CNT_multiloading <: AbstractFPNucleationFunction

Classical Nucleation Theory (CNT) nucleation function.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("CNT")
- `symbols::Vector{Symbol}`: Parameter symbols [:Aⱼ, :γ]
"""
Base.@kwdef @concrete struct nucl_CNT_multiloading <: AbstractFPNucleationFunction
    nparams::Int64 = 2
    string::String = "CNT-multiload"
    symbols::Vector{Symbol} = [:Aⱼ, :γ]
    unique_loadings::Vector{Float64} = [0.0]
end
"""
    nucl_CNT_multiloading(unique_loadings::AbstractArray{<:Real}) -> nucl_CNT_multiloading

Construct a `nucl_CNT_multiloading` for multiple loading conditions.

Creates a nucleation function with separate CNT parameters (Aj, γ) for each unique loading value.

# Arguments
- `unique_loadings::AbstractArray{<:Real}`: Array of unique loading values

# Returns
- `nucl_CNT_multiloading`: Nucleation function with 2×length(unique_loadings) parameters
"""
function nucl_CNT_multiloading(unique_loadings::AbstractArray{<:Real})
    num_unique_loadings = length(unique_loadings)
    nparams = 2 * num_unique_loadings
    symbols = Symbol[]
    for i in 1:num_unique_loadings
        push!(symbols, Symbol("A$i"))
        push!(symbols, Symbol("γ$i"))
    end
    nucl_CNT_multiloading(unique_loadings = unique_loadings, nparams = nparams,
                          symbols = symbols)
end

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
    nucl_sreg <: AbstractFPNucleationFunction

Supersaturation-regulated nucleation function (placeholder/experimental).

Fields:
- `nparams::Int64`: Number of parameters (0)
- `string::String`: String identifier ("SReg Nu")
"""
Base.@kwdef @concrete struct nucl_sreg <: AbstractFPNucleationFunction
    nparams::Int64 = 0
    string::String = "SReg Nu"
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
    growth_energy_multiloading <: AbstractFPScalarGrowthFunction

Growth function with activation energy for multiple loading conditions.

Fields:
- `nparams::Int64`: Number of parameters (2 × number of unique loadings)
- `string::String`: String identifier ("GrEnergy-multiload")
- `symbols::Vector{Symbol}`: Parameter symbols ([:Ag1, :g1, :Ag2, :g2, ...])
- `unique_loadings::Vector{Float64}`: Array of unique loading values
"""
Base.@kwdef @concrete struct growth_energy_multiloading <: AbstractFPScalarGrowthFunction
    nparams::Int64 = 2
    string::String = "GrEnergy-multiload"
    symbols::Vector{Symbol} = [:Ag, :g]
    unique_loadings::Vector{Float64} = [0.0]
end

"""
    growth_energy_multiloading(unique_loadings::AbstractArray{<:Real}) -> growth_energy_multiloading

Construct a `growth_energy_multiloading` for multiple loading conditions.

Creates a growth function with separate parameters (Ag, g) for each unique loading value.

# Arguments
- `unique_loadings::AbstractArray{<:Real}`: Array of unique loading values

# Returns
- `growth_energy_multiloading`: Growth function with 2×length(unique_loadings) parameters
"""
function growth_energy_multiloading(unique_loadings::AbstractArray{<:Real})
    num_unique_loadings = length(unique_loadings)
    nparams = 2 * num_unique_loadings
    symbols = Symbol[]
    for i in 1:num_unique_loadings
        push!(symbols, Symbol("Ag$i"))
        push!(symbols, Symbol("g$i"))
    end
    growth_energy_multiloading(unique_loadings = unique_loadings, nparams = nparams,
                               symbols = symbols)
end

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
    growth_empirical_length <: AbstractFPLengthGrowthFunction

Empirical length-dependent crystal growth rate function.

Fields:
- `nparams::Int64`: Number of parameters (4)
- `string::String`: String identifier ("Emp. Gr Length")
"""
Base.@kwdef @concrete struct growth_empirical_length <: AbstractFPLengthGrowthFunction
    nparams::Int64 = 4
    string::String = "Emp. Gr Length"
end

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
Base.@kwdef @concrete struct growth_dissolution <: AbstractFPScalarGrowthFunction
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
Base.@kwdef @concrete struct growth_dissolution_length <: AbstractFPLengthGrowthFunction
    nparams::Int64 = 5
    string::String = "GrDissolution_length"
    symbols::Vector{Symbol} = [:Ad, :Ead, :d, :κ, :p]
end

paramaxis(::growth_dissolution_length) = ComponentArrays.Axis(Ad = 1, Ead = 2, d = 3, κ = 4, p = 5)

"""
    growth_dissolution <: AbstractFPScalarGrowthFunction

Empirical crystal growth rate function with activation energy.

Fields:
- `nparams::Int64`: Number of parameters (2)
- `string::String`: String identifier ("GrDissolution")
- `symbols::Vector{Symbol}`: Parameter symbols [:Ad, :Ead, :d]
"""
Base.@kwdef @concrete struct growth_energy_dissolution <: AbstractFPScalarGrowthFunction
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

## Loss-function structs
"""
    AbstractPELossFunction

Abstract supertype for parameter estimation loss functions.
"""
abstract type AbstractPELossFunction end

"""
    logMLE <: AbstractPELossFunction

Log Maximum Likelihood Estimation loss function.

Fields:
- `weighting::Tuple{Float64,Float64}`: Weighting factors for different components (default: (1.0, 1.0))
- `string::String`: String identifier
- `symbols::Vector{Symbol}`: Parameter symbols [:logMLE]
"""
Base.@kwdef @concrete struct logMLE <: AbstractPELossFunction
    weighting::Tuple{Float64, Float64} = (1.0, 1.0)
    string::String = weighting == (1.0, 1.0) ? "Log MLE" : "Log MLE wgted $(weighting)"
    symbols::Vector{Symbol} = [:logMLE]
end

"""
    mae <: AbstractPELossFunction

Mean Absolute Error loss function.

Fields:
- `weighting::Tuple{Float64,Float64}`: Weighting factors for different components (default: (1.0, 1.0))
- `string::String`: String identifier
"""
Base.@kwdef @concrete struct mae <: AbstractPELossFunction
    weighting::Tuple{Float64, Float64} = (1.0, 1.0)
    string::String = weighting == (1.0, 1.0) ? "MAE" : "MAE wgted $(weighting)"
end

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
    CrystallisationExperiment{O<:NamedTuple} <: AbstractExperiment

A single crystallisation experiment: a typed `NamedTuple` of observables
plus the run conditions.

Fields:
- `observables::O`: e.g. `(concentration = Observable(...), d43 = Observable(...), d50q = Observable(...))`.
  The `concentration` observable is mandatory for loss evaluation.
- `temperature::Float64`: run temperature in Kelvin
- `loading::Float64`: loading (e.g. volumetric solids fraction)
- `exp_id::Int`: experiment identifier

The `NamedTuple` shape keeps the container type-stable and Tables.jl
compatible; additional observables (pH, mass, PSD, ...) are added as new
fields, not new container types.
"""
Base.@kwdef @concrete struct CrystallisationExperiment{O <: NamedTuple} <:
                 AbstractExperiment
    observables::O
    temperature::Float64
    loading::Float64
    exp_id::Int
end

"""
    initial_concentration(expt::CrystallisationExperiment) -> Real

Initial solute concentration of an experiment, read from the
`concentration` series observable (first timepoint).
"""
initial_concentration(expt::CrystallisationExperiment) =
    expt.observables.concentration.mean[1]

## Solver
"""
    AbstractSolver

Abstract supertype for all numerical solvers used in crystallization simulations.
"""
abstract type AbstractSolver end

"""
    AbstractDiscretisedSolver <: AbstractSolver

Abstract supertype for discretised numerical solvers.
"""
abstract type AbstractDiscretisedSolver <: AbstractSolver end

"""
    FiniteVol <: AbstractDiscretisedSolver

Finite volume solver for crystallization population balance equations.

Fields:
- `meshsize::Int64`: Number of mesh points (default: 200)
- `lmin::Float64`: Minimum crystal size (default: 0.0)
- `lmax::Float64`: Maximum crystal size (default: 50.0e-6)
- `cell_face::LinRange{Float64,Int64}`: Cell face positions
- `cell_centre::LinRange{Float64,Int64}`: Cell center positions
- `cell_dL::Float64`: Cell size
- `string::String`: Solver identifier ("FV")
- `timestepping_algorithm::Symbol`: Time-stepping algorithm selector (e.g. `:tsit5`, `:ssprk43`, `:auto`).
- `reltol::Float64`: Scalar relative tolerance for ODE solver.
- `abstol::Float64`: Scalar absolute tolerance for ODE solver.
"""
Base.@kwdef @concrete struct FiniteVol <: AbstractDiscretisedSolver
    meshsize::Int64 = 200
    lmin::Float64 = 0.0
    lmax::Float64 = 50.0e-6
    cell_face::LinRange{Float64, Int64} = LinRange(lmin, lmax, meshsize + 1)
    cell_centre::LinRange{Float64,
                          Int64} = 0.5 .*
                                   (cell_face[2:end] .+ cell_face[1:(end - 1)])
    cell_dL::Float64 = cell_centre[2] - cell_centre[1]
    string::String = "FV"
    timestepping_algorithm::Symbol = :auto
    reltol::Float64 = 1e-2
    abstol::Float64 = 1e-6
end

"""
    WENO <: AbstractDiscretisedSolver

Weighted Essentially Non-Oscillatory (WENO) solver for crystallization population balance equations.

Fields:
- `meshsize::Int64`: Number of mesh points (default: 200)
- `lmin::Float64`: Minimum crystal size (default: 0.0)
- `lmax::Float64`: Maximum crystal size (default: 50.0e-6)
- `cell_face::LinRange{Float64,Int64}`: Cell face positions
- `cell_centre::LinRange{Float64,Int64}`: Cell center positions
- `cell_dL::Float64`: Cell size
- `string::String`: Solver identifier ("WENO")
- `timestepping_algorithm::Symbol`: Time-stepping algorithm selector (e.g. `:tsit5`, `:ssprk43`, `:auto`).
- `reltol::Float64`: Scalar relative tolerance for ODE solver.
- `abstol::Float64`: Scalar absolute tolerance for ODE solver.
"""
Base.@kwdef @concrete struct WENO <: AbstractDiscretisedSolver
    meshsize::Int64 = 200
    lmin::Float64 = 0.0
    lmax::Float64 = 50.0e-6
    cell_face::LinRange{Float64, Int64} = LinRange(lmin, lmax, meshsize + 1)
    cell_centre::LinRange{Float64,
                          Int64} = 0.5 .*
                                   (cell_face[2:end] .+ cell_face[1:(end - 1)])
    cell_dL::Float64 = cell_centre[2] - cell_centre[1]
    string::String = "WENO"
    timestepping_algorithm::Symbol = :auto
    reltol::Float64 = 1e-1
    abstol::Float64 = 1e-4
end

"""
    MoM <: AbstractSolver

Method of Moments solver for crystallization population balance equations.

Fields:
- `string::String`: Solver identifier ("MoM")
- `nmoments::Int64`: Highest tracked moment order. The state holds
  moments `µ0 .. µ_nmoments` plus the liquid-phase concentration
  (`nmoments + 2` states). Must be `>= 2` (the concentration closure uses
  `µ2`); `d43 = µ4/µ3` requires `nmoments >= 4` (the default).
- `timestepping_algorithm::Symbol`: Time-stepping algorithm selector (e.g. `:tsit5`, `:ssprk43`, `:auto`).
- `reltol::Float64`: Scalar relative tolerance for ODE solver.
- `abstol::Float64`: Scalar absolute tolerance for ODE solver.
"""
Base.@kwdef @concrete struct MoM <: AbstractSolver
    string::String = "MoM"
    nmoments::Int64 = 4
    timestepping_algorithm::Symbol = :auto
    reltol::Float64 = 1e-10
    abstol::Float64 = 1e-8
end

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
## Saturation models
##############################################################

"""
    AbstractSaturationModel

Abstract supertype for solubility/saturation models (see
`ConstantSaturation`, `PolynomialSaturation`, `CallableSaturation`).
"""
abstract type AbstractSaturationModel end

"""
    lysozyme_saturation() -> PolynomialSaturation

The legacy lysozyme solubility polynomial (kg/m³, temperature in °C):
`0.3705 + 7.171e-2 ΔT - 1.924e-3 ΔT² + 17.97e-5 ΔT³`, with
`ΔT = T_K - 273.15`. This is the default `CrystallisationProblem`
saturation model (backward compatible with the original hardcoded curve).
"""
lysozyme_saturation() =
    PolynomialSaturation(; coeffs = [0.3705, 7.171e-2, -1.924e-3, 17.97e-5])

"""
    ConstantSaturation{T<:Real} <: AbstractSaturationModel

Constant solubility `value` (kg/m³), independent of temperature and time.
"""
Base.@kwdef @concrete struct ConstantSaturation{T <: Real} <: AbstractSaturationModel
    value::T
end

"""
    PolynomialSaturation{T<:Real} <: AbstractSaturationModel

Solubility as a polynomial in `T - Tref` (default `Tref = 273.15`, i.e.
temperature in Celsius): `coeffs[1] + coeffs[2] x + coeffs[3] x² + ...`,
evaluated with Horner's scheme.
"""
Base.@kwdef @concrete struct PolynomialSaturation{T <: Real} <: AbstractSaturationModel
    coeffs::Vector{T}
    Tref::T = 273.15
end

"""
    CallableSaturation{F} <: AbstractSaturationModel

Arbitrary solubility as a user function `f(T_K, t)` of temperature (K) and
time (minutes).
"""
Base.@kwdef @concrete struct CallableSaturation{F} <: AbstractSaturationModel
    f::F
end

"""
    saturation_concentration(sm::AbstractSaturationModel, temp_profile, t) -> Real

Solubility (kg/m³) at time `t` under the temperature profile `temp_profile`.
"""
saturation_concentration(sm::ConstantSaturation, temp_profile, t) = sm.value

function saturation_concentration(sm::PolynomialSaturation, temp_profile, t)
    x = temperature(temp_profile, t) - sm.Tref
    c = sm.coeffs
    acc = c[end]
    @inbounds for i in (length(c) - 1):-1:1
        acc = acc * x + c[i]
    end
    return acc
end

saturation_concentration(sm::CallableSaturation, temp_profile, t) =
    sm.f(temperature(temp_profile, t), t)

"""
    saturation_concentration(sm::AbstractSaturationModel, temp_profile) -> Real

Solubility at time `t = 0` under the temperature profile.
"""
saturation_concentration(sm::AbstractSaturationModel, temp_profile) =
    saturation_concentration(sm, temp_profile, 0.0)

## Problem struct

"""
    AbstractCrystallisationProblem

Abstract supertype for crystallization problem definitions.
"""
abstract type AbstractCrystallisationProblem end

"""
    CrystallisationProblem{NuF,GrF,BrF,AggF,solmethod,NuP,GrP,BrP,AggP} <: AbstractCrystallisationProblem

Main structure defining a crystallization problem with kinetics and solver specifications.

Type Parameters:
- `NuF<:AbstractNucleationFunction`: Type of nucleation function
- `GrF<:AbstractGrowthFunction`: Type of growth function
- `BrF<:AbstractBreakageFunction`: Type of breakage function
- `AggF<:AbstractAggregationFunction`: Type of aggregation function
- `solmethod<:AbstractSolver`: Type of solver
- `NuP,GrP,BrP,AggP<:AbstractVector{<:Real}`: Parameter vector types

Fields:
- `temp_profile::TP`: Temperature profile (see `AbstractTemperature`)
- `ρ::Float64`: Crystal density (kg/m³)
- `initial_concentration::Float64`: Initial solute concentration (kg/m³)
- `saturation_model::AbstractSaturationModel`: Solubility model (default `lysozyme_saturation()`)
- `kv::Float64`: Volume shape factor
- `molecular_volume::Float64`: Molecular volume (m³)
- `kinetics_nucleationfunction::NuF`: Nucleation function
- `kinetics_growthfunction::GrF`: Growth function
- `parameterset_nucleation::NuP`: Nucleation parameters
- `parameterset_growth::GrP`: Growth parameters
- `kinetics_breakagefunction::BrF`: Breakage function (default: nobreakage())
- `parameterset_breakage::BrP`: Breakage parameters (default: [0.0])
- `kinetics_aggregationfunction::AggF`: Aggregation function (default: noaggregation())
- `parameterset_aggregation::AggP`: Aggregation parameters (default: [0.0])
- `R::Float64`: Gas constant (J/mol/K)
- `kb::Float64`: Boltzmann constant (J/K)
- `solver::solmethod`: Numerical solver (holds time-stepping configuration).
"""
Base.@kwdef @concrete struct CrystallisationProblem{NuF <: AbstractNucleationFunction,
                                                    GrF <: AbstractGrowthFunction,
                                                    BrF <: AbstractBreakageFunction,
                                                    AggF <: AbstractAggregationFunction,
                                                    solmethod <: AbstractSolver,
                                                    NuP <: AbstractVector{<:Real},
                                                    GrP <: AbstractVector{<:Real},
                                                    BrP <: AbstractVector{<:Real},
                                                    AggP <: AbstractVector{<:Real},
                                                    TP <: AbstractTemperature,
                                                                     SM <: AbstractSaturationModel} <:
                             AbstractCrystallisationProblem

    # Operation
    temp_profile::TP = ConstantTemperature(273.15 + 20.0) # Default to 25°C

    # Loading
    loading::Float64 = 0.0

    # Solute
    ρ::Float64 = 1370.0
    initial_concentration::Float64 = 20.0
    saturation_model::SM = lysozyme_saturation()
    kv::Float64 = 0.81 #0.55
    molecular_volume::Float64 = 2.97e-26

    # Chosen Kinetics
    kinetics_nucleationfunction::NuF = nucl_CNT()
    kinetics_growthfunction::GrF = growth_empirical()
    parameterset_nucleation::NuP = [1.0, 1.0]
    parameterset_growth::GrP = [1.0, 1.0]

    kinetics_breakagefunction::BrF = nobreakage()
    parameterset_breakage::BrP = [0.0]

    kinetics_aggregationfunction::AggF = noaggregation()
    parameterset_aggregation::AggP = [0.0]

    initial_state::Union{Nothing, AbstractVector{<:Real}} = nothing

    # Constants
    R::Float64 = 8.314
    kb::Float64 = 1.380649e-23

    solver::solmethod = MoM()

end

"""
    saturation_concentration(prob::CrystallisationProblem, t) -> Real

Solubility (kg/m³) of the problem at time `t` under its temperature profile.
"""
saturation_concentration(prob::CrystallisationProblem, t) =
    saturation_concentration(prob.saturation_model, prob.temp_profile, t)

"""
    supersaturation(prob::CrystallisationProblem, state, t) -> Real

Supersaturation ratio `state[end] / saturation_concentration(prob, t)`. The
liquid-phase concentration is the last state component in both the MoM and
discretised solver states.
"""
supersaturation(prob::CrystallisationProblem, state, t) =
    state[end] / saturation_concentration(prob, t)
