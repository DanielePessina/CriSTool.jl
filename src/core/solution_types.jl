"""
Typed solution and ensemble-result containers used throughout CriSTool.
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
@concrete struct CrystallisationFVSolution{Tt, TCo, TNn, TVd, TDq, TDm, TL, TStats} <:
                 AbstractSolution where {Tt <: AbstractArray{<:Real},
                                         TCo <: AbstractArray{<:Real},
                                         TNn <: AbstractArray{<:Real},
                                         TVd <: AbstractArray{<:Real},
                                         TDq <: AbstractArray{<:Real},
                                         TDm <: AbstractArray{<:Real},
                                         TL <: NamedTuple,
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

    solvent_state::TL

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
@concrete struct CrystallisationMoMSolution{Tt, TCo, TD1, TD3, TD4, TMu2, TL, TStats} <:
                 AbstractSolution where {Tt <: AbstractArray{<:Real},
                                         TCo <: AbstractArray{<:Real},
                                         TD1 <: AbstractArray{<:Real},
                                         TD3 <: AbstractArray{<:Real},
                                         TD4 <: AbstractArray{<:Real},
                                         TMu2 <: AbstractArray{<:Real},
                                         TL <: NamedTuple,
                                         TStats}
    time::Tt
    concentration::TCo
    d10::TD1
    d32::TD3
    d43::TD4

    mu2::TMu2

    solvent_state::TL

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
