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
    - `reactor_state::NamedTuple`: Reactor-volume and size-boundary-flow
      trajectories. Mesh solutions report signed lower/upper solid-mass flows
      in kg/s, with positive values directed out of the size domain.

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
    moment2::TDm

    solvent_state::TL

    final_state::TCo
    ode_stats::TStats
    success::Bool
    reactor_state::NamedTuple

end

"""
    CrystallisationMoMSolution{Tt,TCo,TD1,TD3,TD4,TM2} <: AbstractSolution where {Tt<:AbstractArray{<:Real},TCo<:AbstractArray{<:Real},TD1<:AbstractArray{<:Real},TD3<:AbstractArray{<:Real},TD4<:AbstractArray{<:Real},TM2<:AbstractArray{<:Real}}

    Stores Method of Moments solver results.

    Fields:
    - `time::Tt`: Time points vector
    - `concentration::TCo`: Concentration vector
    - `d10::TD1`: 10th percentile of number density
    - `d32::TD3`: Sauter mean diameter
    - `d43::TD4`: Volume mean diameter
    - `moment2::TM2`: Second raw moment of number density (m⁻¹)
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

    moment2::TMu2

    solvent_state::TL

    final_state::TCo
    ode_stats::TStats
    success::Bool
    reactor_state::NamedTuple

end

"""
    CrystallisationQMOMSolution

Stores a raw-moment QMOM trajectory together with the Gaussian quadrature
reconstructed at every saved time.  `moments` has shape `(2N, nt)`, while the
quadrature arrays have shape `(N, nt)`; inactive nodes in an empty or
rank-deficient rule are represented by zero node/weight entries.  The
corresponding active rank and inversion status are available in
`inversion_diagnostics` and through `quadrature(solution, index)`.
"""
struct CrystallisationQMOMSolution{Tt, TCo, TM, TN, TW, TD1, TD3, TD4, TMu2,
                                   TL, TF, TS, TDiag} <: AbstractSolution
    time::Tt
    concentration::TCo
    moments::TM
    quadrature_nodes::TN
    quadrature_weights::TW
    d10::TD1
    d32::TD3
    d43::TD4
    moment2::TMu2
    solvent_state::TL
    final_state::TF
    ode_stats::TS
    success::Bool
    reactor_state::NamedTuple
    inversion_diagnostics::TDiag
end

"""
    CrystallisationDQMOMSolution

Stores a seeded direct-quadrature trajectory. `weights` and `nodes` have
shape `(N, nt)` and are physical particle-number weights and crystal lengths
in metres. `moments` are derived from those arrays and have shape
`(2N, nt)`; they are not the ODE state.
"""
struct CrystallisationDQMOMSolution{Tt, TCo, TW, TN, TM, TD1, TD3, TD4, TMu2,
                                    TL, TF, TS, TDiag} <: AbstractSolution
    time::Tt
    concentration::TCo
    weights::TW
    nodes::TN
    moments::TM
    d10::TD1
    d32::TD3
    d43::TD4
    moment2::TMu2
    solvent_state::TL
    final_state::TF
    ode_stats::TS
    success::Bool
    reactor_state::NamedTuple
    projection_diagnostics::TDiag
end

# Preserve the existing positional solution constructors for batch callers.
CrystallisationMoMSolution(args::Vararg{Any,10}) = CrystallisationMoMSolution(args..., (;))
CrystallisationFVSolution(args::Vararg{Any,15}) = CrystallisationFVSolution(args..., (;))
CrystallisationQMOMSolution(args::Vararg{Any,14}) =
    CrystallisationQMOMSolution(args[1:13]..., (;), args[14])
CrystallisationDQMOMSolution(args::Vararg{Any,14}) =
    CrystallisationDQMOMSolution(args[1:13]..., (;), args[14])
