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
  moments `µ0 .. µ_nmoments` plus the solvent-phase concentration
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
