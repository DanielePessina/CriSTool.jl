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
    AbstractMomentSolver <: AbstractSolver

Abstract supertype for solvers that evolve raw moments of the crystal-size
distribution rather than values on a fixed length mesh.
"""
abstract type AbstractMomentSolver <: AbstractSolver end

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
    # Tolerances re-derived for the SI seconds kernel (Step 3): the default
    # loose pair (1e-2/1e-6) made FV gradients disagree with finite differences
    # and allowed negative-density overshoot under the faster nm/s dynamics.
    reltol::Float64 = 1e-4
    abstol::Float64 = 1e-10
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
    # Tolerances re-derived for the SI seconds kernel (Step 3); see FiniteVol.
    reltol::Float64 = 1e-4
    abstol::Float64 = 1e-10
end

"""
    MoM <: AbstractMomentSolver

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
Base.@kwdef @concrete struct MoM <: AbstractMomentSolver
    string::String = "MoM"
    nmoments::Int64 = 4
    timestepping_algorithm::Symbol = :auto
    reltol::Float64 = 1e-10
    abstol::Float64 = 1e-8
end

"""
    QMOM <: AbstractMomentSolver

Classical one-dimensional Quadrature Method of Moments solver.  A QMOM with
`nquadrature = N` evolves the raw physical moments `M₀:M₂N₋₁` and reconstructs
an `N`-node Gaussian quadrature when a size-dependent closure is required.

The configuration values are intentionally kept on the solver, while the
nodes and weights reconstructed from a state live in `QMOMQuadrature`.

Fields:
- `nquadrature::Int64`: Number of Gaussian quadrature nodes (at least 2 for
  the crystallisation volume closure; default: 3).
- `coordinate_scale::Float64`: Physical length represented by one scaled
  quadrature coordinate (default: `1e-6` m).
- `inversion_algorithm::Symbol`: Moment inversion route.  `:wheeler` is the
  supported route and uses Wheeler recurrence coefficients followed by
  Golub-Welsch diagonalisation.
- `realizability_tolerance::Float64`: Absolute tolerance used for support and
  recurrence checks in the scaled coordinate.
- `empty_population_tolerance::Float64`: Absolute `M₀` threshold below which
  the population is represented by an empty rule.
- `node_coalescence_tolerance::Float64`: Threshold for rank deflation when a
  recurrence coefficient indicates coalescing support.
- `minimum_size::Float64`: Lower support bound in physical metres.
- `timestepping_algorithm::Symbol`: Time-stepping algorithm selector.
- `reltol::Float64`, `abstol::Float64`: ODE tolerances.
"""
Base.@kwdef @concrete struct QMOM <: AbstractMomentSolver
    nquadrature::Int64 = 3
    coordinate_scale::Float64 = 1e-6
    inversion_algorithm::Symbol = :wheeler
    realizability_tolerance::Float64 = 1e-10
    empty_population_tolerance::Float64 = 1e-14
    node_coalescence_tolerance::Float64 = 1e-10
    minimum_size::Float64 = 0.0
    string::String = "QMOM"
    timestepping_algorithm::Symbol = :auto
    # QMOM is a closure approximation; a 1e-7 relative solve tolerance keeps
    # the default materially cheaper while preserving the benchmark outputs.
    reltol::Float64 = 1e-7
    abstol::Float64 = 1e-8
end

"""Highest tracked raw-moment order for a moment solver."""
moment_order(solver::MoM) = solver.nmoments
moment_order(solver::QMOM) = 2 * solver.nquadrature - 1

"""Number of raw moments represented by a moment solver state."""
moment_count(solver::MoM) = solver.nmoments + 1
moment_count(solver::QMOM) = 2 * solver.nquadrature

"""Number of population states before the named solvent variables."""
_population_state_count(solver::FiniteVol) = solver.meshsize
_population_state_count(solver::WENO) = solver.meshsize
_population_state_count(solver::MoM) = moment_count(solver)
_population_state_count(solver::QMOM) = moment_count(solver)
_population_state_count(solver::AbstractSolver) =
    throw(ArgumentError("No population-state size is defined for $(typeof(solver))."))

"""Compatibility accessor for the highest QMOM moment order."""
nmoments(solver::QMOM) = moment_order(solver)

"""
    QMOMInversionDiagnostics

Numerical diagnostics returned alongside a QMOM quadrature.  `status` is one
of `:ok`, `:empty`, or `:deflated` for successful inversions; failed
realizability checks throw `ArgumentError` before a quadrature is returned.
"""
struct QMOMInversionDiagnostics{T}
    status::Symbol
    active_nodes::Int
    minimum_node::T
    minimum_weight::T
    minimum_recurrence::T
    reconstruction_error::T
end

"""
    QMOMQuadrature

Gaussian quadrature reconstructed from a raw moment vector.  `nodes` are
physical crystal lengths in metres, `weights` are particle-number weights in
the same number-density units as `M₀`, and `active_nodes` records the rule
rank after empty/rank-deficient support handling.
"""
struct QMOMQuadrature{TN <: AbstractVector, TW <: AbstractVector, TD}
    nodes::TN
    weights::TW
    active_nodes::Int
    diagnostics::TD
end

Base.length(quadrature::QMOMQuadrature) = quadrature.active_nodes
Base.isempty(quadrature::QMOMQuadrature) = quadrature.active_nodes == 0
