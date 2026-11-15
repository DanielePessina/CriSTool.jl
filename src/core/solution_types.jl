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
    AbstractPopulationTrajectory

Solver-specific part of a `CrystallisationSolution`: the population
representation saved at every output time. `MeshPopulation`,
`MomentPopulation` and `QuadraturePopulation` are the built-in subtypes.
"""
abstract type AbstractPopulationTrajectory end

"""
    MeshPopulation

Population trajectory of a discretised solver (`FiniteVol`, `WENO`).

Fields:
- `numberdensity`: number density on the cell centres, `(meshsize, nt)` (m⁻⁴).
- `voldensity`: volume-weighted density `kv L³ n`, same shape.
- `d10q`, `d50q`, `d90q`: volume-distribution quantiles per saved time (m).
"""
struct MeshPopulation{TDensity <: AbstractMatrix, TQuantile <: AbstractVector} <:
       AbstractPopulationTrajectory
    numberdensity::TDensity
    voldensity::TDensity
    d10q::TQuantile
    d50q::TQuantile
    d90q::TQuantile
end

"""
    MomentPopulation

Raw-moment trajectory of a `MoM` solve. `moments` has shape
`(nmoments + 1, nt)`; row `k + 1` holds the SI raw moment `µk`.
"""
struct MomentPopulation{TMoments <: AbstractMatrix} <: AbstractPopulationTrajectory
    moments::TMoments
end

"""
    QuadraturePopulation

Quadrature trajectory of a `QMOM` or `DQMOM` solve. `moments` has shape
`(2N, nt)`; `nodes` (m) and `weights` (particle number per m³) have shape
`(N, nt)`.

For `QMOM` the moments are the integrated state and the quadrature is
reconstructed at every saved time; inactive nodes of an empty or
rank-deficient rule are stored as zeros and `diagnostics` is a vector of
`QMOMInversionDiagnostics`. For `DQMOM` the weights and nodes are the
integrated state, the moments are derived from them, and `diagnostics` is a
vector of `DQMOMProjectionDiagnostics` (or `nothing` for a dual-valued
trajectory). `quadrature(solution, index)` returns one saved rule.
"""
struct QuadraturePopulation{TMoments <: AbstractMatrix, TNodes <: AbstractMatrix, TDiagnostics} <:
       AbstractPopulationTrajectory
    moments::TMoments
    nodes::TNodes
    weights::TNodes
    diagnostics::TDiagnostics
end

"""
    CrystallisationSolution{Population}

Physical result of a transient solve, shared by every solver. Solver-specific
data live in `population` (`MeshPopulation`, `MomentPopulation` or
`QuadraturePopulation`); everything else is common.

Fields:
- `t`: saved times (s).
- `concentration`: solute concentration trajectory (kg/m³), also available as
  `solvent_state.concentration`.
- `population`: solver-specific population trajectory.
- `d10`, `d32`, `d43`: moment-ratio sizes (m); zero for an empty population
  and `NaN` when the solver does not track the required moment.
- `moment2`: second raw moment µ2 (m⁻¹).
- `solvent_state`: named solvent-variable trajectories.
- `reactor_state`: reactor-volume and size-boundary flow trajectories. Mesh
  solutions report signed lower/upper solid-mass flows in kg/s, positive out
  of the size domain.
- `final_state`: the public numerical state at the last saved time.
- `ode_stats`: ODE solver statistics.
- `retcode::ReturnCode.T`: return code of the ODE solve; check it with
  `successful_retcode(solution)`.

Named access does not depend on the solver: `solution[:concentration]`,
`solution[:d43]`, `solution[:d50q]` or any solvent-state name index the
trajectories, as do `state_vars`, `size_metrics`, `reactor_vars` and
`observable_values`.
"""
struct CrystallisationSolution{Population <: AbstractPopulationTrajectory, TTime, TConcentration,
                               TSize, TSolvent <: NamedTuple, TReactor <: NamedTuple,
                               TFinal, TStats} <: AbstractSolution
    t::TTime
    concentration::TConcentration
    population::Population
    d10::TSize
    d32::TSize
    d43::TSize
    moment2::TSize
    solvent_state::TSolvent
    reactor_state::TReactor
    final_state::TFinal
    ode_stats::TStats
    retcode::SciMLBase.ReturnCode.T
end
