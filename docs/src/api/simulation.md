# Simulation and results

`CrystallisationProblem` follows the SciML problem/solve pattern: it holds the
model, conditions, `tspan` and kinetic values `p`, and `solve(problem, solver)`
chooses the numerical method. `remake(problem; ...)` copies it with fields
replaced. Solutions expose `sol.t`, `sol.retcode`, `successful_retcode(sol)` and
`sol[:name]`.

```@docs
CrystallisationProblem
solve(::CrystallisationProblem, ::AbstractSolver)
remake(::CrystallisationProblem)
CrystallisationSystem
LysozymeSystem
CrystallisationModel
KineticModel
RunConditions
FromObservation
CallableGrowth
CallableNucleation
CallableLengthGrowth
KineticContext
system_properties
required_properties
named_parameters
kinetic_parameter_vector
CrystallisationSolution
AbstractPopulationTrajectory
MeshPopulation
MomentPopulation
QuadraturePopulation
CrystallisationSteadyStateSolution
SteadyStateConvergenceError
AbstractSolution
getindex(::CriSTool.AbstractSolution, ::Symbol)
successful_retcode(::CriSTool.AbstractSolution)
CrystallisationSteadyStateProblem
solve(::CrystallisationSteadyStateProblem, ::AbstractSolver)
SteadyStateOptions
AbstractInitialCrystals
LogNormalInitialCrystals
GaussianInitialCrystals
DistributionInitialCrystals
AbstractSeedWeighting
NumberWeighted
VolumeWeighted
number_weighted
seed_domain_diagnostics
initial_state_from_characteristics
state_vars
size_metrics
observable_values
solvent_state
crystallisation_odeproblem
crystallisation_solution
```

## Operations

```@docs
AbstractCrystallisationOperation
BatchOperation
MSMPROperation
FedBatchOperation
CrystallisationFeed
reactor_vars
reactor_volume
operation_flows
```

The reactor volume is a named observable for flow operations:
`observable_values(solution, :volume)`. The steady-state runner for autonomous
fixed-flow MSMPR is `solve(CrystallisationSteadyStateProblem(problem), solver)`
(see the *Steady MSMPR solves* section
below and the [Reactor operations and steady state](../operations.md) guide).

## Solvent dynamics

`default_solvent_dynamics` is the default callable used by
`CrystallisationProblem`. It updates the named `:concentration` solvent
variable from the crystal growth rate and returns zero rates for any
additional named solvent variables. Supply extra initial solvent values through
`RunConditions(solvent = (...))` and a custom `solvent_dynamics` callable
(a `CrystallisationProblem` keyword, or `problem_options` in `EstimationProblem`) when
the process includes coupled variables such as pH or volume.

## Steady MSMPR solves

`solve(CrystallisationSteadyStateProblem(problem; options, u0), solver)` is the
explicit steady-state route for a constant, autonomous `MSMPROperation`.
`CrystallisationSteadyStateProblem` is CriSTool's own type because SciMLBase's
`SteadyStateProblem` wraps an `ODEFunction` and initial state, which here
depend on the solver. The solve relaxes the generated dynamics until the scaled
residual covers every population and solvent-state component, then returns a
`CrystallisationSteadyStateSolution`: scalar equilibrium values through
`steady.equilibrium` (`steady.equilibrium.concentration`, `.d43`), product flows in `steady.product`, the full residual,
a `retcode`, and physical and mass-balance diagnostics in `steady.diagnostics`
(including the relaxation solve's own `ode_retcode`). Settings are passed as
`options = SteadyStateOptions(...)`; `u0` is an optional initial guess in the
solver's state layout. `steady.retcode` is `Success` when every check passes,
`MaxIters` when the relaxation horizon ends before the residual converges,
`Failure` when the physical state or flow balance is invalid, and the ODE code
when the integration itself fails. An unsuccessful result is thrown as
`SteadyStateConvergenceError`, which carries it in its `solution` field.

Batch and fed-batch runs are transient; use `solve(problem, solver)`. Callable flow, feed,
temperature, saturation, or kinetic profiles require
`SteadyStateOptions(autonomous = true)` to declare that they are time invariant;
the declaration is the caller's responsibility.
