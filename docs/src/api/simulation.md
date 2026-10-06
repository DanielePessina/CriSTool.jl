# Simulation and results

```@docs
CrystallisationProblem
runsimulation
CrystallisationFVSolution
CrystallisationMoMSolution
CrystallisationQMOMSolution
CrystallisationDQMOMSolution
CrystallisationSteadyStateSolution
SteadyStateConvergenceError
EnsembleFVSolution
EnsembleMoMSolution
AbstractSolution
solve_steadystate
AbstractInitialCrystals
LogNormalInitialCrystals
GaussianInitialCrystals
initial_state_from_characteristics
get_characteristic_size
getmomentsizes
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
fixed-flow MSMPR is `solve_steadystate` (see the *Steady MSMPR solves* section
below and the [Reactor operations and steady state](../operations.md) guide).

## Solvent dynamics

`default_solvent_dynamics` is the default callable used by
`CrystallisationProblem`. It updates the named `:concentration` solvent
variable from the crystal growth rate and returns zero rates for any
additional named solvent variables. Supply `initial_solvent_state` and a
custom `solvent_dynamics` callable when the process includes coupled variables
such as pH or volume.

## Steady MSMPR solves

`solve_steadystate(problem)` is the explicit steady-state route for a constant,
autonomous `MSMPROperation`. It relaxes the generated dynamics until the scaled
residual covers every population and solvent-state component, then returns a
`CrystallisationSteadyStateSolution` with one value per observable, product
flows, the full residual, the solver retcode, and physical and mass-balance
diagnostics. If a finite relaxation horizon ends before convergence or any
physical check fails, `SteadyStateConvergenceError` carries the unsuccessful
result in its `solution` field with `success = false` and the solver retcode.

Batch and fed-batch problems remain transient simulations through
`runsimulation`. Callable flow, feed, temperature, saturation, or kinetic
profiles require `autonomous = true` to declare that they are time invariant;
the declaration is the caller's responsibility.
