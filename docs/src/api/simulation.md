# Simulation and results

```@docs
CrystallisationProblem
runsimulation
CrystallisationFVSolution
CrystallisationMoMSolution
CrystallisationQMOMSolution
CrystallisationDQMOMSolution
EnsembleFVSolution
EnsembleMoMSolution
AbstractSolution
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
```

## Solvent dynamics

`default_solvent_dynamics` is the default callable used by
`CrystallisationProblem`. It updates the named `:concentration` solvent
variable from the crystal growth rate and returns zero rates for any
additional named solvent variables. Supply `initial_solvent_state` and a
custom `solvent_dynamics` callable when the process includes coupled variables
such as pH or volume.
