# Migration to named model workflows

This redesign intentionally changes the recommended interface:

| Earlier workflow | Model workflow |
| --- | --- |
| Kinetic law plus an ordered flat vector | `KineticModel(law; parameters=(...))` |
| Rebuild physical keyword settings in each routine | Reuse `CrystallisationModel` |
| `runsimulation` returns problem and solution | `simulate(configured_problem)` returns solution |
| Separate fixed and free variants | Select a subset in a study specification |
| Positional PE/MCMC/ABCDE routines | `prepare_fit` then `fit` |
| Prior arrays indexed by kinetic block order | Named `BayesianSpec` or `ABCSpec` |
| First concentration sample treated as C0 | Require mapped time-zero observations |
| Missing likelihood variance gets a fallback | Supplied positive variance or an error loss |
| Seed mesh rescaled to recover omitted mass | Checked support/domain and no silent rescaling |
| Anonymous ensemble matrices | Named joint `ParameterSamples` with sample IDs |

The numerical solver implementations and solution accessors remain reusable.
Legacy inference runners and vector-assembly helpers are internal verification
interfaces, not the tutorial workflow. There is no promise of public backward
compatibility for these old assembly paths. Advanced configured solver and
SciML interfaces retain their distinct capabilities.

Steady fitting is distinct: use steady targets and explicit relaxation initial
conditions, rather than deriving transient initial conditions from steady data.
