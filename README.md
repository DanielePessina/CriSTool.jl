# CriSTool.jl

`CriSTool.jl` is a comprehensive Julia package for the simulation, parameter estimation, and uncertainty quantification of batch crystallisation processes. It provides a flexible framework for researchers and engineers to model complex crystallisation phenomena.

## High-Level Description

The package is designed to handle various aspects of crystallisation modeling:

*   **Population Balance Modeling:** Implements both the Method of Moments (MoM) and Finite Volume (FV) methods (with WENO option) to solve population balance equations, allowing for the tracking of particle size distribution over time.
*   **Flexible Kinetics:** Supports a wide range of kinetic models for nucleation, growth, aggregation, and breakage, with named-parameter access via ComponentArrays. Users can select from built-in empirical and first-principles models or define their own (see Tutorial 4).
*   **Parameter Estimation:** Includes routines for fitting model parameters to experimental data using metaheuristic optimization, Optimization.jl-based search, Turing NUTS, and Approximate Bayesian Computation. ABC inference is unified behind a single `run_abc` entry point with selectable samplers (`ABCDESampler`, `ABCDETurnerSampler`).
*   **Uncertainty & Sensitivity Analysis:** Provides tools to perform uncertainty quantification through ensemble simulations and to analyze model sensitivity to different parameters.
*   **Visualization:** Comes with plotting utilities built on `Makie.jl` for visualizing simulation results, posterior distributions, and measurement data.


## Example Usage

Here is a basic example of how to run a crystallisation simulation using the Method of Moments (MoM) solver.

```julia
using CriSTool

# 1. Define kinetic models for nucleation and growth
# These structs hold information about the model, like the number of parameters.
nucl_model = nucl_CNT() # Empirical nucleation model (2 parameters)
grow_model = growth_empirical() # Empirical growth model (2 parameters)

# 2. Define a vector of kinetic parameters
# The parameters are ordered: [nucleation_params..., growth_params...]
parameters = [38, 0.7, 1.0, 3.0]

# 3. Set up and run the simulation using the keyword-based `runsimulation` function
problem, solution = runsimulation(
    parameters;
    nucl = nucl_model,
    gr = grow_model,
    agg = noaggregation(), # Optional: defaults to no aggregation
    br = nobreakage(),     # Optional: defaults to no breakage
    solver = MoM(),        # Specify the Method of Moments solver
    initial_concentration = 20.0,
    save_idx = 0.0:1.0:100.0 # Time points to save results
)

# 4. Access and print the results
if solution.success
    println("Simulation completed successfully!")
    println("Final concentration: ", solution.concentration[end])
    println("Final d43 (volume-weighted mean size): ", solution.d43[end])
else
    println("Simulation failed.")
end
```

`runsimulation` accepts either a flat `Vector{Float64}` (as above) or a structured `ComponentArray`. The flat form is forwarded into a `ComponentArray` view internally using the composite axis built from the four kinetic models — see [`docs/simulation.md`](docs/simulation.md) and [`docs/kinetics.md`](docs/kinetics.md) for the named-parameter API (`paramaxis`, `_named_params`).

This example demonstrates the core workflow for running a single simulation. The package provides extensive additional functionality for more advanced use cases like parameter estimation, ABC inference, and uncertainty analysis.

## Tutorials

Worked examples in `examples/`:

- `Tutorial 1 Running Simulations.jl` — `runsimulation` under three temperature profiles.
- `Tutorial 2 Parameter Estimation.jl` — PE + ABCDE + Turing NUTS on experimental data.
- `Tutorial 3 Sensitivity Analysis.jl` — global and local sensitivity workflows.
- `Tutorial 4 Defining a Custom Kinetic.jl` — three-step pattern (subtype + `paramaxis` + rate function) for adding a kinetic family from a user script.
- `Tutorial 5 ABCDE and MCMC.jl` — `run_abc` and Turing NUTS on synthetic data, side-by-side posteriors. Self-contained.

Run them from the repository root with the dedicated example environment:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples "examples/Tutorial 1 Running Simulations.jl"
```

## Documentation guides

Short, usage-first guides live in `docs/`:

- `docs/index.md` (navigation)
- `docs/simulation.md` (running simulations)
- `docs/kinetics.md` (nucleation and growth, extending models)
- `docs/solvers.md` (MoM / FiniteVol / WENO)
- `docs/parameter-estimation.md` (loss functions + PE_Routine)
- `docs/optimisation.md` (Optimization.jl routines)
- `docs/abcde.md` (ABCDE and Turner variants)
- `docs/measurements.md` (loading and balancing measurements)
- `docs/temperature-profiles.md` (temperature profiles)
- `docs/uq-ensembles.md` (ensemble UQ utilities)
