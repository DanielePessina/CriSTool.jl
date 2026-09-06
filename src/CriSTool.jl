module CriSTool
"""
Main entry point for CriSTool. Imports dependencies and includes source files
defining models, optimisation routines, uncertainty tools, and plotting.
"""

# Core Julia packages
using Base.Threads
using LinearAlgebra
using Random
using Statistics
using Turing

# Scientific Computing & Differential Equations
using OrdinaryDiffEq
import DiffEqCallbacks: AutoAbstol
using SciMLSensitivity
using OptimizationBase
using PreallocationTools: DiffCache, get_tmp

# Data Structures & Processing
using DataFrames
using ComponentArrays
using Distributions
using MCMCChains
import Base.Iterators

# Parallel Computing & Progress
using ProgressMeter

# File I/O & Serialization
using JLD2
import CSV
import JSON

# Visualization
using PairPlots
import CairoMakie
import Makie
import FlexiChains

# Machine Learning & Optimization
# import Flux
# import MLJFlux
# import MLJGaussianProcesses
# import SymbolicRegression
using Metaheuristics

# Utilities
using ConcreteStructs
using Dates
using MonteCarloMeasurements
import Term
using PrettyTables
using StaticArrays

# Shared numerical policies. Keeping these names centralized makes model,
# measurement, uncertainty, and plotting behavior auditable and configurable
# without scattering magic values across source files.
const CRISTOOL_MOMENT_DENSITY_FLOOR = 1e-8
const CRISTOOL_WENO_EPSILON = 1e-6
const CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE = 1e-3
const CRISTOOL_DISSOLUTION_LREF = 1e-6
const CRISTOOL_CONFIDENCE_Z95 = 1.96
const CRISTOOL_FAILED_SIMULATION_PENALTY = 1e6
const CRISTOOL_MAX_SOLVER_ITERS = 1e8
const CRISTOOL_MAX_PREPARED_SOLVER_ITERS = 1_000
const CRISTOOL_MAX_PREPARED_DISCRETIZED_SOLVER_ITERS = 10_000
const CRISTOOL_MAX_PREPARED_SOLVER_SECONDS = 5.0
const CRISTOOL_PARALLEL_PE_BATCH_SIZE = 8
const CRISTOOL_MAX_OPTIMISER_CALLS = 1e18
const CRISTOOL_PRIOR_PLOT_SAMPLES = 2^18

## Export stuff
export CrystallisationFVSolution, CrystallisationMoMSolution,
       CrystallisationQMOMSolution, CrystallisationDQMOMSolution,
       EnsembleFVSolution, EnsembleMoMSolution,
       CrystallisationProblem
export Observable, ObservableColumns, CrystallisationExperiment, AbstractExperiment,
       initial_concentration
export AbstractInitialCrystals, LogNormalInitialCrystals, GaussianInitialCrystals
export AbstractSolubilityModel, ConstantSolubility, PolynomialSolubility, CallableSolubility,
       AbstractSaturationModel, ConstantSaturation, PolynomialSaturation, CallableSaturation,
       lysozyme_solubility, lysozyme_saturation, saturation_concentration, supersaturation
export AbstractTemperature, ConstantTemperature, LinearTemperature, RampTemperature,
       CallableTemperature, temperature
export AbstractSolution, state_vars, size_metrics, observable_values, solvent_state,
       default_solvent_dynamics
export AbstractNucleationFunction, AbstractGrowthFunction, AbstractFPGrowthFunction,
       AbstractFPScalarGrowthFunction, AbstractFPLengthGrowthFunction, nucl_CNT,
       nucl_empirical, nucl_empirical_energy, nucl_CNTnoS, nucl_secondary,
       nucl_prim_plus_second, nucl_CNT_plus_second, nucl_CNT_fixed,
       nucl_empirical_fixed, AbstractDissolutionFunction,
       AbstractFPScalarDissolutionFunction, AbstractFPLengthDissolutionFunction,
       AbstractScalarDissolutionFunction, AbstractLengthDissolutionFunction,
       nodissolution, nondissolution, dissolutionrate, dissolutionrate!, dissolutionrate_at_length,
       net_growth_rate, net_growth_rate!, net_growth_rate_at_length,
       growth_empirical, growth_empirical_fixed, growth_energy, growth_energy_est,
       growth_empirical_length, growth_BpS, growth_BCF,
       growth_dissolution, growth_dissolution_length, growth_energy_dissolution,
       nucleationrate, growthrate, aggregationrate, breakagerate,
       growthrate!, growthrate_at_length, net_growthrate
export AbstractAggregationFunction, AbstractBreakageFunction, nobreakage,
       breakage_empirical, breakage_uniform, noaggregation,
       aggr_scalar, aggr_linear, aggr_linearvol, aggr_avg
export runsimulation, paramaxis, crystallisation_odeproblem,
       get_characteristic_size, getmomentsizes,
       initial_state_from_characteristics
export AbstractPELossFunction, AbstractVarianceModel, MeasuredVariance, RelativeVariance,
       logMLE, mae, loss, prepare_loss, LossSetup
export experiments_from_table, load_measurements,
       bootstrap_repeatmeasurements, balance_variances, repeatmeasurementbalancer,
       psd_measurementbalancer, bootstrap_measurements
export AbstractSolver, AbstractMomentSolver, FiniteVol, MoM, QMOM, DQMOM, WENO,
       QMOMQuadrature, QMOMInversionDiagnostics, DQMOMProjectionDiagnostics,
       invert_moments,
       moment_order, moment_count, nmoments, quadrature, qmom_quadrature,
       dqmom_quadrature,
       aggregation_moment_source, breakage_moment_source
export run_abc, AbstractABCSampler, ABCDESampler, ABCDETurnerSampler
export run_ensemble, run_ensemble_fixed
export PE_Routine, PE_Routine_Optimisation, ABCDE_Routine, ABCDE_Turner_Routine,
       plot_posterior_pairplot, plot_measurements_vs_ensemble
export ChainPairPlots, ChainStatsPlots, ChainMeasurementPlots
export plot_ps_measurements_vs_ensemble, plot_measurements_vs_simulation,
       plot_ps_measurements_vs_simulation
export chains_to_matrix, distribution_to_matrix, create_product_prior, prior_to_matrix
export nuts_model, MCMC_Routine, kinetic_parameter_symbols, rename_chain
export CRISTOOL_PALETTE

# Core types and public interfaces.
include("core/solution_types.jl")
include("core/kinetic_types.jl")
include("core/loss_types.jl")
include("core/initial_crystal_types.jl")
include("core/measurement_types.jl")
include("core/solver_types.jl")
include("core/temperature_types.jl")
include("core/saturation_types.jl")
include("core/problem_types.jl")

# Measurement ingestion and normalization.
include("measurements/loaders.jl")
include("measurements/balancing.jl")
include("measurements/bootstrap.jl")

# Solution post-processing precedes the model and solver implementations.
include("solvers/post_solution.jl")
include("solvers/initial_state.jl")

# Kinetic rate families and population-balance terms.
include("physics/model_interfaces.jl")
include("physics/nucleation_rates.jl")
include("physics/growth_rates.jl")
include("physics/aggregation_breakage_rates.jl")

# Numerical solver implementations and public simulation wrappers.
include("solvers/solver_helpers.jl")
include("solvers/qmom.jl")
include("solvers/mom.jl")
include("solvers/dqmom.jl")
include("solvers/finite_volume.jl")
include("solvers/weno.jl")
include("solvers/runsimulation.jl")

include("output/pretty_printing.jl")

include("vendor/KissABC.jl")
using .KissABC
using .KissABC: ABCDE, smc, AIS, ApproxPosterior, ApproxKernelizedPosterior

# Inference and posterior utilities.
include("inference/parameter_estimation.jl")

# include("inference/parameter_estimation_bo.jl")

include("inference/abc.jl")

include("inference/chain_utilities.jl")

include("inference/bayesian.jl")

# Uncertainty workflows.
include("uncertainty/sensitivity.jl")
include("uncertainty/ensembles.jl")

# Plotting is included last because it consumes all public solution and
# inference interfaces.
include("plotting/common.jl")
include("plotting/ensemble.jl")
include("plotting/simulation.jl")



end
