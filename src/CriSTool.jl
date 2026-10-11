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
# The solver defaults (Tsit5, SSPRK43) come from the OrdinaryDiffEq subpackages;
# users pass any other algorithm instance from the subpackage that defines it.
using SciMLBase: SciMLBase, ODEProblem, remake, solve, CallbackSet, ContinuousCallback,
                 DiscreteCallback
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqSSPRK: SSPRK43
import DiffEqCallbacks: AutoAbstol, StepsizeLimiter
using OptimizationBase
using PreallocationTools: DiffCache, get_tmp

# Data Structures & Processing
using DataFrames
using ComponentArrays
using Distributions
using QuadGK: quadgk
using MCMCChains
import Base.Iterators

# File I/O
import CSV
import JSON

# Optimization
using Metaheuristics

# Utilities
using ConcreteStructs
using PrecompileTools: @setup_workload, @compile_workload
using MonteCarloMeasurements
using StaticArrays

# Shared numerical policies. Keeping these names centralized makes model,
# measurement, uncertainty, and plotting behavior auditable and configurable
# without scattering magic values across source files.
const CRISTOOL_MOMENT_DENSITY_FLOOR = 1e-8
const CRISTOOL_WENO_EPSILON = 1e-6
const CRISTOOL_DISSOLUTION_EQUILIBRIUM_TOLERANCE = 1e-3
const CRISTOOL_DISSOLUTION_LREF = 1e-6
const CRISTOOL_FAILED_SIMULATION_PENALTY = 1e6
const CRISTOOL_MAX_SOLVER_ITERS = 1e8
const CRISTOOL_MAX_PREPARED_SOLVER_ITERS = 1_000
const CRISTOOL_MAX_PREPARED_DISCRETIZED_SOLVER_ITERS = 10_000
const CRISTOOL_MAX_PREPARED_SOLVER_SECONDS = 5.0
const CRISTOOL_MAX_OPTIMISER_CALLS = 1e18
const CRISTOOL_PRIOR_PLOT_SAMPLES = 2^18

# Universal physical constants (SI).
"""Molar gas constant R (J/mol/K) used by Arrhenius rate laws."""
const CRISTOOL_GAS_CONSTANT = 8.314
"""Boltzmann constant k_B (J/K) used by classical nucleation theory."""
const CRISTOOL_BOLTZMANN_CONSTANT = 1.380649e-23

## Export stuff
export CrystallisationSolution, AbstractPopulationTrajectory, MeshPopulation,
       MomentPopulation, QuadraturePopulation,
       CrystallisationSteadyStateSolution, SteadyStateConvergenceError, SteadyStateOptions,
       CrystallisationProblem
export Observable, ObservableColumns, CrystallisationExperiment,
       initial_concentration
export AbstractCrystallisationOperation, BatchOperation, MSMPROperation, FedBatchOperation,
       CrystallisationFeed, reactor_vars, reactor_volume, operation_flows
export AbstractInitialCrystals, LogNormalInitialCrystals, GaussianInitialCrystals,
       FromObservation, RunConditions
export AbstractSolubilityModel, ConstantSolubility, PolynomialSolubility, CallableSolubility,
       lysozyme_solubility, saturation_concentration, supersaturation
export AbstractTemperature, ConstantTemperature, LinearTemperature, RampTemperature,
       CallableTemperature, temperature
export AbstractSolution, state_vars, size_metrics, observable_values, solvent_state,
       default_solvent_dynamics
export AbstractNucleationFunction, AbstractGrowthFunction,
       AbstractScalarGrowthFunction, AbstractLengthGrowthFunction, CNTNucleation,
       EmpiricalNucleation, ArrheniusNucleation, CNTNucleationNoSupersaturation, SecondaryNucleation,
       CombinedNucleation, NoNucleation, AbstractDissolutionFunction,
       AbstractScalarDissolutionFunction, AbstractLengthDissolutionFunction,
       NoDissolution, dissolutionrate, dissolutionrate!, dissolutionrate_at_length,
       net_growth_rate, net_growth_rate!, net_growth_rate_at_length,
       EmpiricalGrowth, ArrheniusGrowth,
       EmpiricalLengthGrowth, BirthSpreadGrowth, BCFGrowth,
       ArrheniusDissolution, ArrheniusLengthDissolution,
       nucleationrate, growthrate, aggregationrate, breakagerate,
       growthrate!, growthrate_at_length
export AbstractAggregationFunction, AbstractBreakageFunction, NoBreakage,
       EmpiricalBreakage, UniformBreakage, NoAggregation,
       ConstantAggregation, SizeSumAggregation, VolumeSumAggregation, AverageAggregation
export runsimulation, paramaxis, kinetic_parameter_vector, crystallisation_odeproblem,
       crystallisation_solution,
       solve_steadystate,
       initial_state_from_characteristics
export AbstractPELossFunction, AbstractVarianceModel, MeasuredVariance, RelativeVariance,
       GaussianNLL, MeanAbsoluteError, loss
export experiments_from_table, load_measurements,
       bootstrap_repeatmeasurements, balance_variances, repeatmeasurementbalancer,
       psd_measurementbalancer, bootstrap_measurements
export AbstractSolver, AbstractMomentSolver, FiniteVol, MoM, QMOM, DQMOM, WENO,
       QMOMQuadrature, QMOMInversionDiagnostics, DQMOMProjectionDiagnostics,
       invert_moments,
       moment_order, moment_count, quadrature,
       aggregation_moment_source, breakage_moment_source
export AbstractABCSampler, ABCDESampler, ABCDETurnerSampler
export PredictionEnsemble, prediction_summary
export plot_predictions, plot_posterior_pairplot, ChainStatsPlots
export rename_chain
export CrystallisationSystem, LysozymeSystem, KineticModel, CrystallisationModel,
       CallableGrowth, CallableNucleation, CallableLengthGrowth, KineticContext,
       required_properties, named_parameters, simulate, OptimisationSpec, BayesianSpec, ABCSpec,
       PreparedFit, prepare_fit, fit, FitResult, InferenceResult, is_likelihood, loss_terms,
       AbstractSimulationMode, TransientMode, SteadyMode,
       ParameterSamples, parameter_samples, predict, PredictionResult, measurement_samples,
       DistributionInitialCrystals, number_weighted, seed_domain_diagnostics

# Core types and public interfaces.
include("core/solution_types.jl")
include("core/kinetic_types.jl")
include("core/loss_types.jl")
include("core/initial_crystal_types.jl")
include("core/solver_types.jl")
include("core/temperature_types.jl")
include("core/saturation_types.jl")
include("core/operation_types.jl")
include("core/condition_types.jl")
include("core/measurement_types.jl")
include("core/problem_types.jl")

# Measurement ingestion and normalization.
include("measurements/loaders.jl")
include("measurements/balancing.jl")
include("measurements/bootstrap.jl")

# Solution post-processing precedes the model and solver implementations.
include("solvers/post_solution.jl")
include("solvers/initial_state.jl")
include("solvers/seed_distributions.jl")

# Kinetic rate families and population-balance terms.
include("physics/model_interfaces.jl")
include("physics/nucleation_rates.jl")
include("physics/growth_rates.jl")
include("physics/aggregation_breakage_rates.jl")
include("core/model_types.jl")

# Numerical solver implementations and public simulation wrappers.
include("solvers/solver_helpers.jl")
include("solvers/operation_balances.jl")
include("solvers/qmom.jl")
include("solvers/mom.jl")
include("solvers/dqmom.jl")
include("solvers/finite_volume.jl")
include("solvers/weno.jl")
include("solvers/steady_state.jl")

include("vendor/KissABC.jl")
using .KissABC
using .KissABC: ABCDE, smc, AIS, ApproxPosterior, ApproxKernelizedPosterior

# Inference and posterior utilities.
include("inference/parameter_estimation.jl")
include("inference/abc.jl")

include("inference/chain_utilities.jl")

include("inference/bayesian.jl")

include("inference/workflow_interfaces.jl")
include("inference/model_workflows.jl")

# REPL display for user-facing types (after every type it dispatches on)
include("core/show.jl")

# Plotting entry points; their methods live in the Makie/PairPlots extensions.
include("plotting/plots.jl")

include("precompile_workload.jl")

function __init__()
    Base.Experimental.register_error_hint(_plotting_method_error_hint, MethodError)
    return nothing
end

end
