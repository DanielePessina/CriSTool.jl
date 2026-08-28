module CriSTool
"""
Main entry point for CriSTool. Imports dependencies, includes component source files defining models, optimisation routines and uncertainty tools, and creates default output directories.
"""

# Core Julia packages
using Base.Threads
using LinearAlgebra
using Statistics
using Turing

# Scientific Computing & Differential Equations
using OrdinaryDiffEq
using SciMLSensitivity
using OptimizationBase
using ForwardDiff
using PreallocationTools: DiffCache, get_tmp
# using Zygote

# Data Structures & Processing
using DataFrames
using ComponentArrays
using Distributions
using MCMCChains
import Base.Iterators

# Parallel Computing & Progress
using FLoops
using ProgressMeter

# File I/O & Serialization
using JLD2
import XLSX

# Visualization
using PairPlots
import Plots
import CairoMakie
import Makie
import StatsPlots

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

## Export stuff
export CrystallisationFVSolution, CrystallisationMoMSolution, CrystallisationProblem
export Observable, CrystallisationExperiment, AbstractExperiment,
       initial_concentration
export AbstractSaturationModel, ConstantSaturation, PolynomialSaturation, CallableSaturation,
       lysozyme_saturation, saturation_concentration, supersaturation
export AbstractSolution, state_vars, size_metrics
export AbstractNucleationFunction, AbstractGrowthFunction, nucl_CNT,
       nucl_empirical, nucl_CNTnoS, growth_empirical, growth_BpS, growth_BCF
export AbstractAggregationFunction, AbstractBreakageFunction, nobreakage,
       breakage_empirical, noaggregation
export runsimulation, paramaxis, crystallisation_odeproblem,
       get_characteristic_size, getmomentsizes
export AbstractPELossFunction, logMLE, mae, loss, prepare_loss, LossSetup
export load_experiments, load_experiments_legacy, load_experiments_legacy_single,
       bootstrap_repeatmeasurements, repeatmeasurementbalancer, psd_measurementbalancer
export AbstractSolver, FiniteVol, MoM, WENO
export run_abc, AbstractABCSampler, ABCDESampler, ABCDETurnerSampler
export PE_Routine, ABCDE_Routine, ABCDE_Turner_Routine,
       plot_posterior_pairplot, plot_measurements_vs_ensemble
export chains_to_matrix, distribution_to_matrix, create_product_prior, prior_to_matrix
export CRISTOOL_PALETTE

include("Structs.jl")

include("Measurements.jl")

include("PostSolution.jl")

include("Models.jl")

include("PrettyPrinting.jl")

include("KissABC.jl")
using .KissABC
using .KissABC: ABCDE, smc, AIS, ApproxPosterior, ApproxKernelizedPosterior

include("PELossFunctions.jl")

# include("PELossFunctions_BO.jl")

include("ABC.jl")

include("ChainUtilities.jl")

include("ChainsFunctions.jl")

include("SensitivityAnalysis.jl")

include("Uncertainties.jl")

include("plotting.jl")



# Call the function when the module loads
# create_project_directories()

end
