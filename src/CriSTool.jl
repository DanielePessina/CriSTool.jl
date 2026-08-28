module CriSTool
"""
Main entry point for CriSTool. Imports dependencies, includes component source files defining models, optimisation routines and uncertainty tools, and creates default output directories.
"""

# Core Julia packages
using Base.Threads
using LinearAlgebra
using Printf
using Statistics
using Turing

# Scientific Computing & Differential Equations
using OrdinaryDiffEq
using SciMLSensitivity
using OptimizationBase
using ForwardDiff
using PreallocationTools: DiffCache, get_tmp
using DiffEqCallbacks
using SparseArrays
# using Zygote

# Data Structures & Processing
using DataFrames
using ComponentArrays
using StatsBase
using Distributions
using MCMCChains
import Base.Iterators

# Parallel Computing & Progress
using FLoops
using ProgressMeter

# File I/O & Serialization
using JLD2
using CodecZlib
using DelimitedFiles
import XLSX

# Visualization
using LaTeXStrings
using PairPlots
import CairoMakie
import Makie
import Plots
import StatsPlots

# Machine Learning & Optimization
# import Flux
import Lux
import MLJ
# import MLJFlux
# import MLJGaussianProcesses
# import SymbolicRegression
using QuasiMonteCarlo
using Metaheuristics

# Utilities
using ConcreteStructs
using Dates
using MonteCarloMeasurements
import Tables
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
       breakage_empirical, aggregation_empirical, noaggregation
export runsimulation, paramaxis, crystallisation_odeproblem
export AbstractPELossFunction, logMLE, mae, loss, prepare_loss, LossSetup
export load_experiments, load_experiments_legacy, load_experiments_legacy_single,
       bootstrap_repeatmeasurements, repeatmeasurementbalancer, psd_measurementbalancer
export getmomentsizes
export AbstractSolver, FiniteVol, MoM, WENO
export run_abc, AbstractABCSampler, ABCDESampler, ABCDETurnerSampler
export PE_Routine, ABCDE_Routine, ABCDE_Turner_Routine,
       Factored,
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
using .KissABC: ABCDE, smc, Factored, AIS, ApproxPosterior, ApproxKernelizedPosterior

include("PELossFunctions.jl")

# include("PELossFunctions_BO.jl")

# include("Generators.jl")

include("ABC.jl")

include("ChainUtilities.jl")

include("ChainsFunctions.jl")

include("SensitivityAnalysis.jl")

include("Uncertainties.jl")

include("plotting.jl")

#### Make directories:
"""
    create_project_directories()

Creates the necessary directory structure for the CriSTool project if the directories don't already exist.
Returns nothing.
"""
function create_project_directories()
    directories = [
        "R - PE Logs",
        "R - ABCDE Plots",
        joinpath("R - ABCDE Plots", "ABCDE Objects"),
        joinpath("R - MCMC Results", "MCMC Chains"),
        "R - MCMC Results",
        "R - Surrogate Datasets",
        "R - Hybrid Model Datasets",
        "Saved Plots"
    ]

    for dir in directories
        if !isdir(dir)
            mkpath(dir)
        end
    end
    return nothing
end

# Call the function when the module loads
# create_project_directories()

end
