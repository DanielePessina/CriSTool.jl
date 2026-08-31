using CriSTool
using Test
using Distributions
using DataFrames
import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences
import StaticArrays

@testset "CriSTool.jl" begin
    include("test_structs.jl")
    include("test_initial_state.jl")
    include("test_saturation.jl")
    include("test_kinetics.jl")
    include("test_dissolution.jl")
    include("test_weno_signed.jl")
    include("test_qmom.jl")
    include("test_aggregation_breakage.jl")
    include("test_runsimulation.jl")
    include("test_solvers.jl")
    include("test_staticarrays.jl")
    include("test_input_validation.jl")
    include("test_chainutilities.jl")
    include("test_bayesian.jl")
    include("test_generalization.jl")
    include("test_uncertainty_quantification.jl")
    include("test_autodiff.jl")
    include("test_gradients.jl")
    include("test_sensitivity.jl")
    include("test_measurements.jl")
    include("test_gold_fixture.jl")
    include("test_plotting.jl")
end
