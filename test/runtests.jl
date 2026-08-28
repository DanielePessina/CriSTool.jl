using CriSTool
using Test
using Distributions
import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences
import StaticArrays

@testset "CriSTool.jl" begin
    include("test_structs.jl")
    include("test_kinetics.jl")
    include("test_runsimulation.jl")
    include("test_solvers.jl")
    include("test_staticarrays.jl")
    include("test_input_validation.jl")
    include("test_chainutilities.jl")
    include("test_uncertainty_quantification.jl")
    include("test_autodiff.jl")
    include("test_plotting.jl")
end
