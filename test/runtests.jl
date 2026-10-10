using CriSTool
# Internal engine entry points used by a few regression checks.
using CriSTool: run_ensemble, prepare_loss, LossSetup
using Test
using Distributions
using DataFrames
import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences

include("fixtures.jl")

# `Pkg.test()` runs the core groups. Pass group names to run a subset, e.g.
# `Pkg.test(test_args = ["simulate", "steady"])`; "all" adds the optional groups.
const CORE_TEST_GROUPS = ["numerics", "simulate", "operations", "steady", "data", "fit_predict", "gradients"]
const OPTIONAL_TEST_GROUPS = ["plotting", "aqua"]
const SELECTED_TEST_GROUPS = isempty(ARGS) ? CORE_TEST_GROUPS :
                             "all" in ARGS ? vcat(CORE_TEST_GROUPS, OPTIONAL_TEST_GROUPS) : ARGS
for group_name in SELECTED_TEST_GROUPS
    group_name in CORE_TEST_GROUPS || group_name in OPTIONAL_TEST_GROUPS ||
        error("Unknown test group \"$group_name\".")
end

@testset "CriSTool.jl" begin
    for group_name in SELECTED_TEST_GROUPS
        @testset "$group_name" begin
            include("test_$(group_name).jl")
        end
    end
end
