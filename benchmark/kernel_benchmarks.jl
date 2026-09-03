using CriSTool
using Chairmarks
using ComponentArrays
using Statistics
using Printf

"""Benchmark isolated aggregation and breakage RHS terms.

Run with:

    julia --project=benchmark benchmark/kernel_benchmarks.jl

The reported times exclude first-call compilation. Each call includes the
allocation of its returned rate vector, as required by the current out-of-place
kernel interface.
"""

const BENCHMARK_SECONDS = 2
const BENCHMARK_MESHES = (50, 100, 250, 500)

function benchmark_problem(meshsize)
    solver = FiniteVol(meshsize = meshsize, lmax = 50e-6)
    problem = CrystallisationProblem(; solver = solver,
                                     initial_solvent_state = (; concentration = 1.0))
    state = [abs(sin(index)) for index in 1:meshsize]
    return problem, solver, [state; 1.0]
end

function median_sample(benchmark)
    return (time = median(sample.time for sample in benchmark.samples),
            allocations = median(sample.allocs for sample in benchmark.samples),
            bytes = median(sample.bytes for sample in benchmark.samples))
end

function main()
    println("Julia ", VERSION, ", Chairmarks ", pkgversion(Chairmarks))
    println("Median steady-state kernel costs; seconds=", BENCHMARK_SECONDS)
    println()
    @printf("%-8s %18s %18s %18s %18s\n",
            "mesh", "aggregation μs", "aggregation bytes",
            "breakage μs", "breakage bytes")
    println("-"^82)

    for meshsize in BENCHMARK_MESHES
        problem, solver, state = benchmark_problem(meshsize)
        aggregationfunction = aggr_scalar()
        aggregationparameters = ComponentArray([log10(1.0 / 60.0)], paramaxis(aggregationfunction))
        breakagefunction = breakage_empirical()
        breakageparameters = ComponentArray([1.0 / 60.0, 0.0], paramaxis(breakagefunction))

        # Warm-up is excluded from the measured samples.
        CriSTool.aggregationrate(aggregationfunction,
                                 aggregationparameters,
                                 problem,
                                 state,
                                 0.0)
        CriSTool.breakagerate(breakagefunction,
                              breakageparameters,
                              problem,
                              state,
                              0.0)

        aggregationbenchmark = @be CriSTool.aggregationrate($aggregationfunction,
                                                              $aggregationparameters,
                                                              $problem,
                                                              $state,
                                                              0.0) evals = 1 seconds = BENCHMARK_SECONDS
        breakagebenchmark = @be CriSTool.breakagerate($breakagefunction,
                                                       $breakageparameters,
                                                       $problem,
                                                       $state,
                                                       0.0) evals = 1 seconds = BENCHMARK_SECONDS
        aggregation_sample = median_sample(aggregationbenchmark)
        breakage_sample = median_sample(breakagebenchmark)
        @printf("%-8d %15.2f %18d %15.2f %18d\n",
                meshsize,
                aggregation_sample.time * 1e6,
                aggregation_sample.bytes,
                breakage_sample.time * 1e6,
                breakage_sample.bytes)
    end
end

main()
