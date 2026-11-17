# WENO default integrator: SSPRK43 vs Tsit5 (function evaluations, time, error).
# Run from the repository root with the package environment:
#   julia --project=. benchmark/weno_integrator.jl
using CriSTool, Statistics, Printf
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqSSPRK: SSPRK43
include(joinpath(@__DIR__, "..", "test", "fixtures.jl"))

function timed_solve(problem, solver; saveat, alg)
    solution = solve(problem, solver; saveat, alg)          # compile + result
    elapsed = minimum(@elapsed(solve(problem, solver; saveat, alg)) for _ in 1:7)
    return solution, elapsed
end
relative_error(values, reference) = maximum(abs.(values .- reference) ./ max.(abs.(reference), eps()))

benchmark_cases = [
    (label = "translation seed, WENO(40)", problem = CrystallisationProblem(fixture_translation_model(),
        fixture_seeded_conditions(), (0.0, 100.0)), meshsize = 40, lmax = 40e-6, saveat = [0.0, 50.0, 100.0],
        closed_form = (concentration = fixture_translation_concentration.([0.0, 50.0, 100.0]),
                       d43 = fixture_translation_d43.([0.0, 50.0, 100.0]))),
    (label = "translation seed, WENO(200)", problem = CrystallisationProblem(fixture_translation_model(),
        fixture_seeded_conditions(), (0.0, 100.0)), meshsize = 200, lmax = 40e-6, saveat = [0.0, 50.0, 100.0],
        closed_form = (concentration = fixture_translation_concentration.([0.0, 50.0, 100.0]),
                       d43 = fixture_translation_d43.([0.0, 50.0, 100.0]))),
    (label = "lysozyme CNT + growth, WENO(200)", problem = CrystallisationProblem(CrystallisationModel(LysozymeSystem();
            nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
            growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0)),
        RunConditions(temperature = 293.15, concentration = 18.0), (0.0, 28_800.0)),
        meshsize = 200, lmax = 100e-6, saveat = collect(0.0:3600.0:28_800.0), closed_form = nothing),
    (label = "dissolution to extinction, WENO(40)", problem = CrystallisationProblem(CrystallisationModel(fixture_system;
            growth = EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 1.0),
            dissolution = ArrheniusDissolution(dissolution_coefficient = 1e-6, activation_energy = 0.0,
                dissolution_order = 1.0)), fixture_seeded_conditions(concentration = 5.0), (0.0, 60.0);
            solid_mass_concentration_threshold = 5e-4),
        meshsize = 40, lmax = 40e-6, saveat = [0.0, 10.0, 30.0, 60.0], closed_form = nothing),
    (label = "partial dissolution, WENO(200)", problem = CrystallisationProblem(CrystallisationModel(fixture_system;
            growth = EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 1.0),
            dissolution = ArrheniusDissolution(dissolution_coefficient = 1e-7, activation_energy = 0.0,
                dissolution_order = 1.0)), fixture_seeded_conditions(concentration = 9.0), (0.0, 30.0)),
        meshsize = 200, lmax = 40e-6, saveat = [0.0, 10.0, 30.0], closed_form = nothing),
]

for benchmark_case in benchmark_cases
    println("\n== ", benchmark_case.label, " ==")
    reference = solve(benchmark_case.problem, WENO(meshsize = benchmark_case.meshsize, lmax = benchmark_case.lmax,
        reltol = 1e-11, abstol = 1e-16); saveat = benchmark_case.saveat, alg = Tsit5())
    @printf("%-8s %8s %8s %10s %14s %14s %14s\n", "alg", "nf", "naccept", "time ms",
        "C err vs ref", "d43 err vs ref", "d43 err vs exact")
    for (alg_label, alg) in (("Tsit5", Tsit5()), ("SSPRK43", SSPRK43()))
        solver = WENO(meshsize = benchmark_case.meshsize, lmax = benchmark_case.lmax)
        solution, elapsed = timed_solve(benchmark_case.problem, solver; saveat = benchmark_case.saveat, alg)
        exact_error = benchmark_case.closed_form === nothing ? NaN :
            relative_error(solution.d43, benchmark_case.closed_form.d43)
        @printf("%-8s %8d %8d %10.2f %14.2e %14.2e %14.2e\n", alg_label, solution.ode_stats.nf,
            solution.ode_stats.naccept, 1e3 * elapsed, relative_error(solution.concentration, reference.concentration),
            relative_error(solution.d43, reference.d43), exact_error)
    end
end
println("\nWENO_BENCHMARK_DONE")
