# Apples-to-apples rerun of the original Thesis - Benchmarks sweep
# (same params/grids/timesteppings/meshsizes) against the CURRENT CriSTool.
# Run: julia --project=benchmark benchmark/thesis_comparison.jl

using CriSTool, Chairmarks, Printf, StatsBase

using OrdinaryDiffEq: SSPRK43, Tsit5, Kvaerno5

chosen_θ = Float64[38.0, 0.00055, 0.7e-9 / 60, 2.3]
thesis_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = chosen_θ[1], surface_energy = chosen_θ[2]),
    growth = EmpiricalGrowth(growth_coefficient = chosen_θ[3], growth_order = chosen_θ[4]))
thesis_conditions = RunConditions(temperature = 293.15, concentration = 18.0)
thesis_save_times = 0:240:24000        # 0:4:400 minutes, expressed in seconds
thesis_problem = CrystallisationProblem(thesis_model, thesis_conditions,
    (first(thesis_save_times), last(thesis_save_times)), chosen_θ)

meshsizes = [50, 100, 200, 500]
timesteppings = (SSPRK43(), Tsit5(), Kvaerno5())
timestepping_label(algorithm) = string(nameof(typeof(algorithm)))

function ode_stat(sol, name::Symbol)
    stats = sol.ode_stats
    (stats === nothing || !hasproperty(stats, name)) && return missing
    return getproperty(stats, name)
end

rows = Any[]
for ts in timesteppings
    for ms in meshsizes
        solver = CriSTool.FiniteVol(meshsize = ms, timestepping_algorithm = ts)
        b = @be solve($thesis_problem, $solver; saveat = $thesis_save_times) evals=1 seconds=6
        sol = solve(thesis_problem, solver; saveat = thesis_save_times)
        t_ms = StatsBase.median([s.time for s in b.samples]) * 1e3
        a = StatsBase.median([s.allocs for s in b.samples])
        push!(rows, (solver = "FV", ts = timestepping_label(ts), ms = ms, t_ms = t_ms, allocs = a,
                     nf = ode_stat(sol, :nf), naccept = ode_stat(sol, :naccept)))

        solver = CriSTool.WENO(meshsize = ms, timestepping_algorithm = ts)
        b = @be solve($thesis_problem, $solver; saveat = $thesis_save_times) evals=1 seconds=6
        sol = solve(thesis_problem, solver; saveat = thesis_save_times)
        t_ms = StatsBase.median([s.time for s in b.samples]) * 1e3
        a = StatsBase.median([s.allocs for s in b.samples])
        push!(rows, (solver = "WENO", ts = timestepping_label(ts), ms = ms, t_ms = t_ms, allocs = a,
                     nf = ode_stat(sol, :nf), naccept = ode_stat(sol, :naccept)))
    end

    solver = CriSTool.MoM(timestepping_algorithm = ts)
    b = @be solve($thesis_problem, $solver; saveat = $thesis_save_times) evals=1 seconds=6
    sol = solve(thesis_problem, solver; saveat = thesis_save_times)
    t_ms = StatsBase.median([s.time for s in b.samples]) * 1e3
    a = StatsBase.median([s.allocs for s in b.samples])
    push!(rows, (solver = "MoM", ts = timestepping_label(ts), ms = 0, t_ms = t_ms, allocs = a,
                 nf = ode_stat(sol, :nf), naccept = ode_stat(sol, :naccept)))
end

println("\n========== Thesis-sweep rerun on current CriSTool ==========")
@printf("%-6s %-10s %-6s %12s %14s %10s %10s\n", "solver", "ts", "mesh", "med ms", "allocs",
        "nf", "naccept")
for r in rows
    @printf("%-6s %-10s %-6d %12.4f %14d %10s %10s\n", r.solver, string(r.ts), r.ms,
            r.t_ms, r.allocs, r.nf, r.naccept)
end
println("git: ", readchomp(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`))
