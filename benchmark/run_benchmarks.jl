using CriSTool
using Chairmarks
using AllocCheck
using StatsBase
using Printf
using ComponentArrays

# ============================================================================
# CriSTool benchmark + allocation analysis
# Ports the spirit of
#   "Thesis - Benchmarks/benchmarks.jl" (Chairmarks @be, evals=1, ODE stats via
#   sol.ode_stats) into a self-contained setup that also runs AllocCheck.jl.
#
# Usage:  julia --project=benchmark benchmark/run_benchmarks.jl
# ============================================================================

const ROOT      = dirname(@__DIR__)            # repo root
const FIXTURE   = joinpath(ROOT, "test", "fixtures", "real-experimental-dataset.csv")
const FIXED_GRID = 0:1800:16200                # 0:30:270 min, expressed in seconds
const CANONICAL_θ = Float64[38.0, 0.0006, 1e-9 / 60, 3.0]  # SI gold params
const SEEDED_INITIAL_CRYSTALS = LogNormalInitialCrystals(
    mass_concentration = 0.25,
    d43 = 12e-6,
    geometric_std = 1.25)
const SECONDS = 5                              # Chairmarks sampling budget per benchmark

git_hash() = strip(read(`git -C $ROOT rev-parse --short HEAD`, String))

function _get_ode_stat(stats, name::Symbol)
    stats === nothing && return missing
    hasproperty(stats, name) || return missing
    return getproperty(stats, name)
end

const BENCHMARK_MODEL = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = CANONICAL_θ[1], surface_energy = CANONICAL_θ[2]),
    growth = EmpiricalGrowth(growth_coefficient = CANONICAL_θ[3], growth_order = CANONICAL_θ[4]))

"""Configured problem for one experiment: its temperature and measured initial concentration."""
benchmark_problem(e, solver; crystals = nothing) = CrystallisationProblem(BENCHMARK_MODEL,
    merge(e.conditions; concentration = CriSTool.initial_concentration(e), crystals); solver)

experiment_times(e) = e.observables.concentration.time

# Aggregate ODE stats over the 7 experiments of a fit run. NOTE: SciMLBase's
# current DEStats has no `nsteps` field (it was removed) — reported as missing.
function fit_ode_stats(exps, θ, solver; grid=nothing, initial_crystals=nothing)
    nf = naccept = nreject = nsave = 0
    nsolve = 0
    for e in exps
        save_idx = grid === nothing ? experiment_times(e) : grid
        _, sol = CriSTool.runsimulation(θ, benchmark_problem(e, solver; crystals = initial_crystals);
                                        save_idx)
        st = sol.ode_stats
        nf += something(_get_ode_stat(st, :nf), 0)
        naccept += something(_get_ode_stat(st, :naccept), 0)
        nreject += something(_get_ode_stat(st, :nreject), 0)
        nsolve += something(_get_ode_stat(st, :nsolve), 0)
        nsave += length(sol.time)
    end
    return (nf = nf, nsteps = missing, naccept = naccept,
            nreject = nreject, nsave = nsave, nsolve = nsolve)
end

"""Run the compact solvers on the same seeded workload.

DQMOM is intentionally seeded-only in this first implementation.  Keeping
this workload separate from the unseeded gold-fixture table makes that
contract visible and gives QMOM a like-for-like comparison using the same
kinetics, seed characteristics, temperatures, and observation grids.
"""
function run_seeded_compact_fit(exps, θ, solver)
    for e in exps
        CriSTool.runsimulation(θ, benchmark_problem(e, solver; crystals = SEEDED_INITIAL_CRYSTALS);
                               save_idx = experiment_times(e))
    end
    return nothing
end

# --- oracle fit: gold MoM fit over all 7 experiments -----------------------
function run_mom_fit(exps, θ)
    for e in exps
        CriSTool.runsimulation(θ, benchmark_problem(e, mom_solver); save_idx = experiment_times(e))
    end
    return nothing
end

# --- FV/WENO fits: same 7 experiments, common fixed grid -------------------
function run_fixed_grid_fit(exps, θ, solver)
    for e in exps
        CriSTool.runsimulation(θ, benchmark_problem(e, solver); save_idx = FIXED_GRID)
    end
    return nothing
end

# --- AllocCheck target: representative warm MoM call (flat-vector path) ----
warm_mom_call(θ, problem, timegrid) = last(CriSTool.runsimulation(θ, problem; save_idx = timegrid))

# --- results plumbing ------------------------------------------------------
function median_sample(bm::Chairmarks.Benchmark)
    t  = median([s.time for s in bm.samples])
    a  = median([s.allocs for s in bm.samples])
    b  = median([s.bytes for s in bm.samples])
    return (time = t, allocs = a, bytes = b)
end

function alloccheck_summary(errs)
    rows = String[]
    for e in errs
        kind = string(typeof(e).name.name)
        frame = ""
        for fr in e.backtrace
            fn = string(fr)
            if occursin("CriSTool/src", fn)
                frame = fn
                break
            end
        end
        isempty(frame) && (frame = string(first(e.backtrace)))
        msg = sprint(show, e)
        firstline = first(split(msg, '\n'))
        push!(rows, "  [$kind] $firstline")
        isempty(frame) || push!(rows, "        └─ $frame")
    end
    return rows
end

function main()
    exps = load_measurements(FIXTURE)
    @assert length(exps) == 7 "expected 7 experiments, got $(length(exps))"
    println("Loaded $(length(exps)) experiments from $(basename(FIXTURE))\n")

    global mom_solver = CriSTool.MoM()
    fv200  = CriSTool.FiniteVol(meshsize = 200)
    weno200 = CriSTool.WENO(meshsize = 200)
    qmom3 = CriSTool.QMOM(nquadrature = 3)
    dqmom3 = CriSTool.DQMOM(nquadrature = 3)

    # warm-up / compile (excluded from timing)
    run_mom_fit(exps, CANONICAL_θ)
    run_fixed_grid_fit(exps, CANONICAL_θ, fv200)
    run_fixed_grid_fit(exps, CANONICAL_θ, weno200)
    run_fixed_grid_fit(exps, CANONICAL_θ, qmom3)
    run_seeded_compact_fit(exps, CANONICAL_θ, qmom3)
    run_seeded_compact_fit(exps, CANONICAL_θ, dqmom3)

    # ---- benchmarks (Chairmarks, evals=1 like the original script) --------
    println("Benchmarking (Chairmarks, evals=1, seconds=$SECONDS)...")
    bm_mom   = @be run_mom_fit($exps, $CANONICAL_θ) evals=1 seconds=SECONDS
    bm_fv    = @be run_fixed_grid_fit($exps, $CANONICAL_θ, $fv200) evals=1 seconds=SECONDS
    bm_weno  = @be run_fixed_grid_fit($exps, $CANONICAL_θ, $weno200) evals=1 seconds=SECONDS
    bm_qmom  = @be run_fixed_grid_fit($exps, $CANONICAL_θ, $qmom3) evals=1 seconds=SECONDS
    bm_qmom_seeded = @be run_seeded_compact_fit($exps, $CANONICAL_θ, $qmom3) evals=1 seconds=SECONDS
    bm_dqmom = @be run_seeded_compact_fit($exps, $CANONICAL_θ, $dqmom3) evals=1 seconds=SECONDS

    # ---- ODE stats (single warm run each) ---------------------------------
    stats_mom  = fit_ode_stats(exps, CANONICAL_θ, mom_solver)
    stats_fv   = fit_ode_stats(exps, CANONICAL_θ, fv200; grid = FIXED_GRID)
    stats_weno = fit_ode_stats(exps, CANONICAL_θ, weno200; grid = FIXED_GRID)
    stats_qmom = fit_ode_stats(exps, CANONICAL_θ, qmom3; grid = FIXED_GRID)
    stats_qmom_seeded = fit_ode_stats(exps, CANONICAL_θ, qmom3;
                                      initial_crystals = SEEDED_INITIAL_CRYSTALS)
    stats_dqmom = fit_ode_stats(exps, CANONICAL_θ, dqmom3;
                                initial_crystals = SEEDED_INITIAL_CRYSTALS)

    # ---- AllocCheck on a representative warm MoM call ----------------------
    println("Running AllocCheck on the warm MoM call (flat-vector path)...")
    e1 = exps[1]
    warm_problem = benchmark_problem(e1, mom_solver)
    timegrid = collect(Float64, experiment_times(e1))
    warm_mom_call(CANONICAL_θ, warm_problem, timegrid)  # compile first
    t0 = time()
    alloc_errs = AllocCheck.check_allocs(warm_mom_call,
                                         (Vector{Float64}, typeof(warm_problem), Vector{Float64}))
    t_alloc = time() - t0

    # ---- report -------------------------------------------------------------
    io = stdout
    @printf(io, "\n%-6s %12s %12s %14s %8s %8s %8s %8s %10s %10s\n",
            "solver", "med time", "med allocs", "med bytes", "nf", "nacc",
            "nrej", "nsolve", "nsave", "nsteps")
    println(io, "-"^100)
    for (name, bm, st) in (("MoM", bm_mom, stats_mom),
                           ("FV200", bm_fv, stats_fv),
                           ("WENO200", bm_weno, stats_weno),
                           ("QMOM3", bm_qmom, stats_qmom))
        ms = median_sample(bm)
        @printf(io, "%-6s %10.3f ms %12d %14.1f %8d %8d %8d %8d %10d %10s\n",
                name, ms.time * 1e3, ms.allocs, ms.bytes,
                st.nf, st.naccept, st.nreject, st.nsolve, st.nsave,
                string(st.nsteps))
    end
    println(io, "-"^100)
    println(io, "(stats aggregated over the 7 experiments; MoM on per-experiment time ")
    println(io, " grids, FV200/WENO200 on fixed grid 0:1800:16200 seconds; nsteps removed from")
    println(io, " SciMLBase.DEStats, so nsave = length(sol.time) is reported instead)")

    println(io, "\nSeeded compact-solver comparison (same LogNormalInitialCrystals workload):")
    @printf(io, "%-14s %12s %12s %14s %8s %8s %8s %8s %10s %10s\n",
            "solver", "med time", "med allocs", "med bytes", "nf", "nacc",
            "nrej", "nsolve", "nsave", "nsteps")
    println(io, "-"^108)
    for (name, bm, st) in (("QMOM3-seeded", bm_qmom_seeded, stats_qmom_seeded),
                           ("DQMOM3", bm_dqmom, stats_dqmom))
        ms = median_sample(bm)
        @printf(io, "%-14s %10.3f ms %12d %14.1f %8d %8d %8d %8d %10d %10s\n",
                name, ms.time * 1e3, ms.allocs, ms.bytes,
                st.nf, st.naccept, st.nreject, st.nsolve, st.nsave,
                string(st.nsteps))
    end
    println(io, "-"^108)
    println(io, "DQMOM is seeded-only in v1; this table is intentionally separate")
    println(io, "from the unseeded gold-fixture comparisons above.")

    # ---- AllocCheck verdict -------------------------------------------------
    println(io, "\n==== AllocCheck verdict (Julia ", VERSION, ", AllocCheck ", pkgversion(AllocCheck), ") ====")
    println(io, "Analysis took ", round(t_alloc, digits = 1), " s on the warm MoM call.")
    if isempty(alloc_errs)
        println(io, "PASSED: no allocations or dynamic dispatches detected.")
    else
        println(io, "FAILED: ", length(alloc_errs), " potential allocation/dynamic-dispatch sites:")
        foreach(row -> println(io, row), alloccheck_summary(alloc_errs))
    end
    println(io, "(AllocCheck only sees statically resolvable code; the ODE solve core is")
    println(io, " behind kwcall boundaries and is NOT analysed — the reported sites above are all")
    println(io, " in the flat-vector wrapper: paramaxis/ComponentArray axis construction,")
    println(io, " src/physics/model_interfaces.jl and src/inference/workflow_interfaces.jl.)")

    # ---- hotspot: wrapper vs ComponentArray direct call ----------------------
    p = ComponentArray(CANONICAL_θ, paramaxis(warm_problem))
    function run_component_fit(exps, p)
        for e in exps
            CriSTool.runsimulation(p, benchmark_problem(e, mom_solver); save_idx = experiment_times(e))
        end
        return nothing
    end
    run_component_fit(exps, p)
    bm_ca = @be run_component_fit($exps, $p) evals=1 seconds=SECONDS
    m_flat = median_sample(bm_mom)
    m_ca   = median_sample(bm_ca)
    @printf(io, "\nWrapper-overhead probe (MoM fit, 7 exps):\n")
    @printf(io, "  flat Vector{Float64} path : %10.3f ms, %8d allocs, %10.1f bytes\n",
            m_flat.time * 1e3, m_flat.allocs, m_flat.bytes)
    @printf(io, "  ComponentArray direct path: %10.3f ms, %8d allocs, %10.1f bytes\n",
            m_ca.time * 1e3, m_ca.allocs, m_ca.bytes)
    @printf(io, "  wrapper overhead (delta)  : %10.3f ms, %8d allocs, %10.1f bytes\n",
            (m_flat.time - m_ca.time) * 1e3, m_flat.allocs - m_ca.allocs,
            m_flat.bytes - m_ca.bytes)

    # ---- attribution ----------------------------------------------------------
    println(io, "\nPackage git hash (repo root): ", git_hash())
    println(io, "Julia: ", VERSION, "  Chairmarks: ", pkgversion(Chairmarks),
            "  CriSTool: ", pkgversion(CriSTool))
    println(io, "\nRe-run:  julia --project=benchmark benchmark/run_benchmarks.jl")
end

main()
