# Overhead outside the ODE integrator on the 7-experiment gold fixture.
#
# Run:  julia --project=benchmark benchmark/inference_overhead.jl [MoM|FV200|steady ...]
#
# For each solver (MoM and FiniteVol(meshsize = 200)) it times
#   (a) one prepared loss evaluation as `fit` performs it, and its ForwardDiff gradient;
#   (b) the sum of the raw `solve` calls for the same runs, same options, built once;
#       (a) - (b) is the work CriSTool does outside the integrator;
#   (c) `simulate(model, conditions)` against the raw solve of the same run;
#   (d) `predict` over 100 joint parameter draws;
# and (e) a steady MSMPR relaxation (`solve_steadystate` and a steady loss).
# Every row reports the median time, allocations and allocated bytes.
using CriSTool, Chairmarks, ComponentArrays, ForwardDiff, Printf, Random
import OrdinaryDiffEq

const OVERHEAD_FIXTURE = joinpath(dirname(@__DIR__), "test", "fixtures", "real-experimental-dataset.csv")
const OVERHEAD_RUNS = load_measurements(OVERHEAD_FIXTURE)
const OVERHEAD_THETA = [38.0, 0.0006, 1e-9 / 60, 3.0]
const OVERHEAD_MODEL = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = OVERHEAD_THETA[1], surface_energy = OVERHEAD_THETA[2]),
    growth = EmpiricalGrowth(growth_coefficient = OVERHEAD_THETA[3], growth_order = OVERHEAD_THETA[4]))
# All four kinetic parameters are selected, so the gradient has four partials.
const OVERHEAD_BOUNDS = (nucleation = (ln_nucleation_prefactor = (30.0, 45.0), surface_energy = (1e-4, 1e-3)),
    growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0)))
const OVERHEAD_SECONDS = 3.0

overhead_rows = Tuple{String, Float64, Float64, Float64}[]

function record_overhead(row_label, benchmark_result)
    median_sample = Chairmarks.median(benchmark_result)
    push!(overhead_rows, (row_label, median_sample.time, median_sample.allocs, median_sample.bytes))
    @printf("%-46s %12.1f µs %10.0f allocs %12.0f bytes\n", row_label, 1e6 * median_sample.time,
        median_sample.allocs, median_sample.bytes)
    flush(stdout)
    return median_sample.time
end

# The raw solve: the template built by the prepared setup, the same options the
# package passes to `solve`, with the integrator as the only timed work.
function overhead_raw_solve_inputs(prepared_fit, kinetic_values)
    map(prepared_fit.setup.solves.runs) do prepared_run
        configured_problem = prepared_run.problem
        odeproblem, _ = crystallisation_odeproblem(configured_problem, prepared_run.saveat,
            ComponentArray(kinetic_values, CriSTool.paramaxis(configured_problem)))
        configured_solver = configured_problem.solver
        solve_settings = merge((reltol = configured_solver.reltol,
                abstol = CriSTool._solver_absolute_floor(configured_problem, odeproblem),
                dense = false, maxiters = CriSTool.CRISTOOL_MAX_SOLVER_ITERS,
                internalnorm = CriSTool._primal_error_norm, saveat = prepared_run.saveat),
            CriSTool._solver_alg_hints(configured_solver))
        (odeproblem, prepared_run.algorithm, solve_settings)
    end
end

function overhead_raw_solves(raw_inputs)
    for (odeproblem, ode_algorithm, solve_settings) in raw_inputs
        OrdinaryDiffEq.solve(odeproblem, ode_algorithm; solve_settings...)
    end
    return nothing
end

function overhead_dual_values(parameter_values)
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{length(parameter_values), Float64})
    # A registered tag, so nested Jacobian duals of stiff solvers order after it.
    gradient_tag = typeof(ForwardDiff.Tag(overhead_dual_values, Float64))
    return [ForwardDiff.Dual{gradient_tag}(parameter_value, seed)
            for (parameter_value, seed) in zip(parameter_values, seeds)]
end

function benchmark_overhead_solver(solver_label, overhead_solver)
    println("\n== ", solver_label, " ==")
    prepared_fit = prepare_fit(OVERHEAD_MODEL, OVERHEAD_RUNS,
        OptimisationSpec(bounds = OVERHEAD_BOUNDS, algorithm = nothing); solver = overhead_solver)
    loss_objective = selected_values -> loss(prepared_fit, selected_values)
    gradient_config = ForwardDiff.GradientConfig(loss_objective, OVERHEAD_THETA, ForwardDiff.Chunk{4}())
    gradient_buffer = similar(OVERHEAD_THETA)
    loss_objective(OVERHEAD_THETA)
    ForwardDiff.gradient!(gradient_buffer, loss_objective, OVERHEAD_THETA, gradient_config)

    loss_time = record_overhead("(a) prepared loss, 7 runs",
        @be(loss_objective(OVERHEAD_THETA), seconds = OVERHEAD_SECONDS))
    gradient_time = record_overhead("(a) ForwardDiff gradient, 7 runs",
        @be(ForwardDiff.gradient!(gradient_buffer, loss_objective, OVERHEAD_THETA, gradient_config),
            seconds = OVERHEAD_SECONDS))

    raw_inputs = overhead_raw_solve_inputs(prepared_fit, OVERHEAD_THETA)
    raw_dual_inputs = overhead_raw_solve_inputs(prepared_fit, overhead_dual_values(OVERHEAD_THETA))
    raw_time = record_overhead("(b) raw solves, 7 runs",
        @be(overhead_raw_solves(raw_inputs), seconds = OVERHEAD_SECONDS))
    raw_dual_time = record_overhead("(b) raw dual solves, 7 runs",
        @be(overhead_raw_solves(raw_dual_inputs), seconds = OVERHEAD_SECONDS))
    @printf("   overhead outside integrator: loss %.1f µs (%.0f%%), gradient %.1f µs (%.0f%%)\n",
        1e6 * (loss_time - raw_time), 100 * (loss_time - raw_time) / loss_time,
        1e6 * (gradient_time - raw_dual_time), 100 * (gradient_time - raw_dual_time) / gradient_time)

    first_run = first(OVERHEAD_RUNS)
    first_conditions = merge(first_run.conditions; concentration = initial_concentration(first_run))
    first_times = first(prepared_fit.setup.solves.runs).saveat
    simulate(OVERHEAD_MODEL, first_conditions; solver = overhead_solver, saveat = first_times)
    record_overhead("(c) simulate, run 1",
        @be(simulate(OVERHEAD_MODEL, first_conditions; solver = overhead_solver, saveat = first_times),
            seconds = OVERHEAD_SECONDS))
    record_overhead("(c) raw solve, run 1",
        @be(overhead_raw_solves(raw_inputs[1:1]), seconds = OVERHEAD_SECONDS))
    first_problem = first(prepared_fit.setup.solves.runs).problem
    first_ode_solution = OrdinaryDiffEq.solve(raw_inputs[1][1], raw_inputs[1][2]; raw_inputs[1][3]...)
    record_overhead("(c) crystallisation_solution, run 1",
        @be(crystallisation_solution(first_problem, first_ode_solution), seconds = OVERHEAD_SECONDS))

    draw_rng = Xoshiro(20261011)
    prediction_draws = ParameterSamples(nucleation = (
        ln_nucleation_prefactor = 38.0 .+ 0.2 .* randn(draw_rng, 100),),)
    predict(OVERHEAD_MODEL, OVERHEAD_RUNS, prediction_draws; solver = overhead_solver)
    record_overhead("(d) predict, 100 draws x 7 runs",
        @be(predict(OVERHEAD_MODEL, OVERHEAD_RUNS, prediction_draws; solver = overhead_solver),
            seconds = 2 * OVERHEAD_SECONDS))
    return nothing
end

# Lysozyme MSMPR with built-in (time-invariant) kinetics, fed at 40 kg/m³.
const OVERHEAD_STEADY_CONDITIONS = RunConditions(temperature = 293.15, concentration = 40.0,
    operation = MSMPROperation(volume = 1.0, inflow = 1.0 / 3600.0,
        feed = CrystallisationFeed(concentration = 40.0)))

function benchmark_overhead_steady(solver_label, overhead_solver)
    println("\n== steady MSMPR, ", solver_label, " ==")
    steady_problem = CrystallisationProblem(OVERHEAD_MODEL, OVERHEAD_STEADY_CONDITIONS; solver = overhead_solver)
    steady_options = SteadyStateOptions(relaxation_horizon = 360_000.0)
    steady_result = solve_steadystate(steady_problem; options = steady_options)
    record_overhead("(e) solve_steadystate",
        @be(solve_steadystate(steady_problem; options = steady_options), seconds = OVERHEAD_SECONDS))
    steady_concentration = steady_result.equilibrium.concentration
    steady_experiment = CrystallisationExperiment(observables = (
            concentration = Observable(time = [0.0], mean = [steady_concentration], variance = [0.01]),
            d43 = Observable(time = [0.0], mean = [steady_result.equilibrium.d43], variance = [1e-14])),
        conditions = OVERHEAD_STEADY_CONDITIONS, exp_id = 1)
    steady_fit = prepare_fit(OVERHEAD_MODEL, [steady_experiment],
        OptimisationSpec(bounds = OVERHEAD_BOUNDS, algorithm = nothing);
        solver = overhead_solver, mode = SteadyMode(steady_options))
    steady_objective = selected_values -> loss(steady_fit, selected_values)
    steady_objective(OVERHEAD_THETA)
    record_overhead("(e) steady loss",
        @be(steady_objective(OVERHEAD_THETA), seconds = OVERHEAD_SECONDS))
    record_overhead("(e) steady ForwardDiff gradient",
        @be(ForwardDiff.gradient(steady_objective, OVERHEAD_THETA), seconds = OVERHEAD_SECONDS))
    return nothing
end

# Define `const OVERHEAD_SKIP_RUN = true` before `include` to load the helpers only.
const OVERHEAD_SELECTION = @isdefined(OVERHEAD_SKIP_RUN) ? String[] :
    isempty(ARGS) ? ["MoM", "FV200", "steady"] : ARGS
isempty(OVERHEAD_SELECTION) || println("Julia ", VERSION, "; threads=", Threads.nthreads(), "; CPU=", Sys.CPU_NAME)
"MoM" in OVERHEAD_SELECTION && benchmark_overhead_solver("MoM", MoM())
"FV200" in OVERHEAD_SELECTION && benchmark_overhead_solver("FiniteVol(meshsize = 200)", FiniteVol(meshsize = 200))
"steady" in OVERHEAD_SELECTION && (benchmark_overhead_steady("MoM", MoM());
    benchmark_overhead_steady("FiniteVol(meshsize = 200)", FiniteVol(meshsize = 200)))
