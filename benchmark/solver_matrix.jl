using CriSTool
using Chairmarks
using ForwardDiff
using Printf
using Statistics

# ============================================================================
# Solver x kinetics benchmark matrix.
#
# Every case runs on its own fixed 4 h save grid with the solver's default
# tolerances and time stepper. For each supported (case, solver) pair the
# script reports:
#   - the median forward solve (`runsimulation`, the call fits make), its
#     allocations and bytes;
#   - the number of RHS evaluations and the time per evaluation, which splits
#     cost into "how many steps" and "how expensive each step is";
#   - the median ForwardDiff gradient of a scalar trajectory summary with
#     respect to every kinetic parameter (the cost of one gradient-based fit
#     iteration).
# Unsupported combinations (ArgumentError at problem construction) print n/a.
#
# Usage:  julia --project=benchmark benchmark/solver_matrix.jl [seconds]
# ============================================================================

const ROOT = dirname(@__DIR__)
const SECONDS_PER_BENCHMARK = isempty(ARGS) ? 2.0 : parse(Float64, ARGS[1])
const MATRIX_SAVE_TIMES = collect(0.0:300.0:14_400.0)

matrix_git_hash() = strip(read(`git -C $ROOT rev-parse --short HEAD`, String))

const MATRIX_SEED = LogNormalInitialCrystals(mass_concentration = 0.25, d43 = 12e-6, geometric_std = 1.25)
const MATRIX_NUCLEATION = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006)
const MATRIX_GROWTH = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0)
const MATRIX_LYSOZYME = RunConditions(temperature = 293.15, concentration = 18.0)
const MATRIX_SEEDED_LYSOZYME = RunConditions(temperature = 293.15, concentration = 18.0, crystals = MATRIX_SEED)

# Partial dissolution: C0 = 9.8 < Csat = 10 with 2 kg/m³ of seed, so 10% of
# the seed mass dissolves until C = Csat and the crystals survive.
const MATRIX_DISSOLUTION_SYSTEM = CrystallisationSystem(saturation_model = ConstantSolubility(10.0),
    crystal_density = 1400.0, volume_shape_factor = π / 6)

const MATRIX_CASES = (
    (name = "nucleation + growth",
     model = CrystallisationModel(LysozymeSystem(); nucleation = MATRIX_NUCLEATION, growth = MATRIX_GROWTH),
     conditions = MATRIX_LYSOZYME),
    (name = "seeded growth",
     model = CrystallisationModel(LysozymeSystem(); growth = MATRIX_GROWTH),
     conditions = MATRIX_SEEDED_LYSOZYME),
    (name = "seeded nucleation + growth",
     model = CrystallisationModel(LysozymeSystem(); nucleation = MATRIX_NUCLEATION, growth = MATRIX_GROWTH),
     conditions = MATRIX_SEEDED_LYSOZYME),
    (name = "seeded growth + aggregation + breakage",
     model = CrystallisationModel(LysozymeSystem(); growth = MATRIX_GROWTH,
         aggregation = ConstantAggregation(log10_aggregation_coefficient = -14.0),
         breakage = UniformBreakage(ln_breakage_coefficient = log(1e-5), breakage_size_exponent = 1.0)),
     conditions = MATRIX_SEEDED_LYSOZYME),
    (name = "seeded length-dependent growth",
     model = CrystallisationModel(LysozymeSystem();
         growth = EmpiricalLengthGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0,
             size_dependence_coefficient = 1.0, size_dependence_exponent = 0.5)),
     conditions = MATRIX_SEEDED_LYSOZYME),
    (name = "seeded partial dissolution",
     model = CrystallisationModel(MATRIX_DISSOLUTION_SYSTEM;
         growth = EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 1.0),
         dissolution = ArrheniusDissolution(dissolution_coefficient = 1e-8, activation_energy = 0.0,
             dissolution_order = 1.0)),
     conditions = RunConditions(temperature = 293.15, concentration = 9.8,
         crystals = LogNormalInitialCrystals(mass_concentration = 2.0, d43 = 12e-6, geometric_std = 1.25))),
    (name = "MSMPR nucleation + growth",
     model = CrystallisationModel(LysozymeSystem(); nucleation = MATRIX_NUCLEATION, growth = MATRIX_GROWTH),
     conditions = RunConditions(temperature = 293.15, concentration = 18.0,
         operation = MSMPROperation(volume = 1.0, inflow = 1.0 / 3600.0,
             feed = CrystallisationFeed(concentration = 18.0)))),
)

const MATRIX_SOLVERS = (
    ("MoM", MoM()),
    ("QMOM", QMOM()),
    ("DQMOM", DQMOM()),
    ("FV200", FiniteVol(meshsize = 200)),
    ("WENO200", WENO(meshsize = 200)),
)

matrix_problem(case, solver) =
    CrystallisationProblem(case.model, case.conditions; solver)

matrix_forward(parameters, problem) = last(runsimulation(parameters, problem; save_idx = MATRIX_SAVE_TIMES))

# Scalar summary that touches the solute and size outputs every solver shares.
matrix_objective(parameters, problem) = begin
    trajectory = matrix_forward(parameters, problem)
    sum(trajectory.concentration) + 1e6 * sum(trajectory.d43)
end

function matrix_row(case, solver_name, solver)
    problem = try
        matrix_problem(case, solver)
    catch caught_error
        caught_error isa ArgumentError || rethrow()
        return (solver_name, nothing, "constructor: " * first(split(caught_error.msg, '\n')))
    end
    parameters = collect(Float64, kinetic_parameter_vector(problem))
    trajectory = try
        matrix_forward(parameters, problem)
    catch caught_error
        caught_error isa Union{ArgumentError, DomainError} || rethrow()
        return (solver_name, nothing, "solve: " * first(split(sprint(showerror, caught_error), '\n')))
    end
    rhs_evaluations = trajectory.ode_stats.nf
    forward = median(@be matrix_forward($parameters, $problem) evals = 1 seconds = SECONDS_PER_BENCHMARK)
    gradient = try
        ForwardDiff.gradient(parameter_vector -> matrix_objective(parameter_vector, problem), parameters)
        median(@be ForwardDiff.gradient(parameter_vector -> matrix_objective(parameter_vector, $problem),
            $parameters) evals = 1 seconds = SECONDS_PER_BENCHMARK)
    catch caught_error
        @warn "gradient failed" case.name solver_name exception = caught_error
        nothing
    end
    return (solver_name, (; forward, rhs_evaluations, gradient, parameter_count = length(parameters)), "")
end

format_time(seconds) = seconds < 1e-3 ? @sprintf("%7.1f µs", 1e6 * seconds) : @sprintf("%7.2f ms", 1e3 * seconds)

function print_matrix(io, rows_by_case)
    for (case, rows) in rows_by_case
        println(io, "\n## ", case.name)
        @printf(io, "%-8s %11s %9s %11s %7s %10s %11s %9s\n",
            "solver", "forward", "allocs", "bytes", "nf", "per-RHS", "gradient", "grad/fwd")
        for (solver_name, result, reason) in rows
            if result === nothing
                @printf(io, "%-8s n/a (%s)\n", solver_name, reason)
                continue
            end
            forward = result.forward
            gradient_time = result.gradient === nothing ? "failed" : format_time(result.gradient.time)
            gradient_ratio = result.gradient === nothing ? "" : @sprintf("%8.1fx", result.gradient.time / forward.time)
            @printf(io, "%-8s %11s %9d %11d %7d %10s %11s %9s\n", solver_name, format_time(forward.time),
                round(Int, forward.allocs), round(Int, forward.bytes), result.rhs_evaluations,
                format_time(forward.time / result.rhs_evaluations), gradient_time, gradient_ratio)
        end
    end
end

function run_matrix()
    rows_by_case = map(MATRIX_CASES) do case
        @info "benchmarking" case.name
        case => [matrix_row(case, solver_name, solver) for (solver_name, solver) in MATRIX_SOLVERS]
    end
    header = "CriSTool solver matrix — git $(matrix_git_hash()) — Julia $(VERSION), $(Threads.nthreads()) thread(s)\n" *
             "Chairmarks @be evals=1 seconds=$(SECONDS_PER_BENCHMARK), medians; save grid 0:300:14400 s; " *
             "gradient = ForwardDiff over all kinetic parameters"
    println(header)
    print_matrix(stdout, rows_by_case)
    results_directory = joinpath(@__DIR__, "results")
    mkpath(results_directory)
    results_path = joinpath(results_directory, "solver_matrix_$(matrix_git_hash()).txt")
    open(results_path, "w") do io
        println(io, header)
        print_matrix(io, rows_by_case)
    end
    println("\nwritten to ", results_path)
end

(@isdefined(MATRIX_SKIP_RUN) && MATRIX_SKIP_RUN) || run_matrix()
