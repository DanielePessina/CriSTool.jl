# Opt-in static and allocation checks for the hot ODE right-hand sides.
# Run with `test_args = ["static"]` (also part of "all"). Each RHS must be
# allocation-free per call for Float64 and for ForwardDiff.Dual state and
# parameters, and JET must find no runtime-dispatch or error reports in the
# CriSTool frames for Float64 inputs.
using JET: JET
using ComponentArrays: ComponentArray, getaxes
using SciMLBase: isinplace, ODEProblem

const STATIC_SEED = LogNormalInitialCrystals(mass_concentration = 0.25, d43 = 12e-6, geometric_std = 1.25)
const STATIC_NUCLEATION = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006)
const STATIC_GROWTH = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0)
const STATIC_NUCLEATING_MODEL = CrystallisationModel(LysozymeSystem();
    nucleation = STATIC_NUCLEATION, growth = STATIC_GROWTH)
const STATIC_SEEDED_MODEL = CrystallisationModel(LysozymeSystem(); growth = STATIC_GROWTH)
const STATIC_AGGREGATING_MODEL = CrystallisationModel(LysozymeSystem(); growth = STATIC_GROWTH,
    aggregation = ConstantAggregation(log10_aggregation_coefficient = -14.0),
    breakage = UniformBreakage(ln_breakage_coefficient = log(1e-5), breakage_size_exponent = 1.0))
const STATIC_PLAIN_CONDITIONS = RunConditions(temperature = 293.15, concentration = 18.0)
const STATIC_SEEDED_CONDITIONS = RunConditions(temperature = 293.15, concentration = 18.0,
    crystals = STATIC_SEED)

# (label, model, conditions, solver, jet_expected_clean)
const STATIC_CASES = (
    ("MoM nucleation+growth", STATIC_NUCLEATING_MODEL, STATIC_PLAIN_CONDITIONS, MoM()),
    ("QMOM nucleation+growth", STATIC_NUCLEATING_MODEL, STATIC_PLAIN_CONDITIONS, QMOM()),
    ("DQMOM seeded growth", STATIC_SEEDED_MODEL, STATIC_SEEDED_CONDITIONS, DQMOM()),
    ("FiniteVol nucleation+growth", STATIC_NUCLEATING_MODEL, STATIC_PLAIN_CONDITIONS, FiniteVol(meshsize = 100)),
    ("WENO nucleation+growth", STATIC_NUCLEATING_MODEL, STATIC_PLAIN_CONDITIONS, WENO(meshsize = 100)),
    ("QMOM aggregation+breakage", STATIC_AGGREGATING_MODEL, STATIC_SEEDED_CONDITIONS, QMOM()),
    ("DQMOM aggregation+breakage", STATIC_AGGREGATING_MODEL, STATIC_SEEDED_CONDITIONS, DQMOM()),
    ("FiniteVol aggregation+breakage", STATIC_AGGREGATING_MODEL, STATIC_SEEDED_CONDITIONS, FiniteVol(meshsize = 100)),
)

# Same parameter container with every value promoted to a one-partial Dual.
static_dual_parameters(parameters) =
    ComponentArray(map(value -> ForwardDiff.Dual{Nothing}(value, 1.0), collect(parameters)),
                   getaxes(parameters))

# Closure-free RHS evaluators: the type of the ODE function is the dispatch
# barrier, so the measurement covers only the RHS body.
static_rhs_call(::Val{true}, rhs, derivative, state, parameters, time) =
    rhs(derivative, state, parameters, time)
static_rhs_call(::Val{false}, rhs, derivative, state, parameters, time) =
    rhs(state, parameters, time)

function static_rhs_allocated(odeproblem)
    inplace = Val(isinplace(odeproblem))
    rhs, state, parameters = odeproblem.f.f, odeproblem.u0, odeproblem.p
    derivative = similar(state)
    time = first(odeproblem.tspan)
    static_rhs_call(inplace, rhs, derivative, state, parameters, time)  # warm-up
    return static_rhs_measure(inplace, rhs, derivative, state, parameters, time)
end
static_rhs_measure(inplace, rhs, derivative, state, parameters, time) =
    @allocated static_rhs_call(inplace, rhs, derivative, state, parameters, time)

static_problem(model, conditions) = CrystallisationProblem(model, conditions, (0.0, 3600.0))
static_odeproblem(problem, solver, kinetic_values) =
    first(crystallisation_odeproblem(remake(problem; p = kinetic_values), solver))

# Paths that genuinely allocate per call; kept visible as broken rather than hidden.
const STATIC_KNOWN_ALLOCATING = Set{Tuple{String, String}}()

@testset "RHS allocation: zero per call" begin
    for (label, model, conditions, solver) in STATIC_CASES
        problem = static_problem(model, conditions)
        kinetic_values = kinetic_parameter_vector(problem)
        float_problem = static_odeproblem(problem, solver, kinetic_values)
        dual_problem = static_odeproblem(problem, solver, static_dual_parameters(kinetic_values))
        @testset "$label Float64" begin
            if (label, "Float64") in STATIC_KNOWN_ALLOCATING
                @test_broken static_rhs_allocated(float_problem) == 0
            else
                @test static_rhs_allocated(float_problem) == 0
            end
        end
        @testset "$label Dual" begin
            if (label, "Dual") in STATIC_KNOWN_ALLOCATING
                @test_broken static_rhs_allocated(dual_problem) == 0
            else
                @test static_rhs_allocated(dual_problem) == 0
            end
        end
    end
end

@testset "RHS JET: no optimisation or error reports" begin
    for (label, model, conditions, solver) in STATIC_CASES
        problem = static_problem(model, conditions)
        odeproblem = static_odeproblem(problem, solver, kinetic_parameter_vector(problem))
        inplace = Val(isinplace(odeproblem))
        derivative = similar(odeproblem.u0)
        @testset "$label" begin
            JET.@test_opt target_modules = (CriSTool,) static_rhs_call(inplace, odeproblem.f.f,
                derivative, odeproblem.u0, odeproblem.p, first(odeproblem.tspan))
        end
    end
end

# Negative controls: the measurement and the JET target filter must be able to fail.
@testset "static checks detect known regressions" begin
    allocating_rhs = (state, parameters, time) -> [state[1] * parameters[1]]
    controlled = ODEProblem{false}(allocating_rhs, [1.0], (0.0, 1.0), [2.0])
    @test static_rhs_allocated(controlled) > 0
    unstable_rhs(state, parameters, time) = parameters.ref[] * state  # abstract Ref field: dispatch on read
    unstable_report = JET.report_opt(unstable_rhs, Tuple{Vector{Float64}, NamedTuple{(:ref,), Tuple{Base.RefValue{Any}}}, Float64};
        target_modules = (Main,))
    @test !isempty(JET.get_reports(unstable_report))
end
