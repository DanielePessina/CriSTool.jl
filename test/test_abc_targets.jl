# Independent Gaussian likelihood-ratio and observation-count fixtures.
struct TestABCDiscrepancy <: CriSTool.AbstractPELossFunction end
CriSTool.loss(::TestABCDiscrepancy, prepared_setup::LossSetup, kinetic_parameters) = 0.125

struct TestABCTargetSampler <: CriSTool.AbstractABCSampler
    received_target::Base.RefValue{Float64}
end
CriSTool._routine_label(::TestABCTargetSampler) = "ABC target test"
CriSTool._panel_suffix(::TestABCTargetSampler) = ""
CriSTool._save_suffix(::TestABCTargetSampler) = ""
CriSTool._sampler_metadata(::TestABCTargetSampler) = NamedTuple()
function CriSTool._runsampler(sampler::TestABCTargetSampler, prior, lossfn,
                              target, reference_loss; kwargs...)
    sampler.received_target[] = target
    return (P = zeros(0, 0),), true
end

@testset "ABC discrepancy targets" begin
    reference_nll = -10.593986931765556
    @test CriSTool._checked_abc_target(reference_nll, reference_nll + 0.5) == reference_nll + 0.5
    @test_throws ArgumentError CriSTool._checked_abc_target(reference_nll, nothing)
    @test_throws ArgumentError CriSTool._checked_abc_target(Inf, 0.0)
    @test_throws ArgumentError CriSTool._checked_abc_target(reference_nll, reference_nll - 1)
    @test_throws ArgumentError CriSTool._checked_abc_target(reference_nll, NaN)

    configured_system = CrystallisationProblem(
        kinetics_nucleationfunction = nucl_empirical(),
        parameterset_nucleation = [0.0, 1.0],
        kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]),
        parameterset_growth = Float64[],
        initial_concentration = 10.0, saturation_model = ConstantSolubility(10.0),
        initial_solvent_state = (concentration = 10.0, pH = 7.0),
        solvent_dynamics = (system, ode_state, time, growth) ->
            (concentration = 0.0, pH = 0.0), solver = MoM())
    target_experiment = CrystallisationExperiment(
        observables = (concentration = Observable(time = [0.0, 1.0, 2.0],
                          mean = [10.0, 10.0, 10.0], variance = 1e-12),
                       pH = Observable(time = [0.0, 2.0], mean = [7.0, 7.0],
                                       variance = 1.0)), exp_id = 1, conditions = RunConditions(temperature = 293.15, crystals = LogNormalInitialCrystals(
            mass_concentration = 0.5, d43 = 10e-6, geometric_std = 1.2)))
    target_setup = prepare_loss(configured_system, [target_experiment])

    _, unreachable_target_reached = CriSTool._runsampler(ABCDESampler(),
        product_distribution([Uniform(0.0, 1.0), Uniform(0.5, 1.5)]),
        candidate_parameters -> 100.0, 0.0, 100.0;
        nparticles = 8, generations = 1, HPC = true, earlystop = false)
    @test !unreachable_target_reached

    received_target = Ref(NaN)
    target_sampler = TestABCTargetSampler(received_target)
    target_prior = product_distribution([Uniform(0.0, 1.0), Uniform(0.5, 1.5)])
    # Exact Gaussian fit: two variance=1e-12 C points and two variance=1 pH points.
    analytic_nll = log(2π * 1e-12) + log(2π)
    _, target_metadata = run_abc(logMLE(relative_variance_floor = 1e-15), target_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, target = analytic_nll + 1.5, verbosity = 0, saveplot = false)
    @test target_metadata["optmle"] ≈ analytic_nll
    @test received_target[] == analytic_nll + 1.5
    @test_throws ArgumentError run_abc(logMLE(relative_variance_floor = 1e-15), target_setup, [0.0, 1.0],
        target_prior; sampler = target_sampler, target = analytic_nll - 1.0, verbosity = 0, saveplot = false)
    _, explicit_metadata = run_abc(mae(), target_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, target = 0.2, verbosity = 0, saveplot = false)
    @test received_target[] == 0.2
    @test explicit_metadata["target"] == 0.2
    _, custom_metadata = run_abc(TestABCDiscrepancy(), target_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, target = 0.2, verbosity = 0, saveplot = false)
    @test custom_metadata["optmle"] == 0.125
    @test_throws UndefKeywordError run_abc(TestABCDiscrepancy(), target_setup, [0.0, 1.0],
        target_prior; sampler = target_sampler, verbosity = 0, saveplot = false)
end
