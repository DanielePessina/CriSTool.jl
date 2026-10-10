# Independent Gaussian likelihood and observation-count fixtures for ABC fits.
struct TestABCDiscrepancy <: CriSTool.AbstractPELossFunction end
CriSTool.loss(::TestABCDiscrepancy, prepared_setup::LossSetup, kinetic_parameters) = 0.125

struct TestABCTargetSampler <: CriSTool.AbstractABCSampler
    received_target::Base.RefValue{Float64}
    received_reference_loss::Base.RefValue{Float64}
end
function CriSTool._runsampler(sampler::TestABCTargetSampler, prior, lossfn,
                              target, reference_loss; kwargs...)
    sampler.received_target[] = target
    sampler.received_reference_loss[] = reference_loss
    return (P = [0.4 0.6],), true
end

@testset "ABC discrepancy targets" begin
    _, unreachable_target_reached = CriSTool._runsampler(ABCDESampler(),
        product_distribution([Uniform(0.0, 1.0), Uniform(0.5, 1.5)]),
        candidate_parameters -> 100.0, 0.0, 100.0;
        nparticles = 8, generations = 1, HPC = true, earlystop = false)
    @test !unreachable_target_reached

    # Zero nucleation and zero growth keep the concentration at its initial value.
    target_model = CrystallisationModel(CrystallisationSystem(saturation_model = ConstantSolubility(10.0),
            crystal_density = 1000.0, volume_shape_factor = 1.0);
        nucleation = CallableNucleation((rate_values, rate_context) -> 0.0; parameters = (;)),
        growth = CallableGrowth((rate_values, rate_context) -> 0.0 * rate_values.speed; parameters = (speed = 0.5,)))
    target_experiment = CrystallisationExperiment(
        observables = (concentration = Observable(time = [0.0, 1.0, 2.0],
            mean = [10.0, 10.0, 10.0], variance = 1e-12),),
        exp_id = 1, conditions = RunConditions(temperature = 293.15))
    target_priors = (growth = (speed = Uniform(0.0, 1.0),),)
    received_target, received_reference_loss = Ref(NaN), Ref(NaN)
    target_sampler = TestABCTargetSampler(received_target, received_reference_loss)
    abc_fit(abc_loss, abc_target) = fit(prepare_fit(target_model, [target_experiment],
        ABCSpec(priors = target_priors, target = abc_target, sampler = target_sampler, loss = abc_loss)))

    # The time-zero concentration sets the initial condition and is not scored:
    # two exact Gaussian points with variance 1e-12 remain.
    analytic_nll = log(2π * 1e-12)
    likelihood_result = abc_fit(logMLE(relative_variance_floor = 1e-15), analytic_nll + 1.5)
    @test received_reference_loss[] ≈ analytic_nll
    @test received_target[] == analytic_nll + 1.5
    @test likelihood_result.samples.values == [0.4 0.6]
    @test likelihood_result.samples.names == ["growth.speed"]
    @test likelihood_result.samples.source == :abc

    abc_fit(TestABCDiscrepancy(), 0.2)
    @test received_reference_loss[] == 0.125
    @test received_target[] == 0.2
    @test_throws ArgumentError abc_fit(TestABCDiscrepancy(), Inf)
end
