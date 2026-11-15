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
    # For two parameters the 95% chi-square quantile is -2log(0.05).
    reference_nll = -10.593986931765556
    likelihood_increment = -log(0.05)
    derived_target = CriSTool._abcde_target(reference_nll, 10, 2, 0.95)
    @test derived_target ≈ reference_nll + likelihood_increment
    @test derived_target > reference_nll
    @test CriSTool._abcde_target(reference_nll + 123.0, 10, 2, 0.95) ≈
          derived_target + 123.0
    # Candidate acceptance is invariant to Gaussian normalization constants.
    for discrepancy_increment in (1.0, 4.0)
        @test (reference_nll + discrepancy_increment <= derived_target) ==
              (reference_nll + 123.0 + discrepancy_increment <= derived_target + 123.0)
    end
    for policy_alias in (:wilks, :chisq, :chisqtest)
        @test CriSTool._abcde_target(reference_nll, 10, 2, 0.95;
                                    test = policy_alias) ≈ derived_target
    end
    @test CriSTool._abcde_target(reference_nll, 0, 0, 0.95) == reference_nll
    @test CriSTool._abcde_target(reference_nll, 0, 2, 0.95;
                                target = reference_nll + 0.5) == reference_nll + 0.5
    @test_throws ArgumentError CriSTool._abcde_target(reference_nll, 10, 2, 1.0)
    @test_throws ArgumentError CriSTool._abcde_target(Inf, 10, 2, 0.95)
    @test_throws ArgumentError CriSTool._abcde_target(reference_nll, 10, 2, 0.95;
                                                    target = reference_nll - 1)
    @test_throws ArgumentError CriSTool._abcde_target(reference_nll, 10, 2, 0.95;
                                                    target = NaN)
    @test_throws ArgumentError CriSTool._abcde_target(reference_nll, 10, 2, 0.95;
                                                    test = :f)

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
                                       variance = 1.0)),
        temperature = 293.15, exp_id = 1,
        initial_crystals = LogNormalInitialCrystals(
            mass_concentration = 0.5, d43 = 10e-6, geometric_std = 1.2))
    target_setup = prepare_loss(configured_system, [target_experiment])
    @test CriSTool._dofcalculator(logMLE(), target_setup) == 4 # two C and two pH
    include_initial_setup = prepare_loss(configured_system, [target_experiment];
                                         exclude_initial_concentration = false)
    @test CriSTool._dofcalculator(logMLE(), include_initial_setup) == 5
    @test CriSTool._dofcalculator(logMLE(weighting = (concentration = 0.0, pH = 1.0)),
                                  target_setup) == 2
    @test CriSTool._dofcalculator(logMLE(weighting = [1.0, 0.0]), target_setup) == 2
    @test CriSTool._abc_target_policy(logMLE(), target_setup, :auto, nothing) == :wilks
    @test_throws ArgumentError CriSTool._abc_target_policy(mae(), target_setup, :auto, nothing)
    @test_throws ArgumentError CriSTool._abc_target_policy(
        logMLE(weighting = [1.0, 2.0]), target_setup, :wilks, nothing)
    for f_alias in (:f, :fstat, :ftest)
        @test_throws ArgumentError CriSTool._abc_target_policy(logMLE(), target_setup, f_alias, nothing)
        @test_throws ArgumentError CriSTool._abc_target_policy(mae(), target_setup, f_alias, nothing)
    end
    @test CriSTool._abc_target_policy(mae(), target_setup, :auto, 0.2) == :explicit

    _, unreachable_target_reached = CriSTool._runsampler(ABCDESampler(),
        product_distribution([Uniform(0.0, 1.0), Uniform(0.5, 1.5)]),
        candidate_parameters -> 100.0, 0.0, 100.0;
        nparticles = 8, generations = 1, HPC = true, earlystop = false)
    @test !unreachable_target_reached

    received_target = Ref(NaN)
    target_sampler = TestABCTargetSampler(received_target)
    target_prior = product_distribution([Uniform(0.0, 1.0), Uniform(0.5, 1.5)])
    _, target_metadata = run_abc(logMLE(relative_variance_floor = 1e-15), target_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, verbosity = 0, saveplot = false)
    # Exact Gaussian fit: two variance=1e-12 C points and two variance=1 pH points.
    analytic_nll = log(2π * 1e-12) + log(2π)
    @test target_metadata["optmle"] ≈ analytic_nll
    @test received_target[] ≈ analytic_nll + likelihood_increment
    @test target_metadata["target_policy"] == :wilks
    @test target_metadata["included_observations"] == 4
    scaled_experiment = CrystallisationExperiment(
        observables = merge(target_experiment.observables,
            (concentration = Observable(time = [0.0, 1.0, 2.0],
                mean = [1000.0, 1000.0, 1000.0], variance = 1e-8),)),
        temperature = 293.15, exp_id = 2,
        initial_crystals = target_experiment.initial_crystals)
    scaled_system = CrystallisationProblem(
        kinetics_nucleationfunction = configured_system.kinetics_nucleationfunction,
        parameterset_nucleation = configured_system.parameterset_nucleation,
        kinetics_growthfunction = configured_system.kinetics_growthfunction,
        parameterset_growth = configured_system.parameterset_growth,
        initial_concentration = 1000.0, saturation_model = ConstantSolubility(1000.0),
        initial_solvent_state = (concentration = 1000.0, pH = 7.0),
        solvent_dynamics = configured_system.solvent_dynamics, solver = MoM())
    scaled_setup = prepare_loss(scaled_system, [scaled_experiment])
    _, scaled_metadata = run_abc(logMLE(relative_variance_floor = 1e-15),
        scaled_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, verbosity = 0, saveplot = false)
    # Two included concentration observations acquire log(100) each under
    # a factor-100 unit conversion; their likelihood-ratio cutoff is unchanged.
    @test scaled_metadata["optmle"] ≈ analytic_nll + 2log(100.0)
    @test scaled_metadata["target"] - scaled_metadata["optmle"] ≈ likelihood_increment
    _, explicit_metadata = run_abc(mae(), target_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, target = 0.2, verbosity = 0, saveplot = false)
    @test received_target[] == 0.2
    @test explicit_metadata["target_policy"] == :explicit
    _, custom_metadata = run_abc(TestABCDiscrepancy(), target_setup, [0.0, 1.0], target_prior;
        sampler = target_sampler, target = 0.2, verbosity = 0, saveplot = false)
    @test custom_metadata["optmle"] == 0.125
    @test custom_metadata["target_policy"] == :explicit
    @test_throws ArgumentError run_abc(TestABCDiscrepancy(), target_setup, [0.0, 1.0],
        target_prior; sampler = target_sampler, verbosity = 0, saveplot = false)
end
