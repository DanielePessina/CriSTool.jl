struct AccountingBrokenGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::AccountingBrokenGrowth) = CriSTool.ComponentArrays.Axis()
CriSTool.growthrate(::AccountingBrokenGrowth, parameters, configured, state, time) =
    throw(DomainError(time, "deliberate user kinetic error"))

@testset "Named pooled observable objectives" begin
    constant_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -300.0, nucleation_order = 1.0),
        kinetics_growthfunction = growth_empirical_fixed(growth_coefficient = 0.0, growth_order = 1.0),
        parameterset_nucleation = Float64[], parameterset_growth = Float64[],
        initial_solvent_state = (; concentration = 10.0, pH = 7.0),
        initial_concentration = 10.0, solver = MoM())
    concentration_data = Observable(; time = [0.0, 1.0], mean = [10.0, 11.0])
    ph_data = Observable(; time = [0.0, 1.0], mean = [8.0, 8.0])
    conc_only = CrystallisationExperiment(; temperature = 293.15, exp_id = 1, observables = (; concentration = concentration_data))
    both_fields = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; concentration = concentration_data, pH = ph_data))
    reordered_fields = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; pH = ph_data, concentration = concentration_data))
    for experiments in ([conc_only, both_fields], [both_fields, conc_only])
        prepared = prepare_loss(constant_problem, experiments)
        @test loss(mae(), prepared, Float64[]) ≈ 2.0
        @test sum(CriSTool.batchLF_procMO(mae(), prepared, Float64[])) ≈ 2.0
        @test_throws ArgumentError loss(mae(weighting = [1.0, 2.0]), prepared, Float64[])
        @test loss(mae(weighting = (; concentration = 2.0, pH = 3.0)), prepared, Float64[]) ≈ 5.0
    end
    reordered_setup = prepare_loss(constant_problem, [both_fields, reordered_fields])
    @test loss(mae(weighting = (; concentration = 2.0, pH = 3.0)), reordered_setup, Float64[]) ≈ 5.0
    explicit_order = prepare_loss(constant_problem, [conc_only, both_fields];
                                  observable_order = [:pH, :concentration])
    @test CriSTool.batchLF_procMO(mae(weighting = [3.0, 2.0]), explicit_order, Float64[]) ≈ [3.0, 2.0]

    # Two pH points with error one and four with error four: pooled MAE is 3.
    ph_four = Observable(; time = [0.0, 0.25, 0.75, 1.0], mean = fill(11.0, 4))
    uneven_experiment = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; concentration = concentration_data, pH = ph_four))
    uneven_setup = prepare_loss(constant_problem, [both_fields, uneven_experiment])
    @test CriSTool.batchLF_procMO(mae(), uneven_setup, Float64[]) ≈ [1.0, 3.0]
    @test loss(mae(), uneven_setup, Float64[]) ≈ 4.0

    likelihood_concentration = Observable(; time = [0.0, 1.0], mean = [10.0, 11.0], variance = [1.0, 1.0])
    likelihood_ph = Observable(; time = [0.0, 1.0], mean = [8.0, 8.0], variance = [4.0, 4.0])
    likelihood_both = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; concentration = likelihood_concentration, pH = likelihood_ph))
    likelihood_single = CrystallisationExperiment(; temperature = 293.15, exp_id = 2,
        observables = (; concentration = likelihood_concentration))
    likelihood_setup = prepare_loss(constant_problem, [likelihood_both, likelihood_single])
    # Frozen Normal(0,1) NLL at1 twice, Normal(0,2) NLL at1 twice.
    @test loss(logMLE(), likelihood_setup, Float64[]) ≈ 6.312048493938581
    @test CriSTool.batchLF_procMO(logMLE(), likelihood_setup, Float64[]) ≈ [2.8378770664093453, 3.474171427529236]

    explicit_setup = prepare_loss([constant_problem], [both_fields])
    @test loss(mae(), explicit_setup, Float64[]) ≈ 1.5
    @test CriSTool._included_observation_indices(explicit_setup, 1, :concentration) == [1, 2]
    different_conditions = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -300.0, nucleation_order = 1.0),
        kinetics_growthfunction = growth_empirical_fixed(growth_coefficient = 0.0, growth_order = 1.0),
        parameterset_nucleation = Float64[], parameterset_growth = Float64[],
        initial_solvent_state = (; concentration = 20.0, pH = 5.0),
        initial_concentration = 20.0, temp_profile = ConstantTemperature(310.0), solver = MoM())
    authority_setup = prepare_loss([different_conditions], [both_fields])
    @test loss(mae(), authority_setup, Float64[]) ≈ 12.5
    @test temperature(authority_setup.prepared[1].problem.temp_profile, 0.0) == 310.0

    @test_throws DimensionMismatch loss(mae(), explicit_setup, [1.0])
    @test_throws DimensionMismatch CriSTool.batchLF_procSO(mae(), explicit_setup, [1.0])
    @test_throws DimensionMismatch CriSTool.batchLF_procMO(mae(), explicit_setup, [1.0])
    unknown_experiment = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; concentration = concentration_data,
            typo = Observable(; time = [0.0, 1.0], mean = [0.0, 0.0])))
    unknown_setup = prepare_loss(constant_problem, [unknown_experiment])
    @test_throws ArgumentError loss(mae(), unknown_setup, Float64[])
    @test_throws Exception CriSTool.batchLF_procSO(mae(), unknown_setup, zeros(2, 0))
    @test_throws Exception CriSTool.batchLF_procMO(mae(), unknown_setup, zeros(2, 0))
    late_concentration = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; concentration = Observable(; time = [1.0, 2.0], mean = [10.0, 11.0]),
                         pH = ph_data))
    @test_throws ArgumentError prepare_loss(constant_problem, [late_concentration])
    @test_throws ArgumentError prepare_loss(constant_problem, [both_fields]; observable_order = [:pH])
end

@testset "User kinetic exceptions propagate" begin
    broken_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -300.0, nucleation_order = 1.0),
        kinetics_growthfunction = AccountingBrokenGrowth(0),
        parameterset_nucleation = Float64[], parameterset_growth = Float64[],
        initial_concentration = 10.0, solver = MoM())
    broken_experiment = CrystallisationExperiment(; temperature = 293.15, exp_id = 1,
        observables = (; concentration = Observable(; time = [0.0, 1.0], mean = [10.0, 11.0])))
    broken_setup = prepare_loss(broken_problem, [broken_experiment])
    @test_throws DomainError loss(mae(), broken_setup, Float64[])
    @test_throws DomainError loss(logMLE(), broken_setup, Float64[])
    @test_throws DomainError CriSTool.batchLF_procSO(mae(), broken_setup, Float64[])
    @test_throws DomainError CriSTool.batchLF_procMO(mae(), broken_setup, Float64[])
end
