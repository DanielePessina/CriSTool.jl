using Metaheuristics
using Random

@testset "Metaheuristics configuration and ownership" begin
    supplied_de = Metaheuristics.DE(; N = 8, strategy = :rand2,
        F = 0.43, CR = 0.81,
        options = Metaheuristics.Options(; iterations = 3, seed = 921,
            rng = MersenneTwister(921), parallel_evaluation = false,
            f_calls_limit = 500, store_convergence = true, time_limit = 60.0))
    supplied_de.status.f_calls = 123
    supplied_de.status.stop = true
    supplied_snapshot = deepcopy(supplied_de)

    private_de = CriSTool._configured_mh_algorithm(supplied_de)
    @test (private_de.parameters.N, private_de.parameters.strategy,
           private_de.parameters.F, private_de.parameters.CR) == (8, :rand2, 0.43, 0.81)
    @test (private_de.options.iterations, private_de.options.parallel_evaluation,
           private_de.options.f_calls_limit, private_de.options.store_convergence,
           private_de.options.time_limit) == (3, false, 500.0, true, 60.0)
    @test private_de.status.f_calls == 0
    @test !private_de.status.stop
    @test private_de.parameters !== supplied_de.parameters
    @test private_de.options !== supplied_de.options
    @test private_de.options.rng !== supplied_de.options.rng

    explicit_de = CriSTool._configured_mh_algorithm(supplied_de;
        nparticles = 128, generations = 128, parallel_evaluation = true, verbosity = 2)
    @test (explicit_de.parameters.N, explicit_de.options.iterations,
           explicit_de.options.parallel_evaluation, explicit_de.options.verbose) ==
          (128, 128, true, true)
    @test (explicit_de.parameters.strategy, explicit_de.parameters.F,
           explicit_de.parameters.CR) == (:rand2, 0.43, 0.81)
    @test !CriSTool._configured_mh_algorithm(supplied_de; verbosity = 2, HPC = true).options.verbose
    @test_throws ArgumentError CriSTool._configured_mh_algorithm(supplied_de; nparticles = 0)
    @test_throws ArgumentError CriSTool._configured_mh_algorithm(supplied_de; generations = -1)

    for configured_algorithm in (Metaheuristics.NSGA2(; N = 16),
                                  Metaheuristics.SA(; N = 12, x_initial = [0.4, 0.6]),
                                  Metaheuristics.PSO(; N = 14, C1 = 1.3, C2 = 1.8))
        private_algorithm = CriSTool._configured_mh_algorithm(configured_algorithm)
        @test all(getfield(private_algorithm.parameters, field_name) ==
                  getfield(configured_algorithm.parameters, field_name)
                  for field_name in fieldnames(typeof(configured_algorithm.parameters)))
    end

    # Compare `fit` against Metaheuristics directly, with the same objective,
    # configuration and seed. A forced DE strategy or replaced F/CR diverges.
    optimizer_model = CrystallisationModel(LysozymeSystem();
        nucleation = nucl_empirical(log10_nucleation_prefactor = 12.5, nucleation_order = 1.5),
        growth = growth_empirical(growth_coefficient = 3e-8, growth_order = 1.5))
    optimizer_measurements = [CrystallisationExperiment(;
        observables = (; concentration = Observable(;
            time = [0.0, 30.0, 60.0], mean = [18.0, 17.8, 17.5],
            variance = [0.01, 0.01, 0.01])), exp_id = 921, conditions = RunConditions(temperature = 295.0))]
    optimizer_bounds = (nucleation = (log10_nucleation_prefactor = (12.0, 13.0), nucleation_order = (1.0, 2.0)),
        growth = (growth_coefficient = (1e-8, 5e-8), growth_order = (1.0, 2.0)))
    optimizer_prepared = prepare_fit(optimizer_model, optimizer_measurements,
        OptimisationSpec(bounds = optimizer_bounds, algorithm = supplied_de, loss = logMLE(; weighting = [1.0])))
    direct_de = deepcopy(supplied_de)
    direct_de.status = Metaheuristics.State(nothing, [])
    direct_result = Metaheuristics.optimize(candidate -> loss(optimizer_prepared, candidate),
        Metaheuristics.boxconstraints([12.0, 1.0, 1e-8, 1.0], [13.0, 2.0, 5e-8, 2.0]), direct_de)
    public_fit = fit(optimizer_prepared)
    public_result = public_fit.backend_result
    @test Metaheuristics.minimizer(public_result) == Metaheuristics.minimizer(direct_result)
    @test Metaheuristics.minimum(public_result) == Metaheuristics.minimum(direct_result)
    @test public_result.f_calls == direct_result.f_calls
    @test [public_fit.parameters.nucleation.log10_nucleation_prefactor, public_fit.parameters.nucleation.nucleation_order,
           public_fit.parameters.growth.growth_coefficient, public_fit.parameters.growth.growth_order] ==
          Metaheuristics.minimizer(direct_result)
    @test all(getfield(supplied_de.parameters, field_name) ==
              getfield(supplied_snapshot.parameters, field_name)
              for field_name in fieldnames(typeof(supplied_de.parameters)))
    @test all(getfield(supplied_de.options, field_name) ==
              getfield(supplied_snapshot.options, field_name)
              for field_name in fieldnames(typeof(supplied_de.options)) if field_name != :rng)
    @test supplied_de.status.f_calls == 123
    @test supplied_de.status.stop
    @test rand(deepcopy(supplied_de.options.rng), 4) == rand(supplied_snapshot.options.rng, 4)
end
