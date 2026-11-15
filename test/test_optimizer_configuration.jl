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

    # Compare the public routine against Metaheuristics directly, with the same
    # objective/configuration/seed. A forced DE strategy or replaced F/CR diverges.
    optimizer_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical(),
        kinetics_growthfunction = growth_empirical(),
        kinetics_aggregationfunction = noaggregation(),
        kinetics_breakagefunction = nobreakage(), solver = MoM())
    optimizer_measurements = [CrystallisationExperiment(;
        observables = (; concentration = Observable(;
            time = [0.0, 30.0, 60.0], mean = [18.0, 17.8, 17.5],
            variance = [0.01, 0.01, 0.01])), temperature = 295.0, exp_id = 921)]
    optimizer_lower = [12.0, 1.0, 1e-8, 1.0]
    optimizer_upper = [13.0, 2.0, 5e-8, 2.0]
    optimizer_loss = logMLE(; weighting = [1.0])
    optimizer_setup = prepare_loss(optimizer_problem, optimizer_measurements)
    direct_de = deepcopy(supplied_de)
    direct_de.status = Metaheuristics.State(nothing, [])
    direct_result = Metaheuristics.optimize(
        candidate -> CriSTool.batchLF_procSO(optimizer_loss, optimizer_setup, candidate),
        Metaheuristics.boxconstraints(optimizer_lower, optimizer_upper), direct_de)
    public_result = PE_Routine(optimizer_loss, optimizer_measurements,
        optimizer_lower, optimizer_upper, nucl_empirical(), growth_empirical(),
        noaggregation(), nobreakage(); solver = MoM(), MHAlgorithm = supplied_de,
        savetxt = false)
    @test Metaheuristics.minimizer(public_result) == Metaheuristics.minimizer(direct_result)
    @test Metaheuristics.minimum(public_result) == Metaheuristics.minimum(direct_result)
    @test public_result.f_calls == direct_result.f_calls
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
