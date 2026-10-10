# Run: julia --project=benchmark benchmark/model_interface.jl
using CriSTool, BenchmarkTools, ComponentArrays, Test

function benchmark_model_interface()
    benchmark_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38., surface_energy = .0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.))
    benchmark_run = CrystallisationProblem(benchmark_model,
        RunConditions(temperature = 293.15, concentration = 18.); solver = MoM())
    benchmark_times = collect(0.:3600.:14400.)
    native_parameters = [38., .0006, 1e-9 / 60, 3.]
    _, native_trajectory = runsimulation(native_parameters, benchmark_run; save_idx = benchmark_times)
    named_trajectory = simulate(benchmark_run; saveat = benchmark_times)
    @test named_trajectory.concentration ≈ native_trajectory.concentration rtol = 1e-12
    benchmark_experiment = CrystallisationExperiment(observables = (
        concentration = Observable(time = benchmark_times, mean = named_trajectory.concentration,
            variance = fill(.01, length(benchmark_times))),), exp_id = 1, conditions = RunConditions(temperature = 293.15))
    selected_preparation = prepare_fit(benchmark_model, [benchmark_experiment],
        OptimisationSpec(bounds = (growth = (growth_order = (1., 4.),),)))
    full_preparation = selected_preparation.setup
    @test loss(selected_preparation, [3.]) ≈ loss(GaussianNLL(), full_preparation, native_parameters) atol = 1e-10

    named_growth_values = benchmark_run.parameterset_growth
    rate_state = CriSTool._get_initial_state(benchmark_run)
    builtin_growth_law = benchmark_model.growth.law
    callable_growth_law = CallableGrowth((rate_values, rate_context) -> begin
        driving_force = supersaturation(rate_context)
        driving_force > 1.001 ? rate_values.growth_coefficient * (driving_force - 1)^rate_values.growth_order : 0.
    end; parameters = benchmark_model.growth.parameters)
    @test @inferred(growthrate(callable_growth_law, named_growth_values, benchmark_run, rate_state, 0.)) ==
        growthrate(builtin_growth_law, named_growth_values, benchmark_run, rate_state, 0.)

    measurements = [
        "builtin rate" => @benchmark(growthrate($builtin_growth_law, $named_growth_values, $benchmark_run, $rate_state, 0.); seconds = 1),
        "callable rate" => @benchmark(growthrate($callable_growth_law, $named_growth_values, $benchmark_run, $rate_state, 0.); seconds = 1),
        "configured solve" => @benchmark(last(runsimulation($benchmark_run; save_idx = $benchmark_times)); seconds = 1),
        "model simulate" => @benchmark(simulate($benchmark_run; saveat = $benchmark_times); seconds = 1),
        "full loss" => @benchmark(loss(GaussianNLL(), $full_preparation, $native_parameters); seconds = 1),
        "selected loss" => @benchmark(loss($selected_preparation, [3.]); seconds = 1)]
    println("Julia ", VERSION, "; threads=", Threads.nthreads(), "; CPU=", Sys.CPU_NAME)
    for (measurement_name, trial_result) in measurements
        trial_median = BenchmarkTools.median(trial_result)
        println(measurement_name, ": median_ns=", trial_median.time,
            ", bytes=", trial_median.memory, ", allocations=", trial_median.allocs)
    end
end
benchmark_model_interface()
