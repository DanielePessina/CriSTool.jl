import OptimizationOptimJL

struct WorkflowConstantGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
    string::String
    symbols::Vector{Symbol}
end
WorkflowConstantGrowth() = WorkflowConstantGrowth(1, "Workflow constant growth", [:speed_scale])
CriSTool.paramaxis(::WorkflowConstantGrowth) = CriSTool.ComponentArrays.Axis(speed_scale = 1)
CriSTool.growthrate(::WorkflowConstantGrowth, kinetic_parameters, configured_problem,
                    numerical_state, simulation_time) = 1e-7 * kinetic_parameters[1]

@testset "Configured forward and prediction interfaces" begin
    workflow_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed([0.0, 1.0]),
        kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]),
        parameterset_nucleation = Float64[], parameterset_growth = Float64[],
        initial_concentration = 20.0, saturation_model = ConstantSolubility(20.0),
        initial_solvent_state = (; concentration = 20.0, pH = 7.0),
        solvent_dynamics = (configured_problem, numerical_state, saved_time, growth_rate) ->
            (; concentration = -2.0, pH = 1.0), solver = MoM())
    workflow_times = [0.0, 1.0]
    _, workflow_solution = runsimulation(workflow_problem; save_idx = workflow_times)
    @test initial_concentration(workflow_problem) == 20.0
    @test workflow_solution.concentration ≈ [20.0, 18.0]
    @test workflow_solution.solvent_state.pH ≈ [7.0, 8.0]
    workflow_ode, workflow_algorithm = crystallisation_odeproblem(workflow_problem, workflow_times)
    manual_solution = CriSTool.solve(workflow_ode, workflow_algorithm;
        saveat = workflow_times, reltol = 1e-10, abstol = 1e-8)
    converted_solution = crystallisation_solution(workflow_problem, manual_solution)
    @test converted_solution.concentration ≈ [20.0, 18.0]
    @test converted_solution.solvent_state.pH ≈ [7.0, 8.0]

    workflow_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = workflow_times,
            mean = [20.0, 18.0], variance = 1.0),
            pH = Observable(time = workflow_times, mean = [7.0, 8.0], variance = 1.0)),
        temperature = 300.0, exp_id = 701)
    workflow_setup = prepare_loss([workflow_problem], [workflow_experiment])
    @test workflow_setup.prepared[1].problem.temp_profile === workflow_problem.temp_profile
    @test loss(mae(), workflow_setup, Float64[]) ≈ 0.0 atol = 1e-10
    predictions = run_ensemble(zeros(0, 2), workflow_setup; verbosity = 0)
    @test observable_values(predictions[1], :concentration) ≈ [20.0 18.0; 20.0 18.0]
    @test observable_values(predictions[1], :pH) ≈ [7.0 8.0; 7.0 8.0]
    pH_summary = prediction_summary(predictions[1], :pH)
    @test pH_summary.mean ≈ [7.0, 8.0]
    @test pH_summary.std ≈ [0.0, 0.0]
    @test pH_summary.sample_count == 2
    @test pH_summary.failed_count == 0
    @test predictions[1].concentration ≈ [20.0 18.0; 20.0 18.0]
    problem_predictions = run_ensemble(zeros(0, 1), workflow_problem, [workflow_experiment]; verbosity = 0)
    @test observable_values(problem_predictions[1], :pH) ≈ [7.0 8.0]
    @test_throws ArgumentError run_ensemble(zeros(1, 2), workflow_setup; verbosity = 0)
    @test_throws ArgumentError runsimulation([1.0], workflow_problem; save_idx = workflow_times)

    late_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [1.0], mean = [18.0], variance = 1.0)),
        temperature = 300.0, exp_id = 702)
    late_setup = prepare_loss([workflow_problem], [late_experiment])
    @test late_setup.prepared[1].odeproblem.tspan == (0.0, 1.0)
    @test initial_concentration(late_setup.prepared[1].problem) == 20.0
    @test loss(mae(), late_setup, Float64[]) ≈ 0.0 atol = 1e-10

    signal_experiment = CrystallisationExperiment(;
        observables = (; concentration = workflow_experiment.observables.concentration,
            optical_signal = Observable(time = workflow_times, mean = [40.0, 36.0], variance = 1.0)),
        temperature = 300.0, exp_id = 703)
    double_signal_setup = prepare_loss([workflow_problem], [signal_experiment];
        observable_projections = (; optical_signal = physical_solution -> 2 .* physical_solution.concentration))
    triple_signal_setup = prepare_loss([workflow_problem], [signal_experiment];
        observable_projections = (; optical_signal = physical_solution -> 3 .* physical_solution.concentration))
    @test loss(mae(), double_signal_setup, Float64[]) ≈ 0.0 atol = 1e-10
    @test loss(mae(), triple_signal_setup, Float64[]) ≈ 19.0
    @test loss(mae(), double_signal_setup, Float64[]) ≈ 0.0 atol = 1e-10
    signal_predictions = run_ensemble(zeros(0, 1), double_signal_setup; verbosity = 0)
    @test observable_values(signal_predictions[1], :optical_signal) ≈ [40.0 36.0]
    @test_throws ArgumentError prepare_loss([workflow_problem], [workflow_experiment];
        observable_projections = (; concentration = physical_solution -> physical_solution.concentration))

    failed_setup = prepare_loss([workflow_problem], [workflow_experiment];
        algorithm = CriSTool.OrdinaryDiffEq.Euler(),
        solve_options = (; dt = 0.1, adaptive = false, maxiters = 2))
    @test loss(mae(), failed_setup, Float64[]) == 2_000_000.0
    failed_predictions = run_ensemble(zeros(0, 1), failed_setup; verbosity = 0)
    @test failed_predictions[1].success == [false]
    @test failed_predictions[1].diagnostics[1].reason ==
        CriSTool.OrdinaryDiffEq.SciMLBase.ReturnCode.MaxIters
    @test_throws ArgumentError prediction_summary(failed_predictions[1], :concentration)
    @test_throws ArgumentError prediction_summary(failed_predictions[1], :concentration; skip_failed = true)
    inactive_size_experiment = CrystallisationExperiment(;
        observables = merge(workflow_experiment.observables,
            (; d50q = Observable(time = [1.0], mean = [1e-6], variance = 1e-12))),
        temperature = 300.0, exp_id = 705)
    inactive_size_setup = prepare_loss([workflow_problem], [inactive_size_experiment])
    @test loss(mae(weighting = (; concentration = 1.0, pH = 1.0, d50q = 0.0)),
        inactive_size_setup, Float64[]) ≈ 0.0 atol = 1e-10
end

@testset "Explicit seed declaration preserves independent solvent input" begin
    matching_seed = LogNormalInitialCrystals(mass_concentration = 0.25,
        d43 = 10e-6, geometric_std = 1.2)
    seed_configuration = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
            nucleation_order = 1.0), parameterset_nucleation = Float64[],
        kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]),
        parameterset_growth = Float64[], initial_concentration = 20.0,
        saturation_model = ConstantSolubility(20.0), solver = MoM())
    physical_seed_state = initial_state_from_characteristics(seed_configuration, matching_seed)
    physical_seed_state[end] = 12.0
    configured_seed = CriSTool._copy_crystallisation_problem(seed_configuration;
        initial_state = physical_seed_state)
    seed_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [0.0, 1.0],
            mean = [12.0, 12.0], variance = 1.0)), initial_crystals = matching_seed,
        temperature = 300.0, exp_id = 706)
    seed_setup = prepare_loss([configured_seed], [seed_experiment])
    @test initial_concentration(seed_setup.prepared[1].problem) == 12.0
    @test loss(mae(), seed_setup, Float64[]) ≈ 0.0 atol = 1e-10
    negative_solvent_state = copy(physical_seed_state)
    negative_solvent_state[end] = -1.0
    @test_throws ArgumentError runsimulation(CriSTool._copy_crystallisation_problem(configured_seed;
        initial_state = negative_solvent_state); save_idx = [0.0, 1.0])
end

@testset "Configured mesh predictions retain volume" begin
    volume_operation = FedBatchOperation(initial_volume = 2.0, inflow = 0.5,
        feed = CrystallisationFeed(concentration = 2.0))
    volume_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [0.0, 4.0],
            mean = [8.0, 5.0], variance = 1.0)), temperature = 293.15, exp_id = 707)
    for mesh_solver in (FiniteVol(meshsize = 20, lmax = 50e-6), WENO(meshsize = 20, lmax = 50e-6))
        volume_problem = CrystallisationProblem(; operation = volume_operation, solver = mesh_solver,
            kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                nucleation_order = 1.0), parameterset_nucleation = Float64[],
            kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]),
            parameterset_growth = Float64[], initial_concentration = 8.0)
        volume_predictions = run_ensemble(zeros(0, 1), volume_problem, [volume_experiment]; verbosity = 0)
        @test volume_predictions[1].volume ≈ [2.0 4.0]
        @test observable_values(volume_predictions[1], :volume) ≈ [2.0 4.0]
    end
end

@testset "Configured inference physical mass oracle" begin
    mass_oracle_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
            nucleation_order = 1.0), parameterset_nucleation = Float64[],
        kinetics_growthfunction = WorkflowConstantGrowth(), parameterset_growth = [1.0],
        initial_concentration = 20.0, saturation_model = ConstantSolubility(10.0),
        initial_state = [1e12, 1e7, 100.0, 1e-3, 1e-8, 20.0], solver = MoM())
    # Frozen conservation values for N=1e12 monodisperse 10 µm seeds,
    # constant growth 0.1 µm/s, rho*kv=1109.7 kg/m³.
    frozen_concentration = [20.0, 19.9663749803, 19.9320774824]
    mass_oracle_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [0.0, 1.0, 2.0],
            mean = frozen_concentration, variance = 1e-4)), temperature = 293.15, exp_id = 704)
    mass_oracle_setup = prepare_loss([mass_oracle_problem], [mass_oracle_experiment])
    _, mass_oracle_solution = runsimulation([1.0], mass_oracle_problem;
        save_idx = [0.0, 1.0, 2.0])
    @test mass_oracle_solution.concentration ≈ frozen_concentration atol = 1e-8
    @test loss(mae(), mass_oracle_setup, [1.0]) ≈ 0.0 atol = 1e-8
    mass_prediction = run_ensemble(reshape([1.0], 1, 1), mass_oracle_setup; verbosity = 0)
    @test observable_values(mass_prediction[1], :concentration) ≈
        reshape(frozen_concentration, 1, :) atol = 1e-8
    mass_oracle_model = nuts_model(mass_oracle_setup, [Uniform(0.5, 1.5)]; lossfunction = logMLE())
    @test mass_oracle_model isa CriSTool.Turing.DynamicPPL.Model
    # Uniform(0.5,1.5) contributes log density zero. Conditioning includes
    # that prior term and fixing omits it, so both retain this frozen likelihood.
    @test CriSTool.Turing.DynamicPPL.logjoint(mass_oracle_model, (; θ = [1.0])) ≈
        11.058694958350254 atol = 1e-8
    conditioned_mass_model = CriSTool.Turing.DynamicPPL.condition(
        mass_oracle_model; θ = [1.0])
    fixed_mass_model = CriSTool.Turing.DynamicPPL.fix(
        mass_oracle_model; θ = [1.0])
    @test CriSTool.Turing.DynamicPPL.logjoint(conditioned_mass_model, NamedTuple()) ≈
        11.058694958350254 atol = 1e-8
    @test CriSTool.Turing.DynamicPPL.logjoint(fixed_mass_model, NamedTuple()) ≈
        11.058694958350254 atol = 1e-8
    @test_throws ArgumentError nuts_model(mass_oracle_setup, [Uniform(0.5, 1.5), Uniform(0.5, 1.5)])
    gradient_fit = PE_Routine_Optimisation(logMLE(), mass_oracle_setup, [0.5], [1.5];
        searchalgo = OptimizationOptimJL.Optim.Fminbox(OptimizationOptimJL.Optim.BFGS()),
        x0 = [0.8], searchoptions = (; maxiters = 100), verbosity = 0, savetxt = false)
    @test gradient_fit.u[1] ≈ 1.0 atol = 1e-5
    @test gradient_fit.objective ≈ -11.058694958350254 atol = 1e-7
end

@testset "Default predictions follow each prepared system's observable schema" begin
    schema_times = [0.0, 1.0]
    schema_batch = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
            nucleation_order = 1.0), parameterset_nucleation = Float64[],
        kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]),
        parameterset_growth = Float64[], initial_concentration = 20.0,
        saturation_model = ConstantSolubility(20.0), solver = MoM())
    schema_extended = CriSTool._copy_crystallisation_problem(schema_batch;
        initial_solvent_state = (; concentration = 20.0, pH = 7.0),
        solvent_dynamics = (configured_problem, numerical_state, saved_time, growth_rate) ->
            (; concentration = 0.0, pH = 1.0))
    schema_concentration = Observable(time = schema_times, mean = [20.0, 20.0], variance = 1.0)
    schema_plain_experiment = CrystallisationExperiment(;
        observables = (; concentration = schema_concentration), temperature = 300.0, exp_id = 711)
    schema_extended_experiment = CrystallisationExperiment(;
        observables = (; concentration = schema_concentration,
            pH = Observable(time = schema_times, mean = [7.0, 8.0], variance = 1.0)),
        temperature = 300.0, exp_id = 712)
    schema_setup = prepare_loss([schema_batch, schema_extended],
        [schema_plain_experiment, schema_extended_experiment])
    @test loss(mae(), schema_setup, Float64[]) ≈ 0.0 atol = 1e-10
    schema_predictions = run_ensemble(zeros(0, 1), schema_setup; verbosity = 0)
    @test all(prediction -> prediction.success == [true], schema_predictions)
    @test observable_values(schema_predictions[1], :concentration) ≈ [20.0 20.0]
    @test !hasproperty(schema_predictions[1].predictions, :pH)
    @test observable_values(schema_predictions[2], :pH) ≈ [7.0 8.0]
    @test_throws ArgumentError run_ensemble(zeros(0, 1), schema_setup; observables = [:pH])

    schema_msmpr = CriSTool._copy_crystallisation_problem(schema_batch;
        operation = MSMPROperation(volume = 2.0, inflow = 0.5,
            feed = CrystallisationFeed(concentration = 20.0)))
    schema_volume_experiment = CrystallisationExperiment(;
        observables = (; concentration = schema_concentration,
            volume = Observable(time = schema_times, mean = [2.0, 2.0], variance = 1.0)),
        temperature = 300.0, exp_id = 713)
    schema_volume_setup = prepare_loss([schema_batch, schema_msmpr],
        [schema_plain_experiment, schema_volume_experiment])
    schema_volume_predictions = run_ensemble(zeros(0, 1), schema_volume_setup; verbosity = 0)
    @test all(prediction -> prediction.success == [true], schema_volume_predictions)
    @test !hasproperty(schema_volume_predictions[1].predictions, :volume)
    @test observable_values(schema_volume_predictions[2], :volume) == [2.0 2.0]
end

@testset "Univariate ensemble draws preserve sample shape and RNG" begin
    distribution_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
            nucleation_order = 1.0), parameterset_nucleation = Float64[],
        kinetics_growthfunction = WorkflowConstantGrowth(), parameterset_growth = [1.0],
        initial_concentration = 20.0, saturation_model = ConstantSolubility(10.0),
        initial_state = [1e12, 1e7, 100.0, 1e-3, 1e-8, 20.0], solver = MoM())
    distribution_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [0.0, 1.0],
            mean = [20.0, 20.0], variance = 1.0)), temperature = 300.0, exp_id = 714)
    distribution_setup = prepare_loss([distribution_problem], [distribution_experiment])
    for distribution_sample_count in (1, 3)
        distribution_rng = CriSTool.Random.Xoshiro(714)
        expected_draws = rand(CriSTool.Random.Xoshiro(714), Uniform(0.5, 1.5), distribution_sample_count)
        distribution_predictions = only(run_ensemble(Uniform(0.5, 1.5), distribution_setup;
            n_samples = distribution_sample_count, rng = distribution_rng, verbosity = 0))
        @test distribution_predictions.success == fill(true, distribution_sample_count)
        # Closed-system concentration follows the independently known solid
        # inventory rho*kv*N*((L0+G*t)^3-L0^3).
        expected_concentration = 20.0 .- 1109.7e12 .* ((10e-6 .+ 1e-7 .* expected_draws).^3 .- (10e-6)^3)
        @test size(distribution_predictions.concentration) == (distribution_sample_count, 2)
        @test distribution_predictions.concentration[:, 1] == fill(20.0, distribution_sample_count)
        @test distribution_predictions.concentration[:, 2] ≈ expected_concentration atol = 1e-8
    end
    @test_throws ArgumentError run_ensemble(Uniform(0.5, 1.5), distribution_setup; n_samples = 0)
    zero_parameter_problem = CriSTool._copy_crystallisation_problem(distribution_problem;
        kinetics_growthfunction = growth_empirical_fixed([0.0, 1.0]), parameterset_growth = Float64[])
    zero_parameter_setup = prepare_loss([zero_parameter_problem], [distribution_experiment])
    @test_throws ArgumentError run_ensemble(Uniform(0.5, 1.5), zero_parameter_setup; n_samples = 1)
end

struct OutputTimeDomainGrowth <: CriSTool.AbstractFPScalarGrowthFunction
    nparams::Int
end
CriSTool.paramaxis(::OutputTimeDomainGrowth) = CriSTool.ComponentArrays.Axis(output_time_rate = 1)
function CriSTool.growthrate(::OutputTimeDomainGrowth, kinetic_parameters, configured_problem,
                            numerical_state, simulation_time)
    simulation_time == 0.5 && throw(DomainError(simulation_time, "user output-time kinetic error"))
    return zero(kinetic_parameters[1])
end

@testset "User kinetic DomainError at interpolated mesh outputs propagates" begin
    output_error_experiment = CrystallisationExperiment(;
        observables = (; concentration = Observable(time = [0.0, 0.5, 1.0],
            mean = [20.0, 20.0, 20.0], variance = 1.0)), temperature = 300.0, exp_id = 822)
    for output_error_solver in (FiniteVol(meshsize = 4), WENO(meshsize = 4))
        output_error_problem = CrystallisationProblem(; solver = output_error_solver,
            kinetics_nucleationfunction = nucl_empirical_fixed([0.0, 1.0]),
            parameterset_nucleation = Float64[], kinetics_growthfunction = OutputTimeDomainGrowth(1),
            parameterset_growth = [0.0], initial_concentration = 20.0,
            saturation_model = ConstantSolubility(20.0))
        output_error_setup = prepare_loss([output_error_problem], [output_error_experiment];
            algorithm = CriSTool.OrdinaryDiffEq.Euler(),
            solve_options = (; dt = 0.3, adaptive = false))
        @test_throws DomainError loss(mae(), output_error_setup, [0.0])
        ensemble_output_error = try
            run_ensemble(reshape([0.0], 1, 1), output_error_setup; verbosity = 0)
        catch propagated_error
            propagated_error
        end
        @test ensemble_output_error isa CompositeException
        @test ensemble_output_error isa Exception &&
            occursin("user output-time kinetic error", sprint(showerror, ensemble_output_error))
    end
end
