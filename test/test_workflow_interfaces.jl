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
    @test_throws ArgumentError run_ensemble(zeros(1, 2), workflow_setup; verbosity = 0)

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
    # Uniform(0.5,1.5) contributes log density zero. Three exact Gaussian
    # measurements with variance 1e-4 have the frozen log joint below.
    @test CriSTool.Turing.DynamicPPL.logjoint(mass_oracle_model, (; θ = [1.0])) ≈
        11.058694958350254 atol = 1e-8
    @test_throws ArgumentError nuts_model(mass_oracle_setup, [Uniform(0.5, 1.5), Uniform(0.5, 1.5)])
    gradient_fit = PE_Routine_Optimisation(logMLE(), mass_oracle_setup, [0.5], [1.5];
        searchalgo = OptimizationOptimJL.Optim.Fminbox(OptimizationOptimJL.Optim.BFGS()),
        x0 = [0.8], searchoptions = (; maxiters = 100), verbosity = 0, savetxt = false)
    @test gradient_fit.u[1] ≈ 1.0 atol = 1e-5
    @test gradient_fit.objective ≈ -11.058694958350254 atol = 1e-7
end
