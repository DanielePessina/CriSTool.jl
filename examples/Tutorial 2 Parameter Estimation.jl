#=Tutorial 2: fit selected named parameters, then predict with the fitted model.=#
using CriSTool, Metaheuristics

function main()
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    # Each experiment starts from its own time-zero concentration observation.
    measured_runs = load_measurements(joinpath(@__DIR__, "fake-experimental-dataset.csv"))
    # Bounds select the fitted parameters; unlisted values stay as bound in the model.
    fit_specification = OptimisationSpec(
        bounds = (nucleation = (ln_nucleation_prefactor = (20.0, 50.0), surface_energy = (0.0001, 0.002)),
                  growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0))),
        algorithm = DE(N = 16, options = Options(iterations = 4, seed = 42)))
    fitted_result = fit(prepare_fit(crystal_model, measured_runs, fit_specification); parallel = EnsembleThreads())
    println("Named estimates: ", fitted_result.parameters)
    println("Observable objectives: ", fitted_result.objectives)
    # The fit remembers its experiments, solver and settings.
    fitted_trajectories = predict(fitted_result)
    println("Predicted final concentrations: ",
        [observable_values(ensemble, :concentration)[1, end] for ensemble in fitted_trajectories.ensembles])
    # For a larger study increase particles/iterations and inspect backend convergence.
    fitted_result
end
main()
