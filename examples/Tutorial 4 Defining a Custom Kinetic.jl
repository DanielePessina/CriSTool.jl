#=Tutorial 4: define a custom law with a callable and one named parameter schema.=#
using CriSTool, Metaheuristics

function main()
    saturation_growth = CallableGrowth(
        (rate_values, rate_context) -> begin
            driving_force = max(supersaturation(rate_context) - 1, 0)
            rate_values.coefficient * driving_force / (rate_values.half_saturation + driving_force)
        end; parameters = (coefficient = 2e-11, half_saturation = 0.3))
    # Callable laws carry their values and are bound automatically.
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = saturation_growth)
    custom_problem = CrystallisationProblem(crystal_model, RunConditions(temperature = 293.15, concentration = 18.0),
        (0.0, 21600.0))
    trajectory = solve(custom_problem, MoM(); saveat = 600.0)
    @assert successful_retcode(trajectory)
    println("Custom growth: final concentration = ", trajectory.concentration[end])
    # Callable parameters are selected by name like built-in ones.
    measured_runs = load_measurements(joinpath(@__DIR__, "fake-experimental-dataset.csv"))
    coefficient_problem = EstimationProblem(crystal_model, measured_runs;
        bounds = (growth = (coefficient = (1e-12, 1e-10),),))
    println("Fitted coefficient: ", solve(coefficient_problem,
        DE(N = 8, options = Options(iterations = 2, seed = 1))).parameters.growth.coefficient)
    # Advanced users can still define custom law types, paramaxis and rate methods.
    trajectory
end
main()
