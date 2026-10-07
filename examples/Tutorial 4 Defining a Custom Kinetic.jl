#=Tutorial 4: define a custom law with a callable and one named parameter schema.=#
using CriSTool

function main()
    saturation_growth = CallableGrowth(
        (rate_values, rate_context) -> begin
            driving_force = max(supersaturation(rate_context) - 1, 0)
            rate_values.coefficient * driving_force / (rate_values.half_saturation + driving_force)
        end; parameters = (coefficient = 2e-11, half_saturation = 0.3))
    crystal_model = CrystallisationModel(system = LysozymeSystem(),
        nucleation = KineticModel(nucl_CNT(); parameters = (
            ln_nucleation_prefactor = 38.0, surface_energy = 0.0006)),
        growth = KineticModel(saturation_growth))
    configured_run = CrystallisationProblem(crystal_model;
        initial_conditions = (concentration = 18.0,), temperature = ConstantTemperature(293.15))
    trajectory = simulate(configured_run; saveat = 0.0:600.0:21600.0)
    @assert trajectory.success
    println("Custom growth: final concentration = ", trajectory.concentration[end])
    # Fit only coefficient using OptimisationSpec(bounds=(growth=(coefficient=(...),),)).
    # Advanced users can still define custom law types, paramaxis and rate methods.
    trajectory
end
main()
