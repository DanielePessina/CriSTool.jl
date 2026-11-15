#=Tutorial 1: build a named model once and run it under several conditions and solvers.=#
using CriSTool

function main()
    # A model is a material system plus bound kinetic laws. Calling a law with
    # its parameter names binds the values; aggregation and breakage default to none.
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))

    # Run conditions are separate from the model: temperature (K), initial
    # concentration (kg/m³), optional seed and reactor operation.
    isothermal_run = RunConditions(temperature = 293.15, concentration = 18.0)
    # A problem is the model under the conditions over a time span (s); the
    # numerical solver is chosen when it is solved, like an ODE algorithm.
    isothermal_problem = CrystallisationProblem(crystal_model, isothermal_run, (0.0, 21600.0))
    for numerical_solver in (MoM(), QMOM(nquadrature = 3), FiniteVol(meshsize = 100, lmax = 100e-6))
        trajectory = solve(isothermal_problem, numerical_solver; saveat = 600.0)
        @assert successful_retcode(trajectory)
        println(nameof(typeof(numerical_solver)), ": C(final) = ", trajectory[:concentration][end],
            " kg/m³; d43(final) = ", trajectory[:d43][end], " m")
    end

    # The same model under a cooling profile with a lognormal seed.
    cooling_run = RunConditions(temperature = LinearTemperature(293.15, -1e-4), concentration = 18.0,
        crystals = LogNormalInitialCrystals(mass_concentration = 0.1, d43 = 20e-6, geometric_std = 1.2))
    cooled = solve(remake(isothermal_problem; conditions = cooling_run), MoM(); saveat = 600.0)
    println("Seeded cooling: C(final) = ", cooled.concentration[end], " kg/m³")
end
main()
