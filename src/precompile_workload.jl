"""
Precompile workload: one short simulation per solver with the built-in kinetic
laws and one prepared-fit loss evaluation, so the first `simulate` or `loss`
call in a session reuses cached native code instead of compiling the solver
stack. Callable laws (`CallableGrowth`, ...)
are user closures and still compile on first use.
"""

@setup_workload begin
    workload_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    workload_seed = LogNormalInitialCrystals(mass_concentration = 0.1, d43 = 20e-6, geometric_std = 1.2)
    workload_seeded_run = RunConditions(temperature = 293.15, concentration = 18.0, crystals = workload_seed)
    workload_unseeded_run = RunConditions(temperature = 293.15, concentration = 18.0)
    workload_save_times = 0.0:30.0:60.0
    # WENO is meant for sharp growth fronts; nucleation into its first cell is slow.
    workload_growth_model = CrystallisationModel(LysozymeSystem();
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    @compile_workload begin
        for workload_solver in (MoM(), QMOM(), DQMOM(), FiniteVol(meshsize = 8, lmax = 100e-6))
            simulate(workload_model, workload_seeded_run; solver = workload_solver,
                     saveat = workload_save_times)
        end
        simulate(workload_growth_model, workload_seeded_run; solver = WENO(meshsize = 8, lmax = 100e-6),
                 saveat = workload_save_times)
        simulate(workload_model, workload_unseeded_run; saveat = workload_save_times)
        # One prepared-fit loss evaluation (the objective every fit calls).
        workload_experiment = CrystallisationExperiment(exp_id = 1,
            observables = (concentration = Observable(time = [0.0, 30.0, 60.0],
                mean = [18.0, 17.99, 17.98], variance = [0.01, 0.01, 0.01]),),
            conditions = RunConditions(temperature = 293.15, crystals = workload_seed))
        workload_fit = prepare_fit(workload_model, [workload_experiment],
            OptimisationSpec(bounds = (growth = (growth_order = (1.0, 4.0),),), algorithm = nothing))
        loss(workload_fit, [3.0])
    end
end
