# Shared named model for tutorials; all kinetic values are visible here.
function tutorial_model(; system = LysozymeSystem(), growth_coefficient = 1e-9 / 60)
    CrystallisationModel(system = system,
        nucleation = KineticModel(nucl_CNT(); parameters = (
            ln_nucleation_prefactor = 38.0, surface_energy = 0.0006)),
        growth = KineticModel(growth_empirical(); parameters = (
            growth_coefficient = growth_coefficient, growth_order = 3.0)))
end
