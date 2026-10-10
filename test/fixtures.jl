# Shared end-to-end fixtures. Every model here is small and has a closed-form
# answer, so tests can assert against independent truth instead of re-deriving
# solver logic. Names carry a `fixture_` prefix to stay greppable.

using Distributions: Uniform

# Unit material: constant solubility, rho = 1000 kg/m³, kv = 1.
const fixture_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.0),
    crystal_density = 1000.0, volume_shape_factor = 1.0)

# Zero nucleation and a constant growth speed G = speed * 1e-6 m/s, independent of
# supersaturation. Seeds then translate rigidly in length: L(t) = L0 + G t.
const fixture_no_nucleation = CallableNucleation((rate_values, rate_context) -> 0.0; parameters = (;))
fixture_constant_growth(; speed = 0.01) =
    CallableGrowth((rate_values, rate_context) -> rate_values.speed * 1e-6; parameters = (; speed))
# Constant nucleation rate B (#/m³/s) for MSMPR population-balance oracles.
fixture_constant_nucleation(; birth_rate) =
    CallableNucleation((rate_values, rate_context) -> rate_values.birth_rate; parameters = (; birth_rate))

fixture_translation_model(; speed = 0.01, nucleation = fixture_no_nucleation) =
    CrystallisationModel(fixture_system; nucleation, growth = fixture_constant_growth(; speed))

# Uniform seed on [10, 20] µm carrying 0.25 kg/m³ of crystal mass.
const fixture_seed_bounds = (10e-6, 20e-6)
const fixture_seed_mass = 0.25
fixture_uniform_seed(; mass_concentration = fixture_seed_mass) =
    DistributionInitialCrystals(Uniform(fixture_seed_bounds...); mass_concentration)

fixture_seeded_conditions(; concentration = 20.0, temperature = 293.15,
        crystals = fixture_uniform_seed(), kwargs...) =
    RunConditions(; temperature, concentration, crystals, kwargs...)

# E[(L + shift)^k] for L ~ Uniform(a, b).
function fixture_uniform_moment(order, shift = 0.0; bounds = fixture_seed_bounds)
    lower, upper = bounds .+ shift
    return (upper^(order + 1) - lower^(order + 1)) / ((order + 1) * (upper - lower))
end

# Closed-form batch solution for the translation model: solute lost equals the
# crystal mass gained, and d43 is the ratio of shifted fourth to third moments.
function fixture_translation_concentration(elapsed_time; speed = 0.01, concentration = 20.0,
        seed_mass = fixture_seed_mass)
    length_shift = speed * 1e-6 * elapsed_time
    return concentration - seed_mass * (fixture_uniform_moment(3, length_shift) / fixture_uniform_moment(3) - 1)
end
fixture_translation_d43(elapsed_time; speed = 0.01) =
    fixture_uniform_moment(4, speed * 1e-6 * elapsed_time) / fixture_uniform_moment(3, speed * 1e-6 * elapsed_time)

# Synthetic measurements from a trajectory (or any named vectors) at its own times.
function fixture_experiment(trajectory, conditions; observables = (:concentration, :d43),
        variances = (concentration = 0.01, d43 = 1e-14), exp_id = 1)
    named_observables = NamedTuple{observables}(map(observables) do observable_name
        observed_means = collect(getproperty(trajectory, observable_name))
        Observable(time = collect(trajectory.time), mean = observed_means,
            variance = fill(getproperty(variances, observable_name), length(observed_means)))
    end)
    return CrystallisationExperiment(; observables = named_observables, exp_id, conditions)
end

# Realistic nonlinear case (README values) for regressions that need real kinetics.
const fixture_lysozyme_model = CrystallisationModel(LysozymeSystem();
    nucleation = nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 0.0007),
    growth = growth_empirical(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
const fixture_lysozyme_conditions = RunConditions(temperature = 293.15, concentration = 18.0)

# Continuous operations sharing one clear feed (C = 2 kg/m³).
fixture_clear_feed(; concentration = 2.0) = CrystallisationFeed(; concentration)
fixture_msmpr(; volume = 2.0, inflow = 0.2, feed = fixture_clear_feed()) =
    MSMPROperation(; volume, inflow, feed)
fixture_fed_batch(; initial_volume = 2.0, inflow = 0.5, feed = fixture_clear_feed()) =
    FedBatchOperation(; initial_volume, inflow, feed)
