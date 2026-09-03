"""
Tutorial 8: real-data comparison of MoM and FiniteVol parameter estimation.

This is an intentionally substantial workflow, not a smoke test.  Set
`CRISTOOL_DATA_FILE` to your CSV file (the shipped fixture is the default),
then run this script in the examples environment. It performs a point
optimization and a four-chain NUTS run for MoM, plus derivative-free ABCDE
for the FV comparison:

1. CNT + empirical growth with MoM;
2. CNT + empirical growth + scalar aggregation + uniform-in-volume breakage
   with a resolved FiniteVol population balance. The FV representative fit is
   found with derivative-free ABCDE.

The final objective values and posterior means are printed side by side so the
fit improvement can be assessed on the same experiments and loss.
"""

using CriSTool
using Distributions
using Metaheuristics
using Random
using Statistics
using Turing

const DATA_FILE = get(ENV, "CRISTOOL_DATA_FILE",
                      joinpath(pkgdir(CriSTool), "test", "fixtures",
                               "real-experimental-dataset.csv"))
const RESULTS_DIR = get(ENV, "CRISTOOL_RESULTS_DIR",
                        joinpath(@__DIR__, "results", "tutorial8"))

# These are deliberately sized for an actual inference run. NUTS chains own
# their loss setups and ODE templates. PE uses bounded Base-threaded blocks so
# concurrent ODE-remake allocations are collected between blocks.
const OPTIMISER_PARTICLES = 256
const OPTIMISER_GENERATIONS = 256
const MCMC_SAMPLES = 1_000
const MCMC_BURNIN = 2_000
const MCMC_CHAINS = 4

function validate_bounds(lower, upper, expected_length)
    length(lower) == expected_length ||
        throw(ArgumentError("lower bounds have the wrong number of parameters"))
    length(upper) == expected_length ||
        throw(ArgumentError("upper bounds have the wrong number of parameters"))
    all(isfinite, lower) && all(isfinite, upper) && all(lower .< upper) ||
        throw(ArgumentError("bounds must be finite and strictly increasing"))
    return nothing
end

function fit_model(measurements, solver, aggregation, breakage, lower, upper, label;
                   optimiser_particles = OPTIMISER_PARTICLES,
                   optimiser_generations = OPTIMISER_GENERATIONS,
                   mcmc_samples = MCMC_SAMPLES)
    loss_function = logMLE(weighting = [1.0, 1.0])
    nucleation = nucl_CNT()
    growth = growth_empirical()

    println("\n--- ", label, " ---")
    optimisation = PE_Routine(loss_function, measurements, lower, upper,
                              nucleation, growth, aggregation, breakage;
                              solver = solver,
                              nparticles = optimiser_particles,
                              generations = optimiser_generations,
                              parallel_evaluation = true,
                              verbosity = 1,
                              savetxt = true,
                              extrastring = label,
                              outputdir = RESULTS_DIR)
    optimum = minimizer(optimisation)
    objective = minimum(optimisation)
    println("MLE parameters: ", round.(optimum, sigdigits = 5))
    println("MLE objective:  ", objective)

    priors = [TriangularDist(lower[index], upper[index], optimum[index])
              for index in eachindex(optimum)]
    chain = MCMC_Routine(measurements, priors, nucleation, growth,
                         aggregation, breakage;
                         solver = solver,
                         lossfunction = loss_function,
                         sampler = NUTS(MCMC_BURNIN, 0.65;
                                        adtype = AutoForwardDiff(chunksize = 4)),
                         n_samples = mcmc_samples,
                         n_chains = MCMC_CHAINS,
                         extrastring = label,
                         outputdir = RESULTS_DIR,
                         saveplot = true,
                         showplot = false,
                         verbosity = 1)
    posterior_mean = vec(mean(chain).nt.mean)
    println("NUTS posterior mean: ", round.(posterior_mean, sigdigits = 5))
    return optimum, objective, posterior_mean
end

function fit_finite_volume_with_abcde(measurements, solver, aggregation, breakage,
                                      lower, upper, reference;
                                      abcde_particles = OPTIMISER_PARTICLES,
                                      abcde_generations = OPTIMISER_GENERATIONS)
    loss_function = logMLE(weighting = [1.0, 1.0])
    nucleation = nucl_CNT()
    growth = growth_empirical()
    prior = product_distribution([Uniform(lower[index], upper[index])
                                  for index in eachindex(lower)])

    println("\n--- FiniteVol: CNT + scalar aggregation + uniform breakage (ABCDE) ---")
    _, abc_metadata = run_abc(loss_function, measurements, reference, prior,
                              nucleation, growth, aggregation, breakage;
                              solver = solver,
                              sampler = ABCDESampler(α = 0),
                              nparticles = abcde_particles,
                              generations = abcde_generations,
                              confidenceinterval = 0.90,
                              test = :f,
                              extrastring = "FiniteVol ABCDE",
                              outputdir = RESULTS_DIR,
                              saveplot = true,
                              verbosity = 1)
    representative = abc_metadata["meanparameters"]
    problem = CriSTool._build_loss_problem(nucleation, growth, aggregation, breakage,
                                           solver)
    objective = loss(loss_function, prepare_loss(problem, measurements), representative)
    println("ABCDE representative parameters: ",
            round.(representative, sigdigits = 5))
    println("ABCDE representative objective:  ", objective)
    return representative, objective, nothing
end

function main()
    Random.seed!(20260830)
    measurements = load_measurements(DATA_FILE)
    isempty(measurements) && error("No experiments found in $DATA_FILE")
    println("Loaded ", length(measurements), " experiments from ", DATA_FILE)

    mom_lower = [25.0, 0.00030, 0.3e-9 / 60, 2.0]
    mom_upper = [50.0, 0.00100, 3.0e-9 / 60, 4.0]
    validate_bounds(mom_lower, mom_upper, 4)
    mom_result = fit_model(measurements,
                            MoM(reltol = 1e-7),
                            noaggregation(), nobreakage(),
                           mom_lower, mom_upper, "MoM: CNT + empirical growth")

    finite_volume_lower = [mom_lower; -20.0 - log10(60); -20.0 - log(60); 0.0]
    finite_volume_upper = [mom_upper; -12.0 - log10(60); -10.0 - log(60); 1.0]
    validate_bounds(finite_volume_lower, finite_volume_upper, 7)
    finite_volume_reference = [mom_result[1]; -18.0 - log10(60); -18.0 - log(60); 0.5]
    all(finite_volume_lower .<= finite_volume_reference) &&
        all(finite_volume_reference .<= finite_volume_upper) ||
        throw(ArgumentError("FiniteVol ABCDE reference is outside its bounds"))
    # FV evaluates a full mesh for every loss call, so use a 64×64 ABCDE
    # search while retaining the resolved 80-cell PBE. The four-chain NUTS
    # stage belongs to the MoM comparison; FV uses ABCDE only.
    finite_volume_result = fit_finite_volume_with_abcde(
        measurements,
        FiniteVol(meshsize = 80, lmax = 80e-6),
        aggr_scalar(), breakage_uniform(),
        finite_volume_lower, finite_volume_upper, finite_volume_reference;
        abcde_particles = 64,
        abcde_generations = 64)

    println("\n=== Fit comparison ===")
    println("MoM MLE objective:  ", mom_result[2])
    println("FiniteVol ABCDE objective: ", finite_volume_result[2])
    println("Objective change (FiniteVol ABCDE - MoM): ",
            finite_volume_result[2] - mom_result[2])
    println("A lower FiniteVol ABCDE objective indicates a better fit under this loss.")
end

main()
