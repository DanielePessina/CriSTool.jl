"""
Tutorial 8: real-data comparison of MoM and QMOM parameter estimation.

This is an intentionally substantial workflow, not a smoke test.  Set
`CRISTOOL_DATA_WORKBOOK` to your workbook (the shipped fixture is the default),
then run this script in the examples environment.  It performs a point
optimization and a four-chain NUTS run for:

1. CNT + empirical growth with MoM;
2. CNT + empirical growth + scalar aggregation + empirical breakage with QMOM.

The final objective values and posterior means are printed side by side so the
fit improvement can be assessed on the same experiments and loss.
"""

using CriSTool
using Distributions
using Metaheuristics
using Random
using Statistics
using Turing

const DATA_WORKBOOK = get(ENV, "CRISTOOL_DATA_WORKBOOK",
                          joinpath(pkgdir(CriSTool), "test", "fixtures",
                                   "real-experimental-dataset.xlsx"))
const DATA_SHEET = get(ENV, "CRISTOOL_DATA_SHEET", "Unseeded_PE")
const DATA_LOADING = parse(Float64, get(ENV, "CRISTOOL_DATA_LOADING", "0.0"))

# These are deliberately sized for an actual inference run.  Reduce them only
# when developing the script interactively.
const OPTIMISER_PARTICLES = 512
const OPTIMISER_GENERATIONS = 256
const MCMC_SAMPLES = 2_000
const MCMC_CHAINS = 4

function fit_model(measurements, solver, aggregation, breakage, lower, upper, label)
    loss_function = logMLE(weighting = [1.0, 1.0])
    nucleation = nucl_CNT()
    growth = growth_empirical()

    println("\n--- ", label, " ---")
    optimisation = PE_Routine(loss_function, measurements, lower, upper,
                              nucleation, growth, aggregation, breakage;
                              solver = solver,
                              nparticles = OPTIMISER_PARTICLES,
                              generations = OPTIMISER_GENERATIONS,
                              verbosity = 1,
                              savetxt = false)
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
                         sampler = NUTS(1_000, 0.65;
                                        adtype = AutoForwardDiff(chunksize = 4)),
                         n_samples = MCMC_SAMPLES,
                         n_chains = MCMC_CHAINS,
                         saveplot = false,
                         showplot = false,
                         verbosity = 1)
    posterior_mean = vec(mean(chain).nt.mean)
    println("NUTS posterior mean: ", round.(posterior_mean, sigdigits = 5))
    return optimum, objective, posterior_mean
end

function main()
    Random.seed!(20260830)
    measurements = load_experiments(DATA_WORKBOOK, DATA_SHEET, DATA_LOADING)
    isempty(measurements) && error("No experiments found in $DATA_WORKBOOK [$DATA_SHEET, loading=$DATA_LOADING]")
    println("Loaded ", length(measurements), " experiments from ", DATA_WORKBOOK)

    mom_lower = [25.0, 0.30, 0.30, 2.0]
    mom_upper = [50.0, 1.00, 3.00, 4.0]
    mom_result = fit_model(measurements, MoM(), noaggregation(), nobreakage(),
                           mom_lower, mom_upper, "MoM: CNT + empirical growth")

    # QMOM uses the validated volume-additive scalar aggregation and
    # uniform-in-volume empirical breakage closures.
    qmom_lower = [25.0, 0.30, 0.30, 2.0, -10.0, 0.0, 0.5]
    qmom_upper = [50.0, 1.00, 3.00, 4.0, 5.0, 10.0, 2.0]
    qmom_result = fit_model(measurements, QMOM(nquadrature = 3),
                            aggr_scalar(), breakage_empirical(),
                            qmom_lower, qmom_upper,
                            "QMOM: growth + scalar aggregation + empirical breakage")

    println("\n=== Fit comparison ===")
    println("MoM MLE objective:  ", mom_result[2])
    println("QMOM MLE objective: ", qmom_result[2])
    println("Objective change (QMOM - MoM): ", qmom_result[2] - mom_result[2])
    println("A lower QMOM objective indicates a better fit under this loss.")
end

main()
