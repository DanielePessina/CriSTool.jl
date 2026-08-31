"""
Tutorial 2: Parameter estimation from experimental data.

Three steps on the same dataset:
  1. MLE point estimate via Metaheuristics (`PE_Routine`).
  2. ABCDE posterior around the MLE via the unified `run_abc` entry.
  3. Turing NUTS using the same loss function as the ABC discrepancy, via
     the `nuts_model` builder (triangular priors on the MLE).

Reads `fake-experimental-dataset.csv` (5 unseeded experiments, generated
from `[Aj=38, γ=0.6, Ag=1, g=3]` with 3% / 8% heteroscedastic noise).
Swap `DATA_FILE` for your own .csv to fit real data; the file
just needs the columns expected by `load_experiments`.
"""

using CriSTool
using Metaheuristics
using Turing, Distributions
using Random

const DATA_FILE = joinpath(@__DIR__, "fake-experimental-dataset.csv")

function main()
    Random.seed!(11)

    # 1. Load + variance-balance the experimental data. The balancer rescales
    #    concentration / PSD variance by the supplied factors so the loss
    #    function weights the two observation types more comparably.
    raw          = load_experiments(DATA_FILE)
    measurements = CriSTool.psd_measurementbalancer(
                    CriSTool.repeatmeasurementbalancer(raw, 3), 4)

    # Forward model + bounds. Use the same kinetic shape that generated the
    # fake dataset so the MLE should land near the truth.
    nucl, gr  = nucl_CNT(), growth_empirical()
    agg, br   = noaggregation(), nobreakage()
    solver    = MoM()
    loss      = logMLE(weighting = [1.0, 1.0])
    lb        = [25.0, 0.30, 0.30, 2.0]   # [Aj, γ, Ag, g]
    ub        = [50.0, 1.00, 3.00, 4.0]

    # 2. MLE via Metaheuristics differential evolution.
    optres = PE_Routine(loss, measurements, lb, ub, nucl, gr, agg, br;
                         solver = solver,
                         extrastring = "PE tutorial",
                         nparticles = 128, generations = 128,
                         verbosity = 0, savetxt = false)
    optimal = minimizer(optres)
    println("MLE optimum: ", round.(optimal, digits = 3))

    # 3. ABCDE posterior around the MLE.
    prior = product_distribution([TriangularDist(lb[i], ub[i], optimal[i])
                       for i in eachindex(optimal)]...)
    _, abc_meta = run_abc(loss, measurements, optimal, prior, nucl, gr, agg, br;
                           solver = solver,
                           sampler = ABCDESampler(α = 0),
                           extrastring = "ABCDE tutorial",
                           nparticles = 256, generations = 200,
                           confidenceinterval = 0.9,
                           saveplot = false, verbosity = 0)
    println("ABCDE posterior mean: ", round.(abc_meta["meanparameters"], digits = 3))

    # 4. NUTS using the same loss. `nuts_model` builds the Turing model
    #    (one prior per parameter, triangular on the MLE); `rename_chain`
    #    gives the chain the kinetic parameter names.
    prior_list = [TriangularDist(lb[i], ub[i], optimal[i]) for i in eachindex(optimal)]
    model = nuts_model(measurements, prior_list, nucl, gr, agg, br;
                       solver = solver, lossfunction = loss)
    chain = rename_chain(Turing.sample(model,
                                       NUTS(50, 0.65; adtype = AutoForwardDiff(chunksize = 4)),
                                       100; progress = false),
                         kinetic_parameter_symbols(nucl, gr, agg, br))
    nuts_means = [mean(chain[:Aⱼ]), mean(chain[:γ]), mean(chain[:Ag]), mean(chain[:g])]
    println("NUTS posterior mean:  ", round.(nuts_means, digits = 3))
end

main()
