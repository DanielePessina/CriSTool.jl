"""
Tutorial 2: Parameter estimation from experimental data.

Three steps on the same dataset:
  1. MLE point estimate via Metaheuristics (`PE_Routine`).
  2. ABCDE posterior around the MLE via the unified `run_abc` entry.
  3. Turing NUTS using the same loss function as the ABC discrepancy.

Reads `fake-experimental-dataset.xlsx` (5 unseeded experiments, generated
from `[Aj=38, γ=0.6, Ag=1, g=3]` with 3% / 8% heteroscedastic noise).
Swap `DATA_WORKBOOK` for your own .xlsx to fit real data; the workbook
just needs the columns expected by `makerepeatmeasurements`.
"""

using CriSTool
using Metaheuristics
using Turing, Distributions
using Random

const DATA_WORKBOOK = joinpath(@__DIR__, "fake-experimental-dataset.xlsx")

# Turing model at module scope (Turing requires it). Triangular-prior +
# loss-as-likelihood pattern, same as Tutorial 5.
@model function nuts_model(data, lb, ub, optpara, nucl, gr, agg, br, solver, loss)
    Aj ~ TriangularDist(lb[1], ub[1], optpara[1])
    γ  ~ TriangularDist(lb[2], ub[2], optpara[2])
    Ag ~ TriangularDist(lb[3], ub[3], optpara[3])
    g  ~ TriangularDist(lb[4], ub[4], optpara[4])
    L = CriSTool.parameterestimation_lossfunction(loss, data, [Aj, γ, Ag, g],
                                                   nucl, gr, agg, br, solver)
    Turing.@addlogprob!(-L)
end

function main()
    Random.seed!(11)

    # 1. Load + variance-balance the experimental data. The balancer rescales
    #    concentration / PSD variance by the supplied factors so the loss
    #    function weights the two observation types more comparably.
    raw          = CriSTool.makerepeatmeasurements(DATA_WORKBOOK, "Unseeded_PE", [0.0])
    measurements = CriSTool.psd_measurementbalancer(
                    CriSTool.repeatmeasurementbalancer(raw, 3), 4)

    # Forward model + bounds. Use the same kinetic shape that generated the
    # fake dataset so the MLE should land near the truth.
    nucl, gr  = nucl_CNT(), growth_empirical()
    agg, br   = noaggregation(), nobreakage()
    solver    = MoM()
    loss      = logMLE(weighting = (1.0, 1.0))
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
    prior = Factored([TriangularDist(lb[i], ub[i], optimal[i])
                       for i in eachindex(optimal)]...)
    _, abc_meta = run_abc(loss, measurements, optimal, prior, nucl, gr, agg, br;
                           solver = solver,
                           sampler = ABCDESampler(α = 0),
                           extrastring = "ABCDE tutorial",
                           nparticles = 256, generations = 200,
                           confidenceinterval = 0.9,
                           saveplot = false, verbosity = 0)
    println("ABCDE posterior mean: ", round.(abc_meta["meanparameters"], digits = 3))

    # 4. NUTS using the same loss.
    chain = Turing.sample(nuts_model(measurements, lb, ub, optimal,
                                      nucl, gr, agg, br, solver, loss),
                           NUTS(50, 0.65; adtype = AutoForwardDiff(chunksize = 4)),
                           100; progress = false)
    nuts_means = [mean(chain[:Aj]), mean(chain[:γ]), mean(chain[:Ag]), mean(chain[:g])]
    println("NUTS posterior mean:  ", round.(nuts_means, digits = 3))
end

main()
