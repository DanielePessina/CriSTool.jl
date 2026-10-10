"""
Approximate Bayesian computation samplers used by `fit(::PreparedFit{<:Any, <:ABCSpec})`.
To add a sampler, define a new `<:AbstractABCSampler` struct and a `_runsampler`
method returning `(result, reached_target)`; `result.P` must hold the particles.
"""

function _abc_particles_matrix(particles)
    try
        samples = Matrix(particles)
        return isempty(samples) ? nothing : samples
    catch
        return nothing
    end
end

function _abc_particles_matrix(particles::MonteCarloMeasurements.AbstractParticles)
    scalar_draws = collect(particles.particles)
    return isempty(scalar_draws) ? nothing : reshape(scalar_draws, 1, :)
end

"""
    AbstractABCSampler

Abstract supertype for sampler configuration objects accepted by `ABCSpec`.
Implementations provide the algorithm-specific parameters and dispatch through
the internal sampler runner.
"""
abstract type AbstractABCSampler end

"""
    ABCDESampler(α=0)

Standard ABCDE (Differential Evolution) sampler from KissABC.
"""
Base.@kwdef struct ABCDESampler <: AbstractABCSampler
    α::Int64 = 0
end

"""
    ABCDETurnerSampler(K=8, p_migration=0.10, p_crossover=0.90, κ=1.0,
                      γ2_burnin=0.5, kernel=:gaussian, burnin_frac=0.3)

Turner & Sederberg (2012) ABCDE variant with K-group migration, DE crossover/
mutation, and a kernel-based proposal distribution.
"""
Base.@kwdef struct ABCDETurnerSampler <: AbstractABCSampler
    K::Int = 8
    p_migration::Float64 = 0.10
    p_crossover::Float64 = 0.90
    κ::Float64 = 1.0
    γ2_burnin::Float64 = 0.5
    kernel::Symbol = :gaussian
    burnin_frac::Float64 = 0.3
end

# Sampler-specific call into KissABC. Returns (res, reached_ϵ).
function _runsampler(s::ABCDESampler, prior, lossfn, target, optimallossfunction;
                     nparticles, generations, HPC, earlystop)
    res = ABCDE(prior, lossfn, target;
                nparticles = nparticles, generations = generations,
                α = s.α, HPC = HPC, earlystop = earlystop)
    return res, res.reached_ϵ
end

function _runsampler(s::ABCDETurnerSampler, prior, lossfn, target, optimallossfunction;
                     nparticles, generations, HPC, earlystop)
    δ_derived = max(target - optimallossfunction, eps())
    res = ABCDE_Turner(prior, lossfn, target;
                       nparticles = nparticles, generations = generations,
                       K = s.K, p_migration = s.p_migration, p_crossover = s.p_crossover,
                       κ = s.κ, γ2_burnin = s.γ2_burnin, kernel = s.kernel,
                       δ = δ_derived, best_cost = optimallossfunction,
                       burnin_frac = s.burnin_frac, earlystop = earlystop, HPC = HPC)
    return res, res.reached_ϵ
end
