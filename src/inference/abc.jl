"""
Approximate Bayesian computation samplers, the algorithms of
`solve(::EstimationProblem, ::AbstractABCSampler)`. To add a sampler, define a new `<:AbstractABCSampler` struct and a `_runsampler`
method returning the result; `result.P` must hold the particles. The runner receives
`reference_loss`, a zero-argument function evaluating the loss at the model's current
values, so samplers that do not need it never pay for the solve.
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

Abstract supertype of the approximate Bayesian computation algorithms that
`solve(::EstimationProblem, sampler; target, ...)` accepts. Implementations hold
the algorithm-specific parameters and dispatch through the internal sampler runner.
"""
abstract type AbstractABCSampler end

"""
    ABCDE(; α=0)

Standard ABCDE (Differential Evolution) sampler from KissABC.
"""
Base.@kwdef struct ABCDE <: AbstractABCSampler
    α::Int64 = 0
end

"""
    ABCDETurner(; K=8, p_migration=0.10, p_crossover=0.90, κ=1.0,
                γ2_burnin=0.5, kernel=:gaussian, burnin_frac=0.3)

Turner & Sederberg (2012) ABCDE variant with K-group migration, DE crossover/
mutation, and a kernel-based proposal distribution.
"""
Base.@kwdef struct ABCDETurner <: AbstractABCSampler
    K::Int = 8
    p_migration::Float64 = 0.10
    p_crossover::Float64 = 0.90
    κ::Float64 = 1.0
    γ2_burnin::Float64 = 0.5
    kernel::Symbol = :gaussian
    burnin_frac::Float64 = 0.3
end

# Sampler-specific call into KissABC. Returns the sampler result.
function _runsampler(s::ABCDE, prior, lossfn, target, reference_loss;
                     nparticles, generations, HPC, earlystop)
    return KissABC.ABCDE(prior, lossfn, target;
                 nparticles = nparticles, generations = generations,
                 α = s.α, HPC = HPC, earlystop = earlystop)
end

function _runsampler(s::ABCDETurner, prior, lossfn, target, reference_loss;
                     nparticles, generations, HPC, earlystop)
    optimallossfunction = reference_loss()
    δ_derived = max(target - optimallossfunction, eps())
    return KissABC.ABCDE_Turner(prior, lossfn, target;
                       nparticles = nparticles, generations = generations,
                       K = s.K, p_migration = s.p_migration, p_crossover = s.p_crossover,
                       κ = s.κ, γ2_burnin = s.γ2_burnin, kernel = s.kernel,
                       δ = δ_derived, best_cost = optimallossfunction,
                       burnin_frac = s.burnin_frac, earlystop = earlystop, HPC = HPC)
end
