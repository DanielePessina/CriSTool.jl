"""
Approximate Bayesian computation samplers, the algorithms of
`solve(::EstimationProblem, ::AbstractABCSampler)`. To add a sampler, define a
new `<:AbstractABCSampler` struct and a `_runsampler` method returning the
result; `result.P` must hold the particles: KissABC `Particles`, or a
parameters × samples matrix. The runner receives
`reference_loss`, a zero-argument function evaluating the loss at the model's current
values, so samplers that do not need it never pay for the solve.
"""

# KissABC returns one `Particles` per parameter, or a bare `Particles` for a single
# parameter; posterior draws are laid out parameters × samples, which a custom
# sampler may return directly.
_abc_particles_matrix(draws::AbstractMatrix) = draws
_abc_particles_matrix(particles::MonteCarloMeasurements.AbstractParticles) =
    reshape(collect(particles.particles), 1, :)
_abc_particles_matrix(particles::AbstractVector{<:MonteCarloMeasurements.AbstractParticles}) =
    permutedims(reduce(hcat, [parameter_particles.particles for parameter_particles in particles]))

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
                γ2_burnin=0.5, kernel=GaussianABCKernel(), burnin_frac=0.3)

Turner & Sederberg (2012) ABCDE variant with K-group migration, DE crossover/
mutation, and a kernel-based proposal distribution. `kernel` is
`GaussianABCKernel()`, `LaplaceABCKernel()` or `FlatABCKernel()`.
"""
Base.@kwdef struct ABCDETurner{Kernel <: KissABC.AbstractABCKernel} <: AbstractABCSampler
    K::Int = 8
    p_migration::Float64 = 0.10
    p_crossover::Float64 = 0.90
    κ::Float64 = 1.0
    γ2_burnin::Float64 = 0.5
    kernel::Kernel = GaussianABCKernel()
    burnin_frac::Float64 = 0.3
end

"""
    AbstractABCStopping

When an ABC run ends: `AllGenerations()` (the default) runs every generation;
`StopAtTarget()` stops once every particle is within the target discrepancy.
"""
abstract type AbstractABCStopping end
"""Run every requested ABC generation."""
struct AllGenerations <: AbstractABCStopping end
"""Stop an ABC run once every particle is within the target discrepancy."""
struct StopAtTarget <: AbstractABCStopping end
# KissABC takes the stopping rule as a Bool.
_stops_at_target(::AllGenerations) = false
_stops_at_target(::StopAtTarget) = true

# Sampler-specific call into KissABC. Returns the sampler result.
function _runsampler(s::ABCDE, prior, lossfn, target, reference_loss;
                     nparticles, generations, HPC, stopping)
    return KissABC.ABCDE(prior, lossfn, target;
                 nparticles = nparticles, generations = generations,
                 α = s.α, HPC = HPC, earlystop = _stops_at_target(stopping))
end

function _runsampler(s::ABCDETurner, prior, lossfn, target, reference_loss;
                     nparticles, generations, HPC, stopping)
    optimallossfunction = reference_loss()
    δ_derived = max(target - optimallossfunction, eps())
    return KissABC.ABCDE_Turner(prior, lossfn, target;
                       nparticles = nparticles, generations = generations,
                       K = s.K, p_migration = s.p_migration, p_crossover = s.p_crossover,
                       κ = s.κ, γ2_burnin = s.γ2_burnin, kernel = s.kernel,
                       δ = δ_derived, best_cost = optimallossfunction,
                       burnin_frac = s.burnin_frac, earlystop = _stops_at_target(stopping), HPC = HPC)
end
