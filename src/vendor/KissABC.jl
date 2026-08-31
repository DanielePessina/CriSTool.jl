"""
KissABC submodule - Approximate Bayesian Computation algorithms.

Provides SMC and ABCDE samplers for likelihood-free inference.
Support code for ABC sampling and related distributions. Defines mixed priors,
kernels and implementation details used by the ABCDE algorithm.

This is a vendored, locally maintained fork retained by CriSTool. Its source
was imported into the project history in commit `cf84575`; no registry package
version is assumed. The repository's GPL-3.0 license applies to this vendored
copy; see the root `LICENSE` file.

!!! warning "Deprecation Notice"

"""
module KissABC

import AbstractMCMC
import AbstractMCMC: sample, step, MCMCThreads, MCMCDistributed
using Random
using Distributions
using MonteCarloMeasurements
using ProgressMeter
import Base.length

# Public exports
export ABCDE, smc, AIS
export ApproxPosterior, ApproxKernelizedPosterior, CommonLogDensity

## priors.jl

import Distributions.pdf, Distributions.logpdf, Random.rand, Base.length

struct MixedSupport <: ValueSupport end




## types.jl
abstract type AbstractDensity <: AbstractMCMC.AbstractModel end
abstract type AbstractApproxPosterior <: AbstractDensity end
#=
unconditional_sample(rng::AbstractRNG,density::AbstractDensity) = error("define a method to cover unconditional_sample(rng::AbstractRNG,density::"*repr(typeof(density))*").")
loglike(density::AbstractDensity,sample) = error("define a method to cover logpdf(density::"*repr(typeof(density))*",sample). must return a named tuple (logprior = ?, loglikelihood = ?)")
length(density::AbstractDensity) = error("define a method to cover length(density::"*repr(typeof(density))*").")
accept(density::AbstractDensity,rng::AbstractRNG,old_ld,new_ld) = error("define a method to cover accept(density::"*repr(typeof(density))*",rng::AbstractRNG,old_ld,new_ld). must return boolean to accept or reject a transition from old_ld → new_ld")
=#

struct Particle{Xt}
    x::Xt
    Particle(x::T) where {T} = new{T}(x)
end

op(f, a::Particle, b::Particle) = Particle(op(f, a.x, b.x))
op(f, a::Particle, b::Number) = Particle(op(f, a.x, b))
op(f, a::Number, b::Particle) = Particle(op(f, a, b.x))
op(f, a::Number, b::Number) = f(a, b)
op(f, a, b) = op.(Ref(f), a, b)

op(f, a::Particle) = Particle(op(f, a.x))
op(f, a::Number) = f(a)
op(f, a) = op.(Ref(f), a)

op(f, args...) = foldl((x, y) -> op(f, x, y), args)

push_p(density::AbstractDensity, p::Particle) = p
push_p(density::AbstractApproxPosterior, p::Particle) = Particle(push_p(density.prior, p.x))
push_p(density::Distribution, p) = push_p.(Ref(density), p)
push_p(density::ContinuousDistribution, p::Number) = float(p)
push_p(density::DiscreteDistribution, p::Number) = round(Int, p)

function unconditional_sample(rng::AbstractRNG, density::AbstractApproxPosterior)
    Particle(rand(rng, density.prior))
end

length(density::AbstractApproxPosterior) = length(density.prior)


struct ApproxKernelizedPosterior{P <: Distribution, C, S <: Real} <: AbstractApproxPosterior
    prior::P
    cost::C
    scale::S
    function ApproxKernelizedPosterior(prior::T1,
                                       cost::T2,
                                       target_average_cost::T3) where {T1, T2, T3}
        new{T1, T2, T3}(prior, cost, target_average_cost)
    end
end

function loglike(density::ApproxKernelizedPosterior, sample::Particle)
    lp = logpdf(density.prior, sample.x)
    ll = lp
    if isfinite(lp)
        ll = -0.5 * abs2(density.cost(sample.x) / density.scale)
    end
    (logprior = lp, loglikelihood = ll)
end

is_valid_logdensity(density::ApproxKernelizedPosterior, ld) = isfinite(sum(ld))

function accept(density::ApproxKernelizedPosterior,
                rng::AbstractRNG,
                old_ld,
                new_ld,
                ld_correction)
    isfinite(ld_correction) || error("ld_correction is invalid")
    is_valid_logdensity(density, old_ld) || error("starting sample invalid.")
    is_valid_logdensity(density, new_ld) || return false

    lW = ld_correction + sum(new_ld) - sum(old_ld)
    return -randexp(rng) <= lW
end
struct ApproxPosterior{P <: Distribution, C, S <: Real} <: AbstractApproxPosterior
    prior::P
    cost::C
    maxcost::S
    function ApproxPosterior(prior::T1, cost::T2, max_cost::T3) where {T1, T2, T3}
        new{T1, T2, T3}(prior, cost, max_cost)
    end
end

function loglike(density::ApproxPosterior, sample::Particle)
    lp = logpdf(density.prior, sample.x)
    cs = -lp
    if isfinite(lp)
        cs = density.cost(sample.x)
    end
    (logprior = lp, cost = cs)
end

function is_valid_logdensity(density::ApproxPosterior, ld)
    isfinite(ld.cost) && isfinite(ld.logprior)
end

function accept(density::ApproxPosterior, rng::AbstractRNG, old_ld, new_ld, ld_correction)
    isfinite(ld_correction) || error("ld_correction is invalid")
    is_valid_logdensity(density, old_ld) || error("starting sample invalid.")
    is_valid_logdensity(density, new_ld) || return false

    lW = ld_correction + new_ld.logprior - old_ld.logprior
    lW2 = max(density.maxcost, old_ld.cost) - new_ld.cost
    (-randexp(rng) <= lW) && lW2 >= 0
end
struct CommonLogDensity{N, A, B} <: AbstractDensity
    sample_init::A
    lπ::B
    function CommonLogDensity(nparameters::Int, sample_init::A, lπ::B) where {A, B}
        new{nparameters, A, B}(sample_init, lπ)
    end
end

function unconditional_sample(rng::AbstractRNG, density::CommonLogDensity)
    Particle(density.sample_init(rng))
end

length(density::CommonLogDensity{N}) where {N} = N

function loglike(density::CommonLogDensity, sample::Particle)
    density.lπ(sample.x)
end

is_valid_logdensity(density::CommonLogDensity, ld) = isfinite(ld)

function accept(density::CommonLogDensity, rng::AbstractRNG, old_ld, new_ld, ld_correction)
    isfinite(ld_correction) || error("ld_correction is invalid")
    is_valid_logdensity(density, old_ld) || error("starting sample invalid.")
    is_valid_logdensity(density, new_ld) || return false
    return -randexp(rng) <= ld_correction + new_ld - old_ld
end

"""
    ApproxKernelizedPosterior(
        prior::Distribution,
        cost::Function,
        target_average_cost::Real
    )
this function will return a type which can be used in the `sample` function as an ABC density,
this type works by assuming Gaussianly distributed errors 𝒩(0,ϵ), ϵ is specified in the variable `target_average_cost`.
"""
ApproxKernelizedPosterior
"""
    ApproxPosterior(
        prior::Distribution,
        cost::Function,
        max_cost::Real
    )
this function will return a type which can be used in the `sample` function as an ABC density,
this type works by assuming uniformly distributed errors in [-ϵ,ϵ], ϵ is specified in the variable `max_cost`.
"""
ApproxPosterior

"""
    CommonLogDensity(nparameters, sample_init, lπ)
this function will return a type for performing classical MCMC via the `sample` function.

`nparameters`: total number of parameters per sample.

`sample_init`: function which accepts an `RNG::AbstractRNG` and returns a sample for `lπ`.

`lπ`: function which accepts a sample, and returns a log-density float value.
"""
CommonLogDensity

# export CommonLogDensity, ApproxPosterior, ApproxKernelizedPosterior


#### ABCDE_Turner helpers (Turner & Sederberg, 2012)

"""
    logsumexp_stable(logws::AbstractVector)

Compute log(sum(exp.(logws))) in a numerically stable way.
"""
function logsumexp_stable(logws::AbstractVector)
    m = maximum(logws)
    isfinite(m) || return m
    m + log(sum(exp.(logws .- m)))
end

"""
    sample_from_logweights(rng::AbstractRNG, logws::AbstractVector)

Sample an index from a vector of log-weights using inverse transform sampling.
"""
function sample_from_logweights(rng::AbstractRNG, logws::AbstractVector)
    lse = logsumexp_stable(logws)
    probs = exp.(logws .- lse)
    cumprobs = cumsum(probs)
    u = rand(rng)
    idx = findfirst(x -> x >= u, cumprobs)
    return idx === nothing ? length(logws) : idx
end

"""
    log_kernel_weight(cost, best_cost, δ; kernel=:gaussian)

Compute log ψ(ρ|δ) where ρ = cost - best_cost.
Supported kernels: :gaussian, :laplace, :none.
"""
function log_kernel_weight(cost::Real, best_cost::Real, δ::Real; kernel::Symbol = :gaussian)
    d = cost - best_cost
    if kernel == :gaussian
        return -0.5 * (d / δ)^2
    elseif kernel == :laplace
        return -abs(d) / δ
    else  # :none
        return 0.0
    end
end

"""
    group_range(k::Int, G::Int)

Return the index range for group k (1-indexed) where each group has size G.
"""
group_range(k::Int, G::Int) = ((k - 1) * G + 1):(k * G)

"""
    ABCDE_Turner(prior, cost, ϵ_target; kwargs...)

Turner & Sederberg (2012) ABCDE algorithm with K groups, DE crossover,
mutation, and migration steps.

# Arguments
- `prior`: Distribution for parameters
- `cost`: Function θ -> scalar cost (smaller is better)
- `ϵ_target`: Target threshold for convergence

# Keyword Arguments
- `nparticles::Int=256`: Total particles (will be adjusted if not divisible by K)
- `generations::Int=150`: Number of iterations
- `K::Int=8`: Number of groups
- `p_migration::Float64=0.10`: Migration probability (α)
- `p_crossover::Float64=0.90`: Crossover probability (vs mutation)
- `κ::Float64=1.0`: Crossover component probability
- `γ1_range::Tuple=(0.5, 1.0)`: Range for γ1 sampling
- `γ2_burnin::Float64=0.5`: γ2 value during burn-in (0 in sampling mode)
- `b_eps::Float64=1e-3`: Additive noise magnitude
- `kernel::Symbol=:gaussian`: Kernel type (:gaussian, :laplace, :none)
- `δ::Union{Nothing,Float64}=nothing`: Kernel scale (derived from ϵ_target if nothing)
- `best_cost::Union{Nothing,Float64}=nothing`: Reference cost for kernel
- `burnin_frac::Float64=0.3`: Fraction of generations for burn-in mode
- `earlystop::Bool=false`: Stop early if all particles below ϵ_target
- `verbose::Bool=false`: Print progress info
- `rng::AbstractRNG=Random.Xoshiro()`: Random number generator
- `HPC::Bool=false`: Suppress progress bar
"""
function ABCDE_Turner(prior, cost, ϵ_target;
                      nparticles::Int = 256,
                      generations::Int = 150,
                      K::Int = 8,
                      p_migration::Float64 = 0.10,
                      p_crossover::Float64 = 0.90,
                      κ::Float64 = 1.0,
                      γ1_range::Tuple{Float64, Float64} = (0.5, 1.0),
                      γ2_burnin::Float64 = 0.5,
                      b_eps::Float64 = 1e-3,
                      kernel::Symbol = :gaussian,
                      δ::Union{Nothing, Float64} = nothing,
                      best_cost::Union{Nothing, Float64} = nothing,
                      burnin_frac::Float64 = 0.3,
                      earlystop::Bool = false,
                      verbose::Bool = false,
                      rng = Random.Xoshiro(),
                      HPC::Bool = false)
    # Validate and adjust nparticles for K divisibility
    G = nparticles ÷ K
    if G * K != nparticles
        old_nparticles = nparticles
        G = ceil(Int, nparticles / K)
        nparticles = G * K
        @info "ABCDE_Turner: Adjusted nparticles from $old_nparticles to $nparticles for K=$K groups"
    end
    @assert G>=3 "Group size G=$G must be >= 3 for DE crossover"

    # Derive δ from ϵ_target if not provided
    δ_used = δ === nothing ? max(ϵ_target * 0.5, eps()) : δ
    best_cost_used = best_cost === nothing ? 0.0 : best_cost

    # Initialize particles
    θs = [op(float, Particle(rand(rng, prior))) for _ in 1:nparticles]
    costs = fill(Inf, nparticles)
    logπ = fill(-Inf, nparticles)
    logψ = fill(-Inf, nparticles)

    # Initial evaluation
    for i in 1:nparticles
        logπ[i] = logpdf(prior, push_p(prior, θs[i].x))
        if isfinite(logπ[i])
            costs[i] = cost(θs[i].x)
        end
        # Retry invalid particles
        while !isfinite(costs[i]) || !isfinite(logπ[i])
            θs[i] = op(float, Particle(rand(rng, prior)))
            logπ[i] = logpdf(prior, push_p(prior, θs[i].x))
            if isfinite(logπ[i])
                costs[i] = cost(θs[i].x)
            end
        end
        logψ[i] = log_kernel_weight(costs[i], best_cost_used, δ_used; kernel = kernel)
    end

    # Combined log-weights
    logw = logπ .+ logψ

    # Burn-in vs sampling mode
    burnin_gens = ceil(Int, burnin_frac * generations)

    # Progress bar
    prog = Progress(generations, dt = 1.0, desc = "ABCDE_Turner: ", barlen = 25,
                    showspeed = true, color = :light_cyan, enabled = !HPC)

    # Mutation scale from prior (heuristic: use prior width / 10)
    nparams = length(prior)
    σ_mut = fill(0.1, nparams)  # default

    # Early stopping
    stop_after_iter = nothing

    for iter in 1:generations
        # Check early stop
        if earlystop && stop_after_iter !== nothing && iter > stop_after_iter
            break
        end

        # Determine mode: burn-in or sampling
        is_burnin = iter <= burnin_gens
        γ2 = is_burnin ? γ2_burnin : 0.0

        # Update δ at end of burn-in
        if iter == burnin_gens + 1 && δ === nothing
            δ_used = max(minimum(costs) - best_cost_used, eps())
            # Recompute logψ and logw with new δ
            for i in 1:nparticles
                logψ[i] = log_kernel_weight(costs[i], best_cost_used, δ_used;
                                            kernel = kernel)
            end
            logw .= logπ .+ logψ
        end

        # Migration step (with probability p_migration)
        if rand(rng) < p_migration && K > 1
            η = rand(rng, 1:K)  # number of groups to swap
            groups_to_swap = shuffle(rng, 1:K)[1:η]

            if η >= 2
                # Select one particle per group (inverse-weight sampling = poorly performing)
                selected_indices = Int[]
                for gk in groups_to_swap
                    gr = group_range(gk, G)
                    inv_logws = -logw[gr]  # inverse weights
                    local_idx = sample_from_logweights(rng, inv_logws)
                    push!(selected_indices, gr[local_idx])
                end

                # Cyclic swap: θ[G1] ← θ[Gη], θ[G2] ← θ[G1], etc.
                temp_θ = θs[selected_indices[end]]
                temp_cost = costs[selected_indices[end]]
                temp_logπ = logπ[selected_indices[end]]
                temp_logψ = logψ[selected_indices[end]]
                temp_logw = logw[selected_indices[end]]

                for j in length(selected_indices):-1:2
                    idx_to = selected_indices[j]
                    idx_from = selected_indices[j - 1]
                    θs[idx_to] = θs[idx_from]
                    costs[idx_to] = costs[idx_from]
                    logπ[idx_to] = logπ[idx_from]
                    logψ[idx_to] = logψ[idx_from]
                    logw[idx_to] = logw[idx_from]
                end

                θs[selected_indices[1]] = temp_θ
                costs[selected_indices[1]] = temp_cost
                logπ[selected_indices[1]] = temp_logπ
                logψ[selected_indices[1]] = temp_logψ
                logw[selected_indices[1]] = temp_logw
            end
        end

        # Process each group
        for k in 1:K
            gr = group_range(k, G)
            use_crossover = rand(rng) < p_crossover

            for t in gr
                # Early stop check
                if earlystop && costs[t] <= ϵ_target
                    continue
                end

                if use_crossover
                    # DE Crossover (Eq. 4)
                    # Sample base with weights proportional to exp(logw)
                    b_idx = gr[sample_from_logweights(rng, logw[gr])]

                    # Sample m, n uniformly from group, m ≠ n ≠ t
                    m_idx = t
                    while m_idx == t
                        m_idx = rand(rng, gr)
                    end
                    n_idx = t
                    while n_idx == t || n_idx == m_idx
                        n_idx = rand(rng, gr)
                    end

                    # Sample γ1 from range
                    γ1 = γ1_range[1] + rand(rng) * (γ1_range[2] - γ1_range[1])

                    # Sample additive noise b
                    b_noise = op(x -> (2 * rand(rng) - 1) * b_eps, θs[t].x)

                    # Proposal: θ* = θ_t + γ1*(θ_m - θ_n) + γ2*(θ_b - θ_t) + b
                    diff1 = op(-, θs[m_idx], θs[n_idx])
                    θp = op(+, θs[t], op(*, diff1, γ1))

                    if γ2 > 0
                        diff2 = op(-, θs[b_idx], θs[t])
                        θp = op(+, θp, op(*, diff2, γ2))
                    end
                    θp = op(+, θp, Particle(b_noise))

                    # Apply κ mask (reset some components with prob 1-κ)
                    if κ < 1.0
                        mask = Tuple(rand(rng) < κ for _ in 1:nparams)
                        θp_x = ntuple(j -> mask[j] ? θp.x[j] : θs[t].x[j], nparams)
                        θp = Particle(θp_x)
                    end
                else
                    # Mutation step (simple random walk)
                    perturbation = ntuple(j -> randn(rng) * σ_mut[j], nparams)
                    θp = op(+, θs[t], Particle(perturbation))
                end

                # Evaluate proposal
                lπ_new = logpdf(prior, push_p(prior, θp.x))
                if !isfinite(lπ_new)
                    continue  # reject immediately
                end

                cost_new = cost(θp.x)
                if !isfinite(cost_new)
                    continue
                end

                lψ_new = log_kernel_weight(cost_new, best_cost_used, δ_used;
                                           kernel = kernel)
                logpost_new = lπ_new + lψ_new
                logpost_old = logw[t]

                # MH acceptance (Eq. 6 - simplified, ignoring transition kernel)
                loga = min(0.0, logpost_new - logpost_old)
                if log(rand(rng)) < loga
                    θs[t] = θp
                    costs[t] = cost_new
                    logπ[t] = lπ_new
                    logψ[t] = lψ_new
                    logw[t] = logpost_new
                end
            end
        end

        # Update progress
        complete = 1 - sum(costs .> ϵ_target) / nparticles
        if earlystop && stop_after_iter === nothing && complete >= 1.0
            stop_after_iter = iter + min(20, generations - iter)
        end

        next!(prog,
              showvalues = [
                  (:mode, is_burnin ? "burn-in" : "sampling"),
                  (:completion, round(complete, digits = 4)),
                  (:δ, round(δ_used, sigdigits = 4)),
                  (:range_cost, round.(extrema(costs), digits = 4))
              ])
    end

    # Format output
    conv = maximum(costs) <= ϵ_target
    kept_indices = conv ? collect(1:nparticles) : findall(costs .<= ϵ_target)
    if conv == false
        @info "ABCDE_Turner: filtered samples" remaining = length(kept_indices) total = nparticles
    end
    isempty(kept_indices) && @warn("ABCDE_Turner: no samples reached target; returning empty particles")

    θs_final = [push_p(prior, θs[i].x) for i in kept_indices]
    costs_final = costs[kept_indices]

    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs_final, i) for i in 1:l)
    length(P) == 1 && (P = first(P))

    return (P = P, C = Particles(costs_final), reached_ϵ = conv,
            meta = (K = K, δ = δ_used, kernel = kernel, burnin_gens = burnin_gens))
end

export ABCDE_Turner, log_kernel_weight, logsumexp_stable


#### transition.jl


function de_propose(rng::AbstractRNG, density, particles::AbstractVector, i::Int)
    γ = 2.38 / sqrt(2 * length(density)) * exp(randn(rng) * 0.1)
    a = b = i
    while a ∈ (i,)
        a = rand(rng, eachindex(particles))
    end
    while b ∈ (a, i)
        b = rand(rng, eachindex(particles))
    end
    W = op(*, op(-, particles[a], particles[b]), γ)
    T = op(x -> γ * x / 300 * randn(rng),
           op(+,
              op(abs, op(-, particles[a], particles[b])),
              op(abs, op(-, particles[i], particles[b])),
              op(abs, op(-, particles[a], particles[i]))))
    op(+, particles[i], W, T), 0.0
end

function ais_walk_propose(rng::AbstractRNG, density, particles::AbstractVector, i::Int)
    a = b = c = i
    while a ∈ (i,)
        a = rand(rng, eachindex(particles))
    end
    while b ∈ (a, i)
        b = rand(rng, eachindex(particles))
    end
    while c ∈ (b, a, i)
        c = rand(rng, eachindex(particles))
    end
    Xs = op(/, op(+, particles[a], op(+, particles[b], particles[c])), 3)
    W = op(+,
           op(*, randn(rng), op(-, particles[a], Xs)),
           op(*, randn(rng), op(-, particles[b], Xs)),
           op(*, randn(rng), op(-, particles[c], Xs)))
    op(+, particles[i], W), 0.0
end

"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
cdf_g_inv(u, a) = (u * (sqrt(a) - sqrt(1 / a)) + sqrt(1 / a))^2

"Sample from g using inverse transform sampling.  a=2.0 is recommended."
sample_g(rng::AbstractRNG, a) = cdf_g_inv(rand(rng), a)

function stretch_propose(rng::AbstractRNG, density, particles::AbstractVector, i::Int)
    a = i
    while i == a
        a = rand(rng, eachindex(particles))
    end
    Z = sample_g(rng, 3.0)
    W = op(*, op(-, particles[i], particles[a]), Z)
    op(+, particles[a], W), (length(density) - 1) * log(Z)
end

function propose(rng::AbstractRNG, density, particles::AbstractVector, i::Int)
    p = rand(rng, (1, 1, 1, 1, 2, 2, 3))
    pr = (stretch_propose, de_propose, ais_walk_propose)
    pr[p](rng, density, particles, i)
end

function transition!(density::AbstractDensity,
                     particles::AbstractVector,
                     logdensity::AbstractVector,
                     particle_index::Int,
                     rng::AbstractRNG)
    p, ld_correction = propose(rng, density, particles, particle_index)
    ld = loglike(density, push_p(density, p))
    if accept(density, rng, logdensity[particle_index], ld, ld_correction)
        particles[particle_index] = p
        logdensity[particle_index] = ld
        return true
    end
    return false
end



### smc.jl

# macro cthreads(condition::Symbol, loop) #does not work well because of #15276, but seems to work on Julia v0.7
#     return esc(quote
#         if $condition
#             Threads.@threads $loop
#         else
#             $loop
#         end
#     end)
# end

function ess(w)
    sum(w)^2 / sum(abs2, w)
end

function resample_residual(w::AbstractVector{<:Real}, num_particles::Integer) # taken from Turing.jl
    # Pre-allocate array for resampled particles
    indices = Vector{Int}(undef, num_particles)

    # deterministic assignment
    residuals = similar(w)
    i = 1
    @inbounds for j in 1:length(w)
        x = num_particles * w[j]
        floor_x = floor(Int, x)
        for k in 1:floor_x
            indices[i] = j
            i += 1
        end
        residuals[j] = x - floor_x
    end

    # sampling from residuals
    if i <= num_particles
        residuals ./= sum(residuals)
        rand!(Categorical(residuals), view(indices, i:num_particles))
    end

    return indices
end

"""
Adaptive SMC from P. Del Moral 2012, with Affine invariant proposal mechanism, faster that `AIS` for `ABC` targets.
```julia
function smc(
    prior::Distribution,
    cost::Function;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    nparticles::Int = 100,
    M::Int = 1,
    alpha = 0.95,
    mcmc_retrys::Int = 0,
    mcmc_tol = 0.015,
    epstol = 0.0,
    r_epstol = (1 - alpha)^1.5 / 50,
    min_r_ess = alpha^2,
    max_stretch = 2.0,
    verbose::Bool = false,
    parallel::Bool = false,
)
```

- `prior`: a Distribution object representing the parameters prior.
- `cost`: a function that given a prior sample returns the cost for said sample (e.g. a distance between simulated data and target data).
- `rng`: an AbstractRNG object which is used by SMC for inference (it can be useful to make an inference reproducible).
- `nparticles`: number of total particles to use for inference.
- `M`: number of cost evaluations per particle, increasing this can reduce the chance of rejecting good particles.
- `alpha` - used for adaptive tolerance, by solving `ESS(n,ϵ(n)) = α ESS(n-1, ϵ(n-1))` for `ϵ(n)` at step `n`.
- `mcmc_retrys` - if set > 0, whenever the fraction of accepted particles drops below the tolerance `mcmc_tol` the MCMC step is repeated (no more than `mcmc_retrys` times).
- `mcmc_tol` - stopping condition for SMC, if the fraction of accepted particles drops below `mcmc_tol` the algorithm terminates.
- `epstol` - stopping condition for SMC, if the adaptive cost threshold drops below `epstol` the algorithm has converged and thus it terminates.
- `min_r_ess` - whenever the fractional effective sample size drops below `min_r_ess`, a systematic resampling step is performed.
- `max_stretch` - the proposal distribution of `smc` is the stretch move of Foreman-Mackey et al. 2013, the larger the parameters the wider becomes the proposal distribution.
- `verbose` - if set to `true`, enables verbosity.
- `parallel` - if set to `true`, threaded parallelism is enabled, keep in mind that the cost function must be Thread-safe in such case.

# Example

```Julia
using KissABC
prior=product_distribution([Normal(0,5), Normal(0,5)])
cost((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2
results = smc(prior, cost, alpha=0.5, nparticles=5000).P
```

output:
```TTY
2-element Array{Particles{Float64,5000},1}:
 1.0 ± 0.029
 0.999 ± 0.012
```
"""
function smc(prior::Tprior,
             cost;
             rng::AbstractRNG = Random.GLOBAL_RNG,
             nparticles::Int = 100,
             alpha = 0.95,
             mcmc_retrys::Int = 0,
             mcmc_tol = 0.015,
             epstol = 0.0,
             r_epstol = (1 - alpha)^1.5 / 50,
             min_r_ess = alpha^2,
             max_stretch = 2.0,
             verbose::Bool = false,
             parallel::Bool = true,) where {Tprior <: Distribution}
    min_r_ess > 0 || error("min_r_ess must be > 0.")
    mcmc_retrys >= 0 || error("mcmc_retrys must be >= 0.")
    alpha > 0 || error("alpha must be > 0.")
    r_epstol >= 0 || error("r_epstol must be >= 0")
    mcmc_tol >= 0 || error("mcmc_tol must be >= 0")
    max_stretch > 1 || error("max_stretch must be > 1")
    Np = length(prior)
    min_nparticles = ceil(Int,
                          3 * Np / (min(alpha, min_r_ess)))
    nparticles >= min_nparticles || error("nparticles must be >= $min_nparticles.")
    θs = [op(float, Particle(rand(rng, prior))) for i in 1:nparticles]
    Xs = Vector{Float64}(undef, nparticles)
    if parallel
        Threads.@threads :static for i in 1:nparticles
            Xs[i] = fetch(cost(push_p(prior, θs[i].x)))
        end
    else
        for i in 1:nparticles
            Xs[i] = fetch(cost(push_p(prior, θs[i].x)))
        end
    end

    lπs = [logpdf(prior, push_p(prior, θs[i].x)) for i in 1:nparticles]
    α = alpha
    ϵ = Inf
    alive = fill(true, nparticles)
    iteration = 0
    # Step 1 - adaptive threshold
    while true
        iteration += 1
        ϵv = ϵ
        ϵ = quantile(Xs[alive], α)
        flag = false
        if ϵ > minimum(Xs[alive])
            alive = Xs .< ϵ
        else
            alive = Xs .<= ϵ
            flag = true
        end
        ESS = sum(alive)
        verbose && @show iteration, ϵ, ESS
        # Step 2 - Resampling
        if α * ESS <= nparticles * min_r_ess
            idxalive = (1:nparticles)[alive]
            idx = repeat(idxalive, ceil(Int, nparticles / length(idxalive)))[1:nparticles]
            θs = θs[idx]
            Xs = Xs[idx]
            lπs = lπs[idx]
            ESS = nparticles
            alive .= true
        end

        # Step 3 - MCMC
        accepted = Base.Threads.Atomic{Int}(0)#parallel ? Threads.Atomic{Int}(0) : 0
        retry_N = 1 + mcmc_retrys

        for r in 1:retry_N
            new_p = map(1:nparticles) do i
                a = b = i
                alive[i] || return (nothing, nothing, nothing)
                while a == i
                    a = rand(rng, 1:nparticles)
                end
                while b == i || b == a
                    b = rand(rng, 1:nparticles)
                end
                W = op(*, op(-, θs[b], θs[a]), max_stretch * randn(rng) / sqrt(Np))
                (log(rand(rng)), op(+, θs[i], W), 0.0)
            end
            if parallel
                Threads.@threads :static for i in 1:nparticles
                    alive[i] || continue
                    lprob, θp, logcorr = new_p[i]
                    isnothing(lprob) && continue
                    lπp = logpdf(prior, push_p(prior, θp.x))
                    lπp < 0 && (!isfinite(lπp)) && continue
                    lM = min(lπp - lπs[i] + logcorr, 0.0)
                    if lprob < lM
                        Xp = cost(push_p(prior, θp.x))
                        if flag
                            Xp > ϵ && continue
                        else
                            Xp >= ϵ && continue
                        end
                        θs[i] = θp
                        Xs[i] = Xp
                        lπs[i] = lπp
                        Base.Threads.atomic_add!(accepted, 1)
                    end
                end
            else
                for i in 1:nparticles
                    alive[i] || continue
                    lprob, θp, logcorr = new_p[i]
                    isnothing(lprob) && continue
                    lπp = logpdf(prior, push_p(prior, θp.x))
                    lπp < 0 && (!isfinite(lπp)) && continue
                    lM = min(lπp - lπs[i] + logcorr, 0.0)
                    if lprob < lM
                        Xp = cost(push_p(prior, θp.x))
                        if flag
                            Xp > ϵ && continue
                        else
                            Xp >= ϵ && continue
                        end
                        θs[i] = θp
                        Xs[i] = Xp
                        lπs[i] = lπp
                    Base.Threads.atomic_add!(accepted, 1)
                    end
                end
            end
            accepted[] >= mcmc_tol * nparticles && break
        end
        if 2 * abs(ϵv - ϵ) < r_epstol * (abs(ϵv) + abs(ϵ)) ||
           ϵ <= epstol ||
           accepted[] < mcmc_tol * nparticles
            break
        end
    end
    θs = [push_p(prior, θs[i].x) for i in 1:nparticles][alive]

    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i in 1:l)
    length(P) == 1 && (P = first(P))
    (P = P, C = Xs, ϵ = ϵ)
end

# export smc

#=
using KissABC

prior = Uniform(-10, 10)
sim(μ) = μ + rand((randn() * 0.1, randn()))
cost(x) = abs(sim(x) - 0.0)
R=smc(prior,cost,verbose=true,nparticles=10000,alpha=0.99,mcmc_tol=0.0001).P
hist(R.particles,80,density=true)


pp=Normal(0,5)

cc(x) = 50*(x+randn()*0.01-1)^2

R=smc(pp,cc,verbose=true,alpha=0.95,nparticles=500).P


using KissABC
pp=Factored(Normal(0,5), Normal(0,5))
cc((x,y)) = 50*(x+randn()*0.01-y^2)^2+(y-1+randn()*0.01)^2

R=smc(pp,cc,verbose=true,alpha=0.95,nparticles=500).P
using PyPlot
pygui(true)
sP=Particles(sigmapoints(mean(R),cov(R)))
cc((sP[1],sP[2]))
scatter(R[1].particles,R[2].particles)

scatter(sP[1].particles,sP[2].particles)
hist(R[1].particles,20)

Particles(y)

hist(y,20,weights=Ws)



using Random, KissABC

function costfun((u1, p1); raw=false)
    n=10^6
    A=randexp(n)
    B=rand(n)
    u2 = (1.0 - u1*p1)/(1.0 - p1)
    x = A .* ifelse.(B .< p1, u1, u2)
    sqrt(sum(abs2,[std(x)-2.2, median(x)-0.4]./[2.2,0.4]))
end

@time R=smc(Factored(Uniform(0,1), Uniform(0.5,1)), costfun, nparticles=100,alpha=0.75, verbose=true, parallel=true)

plan=ApproxPosterior(Factored(Uniform(0,1), Uniform(0.5,1)), costfun, 0.01)

@time res = sample(plan, AIS(25),MCMCThreads(),25,4,discard_initial=2500)

using PyPlot
pygui(true)
scatter(R.P[1].particles,R.P[2].particles)


Particles(sigmapoints(mean(R.P),cov(R.P)))

=#



# function pfilter(prior, cost, N; rng=Random.GLOBAL_RNG, q=0.7, eff_tol = 0.1, epstol=-Inf, max_iters = Inf, proposal_width=0.75, verbose=false, parallel=false)
#     lowN=4*length(prior)
#     if N*q<=lowN
#         N=ceil(Int,(lowN+1)/q)
#     end
#     sample=[op(float, Particle(rand(rng, prior))) for i = 1:N]
#     logπ = [logpdf(prior, push_p(prior,sample[i].x)) for i = 1:N]
#     C = fill(cost(sample[1].x),N)
#     @cthreads parallel for i = 1:N
#         trng=rng
#         parallel && (trng=Random.default_rng(Threads.threadid());)
#         if isfinite(logπ[i])
#             C[i] = cost(sample[i].x)
#         end
#         while (!isfinite(C[i])) || (!isfinite(logπ[i]))
#             sample[i]=op(float, Particle(rand(trng, prior)))
#             logπ[i] = logpdf(prior, push_p(prior,sample[i].x))
#             C[i] = cost(sample[i].x)
#         end
#     end

#     iters = 0
#     while true
#         iters += 1
#         ϵ = quantile(C,q)
#         filter_bad=C .> ϵ
#         idxok=(1:N)[.!filter_bad]
#         idxbad=(1:N)[filter_bad]
#         nreps= Threads.Atomic{Int}(0)
#         @cthreads parallel for i in idxbad
#             trng=rng
#             parallel && (trng=Random.default_rng(Threads.threadid());)
#             localreps=0
#             @label resample
#             b=c=d=rand(trng,idxok)
#             while c==b; c=rand(trng,idxok); end
#             while d==b || d==c; d=rand(trng,idxok); end
#             p=op(+,sample[b],op(*,op(-,sample[d],sample[c]), randn(trng)*proposal_width))
#             localreps += 1

#             ll = logpdf(prior, push_p(prior,p.x))
#             if log(rand(trng)) > min(0.0,ll-logπ[i])
#                 @goto resample
#             end
#             Cp=cost(p.x)
#             if Cp > ϵ
#                 @goto resample
#             end
#             C[i] = Cp
#             sample[i] = p
#             logπ[i] = ll
#             Threads.atomic_add!(nreps,localreps)
#         end
#         eff=length(idxbad)/nreps[]
#         verbose && @show iters, ϵ, eff
#         eff<eff_tol && break
#         ϵ<epstol && break
#         iters > max_iters && break
#     end

#     θs = [push_p(prior, sample[i].x) for i = 1:N]
#     l = length(prior)
#     P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
#     length(P)==1 && (P=first(P))
#     (P=P, C=Particles(C))
# end


# export pfilter



function ABCDE(prior, cost, ϵ_target; nparticles = 256, generations = 150, α = 0,
               earlystop = false, verbose = false, rng = Random.Xoshiro(),
               proposal_width = 1.0, HPC::Bool = false)
    @assert 0<=α<1 "α must be in 0 <= α < 1."
    θs = [op(float, Particle(rand(rng, prior))) for i in 1:nparticles]

    logπ = [logpdf(prior, push_p(prior, θs[i].x)) for i in 1:nparticles]
    Δs = fill(cost(θs[1].x), nparticles)


    for i in 1:nparticles
        trng = rng

        if isfinite(logπ[i])
            Δs[i] = cost(θs[i].x)
        end
        while (!isfinite(Δs[i])) || (!isfinite(logπ[i]))
            θs[i] = op(float, Particle(rand(trng, prior)))
            logπ[i] = logpdf(prior, push_p(prior, θs[i].x))
            Δs[i] = cost(θs[i].x)
        end
    end
    nsims = zeros(Int, nparticles)
    γ = proposal_width * 2.38 / sqrt(2 * length(prior))
    iters = 0
    complete = 1 - sum(Δs .> ϵ_target) / nparticles
    stop_after_iter = earlystop && complete >= 1.0 ? iters + min(20, generations - iters) :
                      nothing

    prog = Progress(generations, dt = 1.0, desc = "ABCDE Routine: ", barlen = 25,
                    showspeed = true, color = :light_cyan, enabled = !HPC)

    while iters < generations
        if earlystop && stop_after_iter !== nothing && iters >= stop_after_iter
            break
        end
        iters += 1
        nθs = identity.(θs)
        nΔs = identity.(Δs)
        nlogπ = identity.(logπ)
        ϵ_l, ϵ_h = extrema(Δs)
        ϵ_pop = max(ϵ_target, ϵ_l + α * (ϵ_h - ϵ_l))
        for i in 1:nparticles
            if earlystop
                Δs[i] <= ϵ_target && continue
            end
            trng = rng

            s = i
            ϵ = ifelse(Δs[i] <= ϵ_target, ϵ_target, ϵ_pop)
            if Δs[i] > ϵ
                s = rand(trng, (1:nparticles)[Δs .<= Δs[i]])
            end
            a = s
            while a == s
                a = rand(trng, 1:nparticles)
            end
            b = a
            while b == a || b == s
                b = rand(trng, 1:nparticles)
            end
            θp = op(+, θs[s], op(*, op(-, θs[a], θs[b]), γ))
            lπ = logpdf(prior, push_p(prior, θp.x))
            w_prior = lπ - logπ[i]
            log(rand(trng)) > min(0, w_prior) && continue
            nsims[i] += 1
            dp = cost(θp.x)
            if dp <= max(ϵ, Δs[i])
                nΔs[i] = dp
                nθs[i] = θp
                nlogπ[i] = lπ
            end
        end
        θs = nθs
        Δs = nΔs
        logπ = nlogπ
        ncomplete = 1 - sum(Δs .> ϵ_target) / nparticles
        if earlystop && stop_after_iter === nothing && ncomplete >= 1.0
            extra_gens = min(20, generations - iters)
            stop_after_iter = iters + extra_gens
        end
        if verbose && (ncomplete != complete || complete >= (nparticles - 1) / nparticles)
            @info "Finished run:" completion=ncomplete nsim=sum(nsims) range_ϵ=extrema(Δs)
        end
        complete = ncomplete

        # next!(prog, showvalues = [(:completion, complete), (:nsim, sum(nsims)), (:range_ϵ, extrema(Δs))])
        next!(prog,
              showvalues = [
                  (:completion, round(complete, digits = 5)),
                  (:nsim, round(sum(nsims), digits = 5)),
                  (:range_ϵ, round.(extrema(Δs), digits = 5))
              ])
    end
    conv = maximum(Δs) <= ϵ_target
    # if verbose
    #     @info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs)
    # end
    θs = [push_p(prior, θs[i].x) for i in 1:nparticles]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i in 1:l)
    length(P) == 1 && (P = first(P))
    # (P=P, C=Particles(Δs), reached_ϵ=conv)
    return (P = P, C = Particles(Δs), reached_ϵ = conv)
    # return ABCDEResults(P=P, C=Particles(Δs), reached_ϵ=conv)
end

struct ABCDEResults
    P::Particles
    C::Particles
    reached_ϵ::Bool
end

# export ABCDE

### KissABCfork.jl



struct AIS <: AbstractMCMC.AbstractSampler
    nparticles::Int
end

struct AISState{S, L}
    "Sample of the Affine Invariant Sampler."
    sample::S
    "Log-likelihood of the sample."
    loglikelihood::L
    "Current particle"
    i::Int
    AISState(s::S, l::L, i = 1) where {S, L} = new{S, L}(s, l, i)
end

function AbstractMCMC.step(rng::Random.AbstractRNG,
                           model::AbstractMCMC.AbstractModel,
                           spl::AIS;
                           retry_sampling::Int = 100,
                           kwargs...,)
    nparticles = spl.nparticles
    nparticles < length(model) + 5 && error("nparticles = ",
          nparticles,
          " is insufficient, set number of particles in AIS(⋅) atleast to ",
          length(model) + 5)

    particles = [op(float, unconditional_sample(rng, model)) for i in 1:nparticles]
    logdensity = [loglike(model, push_p(model, particles[i])) for i in 1:nparticles]
    retrys = retry_sampling * nparticles
    for i in 1:nparticles
        while !is_valid_logdensity(model, logdensity[i])
            particles[i] = op(float, unconditional_sample(rng, model))
            logdensity[i] = loglike(model, push_p(model, particles[i]))
            retrys -= 1
            retrys < 0 &&
                error("Prior leads to ∞ costs too often, tune the prior or increase `retry_sampling`.")
        end
    end

    push_p(model, particles[end]), AISState(particles, logdensity)
end

function AbstractMCMC.step(rng::Random.AbstractRNG,
                           model::AbstractMCMC.AbstractModel,
                           spl::AIS,
                           state::AISState;
                           ntransitions::Int = 1,
                           kwargs...,)
    i = state.i
    for reps in 1:ntransitions
        transition!(model, state.sample, state.loglikelihood, i, rng)
    end
    push_p(model, state.sample[i]),
    AISState(state.sample, state.loglikelihood, 1 + (i % spl.nparticles))
end

function AbstractMCMC.bundle_samples(samples::Vector{T},
                                     ::AbstractMCMC.AbstractModel,
                                     ::AIS,
                                     ::Any,
                                     ::Type;
                                     kwargs...,) where {T <: Particle}
    l = length(samples[1].x)
    P = map(x -> Particles(x), getindex.(getfield.(samples, :x), i) for i in 1:l)
    length(P) == 1 && return P[1]
    return P
end

function AbstractMCMC.chainsstack(c::Vector{Vector{T}}) where {T <: Particles}
    nc = length(c)
    pl = length(c[1])
    return [Particles(reduce(vcat, c[n][i].particles for n in 1:nc)) for i in 1:pl]
end

function AbstractMCMC.chainsstack(c::Vector{T}) where {T <: Particles}
    return Particles(reduce(vcat, c[i].particles for i in eachindex(c)))
end

"""
    sample(model, AIS(N), Ns[; optional args])
    sample(model, AIS(N), MCMCThreads(), Ns, Nc[; optional keyword args])
    sample(model, AIS(N), MCMCDistributed(), Ns, Nc[; optional keyword args])

# Generalities

This function will run an Affine Invariant MCMC sampler, and will return a `Particles` object for each parameter,
the mandatory parameters are:

`model`: a subtype of `AbstractDensity`, look at `ApproxPosterior`, `ApproxKernelizedPosterior`, `CommonLogDensity`.

`N`: number of particles in the ensemble, this particles will be evolved to generate new samples.

`Ns`: total number of samples which must be recorded.

`Nc`: total number of chains to run in parallel if MCMCThreads or MCMCDistributed is enabled.


the optional arguments available are:


`discard_initial`: number of mcmc particles to discard before saving any sample.

`ntransitions`: number of mcmc steps per particle between each sample.

`retry_sampling`: number of maximum attempts to resample an initial particle whose cost (or log-density) is ±∞ or NaN.

`progress`: a boolean to disable verbosity

# Minimal Example for `CommonLogDensity`:

```julia
using KissABC
D = CommonLogDensity(
    2, #number of parameters
    rng -> randn(rng, 2), # initial sampling strategy
    x -> -100 * (x[1] - x[2]^2)^2 - (x[2] - 1)^2, # rosenbrock banana log-density
)
res = sample(D, AIS(50), 1000, ntransitions = 100, discard_initial = 500, progress = false)
println(res)
```

output:
```
Particles{Float64,1000}[1.43 ± 1.4, 0.99 ± 0.67]
```

# Minimal Example for `ApproxKernelizedPosterior` (`ApproxPosterior`)

```julia
using KissABC
prior = Uniform(-10, 10) # prior distribution for parameter
sim(μ) = μ + rand((randn() * 0.1, randn())) # simulator function
cost(x) = abs(sim(x) - 0.0) # cost function to compare simulations to target data, in this case simply '0'
plan = ApproxPosterior(prior, cost, 0.01) # Approximate model of log-posterior density (ABC)
#                                           ApproxKernelizedPosterior can be used in the same fashion
res = sample(plan, AIS(100), 2000, discard_initial = 10000, progress = false)
println(res)
```

output:
```
0.0 ± 0.46
```

"""
sample

end # module KissABC
