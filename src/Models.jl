"""
Mechanistic population balance models for batch crystallisation. Defines kinetics for nucleation, growth, aggregation and breakage together with solvers for the finite volume and method of moments formulations.
"""

#### Flux limiters for high-resolution finite volume schemes
"""
    fluxlimiter_ospre(r)

Ospre flux limiter for high-resolution schemes.
Computes a smoothness-based limiter value for numerical flux reconstruction.
"""
fluxlimiter_ospre(r) = (1.5 * (r^2) + r) / (r^2 + r + 1)

@inline _named_params(model, p::AbstractVector) = ComponentArray(p, paramaxis(model))
@inline _named_params(_, p::ComponentArrays.ComponentArray) = p

# Generic fallback for kinetics that haven't declared a custom paramaxis.
# Generates `θ1, θ2, ...` from `model.nparams`. Specific paramaxis methods
# (e.g. `paramaxis(::nucl_CNT) = ComponentArrays.Axis(Aj=1, γ=2)`) take precedence.
function paramaxis(model::Union{AbstractNucleationFunction, AbstractGrowthFunction,
                                AbstractAggregationFunction, AbstractBreakageFunction})
    n = model.nparams
    n == 0 && return ComponentArrays.Axis()
    return ComponentArrays.Axis(NamedTuple{Tuple(Symbol("θ", i) for i in 1:n)}(Tuple(1:n)))
end

# Composite axis spanning the four kinetic families. Slot order (nucl, gr,
# agg, br) matches the legacy positional convention.
function paramaxis(nucl::AbstractNucleationFunction,
                   gr::AbstractGrowthFunction,
                   agg::AbstractAggregationFunction,
                   br::AbstractBreakageFunction)
    nν, ng, na, nb = nucl.nparams, gr.nparams, agg.nparams, br.nparams
    return ComponentArrays.Axis(nucl = ViewAxis(1:nν, paramaxis(nucl)),
                gr = ViewAxis((nν + 1):(nν + ng), paramaxis(gr)),
                agg = ViewAxis((nν + ng + 1):(nν + ng + na), paramaxis(agg)),
                br = ViewAxis((nν + ng + na + 1):(nν + ng + na + nb), paramaxis(br)))
end

paramaxis(prob::CrystallisationProblem) = paramaxis(prob.kinetics_nucleationfunction,
                                                    prob.kinetics_growthfunction,
                                                    prob.kinetics_aggregationfunction,
                                                    prob.kinetics_breakagefunction)


"""
    crystal_state(problem, state) -> AbstractVector

Crystal-population part of a solver state: the named `n` component for
ComponentArray states (MoM moments), or the state prefix before the named
solvent-state variables for flat discretised states (mesh densities).
"""
crystal_state(state::ComponentArrays.ComponentVector) = state.n
crystal_state(state::AbstractVector) = @view state[1:(end - 1)]
crystal_state(problem::CrystallisationProblem, state) =
    @view state[1:(end - length(propertynames(problem.initial_solvent_state)))]

#### Kinetics
## Nucleation
"""
    nucleationrate(::nucl_CNT, parameters::AbstractVector, S::Real,
                   system, temperature, numberdensity)

Calculate nucleation rate using Classical Nucleation Theory (CNT).

# Arguments
- `parameters`: Vector containing [A_j, γ] where:
  - A_j: Pre-exponential factor
  - γ: Surface tension (mJ/m²)
- `S`: Supersaturation ratio
 - `system`: Crystallisation system parameters (molecular volume, constants, etc.)
 - `temperature`: Instantaneous temperature in Kelvin (use `temperature(sys.temp_profile, t)`)
 - `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_CNT, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (60 * exp(p.Aj)) *
        S *
        exp(-16π * ((p.γ * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(::nucl_empirical, parameters::AbstractVector, S::Real,
                   system, temperature, numberdensity)

Calculate nucleation rate using an empirical power law model.

# Arguments
- `parameters`: Vector containing [A_j, j] where:
  - A_j: Pre-exponential factor (log10 scale)
  - j: Power law exponent
- `S`: Supersaturation ratio
 - `system`: Crystallisation system parameters
 - `temperature`: Instantaneous temperature in Kelvin
 - `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ? (60 * 10^(p.Aj)) * (S - 1)^p.j : 0.0
end

"""
    nucleationrate(::nucl_empirical_energy, parameters::AbstractVector, S::Real,
                   system, temperature, loading, numberdensity) -> Real

Calculate nucleation rate using empirical model with activation energy.

# Arguments
- `parameters`: Vector [Aj, Ea, j] where Aj is pre-exponential, Ea is activation energy (kJ/mol), j is exponent
- `S`: Supersaturation ratio
- `system`: Crystallisation system parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_empirical_energy, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ?
           (60 * exp(p.Aj)) * exp(-(p.Ea * 1e3) / (8.314 * temp)) *
           (S - 1)^p.j : 0.0
end

"""
    nucleationrate(::nucl_CNTnoS, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, numberdensity::AbstractVector)

Calculate nucleation rate using Classical Nucleation Theory (CNT) without S factor in pre-exponential term.

# Arguments
- `parameters`: Vector containing [A_j, γ] where:
  - A_j: Pre-exponential factor
  - γ: Surface tension (mJ/m²)
- `S`: Supersaturation ratio
 - `system`: Crystallisation system parameters
 - `temperature`: Instantaneous temperature in Kelvin
 - `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_CNTnoS, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (60 * exp(p.Aj)) *
        S *
        exp(-16π * ((p.γ * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(::nucl_secondary, parameters::AbstractVector, S::Real,
                   system, temperature, loading, numberdensity::AbstractVector) -> Real

Calculate secondary nucleation rate proportional to third moment (crystal mass).

# Arguments
- `parameters`: Vector [Aj, Ea, j] where Aj is pre-exponential, Ea is activation energy (kJ/mol), j is exponent
- `S`: Supersaturation ratio
- `system`: Crystallisation system parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Secondary nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_secondary, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        base_rate = (60 * exp(p.Aj)) *
                    exp(-(p.Ea * 1e3) / (8.314 * temp)) *
                    (S - 1)^p.j
        base_rate * _secondary_third_moment(prob.solver, prob,
                                            crystal_state(prob, state))
    else
        0.0
    end
end

"""
    nucleationrate(::nucl_prim_plus_second, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate combined primary (empirical) and secondary nucleation rate.

# Arguments
- `parameters`: Vector [Aj_prim, Ea_prim, j_prim, Aj_sec, Ea_sec, j_sec] (6 parameters total)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Combined nucleation rate (number/m³/s)
"""
function nucleationrate(nf::nucl_prim_plus_second, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    prim_rate = nucleationrate(nucl_empirical_energy(), p.prim, prob, state, t)
    sec_rate = nucleationrate(nucl_secondary(), p.sec, prob, state, t)
    return prim_rate + sec_rate
end

"""
    nucleationrate(::nucl_CNT_plus_second, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate combined CNT primary and secondary nucleation rate.

# Arguments
- `parameters`: Vector [Aj_CNT, γ, Aj_sec, Ea_sec, j_sec] (5 parameters total)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Combined nucleation rate (number/m³/s)
"""
function nucleationrate(nf::nucl_CNT_plus_second, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    cnt_rate = nucleationrate(nucl_CNT(), p.cnt, prob, state, t)
    sec_rate = nucleationrate(nucl_secondary(), p.sec, prob, state, t)
    return cnt_rate + sec_rate
end

"""
    nucleationrate(NuF::nucl_CNT_fixed, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate CNT nucleation rate using pre-fixed parameters embedded in the struct.

# Arguments
- `NuF`: Fixed CNT nucleation function with embedded Aj and γ parameters
- `parameters`: Ignored (parameters are taken from NuF)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(NuF::nucl_CNT_fixed, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (60 * exp(NuF.Aj)) *
        S *
        exp(-16π * ((NuF.γ * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(NuF::nucl_empirical_fixed, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate empirical nucleation rate using pre-fixed parameters embedded in the struct.

# Arguments
- `NuF`: Fixed empirical nucleation function with embedded Aj and j parameters
- `parameters`: Ignored (parameters are taken from NuF)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(NuF::nucl_empirical_fixed, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    S = supersaturation(prob, state, t)
    return S > 1.001 ? (60 * 10^(NuF.Aj)) * (S - 1)^NuF.j : 0.0
end

"""
    nucleationrate(::nucl_CNT_multiloading, parameters::AbstractVector, S::Real,
                   system, temperature, loading, numberdensity)

Calculate nucleation rate using Classical Nucleation Theory (CNT) with loading-dependent parameters.

# Arguments
- `parameters`: Vector containing [A1, γ1, A2, γ2, ...] where each pair corresponds to a unique loading
- `S`: Supersaturation ratio
- `system`: Crystallisation system parameters (molecular volume, constants, etc.)
- `temperature`: Instantaneous temperature in Kelvin
- `loading`: Current loading value (used to select appropriate A and γ parameters)
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) using the parameters corresponding to the current loading
"""
function nucleationrate(nucl_func::nucl_CNT_multiloading, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    # Find which prob.loading corresponds to the current prob.loading value
    loading_idx = findfirst(==(prob.loading), nucl_func.unique_loadings)

    if loading_idx === nothing
        error("Loading value $prob.loading not found in unique_loadings: $(nucl_func.unique_loadings)")
    end

    # Extract the relevant A and γ parameters for this prob.loading
    param_idx = 2 * (loading_idx - 1) + 1
    A_param = parameters[param_idx]
    γ_param = parameters[param_idx + 1]

    # Apply the standard CNT nucleation rate formula
    return if S > 1.001
        (60 * exp(A_param)) *
        S *
        exp(-16π * ((γ_param * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end


"""
    _secondary_third_moment(solver, prob, nd) -> Real

Third-moment contribution to the secondary nucleation rate, dispatched on the
solver: volume-density quadrature for discretised solvers, `max(0, nd[4])`
(µ3) for the MoM state.
"""
_secondary_third_moment(solver::AbstractDiscretisedSolver, prob, nd) =
    momentcalculator(solver.cell_centre, nd, 3)
_secondary_third_moment(solver::MoM, prob, nd) = max(0, nd[4])  # µ3 stored at index 4

## Growth
"""
    growthrate(::growth_empirical, parameters::AbstractVector, S::Real,
               system, temperature, numberdensity)

Calculate crystal growth rate using an empirical power law model.

# Arguments
- `parameters`: Vector containing [A_g, g] where:
  - A_g: Growth rate constant
  - g: Growth rate order
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Instantaneous temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ? (p.Ag * 1e-9) * ((S - 1)^p.g) : 0.0
end
"""
    growthrate(gf::growth_energy, parameters::AbstractVector, S::Real,
               system, temperature, loading, numberdensity) -> Real

Calculate growth rate with Arrhenius temperature dependence using fixed activation energy.

# Arguments
- `gf`: Growth function with embedded activation energy Ea
- `parameters`: Vector [Ag, g] where Ag is pre-exponential (log10 scale), g is exponent
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_energy, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ?
           exp10(p.Ag) * exp(-gf.Ea / (8.314 * temp)) *
           ((S - 1)^p.g) :
           0.0 ### exp(-50kJ / (8.314 * temp)) is 10^-10, therefore I changed the unit scaling, note this when reading Aj
    ## multiply by 1e12 to convert to typical units
end

"""
    growthrate(gf::growth_energy_est, parameters::AbstractVector, S::Real,
               system, temperature, loading, numberdensity) -> Real

Calculate growth rate with Arrhenius temperature dependence and estimated activation energy.

# Arguments
- `gf`: Growth function struct
- `parameters`: Vector [Ag, Eag, g] where Ag is pre-exponential (log10), Eag is activation energy (kJ/mol), g is exponent
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(gf::growth_energy_est, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ?
           exp10(p.Ag) * exp((-p.Eag * 1e3) / (8.314 * temp)) *
           ((S - 1)^p.g) :
           0.0 ### exp(-50kJ / (8.314 * temp)) is 10^-10, therefore I changed the unit scaling, note this when reading Aj
    ## multiply by 1e12 to convert to typical units
end

"""
    growthrate(gf::growth_energy_multiloading, parameters, S, system, temperature,
               loading, numberdensity) -> Real

Calculate growth rate using loading-dependent parameters.

# Arguments
- `gf`: Growth function with unique_loadings array
- `parameters`: Vector [Ag1, g1, Ag2, g2, ...] with pairs for each loading
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `loading`: Current loading value (used to select parameters)
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) using parameters for the matching loading
"""
function growthrate(gf::growth_energy_multiloading, parameters, prob::CrystallisationProblem, state, t)

    S = supersaturation(prob, state, t)

    temp = temperature(prob.temp_profile, t)
# Find which prob.loading corresponds to the current prob.loading value
    loading_idx = findfirst(==(prob.loading), gf.unique_loadings)

    if loading_idx === nothing
        error("Loading value $prob.loading not found in unique_loadings: $(gf.unique_loadings)")
    end

    # Extract the relevant A and γ parameters for this prob.loading
    param_idx = 2 * (loading_idx - 1) + 1
    A_param::Real = parameters[param_idx]
    g_param::Real = parameters[param_idx + 1]

    return S > 1.001 ?
           exp10(A_param) * exp(-gf.Ea / (8.314 * temp)) *
           ((S - 1)^g_param) :
           0.0
end


"""
    growthrate(::growth_BCF, parameters::AbstractVector, S::Real,
               system::CrystallisationProblem, temperature, numberdensity)

Calculate crystal growth rate using Burton-Cabrera-Frank (BCF) surface diffusion model.

# Arguments
- `parameters`: Vector containing [C3, C4] where:
  - C3: Growth rate constant (model parameter based on physical properties)
  - C4: Surface energy barrier parameter (model parameter based on physical properties)
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Instantaneous temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0

# Notes
- Developed by Burton, Cabrera, and Frank (1951) for growth via screw dislocation geometry
- Assumes surface diffusion of adsorbed species is rate-limiting
- Alternative to Birth and Spread model for low supersaturation conditions
"""
function growthrate(gf::growth_BCF, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (1e-9) * p.C3 * temp / p.C4 *
        (S - 1) *
        tanh(p.C4 / (temp * log(S)))
    else
        0.0
    end
end

"""
    growthrate(::growth_BpS, parameters::AbstractVector, S::Real,
               system::CrystallisationProblem, temperature, numberdensity)

Calculate crystal growth rate using Birth and Spread (B+S) model based on nucleation theory.

# Arguments
- `parameters`: Vector containing [C1, C2] where:
  - C1: Growth rate constant (model parameter based on physical properties fitted to experimental data)
  - C2: Energy barrier parameter (model parameter based on physical properties fitted to experimental data)
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Instantaneous temperature in Kelvin
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0

# Notes
- Based on O'Hara (1973) nucleation theory expression
- Critical nuclei form on surface and spread at constant rates
- Assumes smooth crystal surface at high supersaturations
- May deviate from measured values at low supersaturations
"""
function growthrate(gf::growth_BpS, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (1e-9) *
        p.C1 *
        ((S - 1)^(2 / 3)) *
        (log(S))^(1 / 6) *
        exp(-p.C2 / (temp^2 * log(S)))
    else
        0.0
    end
end

### Dissolution

"""
    growthrate(gf::growth_dissolution_length, parameters::AbstractVector, S::Real,
               mesh::AbstractVector, temperature, loading, numberdensity) -> Vector

Calculate length-dependent dissolution rate (negative growth).

# Arguments
- `gf`: Dissolution growth function
- `parameters`: Vector [Ad, Ead, d, κ, p] for dissolution kinetics
- `S`: Supersaturation ratio
- `mesh`: Cell-centre coordinates (length-dependence is evaluated on each mesh cell)
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Vector of dissolution rates (m/s) at each mesh point, zero vector if supersaturated
"""
function growthrate(gf::growth_dissolution_length, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    pn = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    sat_concentration = saturation_concentration(prob, t)
    concentration = S * sat_concentration

    return concentration < sat_concentration ?
           -(pn.Ad * 1e-9) .*
           exp(-(pn.Ead * 1e3) ./ (8.314 .* temp)) .*
           ((sat_concentration - concentration) .^ pn.d) .*
           (1 .+ pn.κ * 1e3 .* prob.solver.cell_centre) .^ pn.p :
           zeros(size(prob.solver.cell_centre))
end

"""
    growthrate(gf::growth_dissolution, parameters::AbstractVector, S::Real,
               system, temperature, loading, numberdensity) -> Real

Calculate scalar dissolution rate (negative growth) with Arrhenius temperature dependence.

# Arguments
- `gf`: Dissolution growth function
- `parameters`: Vector [Ad, Ead, d] for dissolution kinetics
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Dissolution rate (negative m/s) if undersaturated, 0 otherwise
"""
function growthrate(gf::growth_dissolution, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    sat_concentration = saturation_concentration(prob, t)
    concentration = S * sat_concentration

    return concentration < sat_concentration ?
           -(p.Ad * 1e-9) *
           exp(-(p.Ead * 1e3) / (8.314 * temp)) *
           ((1 - S)^p.d) :
           0.0
end

"""
    growthrate(gf::growth_energy_dissolution, parameters::AbstractVector, S::Real,
               system, temperature, loading, numberdensity) -> Real

Calculate growth or dissolution rate depending on supersaturation.

Uses `growth_energy` for supersaturated conditions (S > 1.001) and
`growth_dissolution` for undersaturated conditions.

# Arguments
- `gf`: Combined growth/dissolution function
- `parameters`: Vector [Ag, g, Ad, Ead, d] for combined kinetics
- `S`: Supersaturation ratio
- `system`: System parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if supersaturated, dissolution rate if undersaturated
"""
function growthrate(gf::growth_energy_dissolution, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ?
           growthrate(growth_energy(), p, prob, state, t) :
           growthrate(growth_dissolution(), p, prob, state, t)
end

###

"""
    growthrate(grf::growth_empirical_fixed, parameters, S::Real,
               system::CrystallisationProblem, temperature, loading, numberdensity) -> Real

Calculate empirical growth rate using pre-fixed parameters embedded in the struct.

# Arguments
- `grf`: Fixed empirical growth function with embedded Ag and g parameters
- `parameters`: Ignored (parameters taken from grf)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Growth rate (m/s) if S > 1.001, otherwise 0
"""
function growthrate(grf::growth_empirical_fixed, parameters, prob::CrystallisationProblem, state, t) 
    S = supersaturation(prob, state, t)
    return S > 1.001 ? (grf.Ag * 1e-9) * ((S - 1)^grf.g) : 0.0
end

## Aggregation

"""
    aggregationrate(::noaggregation, parameters::AbstractVector, mesh::AbstractVector,
                    numberdensity::AbstractVector) -> Real

Return zero aggregation rate (placeholder for no aggregation).

# Returns
- 0.0
"""
function aggregationrate(::noaggregation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return 0.0
end

"""
    aggregationrate(::aggr_scalar, parameters::AbstractVector, fullmesh::AbstractVector,
                    fullnumberdensity::AbstractVector) -> Vector

Calculate size-independent (scalar) aggregation rate.

# Arguments
- `parameters`: Vector [log10(β)] where β is aggregation kernel constant
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(af::aggr_scalar, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(af, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    cumul_numberdensity = cumsum(reverse(crystal_state(prob, state)))
    return 10^(p.logβ) * 0.5 .* Base.step(prob.solver.cell_centre) .* crystal_state(prob, state) .*
           cumul_numberdensity .-
           10^(p.logβ) .* Base.step(prob.solver.cell_centre) .* crystal_state(prob, state) *
           sum(crystal_state(prob, state))
end

"""
    aggregationrate(::aggr_linear, parameters::AbstractVector, fullmesh::AbstractVector,
                    fullnumberdensity::AbstractVector) -> Vector

Calculate linear size-dependent aggregation rate (kernel proportional to sum of sizes).

# Arguments
- `parameters`: Vector [log10(β)] where β is aggregation kernel constant
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(af::aggr_linear, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(af, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    cumul_numberdensity = cumsum(reverse(crystal_state(prob, state)))
    cumul_linear = cumsum(prob.solver.cell_centre)
    return 10^(p.logβ) * 0.5 .* Base.step(prob.solver.cell_centre) .* crystal_state(prob, state) .*
           cumul_numberdensity .* (prob.solver.cell_centre .+ cumul_linear) .-
           10^(p.logβ) .* Base.step(prob.solver.cell_centre) .* crystal_state(prob, state) *
           sum(crystal_state(prob, state)) .* (prob.solver.cell_centre .+ cumul_linear)
end

"""
    aggregationrate(::aggr_linearvol, parameters::AbstractVector, fullmesh::AbstractVector,
                    fullnumberdensity::AbstractVector) -> Vector

Calculate linear volume-dependent aggregation rate (kernel proportional to sum of volumes).

# Arguments
- `parameters`: Vector [log10(β)] where β is aggregation kernel constant
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(af::aggr_linearvol, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(af, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    cumul_numberdensity = cumsum(reverse(crystal_state(prob, state)))
    cumul_linear = cumsum(prob.solver.cell_centre)
    return 10 .^ (p.logβ) * 0.5 .* Base.step(prob.solver.cell_centre) .* crystal_state(prob, state) .*
           cumul_numberdensity .* (prob.solver.cell_centre .^ 3 .+ cumul_linear .^ 3) .-
           10^(p.logβ) .* Base.step(prob.solver.cell_centre) .* crystal_state(prob, state) *
           sum(crystal_state(prob, state)) .* (prob.solver.cell_centre .^ 3 .+ cumul_linear .^ 3)
end

"""
    breakagerate(::nobreakage, parameters::AbstractVector, fullmesh::AbstractVector,
                 fullnumberdensity::AbstractVector) -> Real

Return zero breakage rate (placeholder for no breakage).

# Returns
- 0.0
"""
function breakagerate(::nobreakage, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return 0.0
end

"""
    breakagerate(::breakage_empirical, parameters::AbstractVector, mesh::AbstractVector,
                 numberdensity::AbstractVector) -> Real

Calculate empirical breakage rate.

# Arguments
- `parameters`: Vector [breakage constant, breakage exponent]
- `mesh`: Cell center positions
- `numberdensity`: Number density at each cell

# Returns
- Breakage rate contribution
"""
function breakagerate(bf::breakage_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(bf, parameters)
    mesh = prob.solver.cell_centre
    numberdensity = crystal_state(prob, state)
    rates = zeros(promote_type(eltype(numberdensity), typeof(p.b)), length(mesh))
    for idx in eachindex(mesh)
        source = if idx < length(mesh)
            larger_mesh = mesh[(idx + 1):end]
            larger_density = numberdensity[(idx + 1):end]
            trapz(larger_mesh .^ 3,
                  2 ./ (larger_mesh .^ 3) .* p.b .* larger_mesh .^ (3 * p.n) .*
                  larger_density)
        else
            zero(eltype(rates))
        end
        rates[idx] = source - p.b * mesh[idx]^(3 * p.n) * numberdensity[idx]
    end
    return rates
end

"""
    breakagerate(::breakage_uniform, parameters::AbstractVector, fullmesh::AbstractVector,
                 fullnumberdensity::AbstractVector) -> Vector

Calculate uniform breakage rate (daughter fragments uniformly distributed).

# Arguments
- `parameters`: Vector [log(breakage constant), breakage exponent]
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of breakage rates at each cell
"""
function breakagerate(bf::breakage_uniform, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(bf, parameters)
    cumul_numberdensity = cumsum(reverse(crystal_state(prob, state)))
    cumul_linear = cumsum(prob.solver.cell_centre)
    fullmesh = prob.solver.cell_centre
    fullnumberdensity = crystal_state(prob, state)

    cumul_integral = Base.step(CRISTOOL_MICROMETER_SCALE * prob.solver.cell_centre) .^ 3 .*
                     cumsum(2 .* (CRISTOOL_MICROMETER_SCALE .* prob.solver.cell_centre) .^ -3 .* exp(p.logb) .*
                            (CRISTOOL_MICROMETER_SCALE .* prob.solver.cell_centre) .^ (3p.n) .* crystal_state(prob, state))
    return (cumul_integral[end] .- cumul_integral) .-
           exp(p.logb) .* (CRISTOOL_MICROMETER_SCALE .* fullmesh) .^ (3p.n) .* fullnumberdensity
end

###### Solver functions ######

"""
    weno_flux(y::AbstractArray{T}, i::Integer) where {T<:Real}

Calculate the WENO (Weighted Essentially Non-Oscillatory) reconstruction of the flux at cell interface i+1/2.

Uses a 5th order WENO scheme with three candidate stencils for high accuracy in smooth regions while avoiding
oscillations near discontinuities.

# Arguments
- `y`: Array of cell-averaged values
- `i`: Cell index for reconstruction

# Returns
- Reconstructed value at cell interface i+1/2
"""
function weno_flux(y::AbstractArray{T}, i::Integer) where {T <: Real}
    # Constants for WENO scheme
    ε = T(CRISTOOL_WENO_EPSILON)
    γ₁, γ₂, γ₃ = T(0.3), T(0.6), T(0.1)
    c13_12 = T(13 / 12)
    c1_4 = T(1 / 4)

    # Precompute commonly used differences to avoid redundant calculations
    @inbounds begin
        d1 = y[i + 2] - y[i + 1]
        d2 = y[i + 1] - y[i]
        d3 = y[i] - y[i - 1]
        d4 = y[i - 1] - y[i - 2]

        # Calculate candidate stencils (q values)
        q₁ = muladd(T(5 / 6), y[i + 1], muladd(T(1 / 3), y[i], -T(1 / 6) * y[i + 2]))
        q₂ = muladd(T(5 / 6), y[i], muladd(T(1 / 3), y[i + 1], -T(1 / 6) * y[i - 1]))
        q₃ = muladd(T(11 / 6), y[i], muladd(-T(7 / 6), y[i - 1], T(1 / 3) * y[i - 2]))

        # Calculate smoothness indicators (β values)
        β₁ = muladd(c13_12, (d1 - d2)^2, c1_4 * (d1 + d2)^2)
        β₂ = muladd(c13_12, d2^2, c1_4 * (d3 + d1)^2)
        β₃ = muladd(c13_12, d4^2, c1_4 * (3d3 + d4)^2)

        # Calculate non-linear weights
        α₁ = γ₁ / (ε + β₁)^2
        α₂ = γ₂ / (ε + β₂)^2
        α₃ = γ₃ / (ε + β₃)^2

        # Normalize weights
        α_sum_inv = 1 / (α₁ + α₂ + α₃)
        ω₁ = α₁ * α_sum_inv
        ω₂ = α₂ * α_sum_inv
        ω₃ = α₃ * α_sum_inv

        # Compute final reconstruction
        return muladd(ω₁, q₁, muladd(ω₂, q₂, ω₃ * q₃))
    end
end

"""
    _get_initial_state(CryProblem) -> Vector

Get initial state vector for simulation.

Returns zeros for number density/moments plus initial concentration,
or the user-provided initial_state if specified.

# Arguments
- `CryProblem`: Crystallisation problem specification

# Returns
- Initial state vector [number_density_or_moments..., concentration]
"""
_get_initial_state(CryProblem) = isnothing(CryProblem.initial_state) ?
                                     _zero_state(CryProblem.solver, CryProblem) :
                                     CryProblem.initial_state

_zero_state(solver::AbstractDiscretisedSolver, CryProblem) =
    [zeros(solver.meshsize); collect(values(CryProblem.initial_solvent_state))]
_zero_state(solver::MoM, CryProblem) =
    [zeros(solver.nmoments + 1); collect(values(CryProblem.initial_solvent_state))]

"""
    _build_timestepping_algorithm(algorithm_type; step_limiter=nothing, stage_limiter=nothing)

Build an ODE solver algorithm with optional step and stage limiters.

# Arguments
- `algorithm_type`: ODE solver type (e.g., Tsit5, SSPRK43)
- `step_limiter`: Optional step limiter callback
- `stage_limiter`: Optional stage limiter callback

# Returns
- Configured ODE solver algorithm instance
"""
@inline function _build_timestepping_algorithm(algorithm_type;
                                                step_limiter = nothing,
                                                stage_limiter = nothing)
    if isnothing(step_limiter) && isnothing(stage_limiter)
        return algorithm_type()
    end
    if isnothing(step_limiter)
        return algorithm_type(stage_limiter! = stage_limiter)
    end
    if isnothing(stage_limiter)
        return algorithm_type(step_limiter! = step_limiter)
    end
    return algorithm_type(step_limiter! = step_limiter, stage_limiter! = stage_limiter)
end

"""
    _build_timestepping_algorithm_step_only(algorithm_type; step_limiter=nothing)

Build an ODE solver algorithm with optional step limiter only.

Used for solvers that don't support stage limiters (e.g., implicit methods).

# Arguments
- `algorithm_type`: ODE solver type
- `step_limiter`: Optional step limiter callback

# Returns
- Configured ODE solver algorithm instance
"""
@inline function _build_timestepping_algorithm_step_only(algorithm_type;
                                                         step_limiter = nothing)
    if isnothing(step_limiter)
        return algorithm_type()
    end
    return algorithm_type(step_limiter! = step_limiter)
end

"""
    _timestepping_algorithm(::Val{:tsit5}, step_limiter, stage_limiter)

Build Tsit5 algorithm with limiters.
"""
@inline _timestepping_algorithm(::Val{:tsit5}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm(Tsit5; step_limiter = step_limiter,
                                  stage_limiter = stage_limiter)

"""
    _timestepping_algorithm(::Val{:ssprk43}, step_limiter, stage_limiter)

Build SSPRK43 algorithm with limiters.
"""
@inline _timestepping_algorithm(::Val{:ssprk43}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm(SSPRK43; step_limiter = step_limiter,
                                  stage_limiter = stage_limiter)

"""
    _timestepping_algorithm(::Val{:kvaerno5}, step_limiter, stage_limiter)

Build Kvaerno5 algorithm with step limiter only.
"""
@inline _timestepping_algorithm(::Val{:kvaerno5}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm_step_only(Kvaerno5; step_limiter = step_limiter)

"""
    _timestepping_algorithm(::Val{:kencarp4}, step_limiter, stage_limiter)

Build KenCarp4 algorithm with step limiter only.
"""
@inline _timestepping_algorithm(::Val{:kencarp4}, step_limiter, stage_limiter) =
    _build_timestepping_algorithm_step_only(KenCarp4; step_limiter = step_limiter)

"""
    _resolve_timestepping_algorithm(solver::AbstractSolver, default_algorithm::Symbol;
                                    step_limiter=nothing, stage_limiter=nothing)

Resolve and build the timestepping algorithm for a solver.

Uses the solver's configured algorithm if not `:auto`, otherwise uses the default.

# Arguments
- `solver`: Solver with timestepping_algorithm field
- `default_algorithm`: Default algorithm symbol if solver uses :auto
- `step_limiter`: Optional step limiter callback
- `stage_limiter`: Optional stage limiter callback

# Returns
- Configured ODE solver algorithm instance
"""
@inline function _resolve_timestepping_algorithm(solver::AbstractSolver,
                                                 default_algorithm::Symbol;
                                                 step_limiter = nothing,
                                                 stage_limiter = nothing)
    alg_symbol = solver.timestepping_algorithm === :auto ? default_algorithm :
                 solver.timestepping_algorithm
    return _timestepping_algorithm(Val(alg_symbol), step_limiter, stage_limiter)
end

@inline function _concentration_depletion(cell_dL, numberdensity, scalargrowth, cell_centre)
    acc = 0.0
    @inbounds for i in eachindex(numberdensity, cell_centre)
        acc += cell_dL * numberdensity[i] * scalargrowth * (cell_centre[i] * cell_centre[i])
    end
    return acc
end

"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., MoM, ...}, saveat) -> CrystallisationMoMSolution

Simulate crystallisation using Method of Moments solver.

Internal function that solves the moment equations for nucleation and growth
without aggregation or breakage.

# Arguments
- `CryProblem`: Crystallisation problem with MoM solver
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationMoMSolution` containing time, concentration, and moment-derived sizes
"""
@inline function _mom_rhs(CryProblem, scalargrowth, B, solvent_rates, u::SVector{6})
    return SVector(B, scalargrowth * u[1], 2 * scalargrowth * u[2],
                   3 * scalargrowth * u[3], 4 * scalargrowth * u[4],
                   solvent_rates[1])
end

@inline function _mom_rhs(CryProblem, scalargrowth, B, solvent_rates,
                          u::SVector{N}) where {N}
    n_states = N
    n_solvent = length(solvent_rates)
    n_population = n_states - n_solvent
    return SVector(ntuple(Val(n_states)) do k
        k <= n_population ?
            (k == 1 ? B : (k - 1) * scalargrowth * u[k - 1]) :
            solvent_rates[k - n_population]
    end)
end

function _solvent_derivatives(problem::CrystallisationProblem, state, time, growth)
    return _solvent_derivatives(problem, state, time, growth,
                                problem.solvent_dynamics)
end

function _solvent_derivatives(problem::CrystallisationProblem, state, time, growth,
                              solvent_dynamics)
    rates = solvent_dynamics(problem, state, time, growth)
    names = propertynames(problem.initial_solvent_state)
    if rates isa NamedTuple
        return ntuple(index -> begin
            name = names[index]
            hasproperty(rates, name) ? getproperty(rates, name) : zero(growth)
        end, Val(length(names)))
    end
    length(rates) == length(names) ||
        throw(ArgumentError("solvent_dynamics must return one rate per initial solvent variable."))
    return rates
end

function _write_solvent_derivatives!(dst, problem::CrystallisationProblem, rates)
    n_solvent = length(propertynames(problem.initial_solvent_state))
    first_solvent = length(dst) - n_solvent + 1
    @inbounds for index in 1:n_solvent
        dst[first_solvent + index - 1] = rates[index]
    end
    return nothing
end

function _solvent_solution_state(problem::CrystallisationProblem, solution)
    names = propertynames(problem.initial_solvent_state)
    n_solvent = length(names)
    first_solvent = size(solution, 1) - n_solvent + 1
    values = ntuple(index -> solution[first_solvent + index - 1, :], Val(n_solvent))
    return NamedTuple{names}(values)
end

function _solvent_solution_index(problem::CrystallisationProblem, solution, name::Symbol)
    position = findfirst(==(name), propertynames(problem.initial_solvent_state))
    position === nothing && throw(ArgumentError("Unknown solvent-state variable :$name."))
    return size(solution, 1) - length(propertynames(problem.initial_solvent_state)) + position
end

function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                                       noaggregation, MoM,
                                                                       NuP, GrP, BrP, AggP,
                                                                       TP},
                                    saveat) where {NuclF <:
                                                   AbstractNucleationFunction,
                                                   GrF <:
                                                   AbstractGrowthFunction,
                                                   NuP <:
                                                   AbstractVector{<:Real},
                                                   GrP <:
                                                   AbstractVector{<:Real},
                                                   BrP <:
                                                   AbstractVector{<:Real},
                                                   AggP <:
                                                   AbstractVector{<:Real},
                                                   TP <:
                                                   AbstractTemperature}
    n_mom = CryProblem.solver.nmoments
    @assert n_mom >= 2 "MoM solver requires nmoments >= 2 (concentration closure uses µ2)"

    function MoM_model(u, p, t)
        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, u, t)
        B = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                           CryProblem, u, t)
        solvent_rates = _solvent_derivatives(CryProblem, u, t, scalargrowth)
        return _mom_rhs(CryProblem, scalargrowth, B, solvent_rates, u)
    end

    θ = ComponentArray(;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth)

    ET = eltype(CryProblem.parameterset_nucleation)
    n_mom = CryProblem.solver.nmoments
    u0_vec = ET.(_get_initial_state(CryProblem))
    n_states = n_mom + 1 + length(propertynames(CryProblem.initial_solvent_state))
    u0_typed = SVector(ntuple(k -> u0_vec[k], Val(n_states)))
    ODEprob = ODEProblem(MoM_model, u0_typed, (saveat[1], saveat[end]), θ)

    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5)
    return (ODEprob, tstep_solver)
end

function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                           noaggregation, MoM,
                                                           NuP, GrP, BrP, AggP,
                                                           TP},
                        sol) where {NuclF <: AbstractNucleationFunction,
                                    GrF <: AbstractGrowthFunction,
                                    NuP <: AbstractVector{<:Real},
                                    GrP <: AbstractVector{<:Real},
                                    BrP <: AbstractVector{<:Real},
                                    AggP <: AbstractVector{<:Real},
                                    TP <: AbstractTemperature}
    final_state = collect(sol[:, end])

    n_mom = CryProblem.solver.nmoments
    n_states = n_mom + 1 + length(propertynames(CryProblem.initial_solvent_state))
    # Fixed moment indices: state k holds µ_{k-1}; µ2 = state 3, µ3 = state 4,
    # µ4 = state 5. Higher moments (if any) do not change these metrics.
    d32 = n_mom >= 3 ? CRISTOOL_MICROMETER_SCALE .* sol[4, :] ./
                       (sol[3, :] .+ CRISTOOL_MOMENT_RATIO_FLOOR) :
          fill(NaN, length(sol.t))
    d43 = n_mom >= 4 ? CRISTOOL_MICROMETER_SCALE .* sol[5, :] ./
                       (sol[4, :] .+ CRISTOOL_MOMENT_RATIO_FLOOR) :
          fill(NaN, length(sol.t))
    mu2 = n_mom >= 2 ? sol[3, :] : fill(NaN, length(sol.t))
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

    return CrystallisationMoMSolution(sol.t,
                                      solvent_solution_state.concentration,
                                      CRISTOOL_MICROMETER_SCALE * sol[2, :] ./
                                      (sol[1, :] .+ CRISTOOL_MOMENT_RATIO_FLOOR),
                                      d32,
                                      d43,
                                      mu2,
                                      solvent_solution_state,
                                      final_state,
                                      sol.stats,
                                      OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end

function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, nobreakage,
                                                                     noaggregation, MoM,
                                                                     NuP, GrP, BrP, AggP,
                                                                     TP},
                                  saveat)::CrystallisationMoMSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractGrowthFunction,
                                                                             NuP <:
                                                                             AbstractVector{<:Real},
                                                                             GrP <:
                                                                             AbstractVector{<:Real},
                                                                             BrP <:
                                                                             AbstractVector{<:Real},
                                                                             AggP <:
                                                                             AbstractVector{<:Real},
                                                                             TP <:
                                                                             AbstractTemperature}
    ODEprob, tstep_solver = crystallisation_odeproblem(CryProblem, saveat)
    sol = solve(ODEprob,
                tstep_solver;
                saveat = saveat,
                reltol = CryProblem.solver.reltol,
                abstol = CryProblem.solver.abstol,)
    return _wrap_solution(CryProblem, sol)
end

"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., FiniteVol, ...}, saveat) -> CrystallisationFVSolution

Simulate crystallisation using Finite Volume solver with scalar growth.

Internal function that solves the population balance equation using high-resolution
finite volume method with flux limiters.

# Arguments
- `CryProblem`: Crystallisation problem with FiniteVol solver and scalar growth function
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationFVSolution` containing time, concentration, number density, and size quantiles
"""
    function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPScalarGrowthFunction,
                                                                             BrF <:
                                                                             AbstractBreakageFunction,
                                                                             AggF <:
                                                                             AbstractAggregationFunction,
                                                                             NuP <:
                                                                             AbstractVector{<:Real},
                                                                             GrP <:
                                                                             AbstractVector{<:Real},
                                                                             BrP <:
                                                                             AbstractVector{<:Real},
                                                                             AggP <:
                                                                             AbstractVector{<:Real},
                                                                             TP <:
                                                                             AbstractTemperature}

    # Pre-allocate flux cache using DiffCache for ForwardDiff compatibility
    # DiffCache automatically handles type conversion for dual numbers during AD
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))

    function HRFV_FLWmodel(dstdt, st, p, t)
        # Get properly-typed cache based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)

    numberdensity = crystal_state(CryProblem, st)

        cell_centre = CryProblem.solver.cell_centre

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, st, t)

        # Calculate flux into _flux_cache
        _flux_cache[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                                        p.nucl,
                                        CryProblem, st, t) ## Inflow

        _flux_cache[2] = scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2])

        for idx_cell in 3:length(numberdensity) # Corresponds to _flux_cache[idx_cell]
            grad_up = numberdensity[idx_cell - 1] - numberdensity[idx_cell - 2]
            grad_down = numberdensity[idx_cell] - numberdensity[idx_cell - 1]
            r = grad_up / max(eps(eltype(st)), grad_down)
            _flux_cache[idx_cell] = scalargrowth * (numberdensity[idx_cell - 1] +
                                      0.5 *
                                      fluxlimiter_ospre(r) *
                                      grad_down)
        end

        _flux_cache[length(numberdensity) + 1] = scalargrowth * (numberdensity[end] +
                                                   0.5 * (numberdensity[end] -
                                                    numberdensity[end - 1]))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)

        for i in 1:length(dstdt_nd_view) # i is cell index
            dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                               CryProblem.solver.cell_dL
        end

        # Add aggregation and breakage terms
        agg_rate = aggregationrate(CryProblem.kinetics_aggregationfunction,
                                   p.agg,
                                   CryProblem, st, t)
        br_rate = breakagerate(CryProblem.kinetics_breakagefunction,
                               p.br,
                               CryProblem, st, t)

        if !(typeof(agg_rate) <: Real && agg_rate == 0.0)
            dstdt_nd_view .+= agg_rate
        end
        if !(typeof(br_rate) <: Real && br_rate == 0.0)
            dstdt_nd_view .+= br_rate
        end

        solvent_rates = _solvent_derivatives(CryProblem, st, t, scalargrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

        return nothing
    end

    # function CFLcallback(c)
    #     return CryProblem.solver.cell_dL[1] /
    #            growthrate(CryProblem.kinetics_growthfunction,
    #                       CryProblem.parameterset_growth,
    #                       c / _get_saturationconcentration(CryProblem.temp_profile, t),
    #                       CryProblem, temperature(CryProblem.temp_profile, t),
    #                       zeros(Float64, CryProblem.solver.meshsize))
    # end
    function CFLcallback(u, integrator, p, t)
        return 0.99 * CryProblem.solver.cell_dL[1] /
               growthrate(CryProblem.kinetics_growthfunction,
                          p.gr, CryProblem, u, t)
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)

    # Construct initial conditions with the correct element type
    u0_typed = eltype(CryProblem.parameterset_nucleation).(_get_initial_state(CryProblem))

    # println("Initial state shape: ", size(u0_typed))

    ODEprob = ODEProblem(HRFV_FLWmodel,
                         u0_typed, # eltype-matched plain state (do NOT convert to the params type)
                         (saveat[1], saveat[end]),
                         θ)
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5;
                                        step_limiter = CFLcallback);
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                       FiniteVol, NuP, GrP,
                                                                       BrP, AggP, TP},
                                    sol) where {NuclF <:
                                                                              AbstractNucleationFunction,
                                                                              GrF <:
                                                                              AbstractFPScalarGrowthFunction,
                                                                              BrF <:
                                                                              AbstractBreakageFunction,
                                                                              AggF <:
                                                                              AbstractAggregationFunction,
                                                                              NuP <:
                                                                              AbstractVector{<:Real},
                                                                              GrP <:
                                                                              AbstractVector{<:Real},
                                                                              BrP <:
                                                                              AbstractVector{<:Real},
                                                                              AggP <:
                                                                              AbstractVector{<:Real},
                                                                              TP <:
                                                                              AbstractTemperature}

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    nd_matrix = sol[1:(end - n_solvent), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre,
                                              nd_matrix,
                                              CryProblem.kv)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

    return CrystallisationFVSolution(sol.t,
                                     solvent_solution_state.concentration,
                                     nd_matrix,
                                     vol_weighted_dens,
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.1),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.5),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.9),
                                     moments.d10,
                                     moments.d32,
                                     moments.d43,
                                     moments.mu2,
                                     solvent_solution_state,
                                     vec(sol[:, end]),
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat)::CrystallisationFVSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPScalarGrowthFunction,
                                                                             BrF <:
                                                                             AbstractBreakageFunction,
                                                                             AggF <:
                                                                             AbstractAggregationFunction,
                                                                             NuP <:
                                                                             AbstractVector{<:Real},
                                                                             GrP <:
                                                                             AbstractVector{<:Real},
                                                                             BrP <:
                                                                             AbstractVector{<:Real},
                                                                             AggP <:
                                                                             AbstractVector{<:Real},
                                                                             TP <:
                                                                             AbstractTemperature}

    ODEprob, tstep_solver = crystallisation_odeproblem(CryProblem, saveat)
    ODEsol = solve(ODEprob,
                   tstep_solver;
                   reltol = CryProblem.solver.reltol,
                   abstol = CryProblem.solver.abstol,
                   dense = false,
                   alg_hints = [:stiff],
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS,)
    return _wrap_solution(CryProblem, ODEsol)
end

"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., FiniteVol, ...}, saveat) -> CrystallisationFVSolution

Simulate crystallisation using Finite Volume solver with length-dependent growth.

Internal function that solves the population balance equation using high-resolution
finite volume method with size-dependent growth rates.

# Arguments
- `CryProblem`: Crystallisation problem with FiniteVol solver and length-based growth function
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationFVSolution` containing time, concentration, number density, and size quantiles
"""
    function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPLengthGrowthFunction,
                                                                             BrF <:
                                                                             AbstractBreakageFunction,
                                                                             AggF <:
                                                                             AbstractAggregationFunction,
                                                                             NuP <:
                                                                             AbstractVector{<:Real},
                                                                             GrP <:
                                                                             AbstractVector{<:Real},
                                                                             BrP <:
                                                                             AbstractVector{<:Real},
                                                                             AggP <:
                                                                             AbstractVector{<:Real},
                                                                             TP <:
                                                                             AbstractTemperature}
    # Pre-allocate flux cache using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))

    function HRFV_FLWmodel(dstdt, st, p, t)
        # Get properly-typed cache based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)

        numberdensity = crystal_state(CryProblem, st)

        cell_centre = CryProblem.solver.cell_centre

        lengthbasedgrowth = growthrate(CryProblem.kinetics_growthfunction,
                                       p.gr, CryProblem, st, t)

        # Calculate flux into _flux_cache
        _flux_cache[1] = nucleationrate(CryProblem.kinetics_nucleationfunction,
                                        p.nucl, CryProblem, st, t) ## Inflow

        # Central difference for flux at first interior interface
        g_interface = 0.5 * (lengthbasedgrowth[1] + lengthbasedgrowth[2])
        _flux_cache[2] = g_interface * 0.5 * (numberdensity[1] + numberdensity[2])

        # High-resolution scheme for other interior interfaces
        for idx_cell in 3:length(numberdensity) # Corresponds to _flux_cache[idx_cell]
            g_interface = 0.5 *
                          (lengthbasedgrowth[idx_cell - 1] + lengthbasedgrowth[idx_cell])
            grad_up = numberdensity[idx_cell - 1] - numberdensity[idx_cell - 2]
            grad_down = numberdensity[idx_cell] - numberdensity[idx_cell - 1]
            r = grad_up / max(eps(eltype(st)), grad_down)
            reconstructed_n = numberdensity[idx_cell - 1] +
                              0.5 *
                              fluxlimiter_ospre(r) *
                              grad_down
            _flux_cache[idx_cell] = g_interface * reconstructed_n
        end

        # Outflow boundary condition
        _flux_cache[length(numberdensity) + 1] = lengthbasedgrowth[end] *
                                                 (numberdensity[end] +
                                                  0.5 * (numberdensity[end] -
                                                   numberdensity[end - 1]))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)

        for i in 1:length(dstdt_nd_view) # i is cell index
            dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                               CryProblem.solver.cell_dL
        end

        # Add aggregation and breakage terms
        agg_rate = aggregationrate(CryProblem.kinetics_aggregationfunction,
                                   p.agg,
                                   CryProblem, st, t)
        br_rate = breakagerate(CryProblem.kinetics_breakagefunction,
                               p.br,
                               CryProblem, st, t)

        if !(typeof(agg_rate) <: Real && agg_rate == 0.0)
            dstdt_nd_view .+= agg_rate
        end
        if !(typeof(br_rate) <: Real && br_rate == 0.0)
            dstdt_nd_view .+= br_rate
        end

        solvent_rates = _solvent_derivatives(CryProblem, st, t, lengthbasedgrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

        return nothing
    end

    function CFLcallback(u, integrator, p, t)
        g_vec = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                           CryProblem, u, t)
        return 0.99 * CryProblem.solver.cell_dL / maximum(g_vec)
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)

    # Construct initial conditions with the correct element type
    u0_typed = eltype(CryProblem.parameterset_nucleation).(_get_initial_state(CryProblem))

    ODEprob = ODEProblem(HRFV_FLWmodel,
                         u0_typed, # eltype-matched plain state (do NOT convert to the params type)
                         (saveat[1], saveat[end]),
                         θ)
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :ssprk43;
                                        step_limiter = CFLcallback);
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   sol) where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPLengthGrowthFunction,
                                                                             BrF <:
                                                                             AbstractBreakageFunction,
                                                                             AggF <:
                                                                             AbstractAggregationFunction,
                                                                             NuP <:
                                                                             AbstractVector{<:Real},
                                                                             GrP <:
                                                                             AbstractVector{<:Real},
                                                                             BrP <:
                                                                             AbstractVector{<:Real},
                                                                             AggP <:
                                                                             AbstractVector{<:Real},
                                                                             TP <:
                                                                             AbstractTemperature}

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    nd_matrix = sol[1:(end - n_solvent), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre,
                                              nd_matrix,
                                              CryProblem.kv)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

    return CrystallisationFVSolution(sol.t,
                                     solvent_solution_state.concentration,
                                     nd_matrix,
                                     vol_weighted_dens,
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.1),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.5),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens,
                                                        0.9),
                                     moments.d10,
                                     moments.d32,
                                     moments.d43,
                                     moments.mu2,
                                     solvent_solution_state,
                                     vec(sol[:, end]),
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                      FiniteVol, NuP, GrP,
                                                                      BrP, AggP, TP},
                                   saveat)::CrystallisationFVSolution where {NuclF <:
                                                                             AbstractNucleationFunction,
                                                                             GrF <:
                                                                             AbstractFPLengthGrowthFunction,
                                                                             BrF <:
                                                                             AbstractBreakageFunction,
                                                                             AggF <:
                                                                             AbstractAggregationFunction,
                                                                             NuP <:
                                                                             AbstractVector{<:Real},
                                                                             GrP <:
                                                                             AbstractVector{<:Real},
                                                                             BrP <:
                                                                             AbstractVector{<:Real},
                                                                             AggP <:
                                                                             AbstractVector{<:Real},
                                                                             TP <:
                                                                             AbstractTemperature}
    # Pre-allocate flux cache using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))

    ODEprob, tstep_solver = crystallisation_odeproblem(CryProblem, saveat)
    ODEsol = solve(ODEprob,
                   tstep_solver;
                   reltol = CryProblem.solver.reltol,
                   abstol = CryProblem.solver.abstol,
                   dense = false,
                   alg_hints = [:stiff],
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS,)
    return _wrap_solution(CryProblem, ODEsol)
end


"""
    _simulatecrystallisation(CryProblem::CrystallisationProblem{..., WENO, ...}, saveat) -> CrystallisationFVSolution

Simulate crystallisation using WENO (Weighted Essentially Non-Oscillatory) solver.

Internal function that solves the population balance equation using 5th-order WENO
reconstruction for high accuracy near discontinuities.

# Arguments
- `CryProblem`: Crystallisation problem with WENO solver
- `saveat`: Time points at which to save solution

# Returns
- `CrystallisationFVSolution` containing time, concentration, number density, and size quantiles
"""
function crystallisation_odeproblem(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  saveat) where {NuclF <:
                                                                            AbstractNucleationFunction,
                                                                            GrF <:
                                                                            AbstractGrowthFunction,
                                                                            BrF <:
                                                                            AbstractBreakageFunction,
                                                                            AggF <:
                                                                            AbstractAggregationFunction,
                                                                            NuP <:
                                                                            AbstractVector{<:Real},
                                                                            GrP <:
                                                                            AbstractVector{<:Real},
                                                                            BrP <:
                                                                            AbstractVector{<:Real},
                                                                            AggP <:
                                                                            AbstractVector{<:Real},
                                                                            TP <:
                                                                            AbstractTemperature}
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))

    function WENO_Model(dstdt, st, p, t)
        # Get properly-typed caches based on state vector type
        _flux_cache = get_tmp(_flux_cache_dc, st)
        _ndens_pad_cache = get_tmp(_ndens_pad_cache_dc, st)

        numberdensity = crystal_state(CryProblem, st)
        cell_centre = CryProblem.solver.cell_centre
        temp = temperature(CryProblem.temp_profile, t)
        S = supersaturation(CryProblem, st, t)

        scalargrowth = growthrate(CryProblem.kinetics_growthfunction, p.gr,
                                  CryProblem, st, t)
        inflowbc = nucleationrate(CryProblem.kinetics_nucleationfunction, p.nucl,
                                  CryProblem, st, t)

        # Fill padded density cache
        _ndens_pad_cache[1] = inflowbc
        _ndens_pad_cache[2] = inflowbc
        _ndens_pad_cache[3:(end - 2)] .= numberdensity
        _ndens_pad_cache[end - 1] = zero(eltype(st))
        _ndens_pad_cache[end] = zero(eltype(st))

        # Calculate flux into _flux_cache
        _flux_cache[1] = inflowbc
        _flux_cache[2] = scalargrowth * 0.5 * (numberdensity[1] + numberdensity[2])
        for i in 3:length(numberdensity)
            _flux_cache[i] = scalargrowth * weno_flux(_ndens_pad_cache, i + 1)
        end
        _flux_cache[end] = scalargrowth * (numberdensity[end] +
                            0.5 * (numberdensity[end] - numberdensity[end - 1]))

        # Calculate dstdt for numberdensity part
        dstdt_nd_view = crystal_state(CryProblem, dstdt)
        for i in 1:length(dstdt_nd_view)
            dstdt_nd_view[i] = -(_flux_cache[i + 1] - _flux_cache[i]) /
                               CryProblem.solver.cell_dL
        end

        # Add aggregation and breakage terms
        agg_rate = aggregationrate(CryProblem.kinetics_aggregationfunction, p.agg,
                                   CryProblem, st, t)
        br_rate = breakagerate(CryProblem.kinetics_breakagefunction, p.br,
                               CryProblem, st, t)

        if !(typeof(agg_rate) <: Real && agg_rate == 0.0)
            dstdt_nd_view .+= agg_rate
        end
        if !(typeof(br_rate) <: Real && br_rate == 0.0)
            dstdt_nd_view .+= br_rate
        end

        solvent_rates = _solvent_derivatives(CryProblem, st, t, scalargrowth)
        _write_solvent_derivatives!(dstdt, CryProblem, solvent_rates)

        return nothing
    end

    θ = (;
                       nucl = CryProblem.parameterset_nucleation,
                       gr = CryProblem.parameterset_growth,
                       br = CryProblem.parameterset_breakage,
                       agg = CryProblem.parameterset_aggregation)

    ET = eltype(CryProblem.parameterset_nucleation) # Assumes this reflects TPara
    utyped = ET.(_get_initial_state(CryProblem)) # Convert initial state to correct type
    ODEprob = ODEProblem(WENO_Model,
                         utyped, # NuP is type of parameterset_nucleation
                         (saveat[1], saveat[end]),
                         θ)

    function CFLcallback!(u, integrator, p, t)
        return 0.9 * CryProblem.solver.cell_dL[1] /
               growthrate(CryProblem.kinetics_growthfunction,
                          p.gr, CryProblem, u, t)
    end
    tstep_solver = _resolve_timestepping_algorithm(CryProblem.solver, :tsit5;
                                                   stage_limiter = CFLcallback!);
    return (ODEprob, tstep_solver)
end
function _wrap_solution(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  sol) where {NuclF <:
                                                                            AbstractNucleationFunction,
                                                                            GrF <:
                                                                            AbstractGrowthFunction,
                                                                            BrF <:
                                                                            AbstractBreakageFunction,
                                                                            AggF <:
                                                                            AbstractAggregationFunction,
                                                                            NuP <:
                                                                            AbstractVector{<:Real},
                                                                            GrP <:
                                                                            AbstractVector{<:Real},
                                                                            BrP <:
                                                                            AbstractVector{<:Real},
                                                                            AggP <:
                                                                            AbstractVector{<:Real},
                                                                            TP <:
                                                                            AbstractTemperature}
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))

    n_solvent = length(propertynames(CryProblem.initial_solvent_state))
    nd_matrix = sol[1:(end - n_solvent), :]
    vol_weighted_dens = volumeweighteddensity(CryProblem.solver.cell_centre, nd_matrix,
                                              CryProblem.kv)
    moments = _momentsizes(CryProblem.solver.cell_centre, nd_matrix)
    solvent_solution_state = _solvent_solution_state(CryProblem, sol)

    return CrystallisationFVSolution(sol.t, ### Will eventually have to be changed to discretised solution
                                     solvent_solution_state.concentration,
                                     nd_matrix,
                                     vol_weighted_dens,
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens, 0.1),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens, 0.5),
                                     quantilecalculator(CryProblem.solver.cell_centre,
                                                        vol_weighted_dens, 0.9),
                                     moments.d10,
                                     moments.d32,
                                     moments.d43,
                                     moments.mu2,
                                     solvent_solution_state,
                                     sol[:, end],
                                     sol.stats,
                                     OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode))
end
function _simulatecrystallisation(CryProblem::CrystallisationProblem{NuclF, GrF, BrF, AggF,
                                                                     WENO, NuP, GrP, BrP,
                                                                     AggP, TP},
                                  saveat)::CrystallisationFVSolution where {NuclF <:
                                                                            AbstractNucleationFunction,
                                                                            GrF <:
                                                                            AbstractGrowthFunction,
                                                                            BrF <:
                                                                            AbstractBreakageFunction,
                                                                            AggF <:
                                                                            AbstractAggregationFunction,
                                                                            NuP <:
                                                                            AbstractVector{<:Real},
                                                                            GrP <:
                                                                            AbstractVector{<:Real},
                                                                            BrP <:
                                                                            AbstractVector{<:Real},
                                                                            AggP <:
                                                                            AbstractVector{<:Real},
                                                                            TP <:
                                                                            AbstractTemperature}
    # Pre-allocate caches using DiffCache for ForwardDiff compatibility
    _flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    _ndens_pad_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 4))

    ODEprob, tstep_solver = crystallisation_odeproblem(CryProblem, saveat)
    ODEsol = solve(ODEprob,
                   tstep_solver;
                   reltol = CryProblem.solver.reltol,
                   abstol = CryProblem.solver.abstol,
                   dense = false,
                   alg_hints = [:stiff],
                   saveat = saveat,
                   maxiters = CRISTOOL_MAX_SOLVER_ITERS,)

    return _wrap_solution(CryProblem, ODEsol)
end

### One-pass Runner functions
"""
    runsimulation(parameters, nucleationfunction, growthfunction,
                  aggregationfunction, breakagefunction, initialconc;
                  save_idx = 0:5.0:480.0,
                  solver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                  initial_state = nothing)

Simulate batch crystallisation for a given set of kinetic models.

# Arguments
- `parameters::AbstractArray{<:Real}`: concatenated parameter vector
  `[p_ν; p_g; p_agg; p_br]` whose length must equal
  `nucleationfunction.nparams + growthfunction.nparams +
  aggregationfunction.nparams + breakagefunction.nparams`.
- `nucleationfunction`, `growthfunction`, `aggregationfunction`,
  `breakagefunction`: kinetic models describing nucleation, growth,
  aggregation and breakage respectively.
- `initialconc::Float64`: initial solute concentration.
- `save_idx::AbstractVector{<:Real}`: times (in minutes) at which the
  solution is saved. The number of time steps is `length(save_idx)`.
- `solver::AbstractSolver`: numerical solver (finite volume or method of
  moments).
- `initial_state::AbstractVector`: optional initial number density of
  length equal to the solver mesh size when using a discretised solver.

# Returns
A tuple `(problem, solution)` where `problem` is a
`CrystallisationProblem` and `solution` is either
`CrystallisationFVSolution` or `CrystallisationMoMSolution` depending on
`solver`.
"""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractFPNucleationFunction,
                       growthfunction::AbstractFPGrowthFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing,) where {TPara <:
                                                                                               Real,
                                                                                               S <:
                                                                                               AbstractArray{<:Real}}

    ## Parameter vector length validation
    expected_nparams = nucleationfunction.nparams + growthfunction.nparams +
                       aggregationfunction.nparams + breakagefunction.nparams
    if length(parameters) != expected_nparams
        throw(ArgumentError("Parameter vector has length $(length(parameters)) but expected $expected_nparams " *
                            "(nucleation: $(nucleationfunction.nparams), growth: $(growthfunction.nparams), " *
                            "aggregation: $(aggregationfunction.nparams), breakage: $(breakagefunction.nparams))"))
    end

    ## New convention for parameter arrays will be: [nucleation, growth, aggregation, breakage]

    crproblem = CrystallisationProblem(;
                                       kinetics_nucleationfunction = nucleationfunction,
                                       kinetics_growthfunction = growthfunction,
                                       parameterset_nucleation = if nucleationfunction.nparams >
                                                                    1
                                           parameters[1:(nucleationfunction.nparams)]
                                       else
                                           TPara[0.0] # Use TPara for fixed value
                                       end,
                                       parameterset_growth = parameters[(nucleationfunction.nparams + 1):(growthfunction.nparams + nucleationfunction.nparams)],
                                       kinetics_aggregationfunction = aggregationfunction,
                                       kinetics_breakagefunction = breakagefunction,
                                       parameterset_aggregation = convert(Vector{eltype(parameters)},
                                                                          parameters[(growthfunction.nparams + nucleationfunction.nparams + 1):(growthfunction.nparams + nucleationfunction.nparams + aggregationfunction.nparams)]),
                                       parameterset_breakage = convert(Vector{eltype(parameters)},
                                                                       parameters[(growthfunction.nparams + nucleationfunction.nparams + aggregationfunction.nparams + 1):(growthfunction.nparams + nucleationfunction.nparams + aggregationfunction.nparams + breakagefunction.nparams)]),
                                       initial_concentration = initialconc,
                                       solver = solver,
                                       initial_state = initial_state,) # Allow passing an initial state
    #

    crsolution = _simulatecrystallisation(crproblem, save_idx)

    return crproblem, crsolution
end
"""
    runsimulation(parameters, nucleationfunction::AbstractDDNucleationFunction,
                  growthfunction::AbstractFPGrowthFunction, ...) -> (problem, solution)

Simulate with data-driven nucleation and first-principles growth.

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractDDNucleationFunction,
                       growthfunction::AbstractFPGrowthFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing) where {TPara <:
                                                                                              Real,
                                                                                              S <:
                                                                                              AbstractArray{<:Real}}

    ## New convention for parameter arrays will be: [nucleation, growth, aggregation, breakage]

    crproblem = CrystallisationProblem(;
                                       kinetics_nucleationfunction = nucleationfunction,
                                       kinetics_growthfunction = growthfunction,
                                       parameterset_nucleation = TPara[0.0], # Use TPara for fixed value
                                       parameterset_growth = parameters[1:(growthfunction.nparams)],
                                       kinetics_aggregationfunction = aggregationfunction,
                                       kinetics_breakagefunction = breakagefunction,
                                       parameterset_aggregation = convert(Vector{eltype(parameters)},
                                                                          parameters[(growthfunction.nparams + 1):(growthfunction.nparams + aggregationfunction.nparams)]),
                                       parameterset_breakage = convert(Vector{eltype(parameters)},
                                                                       parameters[(growthfunction.nparams + aggregationfunction.nparams + 1):(growthfunction.nparams + aggregationfunction.nparams + breakagefunction.nparams)]),
                                       initial_concentration = initialconc,
                                       solver = solver,
                                       initial_state = initial_state,) # Allow passing an initial state
    #

    crsolution = _simulatecrystallisation(crproblem, save_idx)

    return crproblem, crsolution
end
"""
    runsimulation(parameters, nucleationfunction::AbstractFPNucleationFunction,
                  growthfunction::AbstractDDGrowthFunction, ...) -> (problem, solution)

Simulate with first-principles nucleation and data-driven growth.

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractFPNucleationFunction,
                       growthfunction::AbstractDDGrowthFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing) where {TPara <:
                                                                                              Real,
                                                                                              S <:
                                                                                              AbstractArray{<:Real}}

    ## New convention for parameter arrays will be: [nucleation, growth, aggregation, breakage]

    crproblem = CrystallisationProblem(;
                                       kinetics_nucleationfunction = nucleationfunction,
                                       kinetics_growthfunction = growthfunction,
                                       parameterset_nucleation = parameters[1:(nucleationfunction.nparams)],
                                       parameterset_growth = TPara[0.0], # Use TPara for fixed value
                                       kinetics_aggregationfunction = aggregationfunction,
                                       kinetics_breakagefunction = breakagefunction,
                                       parameterset_aggregation = convert(Vector{eltype(parameters)},
                                                                          parameters[(nucleationfunction.nparams + 1):(nucleationfunction.nparams + aggregationfunction.nparams)]),
                                       parameterset_breakage = convert(Vector{eltype(parameters)},
                                                                       parameters[(nucleationfunction.nparams + aggregationfunction.nparams + 1):(nucleationfunction.nparams + aggregationfunction.nparams + breakagefunction.nparams)]),
                                       initial_concentration = initialconc,
                                       solver = solver,
                                       initial_state = initial_state,) # Allow passing an initial state
    #

    crsolution = _simulatecrystallisation(crproblem, save_idx)

    return crproblem, crsolution
end
"""
    runsimulation(parameters, nucleationfunction::AbstractDDNucleationFunction,
                  growthfunction::AbstractDDGrowthFunction, ...) -> (problem, solution)

Simulate with data-driven nucleation and data-driven growth.

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractDDNucleationFunction,
                       growthfunction::AbstractDDGrowthFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing) where {TPara <:
                                                                                              Real,
                                                                                              S <:
                                                                                              AbstractArray{<:Real}}

    ## New convention for parameter arrays will be: [nucleation, growth, aggregation, breakage]

    crproblem = CrystallisationProblem(;
                                       kinetics_nucleationfunction = nucleationfunction,
                                       kinetics_growthfunction = growthfunction,
                                       parameterset_nucleation = TPara[0.0], # Use TPara for fixed value
                                       parameterset_growth = TPara[0.0], # Use TPara for fixed value
                                       kinetics_aggregationfunction = aggregationfunction,
                                       kinetics_breakagefunction = breakagefunction,
                                       parameterset_aggregation = parameters[1:(aggregationfunction.nparams)],
                                       parameterset_breakage = parameters[(aggregationfunction.nparams + 1):(aggregationfunction.nparams + breakagefunction.nparams)],
                                       initial_concentration = initialconc,
                                       solver = solver,
                                       initial_state = initial_state,) # Allow passing an initial state
    #

    crsolution = _simulatecrystallisation(crproblem, save_idx)

    return crproblem, crsolution
end
"""
    runsimulation(nucleationfunction::AbstractDDNucleationFunction,
                  growthfunction::AbstractDDGrowthFunction, noaggregation, nobreakage,
                  initialconc; ...) -> (problem, solution)

Simulate with data-driven kinetics and no aggregation/breakage (no parameters needed).

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(nucleationfunction::AbstractDDNucleationFunction,
                       growthfunction::AbstractDDGrowthFunction,
                       aggregationfunction::noaggregation,
                       breakagefunction::nobreakage,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = FiniteVol(; meshsize = 500, lmax = 50e-6),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing) where {S <:
                                                                                              AbstractArray{<:Real}}

    ## New convention for parameter arrays will be: [nucleation, growth, aggregation, breakage]

    crproblem = CrystallisationProblem(;
                                       kinetics_nucleationfunction = nucleationfunction,
                                       kinetics_growthfunction = growthfunction,
                                       parameterset_nucleation = [0.0],
                                       parameterset_growth = [0.0],
                                       kinetics_aggregationfunction = aggregationfunction,
                                       kinetics_breakagefunction = breakagefunction,
                                       initial_concentration = initialconc,
                                       solver = solver,
                                       initial_state = initial_state,) # Allow passing an initial state
    #

    crsolution = _simulatecrystallisation(crproblem, save_idx)

    return crproblem, crsolution
end
"""
    runsimulation(parameters, nucleationfunction::AbstractFPNucleationFunction,
                  growthfunction::AbstractFPGrowthFunction, initialconc; ...) -> (problem, solution)

Simplified interface for nucleation and growth only (no aggregation/breakage).

Calls the full runsimulation with noaggregation() and nobreakage().

See main `runsimulation` docstring for full argument descriptions.
"""
function runsimulation(parameters::AbstractArray{TPara},
                       nucleationfunction::AbstractFPNucleationFunction,
                       growthfunction::AbstractFPGrowthFunction,
                       initialconc::Float64;
                       save_idx::S = 0:5.0:480.0,
                       solver::AbstractSolver = MoM(),
                       initial_state::Union{Nothing, AbstractArray{<:Real}} = nothing) where {TPara <:
                                                                                              Real,
                                                                                              S <:
                                                                                              AbstractArray{<:Real}}
    return runsimulation(parameters,
                         nucleationfunction,
                         growthfunction,
                         noaggregation(),
                         nobreakage(),
                         initialconc;
                         save_idx = save_idx,
                         solver = solver,
                         initial_state = initial_state)
end

"""
    runsimulation(parameters::ComponentArray; nucl, gr, agg, br, solver,
                  initial_concentration = 18.0, initial_state = nothing,
                  save_idx = 0:5.0:480.0, cry_kwargs...)

Core kwarg-form entry. Expects `parameters` to be a ComponentArray laid out
with the top-level axis built by `paramaxis(nucl, gr, agg, br)` — that is, with
named slices `parameters.nucl`, `parameters.gr`, `parameters.agg`, `parameters.br`.

Use this when you already hold a structured parameter vector (e.g. from an
optimiser that supports ComponentArrays). For a flat `Vector{Float64}`, call
the AbstractVector overload — it wraps the input here.
"""
function runsimulation(parameters::ComponentArrays.ComponentArray;
                       nucl::AbstractNucleationFunction = noaggregation(),
                       gr::AbstractGrowthFunction = noaggregation(),
                       agg::AbstractAggregationFunction = noaggregation(),
                       br::AbstractBreakageFunction = nobreakage(),
                       solver::AbstractSolver = FiniteVol(meshsize = 500, lmax = 50e-6),
                       initial_concentration = 18.0,
                       initial_state::Union{Nothing, AbstractVector} = nothing,
                       save_idx = 0:5.0:480.0,
                       cry_kwargs...)
    cry = CrystallisationProblem(; kinetics_nucleationfunction = nucl,
                                 kinetics_growthfunction = gr,
                                 kinetics_aggregationfunction = agg,
                                 kinetics_breakagefunction = br,
                                 parameterset_nucleation = parameters.nucl,
                                 parameterset_growth = parameters.gr,
                                 parameterset_aggregation = parameters.agg,
                                 parameterset_breakage = parameters.br,
                                 solver = solver,
                                 initial_concentration,
                                 initial_state,
                                 cry_kwargs...)

    return cry, _simulatecrystallisation(cry, save_idx)
end

"""
    runsimulation(parameters::AbstractVector; nucl, gr, agg, br, ...)

Backward-compatible wrapper for callers (optimisers, scripts) that hand in a
flat parameter vector. Validates length, builds a ComponentArray view with the
composite axis from the kinetic models, and forwards to the ComponentArray
core. AD types (e.g. `Vector{Dual}`) flow through unchanged.
"""
function runsimulation(parameters::AbstractVector;
                       nucl::AbstractNucleationFunction = noaggregation(),
                       gr::AbstractGrowthFunction = noaggregation(),
                       agg::AbstractAggregationFunction = noaggregation(),
                       br::AbstractBreakageFunction = nobreakage(),
                       kwargs...)
    nν, ng, na, nb = nucl.nparams, gr.nparams, agg.nparams, br.nparams
    expected_nparams = nν + ng + na + nb
    if length(parameters) != expected_nparams
        throw(ArgumentError("Parameter vector has length $(length(parameters)) but expected $expected_nparams " *
                            "(nucleation: $nν, growth: $ng, aggregation: $na, breakage: $nb)"))
    end
    p = ComponentArray(parameters, paramaxis(nucl, gr, agg, br))
    return runsimulation(p; nucl = nucl, gr = gr, agg = agg, br = br, kwargs...)
end
