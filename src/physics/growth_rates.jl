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
