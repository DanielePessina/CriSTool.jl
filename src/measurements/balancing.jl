function _variance_floor_observable(observable::Observable{T, Tt, Nothing},
                                    min_rel_std_pc) where {T <: AbstractArray, Tt}
    floor = (min_rel_std_pc * 1e-2 .* abs.(observable.mean)) .^ 2
    return Observable(; time = observable.time, mean = observable.mean,
                      variance = floor)
end

function _variance_floor_observable(observable::Observable{T, Tt, Tv},
                                    min_rel_std_pc) where {T <: AbstractArray, Tt, Tv}
    floor = (min_rel_std_pc * 1e-2 .* abs.(observable.mean)) .^ 2
    variance = max.(observable.variance, floor)
    return Observable(; time = observable.time, mean = observable.mean, variance = variance)
end

function _variance_floor_observable(observable::Observable{T, Tt, Nothing},
                                    min_rel_std_pc) where {T <: Real, Tt}
    floor = (min_rel_std_pc * 1e-2 * abs(observable.mean))^2
    return Observable(; time = observable.time, mean = observable.mean, variance = floor)
end

function _variance_floor_observable(observable::Observable{T, Tt, Tv},
                                    min_rel_std_pc) where {T <: Real, Tt, Tv}
    floor = (min_rel_std_pc * 1e-2 * abs(observable.mean))^2
    variance = max(observable.variance, floor)
    return Observable(; time = observable.time, mean = observable.mean, variance = variance)
end

"""
    balance_variances(experiments; obs, min_rel_std_pc=10)
        -> Vector{CrystallisationExperiment}

Apply a relative standard-deviation floor to one named observable across all
experiments. The experiment metadata and all other observables are preserved.
"""
function balance_variances(experiments::Vector{<:CrystallisationExperiment};
                           obs::Symbol,
                           min_rel_std_pc::Real = 10)
    newmeasurements = Vector{CrystallisationExperiment}(undef, length(experiments))
    for (index, expt) in enumerate(experiments)
        hasproperty(expt.observables, obs) ||
            throw(ArgumentError("Experiment $(expt.exp_id) has no observable :$obs."))
        measured = getproperty(expt.observables, obs)
        balanced = _variance_floor_observable(measured, min_rel_std_pc)
        newmeasurements[index] = CrystallisationExperiment(;
            observables = merge(expt.observables, NamedTuple{(obs,)}((balanced,))),
            temperature = expt.temperature,
            initial_crystals = expt.initial_crystals,
            exp_id = expt.exp_id,
            metadata = expt.metadata)
    end
    return newmeasurements
end

"""
    repeatmeasurementbalancer(experiments::Vector{CrystallisationExperiment}, minconcstd::Real=10) -> Vector{CrystallisationExperiment}

Balance concentration measurement variances by enforcing a minimum relative
variance (at least `minconcstd`% of the mean value at each timepoint).
"""
function repeatmeasurementbalancer(experiments::Vector{CrystallisationExperiment},
                                   minconcstd::Real = 10)
    return balance_variances(experiments; obs = :concentration,
                             min_rel_std_pc = minconcstd)
end

"""
    psd_measurementbalancer(experiments::Vector{CrystallisationExperiment}, psd_std_pc::Real=10) -> Vector{CrystallisationExperiment}

Balance particle size measurement variances by enforcing a minimum relative
variance (at least `psd_std_pc`% of the value for the `d43`/`d50q` scalar
observables).
"""
function psd_measurementbalancer(experiments::Vector{CrystallisationExperiment},
                                 psd_std_pc::Real = 10)
    balanced = experiments
    for observable_name in (:d43, :d50q)
        if !isempty(balanced) && hasproperty(first(balanced).observables, observable_name)
            balanced = balance_variances(balanced; obs = observable_name,
                                         min_rel_std_pc = psd_std_pc)
        end
    end
    return balanced
end
