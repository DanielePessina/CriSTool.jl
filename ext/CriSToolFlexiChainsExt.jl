"""
Resolves the method ambiguity between the vendored KissABC `AIS`
`bundle_samples` and FlexiChains' `bundle_samples(..., ::Type{VNChain})`. The
AIS method ignores the requested chain type, so both dispatch to the shared
body. Loaded only when FlexiChains is, since the ambiguity needs both methods.
"""
module CriSToolFlexiChainsExt

import AbstractMCMC
import FlexiChains
using CriSTool.KissABC: AIS, Particle, _ais_bundle_samples

function AbstractMCMC.bundle_samples(samples::Vector{T},
                                     m::AbstractMCMC.AbstractModel,
                                     s::AIS,
                                     lss,
                                     ::Type{FlexiChains.VNChain};
                                     kwargs...,) where {T <: Particle}
    return _ais_bundle_samples(samples)
end

end
