@testset "FV linear reconstruction and extrema" begin
    # A limiter must reproduce a linear profile in either slope direction.
    @test CriSTool.fluxlimiter_ospre(1.0) == 1.0
    @test CriSTool.fluxlimiter_ospre(0.0) == 0.0
    @test CriSTool.fluxlimiter_ospre(-2.0) == 0.0
    reconstructed_flux = zeros(5)
    CriSTool._fill_fv_scalar_flux!(reconstructed_flux, [4.0, 3.0, 2.0, 1.0], 1.0, 4.5)
    @test reconstructed_flux ≈ [4.5, 3.5, 2.5, 1.5, 0.5]
    @test -diff(reconstructed_flux) ≈ ones(4)
    CriSTool._fill_fv_scalar_flux!(reconstructed_flux, [1.0, 2.0, 3.0, 4.0], 1.0, 0.5)
    @test reconstructed_flux ≈ [0.5, 1.5, 2.5, 3.5, 4.5]
    @test -diff(reconstructed_flux) ≈ -ones(4)
    # At a peak the reconstruction uses the cell value, rather than an
    # anti-diffusive slope inferred from gradients of opposite sign.
    CriSTool._fill_fv_scalar_flux!(reconstructed_flux, [1.0, 3.0, 2.0, 1.0], 1.0, 0.0)
    @test reconstructed_flux[3] == 3.0
end
