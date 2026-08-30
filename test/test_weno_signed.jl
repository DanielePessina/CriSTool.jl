"""Focused checks for the signed fifth-order WENO transport path."""

function _weno_test_padding(numberdensity)
    padded_density = zeros(eltype(numberdensity), length(numberdensity) + 4)
    padded_density[3:(end - 2)] .= numberdensity
    padded_density[1:2] .= numberdensity[1]
    padded_density[(end - 1):end] .= zero(eltype(numberdensity))
    return padded_density
end

@testset "Signed WENO transport" begin
    @testset "Left and mirrored right traces translate a smooth profile" begin
        meshsize = 64
        cell_width = 1.0 / meshsize
        # Cell averages of a smooth periodic profile are the independent
        # finite-volume data expected by the reconstruction.
        cell_average_factor = sinpi(cell_width) / (pi * cell_width)
        numberdensity = [1.0 + 0.2 * sin(2pi * (cell_index - 0.5) * cell_width) *
                         cell_average_factor for cell_index in 1:meshsize]
        padded_density = _weno_test_padding(numberdensity)
        face_index = 32
        exact_face_value = 1.0 + 0.2 * sin(2pi * (face_index - 1) * cell_width)

        left_trace = CriSTool._weno_flux_left(padded_density, face_index + 1)
        right_trace = CriSTool._weno_flux_right(padded_density, face_index + 2)
        @test left_trace ≈ exact_face_value atol = 2e-6
        @test right_trace ≈ exact_face_value atol = 2e-6
    end

    @testset "Negative scalar growth uses mirrored WENO and open boundaries" begin
        meshsize = 32
        numberdensity = [1.0 + 0.1 * sin(2pi * index / meshsize)
                         for index in 1:meshsize]
        padded_density = zeros(meshsize + 4)
        flux = zeros(meshsize + 1)
        signed_growth_rate = -2.0
        CriSTool._fill_signed_weno_flux!(flux, numberdensity, signed_growth_rate,
                                         0.0, padded_density)

        @test flux[1] == signed_growth_rate * numberdensity[1]
        @test flux[end] == 0.0
        @test all(flux[2:(end - 1)] .<= 0.0)
        # A smooth profile should use a high-order trace rather than the
        # right-cell value at an interior face.
        @test flux[16] / signed_growth_rate != numberdensity[16]
    end

    @testset "Mesh-aligned rates choose WENO direction face by face" begin
        meshsize = 24
        numberdensity = collect(range(0.5, 2.0; length = meshsize))
        signed_growth_rates = vcat(fill(1.0, meshsize ÷ 2),
                                   fill(-1.0, meshsize - meshsize ÷ 2))
        flux = zeros(meshsize + 1)
        padded_density = zeros(meshsize + 4)
        CriSTool._fill_signed_weno_flux!(flux, numberdensity,
                                         signed_growth_rates, 3.0,
                                         padded_density)

        @test flux[1] == 3.0
        @test all(flux[2:(meshsize ÷ 2)] .>= 0.0)
        @test flux[meshsize ÷ 2 + 1] == 0.0
        @test all(flux[(meshsize ÷ 2 + 2):end] .<= 0.0)
        @test flux[end] == 0.0
    end

    @testset "Nonnegative interface correction and conservative balance" begin
        meshsize = 20
        numberdensity = zeros(meshsize)
        numberdensity[8:12] .= 1.0
        signed_growth_rates = vcat(fill(1.0, 10), fill(-1.0, 10))
        flux = zeros(meshsize + 1)
        padded_density = zeros(meshsize + 4)
        CriSTool._fill_signed_weno_flux!(flux, numberdensity,
                                         signed_growth_rates, 0.0,
                                         padded_density)

        # Flux signs follow the signed physical rate, while the reconstructed
        # interface states remain nonnegative after the local correction.
        for face_index in 2:meshsize
            face_rate = 0.5 * (signed_growth_rates[face_index - 1] +
                               signed_growth_rates[face_index])
            if face_rate > 0.0
                @test flux[face_index] >= 0.0
            elseif face_rate < 0.0
                @test flux[face_index] <= 0.0
            else
                @test flux[face_index] == 0.0
            end
        end

        cell_width = 0.25
        density_derivative = [-(flux[cell_index + 1] - flux[cell_index]) /
                              cell_width for cell_index in 1:meshsize]
        @test sum(density_derivative) * cell_width ≈ -(flux[end] - flux[1])
    end
end
