@testset "Positive WENO uses analytic cell-average reconstruction" begin
    linear_cell_averages = [2.0 + 0.4 * centre for centre in -2:2]
    @test CriSTool.weno_flux(linear_cell_averages, 3) ≈ 2.2 atol = 2e-15

    reconstruction_errors = Float64[]
    for cell_width in (0.2, 0.1, 0.05)
        cell_centres = [1.0 + stencil_offset * cell_width for stencil_offset in -2:2]
        cell_averages = [2.0 + (cos(cell_centre - cell_width / 2) -
            cos(cell_centre + cell_width / 2)) / cell_width for cell_centre in cell_centres]
        reconstructed_density = CriSTool.weno_flux(cell_averages, 3)
        exact_face_density = 2.0 + sin(1.0 + cell_width / 2)
        push!(reconstruction_errors, abs(reconstructed_density - exact_face_density))
    end
    @test reconstruction_errors[1] / reconstruction_errors[2] > 20.0
    @test reconstruction_errors[2] / reconstruction_errors[3] > 20.0
    @test reconstruction_errors[3] < 1e-7
end

@testset "Positive WENO reconstructs the inlet-adjacent interior face" begin
    positive_flux = zeros(5)
    padded_density = zeros(8)
    CriSTool._fill_positive_weno_flux!(positive_flux, [1.0, 2.0, 4.0, 5.0],
                                       2.0, 6.0, padded_density)

    # For the inlet ghosts [3, 3] and first five stencil values [3, 3, 1, 2, 4],
    # the hand-evaluated fifth-order Jiang-Shu left trace is 1.3155079987669.
    # The old midpoint-only boundary face would instead give flux 3.0 here.
    @test padded_density[1:2] == [3.0, 3.0]
    @test positive_flux[1] == 6.0
    @test positive_flux[2] ≈ 2.631015997533807 atol = 2e-14
    @test positive_flux[end] == 11.0
end
