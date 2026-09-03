@testset "StaticArrays - MoM" begin
    using StaticArrays

    nucl_func = nucl_CNT()
    grow_func = growth_empirical()
    params = [38.0, 0.0007, 1e-9 / 60, 3.0]
    initial_conc = 18.0

    # Provide a static initial state to ensure the MoM path handles StaticArrays.
    initial_state = StaticArrays.@SVector [0.0, 0.0, 0.0, 0.0, 0.0, initial_conc]

    _,
    sol = runsimulation(params,
                        nucl_func,
                        grow_func,
                        initial_conc;
                        save_idx = 0:3600.0:14400.0,
                        solver = MoM(),
                        initial_state = initial_state)

    @test sol.success
    @test sol.concentration[end] > 0
end
