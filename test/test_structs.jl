@testset "Structs and Type Stability" begin

    @testset "Kinetic Function Constructors" begin
        # Nucleation functions
        @test nucl_CNT().nparams == 2
        @test nucl_CNT().string == "CNT"
        @test nucl_CNT().symbols == [:Aⱼ, :γ]

        @test nucl_empirical().nparams == 2
        @test nucl_empirical().symbols == [:Aj, :j]

        @test nucl_CNTnoS().nparams == 2

        # Growth functions
        @test growth_empirical().nparams == 2
        @test growth_empirical().string == "Emp. Gr"
        @test growth_empirical().symbols == [:Ag, :g]

        @test growth_BCF().nparams == 2
        @test growth_BpS().nparams == 2

        # Aggregation and breakage
        @test noaggregation().nparams == 0
        @test nobreakage().nparams == 0
    end

    @testset "Solver Constructors" begin
        # FiniteVol default construction
        fv = FiniteVol()
        @test fv.meshsize == 200
        @test fv.lmin == 0.0
        @test fv.lmax == 50.0e-6
        @test fv.string == "FV"
        @test length(fv.cell_centre) == fv.meshsize
        @test fv.cell_dL > 0

        # FiniteVol custom construction
        fv_custom = FiniteVol(meshsize = 100, lmax = 100e-6)
        @test fv_custom.meshsize == 100
        @test fv_custom.lmax == 100e-6

        # MoM construction
        mom = MoM()
        @test mom.string == "MoM"

        # WENO construction
        weno = WENO()
        @test weno.meshsize == 200
        @test weno.string == "WENO"
    end

    @testset "Temperature Profiles" begin
        # ConstantTemperature
        ct = CriSTool.ConstantTemperature(293.15)
        @test ct.value == 293.15

        # LinearTemperature
        lt = CriSTool.LinearTemperature(293.15, -0.5)
        @test lt.T0 == 293.15
        @test lt.slope == -0.5
    end

    @testset "Temperature Type Stability" begin
        # ConstantTemperature type stability
        ct = CriSTool.ConstantTemperature(293.15)
        @test @inferred(CriSTool.temperature(ct, 0.0)) == 293.15
        @test @inferred(CriSTool.temperature(ct, 100.0)) == 293.15

        # LinearTemperature type stability
        lt = CriSTool.LinearTemperature(293.15, -0.5)
        @test @inferred(CriSTool.temperature(lt, 0.0)) == 293.15
        @test @inferred(CriSTool.temperature(lt, 60.0)) ≈ 293.15 - 30.0

        # CallableTemperature type stability
        callable = CriSTool.CallableTemperature(t -> 273.15 + 20.0 * exp(-t/100))
        result = @inferred(CriSTool.temperature(callable, 0.0))
        @test result ≈ 293.15
    end

    @testset "Saturation Model Type Stability" begin
        temp_profile = CriSTool.ConstantTemperature(293.15)

        # Default lysozyme polynomial: hand-computed value at 20°C
        sm = lysozyme_saturation()
        @test sm isa PolynomialSaturation
        sat_conc = @inferred(saturation_concentration(sm, temp_profile, 0.0))
        @test sat_conc isa Float64
        # 0.3705 + 7.171e-2*20 - 1.924e-3*400 + 17.97e-5*8000
        @test sat_conc ≈ 0.3705 + 7.171e-2 * 20.0 - 1.924e-3 * 400.0 + 17.97e-5 * 8000.0
        @test sat_conc > 0

        # Without time argument == t = 0
        @test @inferred(saturation_concentration(sm, temp_profile)) ≈ sat_conc

        # Constant and callable models
        @test saturation_concentration(ConstantSaturation(2.47), temp_profile, 123.0) == 2.47
        callable = CallableSaturation((T, t) -> 0.5 * T)
        @test saturation_concentration(callable, temp_profile, 0.0) ≈ 0.5 * 293.15
    end

    @testset "Loss Function Constructors" begin
        @test logMLE().weighting == (1.0, 1.0)
        @test mae().weighting == (1.0, 1.0)

        # Custom weighting
        weighted_mle = logMLE(weighting = (0.5, 2.0))
        @test weighted_mle.weighting == (0.5, 2.0)
    end
end
