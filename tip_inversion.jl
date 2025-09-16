# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac tip_inversion module on Julia language.

Converted from PyFrac/tip_inversion.py
Created by Haseeb Zia on Tue Nov 01 15:22:00 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""

module TipInversion

    using Logging
    using Roots
    using Optim

    include("properties.jl")
    using .Properties: instrument_start, instrument_close

    export C1, C2, TipAsym_k_exp, TipAsym_m_exp, TipAsym_mt_exp, TipAsym_viscStor_Res,
        TipAsym_MDR_Res, TipAsym_M_MDR_Res, TipAsym_viscLeakOff_Res, TipAsym_MK_zrthOrder_Res,
        TipAsym_MK_deltaC_Res, TipAsym_MTildeK_zrthOrder_Res, TipAsym_MTildeK_deltaC_Res,
        f, TipAsym_Universal_1stOrder_Res, TipAsym_Universal_zrthOrder_Res,
        TipAsym_Hershcel_Burkley_Res, TipAsym_power_law_Res, TipAsym_Hershcel_Burkley_MK_Res,
        TipAsym_power_law_MK_Res, TipAsym_PowerLaw_M_vertex_Res, TipAsym_variable_Toughness_Res,
        Vm_residual, FindBracket_dist, TipAsymInversion, StressIntensityFactor,
        TipAsymInversion_hetrogenous_toughness, find_zero_vertex

    # Константы
    const beta_m = 2^(1/3) * 3^(5/6)
    const beta_mtld = 4/(15^(1/4) * (2^0.5 - 1)^(1/4))
    const cnst_mc = 3 * beta_mtld^4 / (4 * beta_m^3)
    const cnst_m = beta_m^3 / 3
    const Ki_c = 3000

    """
        C1(delta)

        # Arguments
        - `delta::Float64`: delta parameter

        # Returns
        - `Float64`: C1 value
    """
    function C1(delta::Float64)::Float64
        if (delta >= 1 || delta <= 0)
            return cnst_m
        else
            return 4 * (1 - 2 * delta) / (delta * (1 - delta)) * tan(π * delta)
        end
    end

    """
        C2(delta)

        # Arguments
        - `delta::Float64`: delta parameter

        # Returns
        - `Float64`: C2 value
    """
    function C2(delta::Float64)::Float64
        if delta == 1/3
            return beta_mtld^4 / 4
        else
            return 16 * (1 - 3 * delta) / (3 * delta * (2 - 3 * delta)) * tan(3 * π / 2 * delta)
        end
    end

    """
        TipAsym_k_exp(dist, args...)

        Residual function for the near-field k expansion (Garagash & Detournay, 2011)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_k_exp(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        V = (dist - DistLstTSEltRibbon) / dt
        l_mk = (Kprime^3 / (Eprime^2 * fluidProp.muPrime * V))^2
        l_mtk = Kprime^8 / (Eprime^6 * fluidProp.muPrime^2 * (2 * Cbar)^2 * V)
        l1 = (l_mk^(-1/2) + l_mtk^(-1/2))^(-2)
        l2 = (2 / 3 * l_mk^(-1/2) + l_mtk^(-1/2))^(-2)

        return -wEltRibbon + (Kprime / Eprime)^2 * dist^(1/2) * (1 + 4 * π * (dist/l1)^(1/2) + 64 *
                                                                        (dist * log(dist) / (l1 * l2)^(1/2)))
    end

    """
        TipAsym_m_exp(dist, args...)

        Residual function for the far-field m expansion (Garagash & Detournay, 2011)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_m_exp(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        V = (dist - DistLstTSEltRibbon) / dt
        l_mmt = (2 * Cbar)^6 * Eprime^2 / (V^5 * fluidProp.muPrime^2)

        return -wEltRibbon + (V * fluidProp.muPrime / Eprime)^(1/3) * dist^(2/3) * (beta_m + 1 / 2 * (l_mmt / dist)^(1/6)
                                                                                - 3^(1/6) / 2^(7/3) * (l_mmt / dist)^(1/3)
                                                                                + 2^(7/3) / 3^(5/3) * (l_mmt / dist)^(1/2)
                                                                                - 0.7406 * (l_mmt / dist)^(2/3 - 0.1387))
    end

    """
        TipAsym_mt_exp(dist, args...)

        Residual function for the intermediate-field m expansion (Garagash & Detournay, 2011)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_mt_exp(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        V = (dist - DistLstTSEltRibbon) / dt
        l_mtk = Kprime^8 / (Eprime^6 * fluidProp.muPrime^2 * (2 * Cbar)^2 * V)
        l_mmt = (2 * Cbar)^6 * Eprime^2 / (V^5 * fluidProp.muPrime^2)

        return -wEltRibbon + (2 * Cbar * V^(1/2) * fluidProp.muPrime / Eprime)^(1/4) * dist^(5/8) * (0.0161 * (l_mtk / dist)^(5/8 - 0.06999)
                                                                                                    + 2.53356 + 1.30165 * (dist/l_mmt)^(1/8)
                                                                                                    - 0.451609 * (dist/l_mmt)^(1/4)
                                                                                                    + 0.183355 * (dist/l_mmt)^(3/8))
    end

    """
        TipAsym_viscStor_Res(dist, args...)

        Residual function for viscosity dominate regime, without leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_viscStor_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        return wEltRibbon - (18 * 3^0.5 * (dist - DistLstTSEltRibbon) / dt * muPrime / Eprime)^(1 / 3) * dist^(
                2 / 3)
    end

    """
        TipAsym_MDR_Res(dist, args...)

        Residual function for viscosity dominate regime, without leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MDR_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        density = 1000

        return wEltRibbon - (1.89812 * dist^0.740741 * ((dist - DistLstTSEltRibbon) / dt)^0.481481 * (
                    muPrime^0.7 * density^0.3)^0.37037) / Eprime^0.37037
    end

    """
        TipAsym_M_MDR_Res(dist, args...)

        Residual function for viscosity dominate regime, without leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_M_MDR_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        density = 1000
        Vel = (dist - DistLstTSEltRibbon) / dt

        return wEltRibbon - 3.14735 * dist^(2/3) * ((dist - DistLstTSEltRibbon) / dt)^(1/3) * muPrime^(1/3) * (1 +
        0.255286 * dist^0.2 * Vel^0.4 * density^0.3 / (Eprime^0.1 * muPrime^0.2))^0.37037 / Eprime^(1/3)
    end

    """
        TipAsym_viscLeakOff_Res(dist, args...)

        Residual function for viscosity dominated regime, with leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_viscLeakOff_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        return wEltRibbon - 4 / (15 * tan(π / 8))^0.25 * (2 * Cbar * muPrime / Eprime)^0.25 * ((dist -
                DistLstTSEltRibbon) / dt)^0.125 * dist^(5 / 8)
    end

    """
        TipAsym_MK_zrthOrder_Res(dist, args...)

        Residual function for viscosity to toughness regime with transition, without leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MK_zrthOrder_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        if Kprime == 0
            return TipAsym_viscStor_Res(dist, args...)
        end
        if muPrime == 0
            # return toughness dominated asymptote
            return dist - wEltRibbon^2 * (Eprime / Kprime)^2
        end

        w_tld = Eprime * wEltRibbon / (Kprime * dist^0.5)
        V = (dist - DistLstTSEltRibbon) / dt
        return w_tld - (1 + beta_m^3 * Eprime^2 * V * dist^0.5 * muPrime / Kprime^3)^(1/3)
    end

    """
        TipAsym_MK_deltaC_Res(dist, args...)

        Residual function for viscosity to toughness regime with transition, without leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MK_deltaC_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        if Kprime == 0
            return TipAsym_viscStor_Res(dist, args..., fluidProp.muPrime)
        end
        if fluidProp.muPrime == 0
            # return toughness dominated asymptote
            return dist - wEltRibbon^2 * (Eprime / Kprime)^2
        end

        w_tld = Eprime * wEltRibbon / (Kprime * dist^0.5)

        V = (dist - DistLstTSEltRibbon) / dt
        l_mk = (Kprime^3 / (Eprime^2 * fluidProp.muPrime * V))^2
        x_tld = (dist / l_mk)^(1/2)
        delta = 1 / 3 * beta_m^3 * x_tld / (1 + beta_m^3 * x_tld)
        return w_tld - (1 + 3 * C1(delta) * x_tld)^(1/3)
    end

    """
        TipAsym_MTildeK_zrthOrder_Res(dist, args...)

        Residual function for zeroth-order solution for M~K edge tip asymptote

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MTildeK_zrthOrder_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        w_tld = Eprime * wEltRibbon / (Kprime * dist^0.5)
        V = (dist - DistLstTSEltRibbon) / dt
        return -w_tld + (1 + beta_mtld^4 * 2 * Cbar * Eprime^3 * dist^0.5 * V^0.5 * fluidProp.muPrime / Kprime^4)^(1/4)
    end

    """
        TipAsym_MTildeK_deltaC_Res(dist, args...)

        Residual function for viscosity to toughness regime with transition, without leak off

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MTildeK_deltaC_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        w_tld = Eprime * wEltRibbon / (Kprime * dist^0.5)

        V = (dist - DistLstTSEltRibbon) / dt
        l_mk = (Kprime^3 / (Eprime^2 * fluidProp.muPrime * V))^2
        chi = 2 * Cbar * Eprime / (V^0.5 * Kprime)
        x_tld = (dist / l_mk)^(1/2)
        delta = 1 / 4 * beta_mtld^4 * chi * x_tld / (1 + beta_mtld^4 * chi * x_tld)
        return w_tld - (1 + 4 * C2(delta) * x_tld * chi)^(1/3)
    end

    """
        f(K, Cb, Con)

        # Arguments
        - `K::Float64`: K parameter
        - `Cb::Float64`: Cb parameter
        - `Con::Float64`: Con parameter

        # Returns
        - `Float64`: f value
    """
    function f(K::Float64, Cb::Float64, Con::Float64)::Float64
        if K >= 1
            return 0.0
        elseif Cb > 100
            return (1 - K^4) / (4 * cnst_m * Cb)
        elseif Cb == 0 && K == 0
            return 1 / (3 * Con)
        elseif Cb == 0
            return 1 / (3 * Con) * (1 - K^3)
        else
            return 1 / (3 * Con) * (
                1 - K^3 - 3 * Cb * (1 - K^2) / 2 + 3 * Cb^2 * (1 - K) - 3 * Cb^3 * log((Cb + 1) / (Cb + K)))
        end
    end

    """
        TipAsym_Universal_1stOrder_Res(dist, args...)

        More precise function to be minimized to find root for universal Tip asymptote (see Donstov and Pierce)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_Universal_1stOrder_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        if Cbar == 0
            return TipAsym_MK_deltaC_Res(dist, args[1:end-1]...)
        end

        Vel = (dist - DistLstTSEltRibbon) / dt
        Kh = Kprime * dist^0.5 / (Eprime * wEltRibbon)
        Ch = 2 * Cbar * dist^0.5 / (Vel^0.5 * wEltRibbon)
        sh = muPrime * Vel * dist^2 / (Eprime * wEltRibbon^3)

        g0 = f(Kh, cnst_mc * Ch, cnst_m)
        delt = cnst_m * (1 + cnst_mc * Ch) * g0
        gdelt = f(Kh, Ch * C2(delt) / C1(delt), C1(delt))

        return sh - gdelt
    end

    """
        TipAsym_Universal_zrthOrder_Res(dist, args...)

        Function to be minimized to find root for universal Tip asymptote (see Donstov and Pierce 2017)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_Universal_zrthOrder_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, muPrime) = args

        if Cbar == 0
            return TipAsym_MK_zrthOrder_Res(dist, args...)
        end

        Vel = (dist - DistLstTSEltRibbon) / dt

        Kh = Kprime * dist^0.5 / (Eprime * wEltRibbon)
        Ch = 2 * Cbar * dist^0.5 / (Vel^0.5 * wEltRibbon)
        g0 = f(Kh, cnst_mc * Ch, cnst_m)
        sh = muPrime * Vel * dist^2 / (Eprime * wEltRibbon^3)

        return sh - g0
    end

    """
        TipAsym_Hershcel_Burkley_Res(dist, args...)

        Function to be minimized to find root for Herschel Bulkley (see Bessmertnykh and Donstov 2019)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_Hershcel_Burkley_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, _) = args
        
        if Cbar == 0
            return TipAsym_power_law_MK_Res(dist, args...)
        end
        
        Vel = (dist - DistLstTSEltRibbon) / dt
        n = fluidProp.n
        alpha = -0.3107 * n + 1.9924
        X = 2 * Cbar * Eprime / sqrt(Vel) / Kprime
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * fluidProp.k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        T0t = fluidProp.T0 * 2 * Eprime * ell / Kprime / Kprime
        wtTau = 2 * sqrt(π * T0t) * xt
        wt = ((wEltRibbon * Eprime / Kprime / sqrt(dist))^alpha - wtTau^alpha)^(1 / alpha)

        theta = 0.0452 * n^2 - 0.178 * n + 0.1753
        Vm = 1 - wt^(-((2 + n) / (1 + theta)))
        Vmt = 1 - wt^(-((2 + 2 * n) / (1 + theta)))
        dm = (2 - n) / (2 + n)
        dmt = (2 - n) / (2 + 2 * n)
        Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))
        Bmt = (64 * (1 + n)^2 / (3 * n * (4 + n)) * tan(3 * π * n / (4 * (1 + n))))^(1 / (2 + 2 * n))
        
        dt1 = dmt * dm * Vmt * Vm * 
            (Bm^((2 + n) / n) * Vmt^((1 + theta) / n) + X / wt * Bmt^(2 * (1 + n) / n) * Vm^((1 + theta) / n)) / 
            (dmt * Vmt * Bm^((2 + n) / n) * Vmt^((1 + theta) / n) +
            dm * Vm * X / wt * Bmt^(2 * (1 + n) / n) * Vm^((1 + theta) / n))

        return xt^((2 - n) / (1 + theta)) - dt1 * wt^((2 + n) / (1 + theta)) * (dm^(1 + theta) * Bm^(2 + n) +
                                dmt^(1 + theta) * Bmt^(2 * (1 + n)) * ((1 + X / wt)^n - 1))^(-1 / (1 + theta))
    end

    """
        TipAsym_power_law_Res(dist, args...)

        Function to be minimized to find root for power-law fluid (see e.g. Bessmertnykh and Donstov 2019)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_power_law_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, _) = args
        
        if Cbar == 0
            return TipAsym_power_law_MK_Res(dist, args...)
        end
        
        Vel = (dist - DistLstTSEltRibbon) / dt
        n = fluidProp.n
        X = 2 * Cbar * Eprime / sqrt(Vel) / Kprime
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * fluidProp.k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        wt = wEltRibbon * Eprime / Kprime / sqrt(dist)

        theta = 0.0452 * n^2 - 0.178 * n + 0.1753
        Vm = 1 - wt^(-((2 + n) / (1 + theta)))
        Vmt = 1 - wt^(-((2 + 2 * n) / (1 + theta)))
        dm = (2 - n) / (2 + n)
        dmt = (2 - n) / (2 + 2 * n)
        Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))
        Bmt = (64 * (1 + n)^2 / (3 * n * (4 + n)) * tan(3 * π * n / (4 * (1 + n))))^(1 / (2 + 2 * n))

        dt1 = dmt * dm * Vmt * Vm * 
            (Bm^((2 + n) / n) * Vmt^((1 + theta) / n) + X / wt * Bmt^(2 * (1 + n) / n) * Vm^((1 + theta) / n)) / 
            (dmt * Vmt * Bm^((2 + n) / n) * Vmt^((1 + theta) / n) +
            dm * Vm * X / wt * Bmt^(2 * (1 + n) / n) * Vm^((1 + theta) / n))

        return xt^((2 - n) / (1 + theta)) - dt1 * wt^((2 + n) / (1 + theta)) * (dm^(1 + theta) * Bm^(2 + n) +
                                dmt^(1 + theta) * Bmt^(2 * (1 + n)) * ((1 + X / wt)^n - 1))^(-1 / (1 + theta))
    end

    """
        TipAsym_Hershcel_Burkley_MK_Res(dist, args...)

        Function to be minimized to find root for power-law fluid (see e.g. Bessmertnykh and Donstov 2019)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_Hershcel_Burkley_MK_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        Vel = (dist - DistLstTSEltRibbon) / dt
        n = fluidProp.n
        alpha = -0.3107 * n + 1.9924
        X = 2 * Cbar * Eprime / sqrt(Vel) / Kprime
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * fluidProp.k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        T0t = fluidProp.T0 * 2 * Eprime * ell / Kprime / Kprime
        wtTau = 2 * sqrt(π * T0t) * xt
        wt = ((wEltRibbon * Eprime / Kprime / sqrt(dist))^alpha - wtTau^alpha)^(1 / alpha)

        theta = 0.0452 * n^2 - 0.178 * n + 0.1753
        dm = (2 - n) / (2 + n)
        Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))

        return wt - (1 + (Bm^(2 + n) * xt^(2 - n))^(1 / (1 + theta)))^((1 + theta) / (2 + n)) 
    end

    """
        TipAsym_power_law_MK_Res(dist, args...)

        Function to be minimized to find root for power-law fluid (see e.g. Bessmertnykh and Donstov 2019)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_power_law_MK_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args

        Vel = (dist - DistLstTSEltRibbon) / dt
        n = fluidProp.n
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * fluidProp.k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        wt = wEltRibbon * Eprime / Kprime / sqrt(dist)

        theta = 0.0452 * n^2 - 0.178 * n + 0.1753
        dm = (2 - n) / (2 + n)
        Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))

        return wt - (1 + (Bm^(2 + n) * xt^(2 - n))^(1 / (1 + theta)))^((1 + theta) / (2 + n)) 
    end

    """
        TipAsym_PowerLaw_M_vertex_Res(dist, args...)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_PowerLaw_M_vertex_Res(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt, _) = args
        n = fluidProp.n    
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * fluidProp.k
        Vel = (dist - DistLstTSEltRibbon) / dt
        Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))
        
        return wEltRibbon - Bm * (Mprime * Vel^n / Eprime)^(1 / (2 + n)) * dist^(2 / (2 + n))
    end

    """
        TipAsym_variable_Toughness_Res(dist, args...)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_variable_Toughness_Res(dist::Float64, args...)
        (wEltRibbon, Eprime, Kprime_func, anisotropic_flag, alpha, zero_vertex, center_coord) = args

        if zero_vertex == 0
            x = center_coord[1] + dist * cos(alpha)
            y = center_coord[2] + dist * sin(alpha)
        elseif zero_vertex == 1
            x = center_coord[1] - dist * cos(alpha)
            y = center_coord[2] + dist * sin(alpha)
        elseif zero_vertex == 2
            x = center_coord[1] - dist * cos(alpha)
            y = center_coord[2] - dist * sin(alpha)
        elseif zero_vertex == 3
            x = center_coord[1] + dist * cos(alpha)
            y = center_coord[2] - dist * sin(alpha)
        end

        if anisotropic_flag
            Kprime = Kprime_func(alpha)
        else
            Kprime = Kprime_func(x, y)
        end

        return dist - wEltRibbon^2 * (Eprime / Kprime)^2
    end

    """
        Vm_residual(dist, args...)

        # Arguments
        - `dist::Float64`: distance
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function Vm_residual(dist::Float64, args...)
        (wEltRibbon, Kprime, Eprime, fluidProp, Cbar, DistLstTSEltRibbon, dt) = args
        
        Vel = (dist - DistLstTSEltRibbon) / dt
        n = fluidProp.n
        alpha = -0.3107 * n + 1.9924
        X = 2 * Cbar * Eprime / sqrt(Vel) / Kprime
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * fluidProp.k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        T0t = fluidProp.T0 * 2 * Eprime * ell / Kprime / Kprime
        wtTau = 2 * sqrt(π * T0t) * xt
        wt = ((wEltRibbon * Eprime / Kprime / sqrt(dist))^alpha - wtTau^alpha)^(1 / alpha)
        theta = 0.0452 * n^2 - 0.178 * n + 0.1753
        
        return 100 * eps() - 1 + wt^(-((2 + 2 * n) / (1 + theta)))
    end

    """
        FindBracket_dist(w, Kprime, Eprime, fluidProp, Cprime, DistLstTS, dt, mesh, ResFunc, simProp, muPrime)

        Find the valid bracket for the root evaluation function.

        # Arguments
        - `w::Vector{Float64}`: fracture width
        - `Kprime::Vector{Float64}`: stress intensity factor
        - `Eprime::Vector{Float64}`: plane strain modulus
        - `fluidProp`: fluid properties
        - `Cprime::Vector{Float64}`: Carter's leak off coefficient multiplied by 2
        - `DistLstTS::Vector{Float64}`: distance from last time step
        - `dt::Float64`: time step
        - `mesh`: mesh object
        - `ResFunc`: residual function
        - `simProp`: simulation properties
        - `muPrime::Float64`: viscosity

        # Returns
        - `(a, b)`: tuple of brackets
    """
    function FindBracket_dist(w::Vector{Float64}, Kprime::Vector{Float64}, Eprime::Vector{Float64}, 
                            fluidProp, Cprime::Vector{Float64}, DistLstTS::Vector{Float64}, dt::Float64, 
                            mesh, ResFunc, simProp, muPrime::Union{Float64, Vector{Float64}})
        a = -DistLstTS * (1 + eps())
        
        if fluidProp.rheology == "Newtonian" || sum(Cprime) == 0
            b = fill(6 * (mesh.hx^2 + mesh.hy^2)^0.5, length(w))
        elseif simProp.get_tipAsymptote() in ["HBF", "HBF_aprox", "HBF_num_quad", "PLF", "PLF_aprox", "PLF_num_quad"]
            b = zeros(Float64, length(w))
            for i in 1:length(w)
                TipAsmptargs = (w[i], Kprime[i], Eprime[i], fluidProp, Cprime[i], -DistLstTS[i], dt)
                result = optimize(x -> Vm_residual(x, TipAsmptargs...)^2, (w[i] * Eprime[i] / Kprime[i])^2)
                b[i] = Optim.minimizer(result)
            end
        end

        for i in 1:length(w)
            TipAsmptargs = (w[i], Kprime[i], Eprime[i], fluidProp, Cprime[i], -DistLstTS[i], dt, muPrime)
            Res_a = ResFunc(a[i], TipAsmptargs...)
            Res_b = ResFunc(b[i], TipAsmptargs...)

            cnt = 0
            mid = b[i]
            while Res_a * Res_b > 0
                mid = (a[i] + 2 * mid) / 3  # weighted
                Res_a = ResFunc(mid, TipAsmptargs...)
                cnt += 1
                if Res_a * Res_b < 0
                    a[i] = mid
                    break
                elseif Res_a > 0.0 && Res_b > 0.0
                    mid_b = b[i] * 2^cnt
                    Res_b = ResFunc(mid_b, TipAsmptargs...)
                    if Res_a * Res_b < 0
                        a[i] = mid
                        b[i] = mid_b
                        break
                    end
                end
                if cnt >= 100  # Should assume not propagating. not set to check how frequently it happens.
                    a[i] = NaN
                    b[i] = NaN
                    break
                end
            end
        end

        return a, b
    end

    """
        TipAsymInversion(w, frac, matProp, fluidProp, simParmtrs, dt=nothing, Kprime_k=nothing, Eprime_k=nothing, perfNode=nothing)

        Evaluate distance from the front using tip asymptotics according to the given regime, given the fracture width in
        the ribbon cells.

        # Arguments
        - `w::Vector{Float64}`: fracture width.
        - `frac`: current fracture object.
        - `matProp`: material properties.
        - `fluidProp`: fluid properties.
        - `simParmtrs`: Simulation parameters.
        - `dt::Union{Float64, Nothing}`: time step.
        - `Kprime_k::Union{Vector{Float64}, Nothing}`: Kprime for current iteration of toughness loop. if not given, the Kprime from the given material properties object will be used.
        - `Eprime_k::Union{Vector{Float64}, Nothing}`: the plain strain modulus.
        - `perfNode`: performance node.

        # Returns
        - `Vector{Float64}`: distance (unsigned) from the front to the ribbon cells.
    """
    function TipAsymInversion(w::Vector{Float64}, frac, matProp, fluidProp, simParmtrs, 
                            dt::Union{Float64, Nothing}=nothing, Kprime_k::Union{Vector{Float64}, Nothing}=nothing, 
                            Eprime_k::Union{Vector{Float64}, Nothing}=nothing, perfNode=nothing)
        @debug "TipAsymInversion called" _group="JFrac.TipAsymInversion"
        
        if Kprime_k === nothing
            Kprime = matProp.Kprime[frac.EltRibbon]
        else
            Kprime = Kprime_k
        end

        if Eprime_k === nothing
            Eprime = fill(matProp.Eprime, length(frac.EltRibbon))
        else
            Eprime = Eprime_k
        end

        ResFunc = nothing
        if simParmtrs.get_tipAsymptote() == "U"
            ResFunc = TipAsym_Universal_zrthOrder_Res
        elseif simParmtrs.get_tipAsymptote() == "U1"
            ResFunc = TipAsym_Universal_1stOrder_Res
        elseif simParmtrs.get_tipAsymptote() == "K"
            return w[frac.EltRibbon] .^ 2 .* (Eprime ./ Kprime) .^ 2
        elseif simParmtrs.get_tipAsymptote() == "Kt"
            return w[frac.EltRibbon] .^ 2 .* (Eprime ./ Kprime) .^ 2
        elseif simParmtrs.get_tipAsymptote() == "M"
            ResFunc = TipAsym_viscStor_Res
        elseif simParmtrs.get_tipAsymptote() == "Mt"
            ResFunc = TipAsym_viscLeakOff_Res
        elseif simParmtrs.get_tipAsymptote() == "MK"
            ResFunc = TipAsym_MK_zrthOrder_Res
        elseif simParmtrs.get_tipAsymptote() == "MDR"
            ResFunc = TipAsym_MDR_Res
        elseif simParmtrs.get_tipAsymptote() == "M_MDR"
            ResFunc = TipAsym_M_MDR_Res
        elseif simParmtrs.get_tipAsymptote() in ["HBF", "HBF_aprox", "HBF_num_quad"]
            ResFunc = TipAsym_Hershcel_Burkley_Res
        elseif simParmtrs.get_tipAsymptote() in ["PLF", "PLF_aprox", "PLF_num_quad"]
            ResFunc = TipAsym_power_law_Res
        elseif simParmtrs.get_tipAsymptote() == "PLF_M"
            ResFunc = TipAsym_PowerLaw_M_vertex_Res
        else
            error("Tip asymptote type not supported!")
        end

        # checking propagation condition
        stagnant_indices = findall(Kprime .* (abs.(frac.sgndDist[frac.EltRibbon])) .^ 0.5 ./ 
                                            (Eprime .* w[frac.EltRibbon]) .> 1)
        stagnant = falses(length(frac.EltRibbon))
        stagnant[stagnant_indices] .= true
        
        moving = setdiff(1:length(frac.EltRibbon), stagnant_indices)

        a, b = FindBracket_dist(w[frac.EltRibbon[moving]],
                                Kprime[moving],
                                Eprime[moving],
                                fluidProp,
                                matProp.Cprime[frac.EltRibbon[moving]],
                                frac.sgndDist[frac.EltRibbon[moving]],
                                dt,
                                frac.mesh,
                                ResFunc,
                                simParmtrs,
                                fluidProp.muPrime)
        
        ## AM: part added to take care of nan's in the bracketing if bracketing is no longer possible.
        nan_indices = findall(isnan.(a))
        if !isempty(nan_indices)
            stagnant_from_bracketing = nan_indices
            deleteat!(a, nan_indices)
            deleteat!(b, nan_indices)
            if !isempty(stagnant_indices)
                stagnant_indices = sort(unique(vcat(stagnant_indices, moving[stagnant_from_bracketing])))
                stagnant = falses(length(frac.EltRibbon))
                stagnant[stagnant_indices] .= true
            else
                stagnant_indices = stagnant_from_bracketing
                stagnant = falses(length(frac.EltRibbon))
                stagnant[stagnant_indices] .= true
            end
            moving = setdiff(1:length(frac.EltRibbon), stagnant_indices)
        end
        ## End of adaption

        dist = -frac.sgndDist[frac.EltRibbon]
        for i in 1:length(moving)
            TipAsmptargs = (w[frac.EltRibbon[moving[i]]],
                            Kprime[moving[i]],
                            Eprime[moving[i]],
                            fluidProp,
                            matProp.Cprime[frac.EltRibbon[moving[i]]],
                            -frac.sgndDist[frac.EltRibbon[moving[i]]],
                            dt,
                            fluidProp.muPrime)
            try
                if perfNode === nothing
                    dist[moving[i]] = find_zero(x -> ResFunc(x, TipAsmptargs...), (a[i], b[i]))
                else
                    brentq_itr = instrument_start("Brent method", perfNode)
                    dist[moving[i]] = find_zero(x -> ResFunc(x, TipAsmptargs...), (a[i], b[i]))
                    instrument_close(perfNode, brentq_itr, nothing, nothing, true, nothing, nothing)
                    # push!(perfNode.brentMethod_data, brentq_itr) # Предполагается, что поле существует
                end
            catch e
                if isa(e, ArgumentError)
                    dist[moving[i]] = NaN
                else
                    rethrow(e)
                end
            end
            
            if isnan(dist[moving[i]]) && simParmtrs.get_tipAsymptote() == "U1"
                @warn "First order did not converged: try with zero order." _group="JFrac.TipAsymInversion"
                try
                    if perfNode === nothing
                        dist[moving[i]] = find_zero(x -> TipAsym_Universal_zrthOrder_Res(x, TipAsmptargs...), (a[i], b[i]))
                    else
                        brentq_itr = instrument_start("Brent method", perfNode)
                        dist[moving[i]] = find_zero(x -> TipAsym_Universal_zrthOrder_Res(x, TipAsmptargs...), (a[i], b[i]))
                        instrument_close(perfNode, brentq_itr, nothing, nothing, true, nothing, nothing)
                        # push!(perfNode.brentMethod_data, brentq_itr)
                    end
                catch e
                    if isa(e, ArgumentError)
                        dist[moving[i]] = NaN
                    else
                        rethrow(e)
                    end
                end
            end
        end
        
        return dist
    end

    """
        StressIntensityFactor(w, lvlSetData, EltTip, EltRibbon, stagnant, mesh, Eprime)

        This function evaluate the stress intensity factor. See Donstov & Pierce Comput. Methods Appl. Mech. Engrn. 2017

        # Arguments
        - `w::Vector{Float64}`: fracture width
        - `lvlSetData::Vector{Float64}`: the level set values, i.e. distance from the fracture front
        - `EltTip::Vector{Int}`: tip elements
        - `EltRibbon::Vector{Int}`: ribbon elements
        - `stagnant::Vector{Bool}`: the stagnant tip cells
        - `mesh`: mesh object
        - `Eprime::Vector{Float64}`: the plain strain modulus

        # Returns
        - `Vector{Float64}`: the stress intensity factor of the stagnant cells. Zero is returned for the 
                            tip cells that are moving.
    """
    function StressIntensityFactor(w::Vector{Float64}, lvlSetData::Vector{Float64}, EltTip::Vector{Int}, 
                                EltRibbon::Vector{Int}, stagnant::Vector{Bool}, mesh, Eprime::Vector{Float64})::Vector{Float64}
        KIPrime = zeros(Float64, length(EltTip))
        for i in 1:length(EltTip)
            if stagnant[i]
                neighbors = mesh.NeiElements[EltTip[i], :]
                enclosing = vcat(neighbors, [neighbors[3] - 1, neighbors[3] + 1, neighbors[4] - 1, neighbors[4] + 1])

                InRibbon = Int[]  # find neighbors in Ribbon cells
                for e in 1:length(enclosing)
                    found_indices = findall(==(enclosing[e]), EltRibbon)
                    if !isempty(found_indices)
                        push!(InRibbon, EltRibbon[found_indices[1]])
                    end
                end

                if length(InRibbon) == 1
                    KIPrime[i] = w[InRibbon[1]] * Eprime[i] / (-lvlSetData[InRibbon[1]])^0.5
                elseif length(InRibbon) > 1  # evaluate using least squares method
                    KIPrime[i] = Eprime[i] * (w[InRibbon[1]] * (-lvlSetData[InRibbon[1]])^0.5 + w[InRibbon[2]] * (
                        -lvlSetData[InRibbon[2]])^0.5) / (-lvlSetData[InRibbon[1]] - lvlSetData[InRibbon[2]])
                else  # ribbon cells not found in enclosure, evaluating with the closest ribbon cell
                    RibbonCellsDist = ((mesh.CenterCoor[EltRibbon, 1] - mesh.CenterCoor[EltTip[i], 1]) .^ 2 + (
                        mesh.CenterCoor[EltRibbon, 2] - mesh.CenterCoor[EltTip[i], 2]) .^ 2) .^ 0.5
                    closest_idx = argmin(RibbonCellsDist)
                    closest = EltRibbon[closest_idx]
                    KIPrime[i] = w[closest] * Eprime[i] / (-lvlSetData[closest])^0.5
                end

                if KIPrime[i] < 0.0
                    KIPrime[i] = 0.0
                end
            end
        end

        return KIPrime
    end

    """
        TipAsymInversion_hetrogenous_toughness(w, frac, mat_prop, level_set)

        This function inverts the tip asymptote with the toughness value taken at the tip instead of taking at the ribbon
        cell.

        # Arguments
        - `w::Vector{Float64}`: fracture width
        - `frac`: current fracture object
        - `mat_prop`: material properties
        - `level_set::Vector{Float64}`: the level set values, i.e. signed distance from the fracture front

        # Returns
        - `Vector{Float64}`: the inverted tip asymptote for the ribbon cells
    """
    function TipAsymInversion_hetrogenous_toughness(w::Vector{Float64}, frac, mat_prop, level_set::Vector{Float64})::Vector{Float64}
        zero_vrtx = find_zero_vertex(frac.EltRibbon, level_set, frac.mesh)
        dist = -level_set
        alpha = zeros(Float64, length(frac.EltRibbon))

        for i in 1:length(frac.EltRibbon)
            if zero_vrtx[i] == 0
                # north-east direction of propagation
                alpha[i] = acos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 2]]) / frac.mesh.hx)
            elseif zero_vrtx[i] == 1
                # north-west direction of propagation
                alpha[i] = acos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 1]]) / frac.mesh.hx)
            elseif zero_vrtx[i] == 2
                # south-west direction of propagation
                alpha[i] = acos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 1]]) / frac.mesh.hx)
            elseif zero_vrtx[i] == 3
                # south-east direction of propagation
                alpha[i] = acos((dist[frac.EltRibbon[i]] - dist[frac.mesh.NeiElements[frac.EltRibbon[i], 2]]) / frac.mesh.hx)
            end

            # Обработка предупреждений (warnings) из Python кода
            if abs(dist[frac.mesh.NeiElements[frac.EltRibbon[i], 1]] / dist[frac.mesh.NeiElements[frac.EltRibbon[i], 2]] - 1) < 1e-7
                # if the angle is 90 degrees
                alpha[i] = π / 2
            end
            if abs(dist[frac.mesh.NeiElements[frac.EltRibbon[i], 3]] / dist[frac.mesh.NeiElements[frac.EltRibbon[i], 4]] - 1) < 1e-7
                # if the angle is 0 degrees
                alpha[i] = 0.0
            end
        end

        sol = zeros(Float64, length(frac.EltRibbon))
        for i in 1:length(frac.EltRibbon)
            TipAsmptargs = (w[frac.EltRibbon[i]],
                            mat_prop.Eprime,
                            mat_prop.KprimeFunc,
                            mat_prop.anisotropic,
                            alpha[i],
                            zero_vrtx[i],
                            frac.mesh.CenterCoor[frac.EltRibbon[i], :])

            # residual for zero distance; used as lower bracket
            residual_zero = TipAsym_variable_Toughness_Res(0.0, TipAsmptargs...)

            # the lower bracket (0) and the upper bracker (4x the maximum possible length in a cell) is divided into 16
            # equally distant points to sample the sign of the residual function. This is necessary to avoid missing a high
            # resolution variation in toughness. This also means that the toughness variations below the upper_bracket/16 is
            # not guaranteed to be caught.
            sample_lngths = range(4*(frac.mesh.hx^2 + frac.mesh.hy^2)^0.5 /
                                        16, stop=4*(frac.mesh.hx^2 + frac.mesh.hy^2)^0.5, length=16)
            cnt = 0
            res_prdct = 0.0
            while res_prdct >= 0 && cnt < 16
                res_prdct = residual_zero * TipAsym_variable_Toughness_Res(sample_lngths[cnt+1], TipAsmptargs...)
                cnt += 1
            end

            if cnt == 16
                sol[i] = NaN
                return sol
            else
                upper_bracket = sample_lngths[cnt]
            end

            try
                sol[i] = find_zero(x -> TipAsym_variable_Toughness_Res(x, TipAsmptargs...), (0.0, upper_bracket))
            catch e
                if isa(e, ArgumentError)
                    sol[i] = NaN
                else
                    rethrow(e)
                end
            end
        end

        return sol .- sol .* 1e-10
    end

    """
        find_zero_vertex(Elts, level_set, mesh)

        find the vertex opposite to the propagation direction from which the perpendicular on the front is drawn

        # Arguments
        - `Elts::Vector{Int}`: elements
        - `level_set::Vector{Float64}`: level set values
        - `mesh`: mesh object

        # Returns
        - `Vector{Int}`: zero vertex indices
    """
    function find_zero_vertex(Elts::Vector{Int}, level_set::Vector{Float64}, mesh)::Vector{Int}
        """ find the vertex opposite to the propagation direction from which the perpendicular on the front is drawn"""

        zero_vertex = zeros(Int, length(Elts))
        for i in 1:length(Elts)
            neighbors = mesh.NeiElements[Elts[i], :]

            if level_set[neighbors[1]] <= level_set[neighbors[2]] && level_set[neighbors[3]] <= level_set[neighbors[4]]
                zero_vertex[i] = 0
            elseif level_set[neighbors[1]] > level_set[neighbors[2]] && level_set[neighbors[3]] <= level_set[neighbors[4]]
                zero_vertex[i] = 1
            elseif level_set[neighbors[1]] > level_set[neighbors[2]] && level_set[neighbors[3]] > level_set[neighbors[4]]
                zero_vertex[i] = 2
            elseif level_set[neighbors[1]] <= level_set[neighbors[2]] && level_set[neighbors[3]] > level_set[neighbors[4]]
                zero_vertex[i] = 3
            end
        end

        return zero_vertex
    end

end # module TipInversion