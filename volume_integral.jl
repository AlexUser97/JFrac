# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""
module VolumeIntegral

    include("tip_inversion.jl")
    using .TipInversion: f, C1, C2

    using Logging
    using QuadGK
    using Roots

    export width_dist_product_HBF, width_HBF, TipAsym_UniversalW_zero_Res, TipAsym_UniversalW_delt_Res,
        TipAsym_MK_W_zrthOrder_Res, TipAsym_MK_W_deltaC_Res, TipAsym_viscStor_Res,
        MomentsTipAssympGeneral, TipAsym_res_Herschel_Bulkley_d_given, MomentsTipAssymp_HBF_approx,
        Pdistance, VolumeTriangle, Area, Integral_over_cell, FindBracket_w, FindBracket_w_HB,
        find_corresponding_ribbon_cell, leak_off_stagnant_tip

    const beta_m = 2^(1/3) * 3^(5/6)
    const beta_mtld = 4/(15^(1/4) * (2^0.5 - 1)^(1/4))
    const cnst_mc = 3 * beta_mtld^4 / (4 * beta_m^3)
    const cnst_m = beta_m^3 / 3
    const Ki_c = 3000


    """
        width_dist_product_HBF(s, HB_args...)

        This function is used to evaluate the first moment of HBF tip solution with numerical quadrature.

        # Arguments
        - `s::Float64`: distance
        - `HB_args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: w * s
    """
    function width_dist_product_HBF(s::Float64, HB_args...)
        HB_args_ext = (HB_args, HB_args[1] - s)
        a = 1e-4
        b = 1e1
        a, b = FindBracket_w_HB(a, b, HB_args_ext...)
        if isnan(a)
            return NaN
        end
        w = find_zero(x -> TipAsym_res_Herschel_Bulkley_d_given(x, HB_args_ext...), (a, b))
        
        return w * s
    end

    """
        width_HBF(s, HB_args...)

        This function is used to evaluate the zeroth moment of HBF tip solution with numerical quadrature.

        # Arguments
        - `s::Float64`: distance
        - `HB_args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: w
    """
    function width_HBF(s::Float64, HB_args...)
        HB_args_ext = (HB_args, s)
        a = 1e-8
        b = 1e1
        a, b = FindBracket_w_HB(a, b, HB_args_ext...)
        if isnan(a)
            return NaN
        end
        w = find_zero(x -> TipAsym_res_Herschel_Bulkley_d_given(x, HB_args_ext...), (a, b))
        
        return w
    end

    """
        TipAsym_UniversalW_zero_Res(w, args...)

        Function to be minimized to find root for universal Tip asymptote (see Donstov and Pierce 2017)

        # Arguments
        - `w::Float64`: width
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_UniversalW_zero_Res(w::Float64, args...)
        dist, Kprime, Eprime, muPrime, Cbar, Vel = args

        if Cbar == 0
            return TipAsym_MK_W_zrthOrder_Res(w, args...)
        end

        Kh = Kprime * dist^0.5 / (Eprime * w)
        Ch = 2 * Cbar * dist^0.5 / (Vel^0.5 * w)
        g0 = f(Kh, 0.9911799823 * Ch, 6 * 3^0.5)
        sh = muPrime * Vel * dist^2 / (Eprime * w^3)

        return sh - g0
    end

    """
        TipAsym_UniversalW_delt_Res(w, args...)

        The residual function zero of which will give the General asymptote

        # Arguments
        - `w::Float64`: width
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_UniversalW_delt_Res(w::Float64, args...)
        dist, Kprime, Eprime, muPrime, Cbar, Vel = args

        if Cbar == 0
            return TipAsym_MK_W_deltaC_Res(w, args...)
        end

        Kh = Kprime * dist^0.5 / (Eprime * w)
        Ch = 2 * Cbar * dist^0.5 / (Vel^0.5 * w)
        sh = muPrime * Vel * dist^2 / (Eprime * w^3)

        g0 = f(Kh, 0.9911799823 * Ch, 10.392304845)
        delt = 10.392304845 * (1 + 0.9911799823 * Ch) * g0

        b = C2(delt) / C1(delt)
        con = C1(delt)
        gdelt = f(Kh, Ch * b, con)

        return sh - gdelt
    end

    """
        TipAsym_MK_W_zrthOrder_Res(w, args...)

        Residual function for viscosity to toughness regime with transition, without leak off

        # Arguments
        - `w::Float64`: width
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MK_W_zrthOrder_Res(w::Float64, args...)
        dist, Kprime, Eprime, muPrime, Cbar, Vel = args

        if Kprime == 0
            return TipAsym_viscStor_Res(w, args...) # todo: make this
        end
        if muPrime == 0
            # return toughness dominated asymptote
            return dist - w^2 * (Eprime / Kprime)^2
        end

        w_tld = Eprime * w / (Kprime * dist^0.5)
        return w_tld - (1 + beta_m^3 * Eprime^2 * Vel * dist^0.5 * muPrime / Kprime^3)^(1/3)
    end

    """
        TipAsym_MK_W_deltaC_Res(w, args...)

        Residual function for viscosity to toughness regime with transition, without leak off

        # Arguments
        - `w::Float64`: width
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_MK_W_deltaC_Res(w::Float64, args...)
        dist, Kprime, Eprime, muPrime, Cbar, Vel = args

        if Kprime == 0
            return TipAsym_viscStor_Res(w, args...)
        end
        if muPrime == 0
            # return toughness dominated asymptote
            return dist - w^2 * (Eprime / Kprime)^2
        end

        w_tld = Eprime * w / (Kprime * dist^0.5)

        l_mk = (Kprime^3 / (Eprime^2 * muPrime * Vel))^2
        x_tld = (dist / l_mk)^(1/2)
        delta = 1 / 3 * beta_m^3 * x_tld / (1 + beta_m^3 * x_tld)
        return w_tld - (1 + 3 * C1(delta) * x_tld)^(1/3)
    end

    """
        TipAsym_viscStor_Res(w, args...)

        Residual function for viscosity dominate regime, without leak off

        # Arguments
        - `w::Float64`: width
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_viscStor_Res(w::Float64, args...)
        dist, Kprime, Eprime, muPrime, Cbar, Vel = args

        return w - (18 * 3^0.5 * Vel * muPrime / Eprime)^(1 / 3) * dist^(2 / 3)
    end

    """
        MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime, regime)

        Moments of the General tip asymptote to calculate the volume integral (see Donstov and Pierce, 2017)

        # Arguments
        - `dist::Float64`: distance
        - `Kprime::Float64`: stress intensity factor
        - `Eprime::Float64`: plane strain modulus
        - `muPrime::Float64`: viscosity
        - `Cbar::Float64`: leak-off coefficient
        - `Vel::Float64`: velocity
        - `stagnant::Bool`: stagnant flag
        - `KIPrime::Float64`: stress intensity factor for stagnant cells
        - `regime::String`: propagation regime

        # Returns
        - `(M0, M1)`: tuple of moments
    """
    function MomentsTipAssympGeneral(dist::Float64, Kprime::Float64, Eprime::Float64, muPrime::Float64, 
                                    Cbar::Float64, Vel::Float64, stagnant::Bool, KIPrime::Float64, regime::String)
        TipAsmptargs = (dist, Kprime, Eprime, muPrime, Cbar, Vel)

        if dist == 0
            w = 0.0
        elseif stagnant
            w = KIPrime * dist^0.5 / Eprime
        else
            a, b = FindBracket_w(dist, Kprime, Eprime, muPrime, Cbar, Vel, regime)
            try
                if regime == "U"
                    w = find_zero(x -> TipAsym_UniversalW_zero_Res(x, TipAsmptargs...), (a, b))  # root finding
                else
                    w = find_zero(x -> TipAsym_UniversalW_delt_Res(x, TipAsmptargs...), (a, b))  # root finding
                end
            catch e
                if isa(e, ArgumentError) || isa(e, DomainError)
                    M0, M1 = NaN, NaN
                    return M0, M1
                else
                    rethrow(e)
                end
            end

            if w < -1e-15
                @warn "Negative width encountered in volume integral" _group="JFrac.MomentsTipAssympGeneral"
                w = abs(w)
            end
        end

        if Vel < 1e-6 || w == 0
            delt = 1 / 6
        else
            Kh = Kprime * dist^0.5 / (Eprime * w)
            Ch = 2 * Cbar * dist^0.5 / (Vel^0.5 * w)
            g0 = f(Kh, 0.9911799823 * Ch, 10.392304845)
            delt = 10.392304845 * (1 + 0.9911799823 * Ch) * g0
        end

        M0 = 2 * w * dist / (3 + delt)
        M1 = 2 * w * dist^2 / (5 + delt)

        if isnan(M0) || isnan(M1)
        M0, M1 = NaN, NaN
        end

        return M0, M1
    end

    """
        TipAsym_res_Herschel_Bulkley_d_given(w, args...)

        Residual function for Herschel-Bulkley fluid model (see Besmertnykh and Dontsov, JAM 2019)

        # Arguments
        - `w::Float64`: width
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: residual
    """
    function TipAsym_res_Herschel_Bulkley_d_given(w::Float64, args...)
        (l, Kprime, Eprime, muPrime, Cbar, Vel, n, k, T0), dist = args
        alpha = -0.3107 * n + 1.9924
        X = 2 * Cbar * Eprime / sqrt(Vel) / Kprime
        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        T0t = T0 * 2 * Eprime * ell / Kprime^2
        wtTau = sqrt(4 * π * T0t) * xt
        wt = ((w * Eprime / Kprime / sqrt(dist))^alpha - wtTau^alpha)^(1 / alpha)

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
        MomentsTipAssymp_HBF_approx(s, HB_args...)

        Approximate moments of the Herschel-Bulkley fluid. Delta is taken to be 1/6.

        # Arguments
        - `s::Float64`: distance
        - `HB_args`: Variable arguments passed as tuple.

        # Returns
        - `(M0, M1)`: tuple of moments
    """
    function MomentsTipAssymp_HBF_approx(s::Float64, HB_args...)
        HB_args_ext = (HB_args, s)
        a = 1e-8
        b = 1e1
        a, b = FindBracket_w_HB(a, b, HB_args_ext...)
        if isnan(a)
            return NaN, NaN
        end
        w = find_zero(x -> TipAsym_res_Herschel_Bulkley_d_given(x, HB_args_ext...), (a, b))

        M0 = 2 * w * s / (3 + 1 / 6)
        M1 = 2 * w * s^2 / (5 + 1 / 6)

        if isnan(M0) || isnan(M1)
        M0, M1 = NaN, NaN
        end

        return M0, M1
    end

    """
        Pdistance(x, y, slope, intercpt)

        distance of a point from a line

        # Arguments
        - `x::Float64`: x coordinate
        - `y::Float64`: y coordinate
        - `slope::Float64`: slope of the line
        - `intercpt::Float64`: y-intercept of the line

        # Returns
        - `Float64`: distance
    """
    function Pdistance(x::Float64, y::Float64, slope::Float64, intercpt::Float64)::Float64
        return (slope * x - y + intercpt) / (slope^2 + 1)^0.5
    end

    """
        VolumeTriangle(dist, param...)

        Volume of the triangle defined by perpendicular distance (dist) and em (em=1/sin(alpha)cos(alpha), where alpha
        is the angle of the perpendicular). The regime variable identifies the propagation regime.

        # Arguments
        - `dist::Float64`: distance
        - `param`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: volume
    """
    function VolumeTriangle(dist::Float64, param...)
        regime, fluid_prop, Kprime, Eprime, Cbar, Vel, stagnant, KIPrime, arrival_t, em, t_lstTS, dt, muPrime = param

        if stagnant
            regime = "U1"
        end

        if regime == "A"
            return dist^2 * em / 2

        elseif regime == "K"
            return 4 / 15 * Kprime / Eprime * dist^2.5 * em

        elseif regime == "M"
            return 0.7081526678 * (Vel * muPrime / Eprime)^(1 / 3) * em * dist^(8 / 3)

        elseif regime == "Lk"
            t = t_lstTS + dt
            if Vel <= 0
                t_e = arrival_t
            else
                t_e = t - dist / Vel
            end

            intgrl_0_t = 4 / 15 * em * (t - t_e)^(5 / 2) * Vel^2
            if (t - t_e - dt) < 0
                intgrl_0_tm1 = 0.0
            else
                intgrl_0_tm1 = 4 / 15 * em * (t - t_e - dt)^(5 / 2) * Vel^2
            end

            return intgrl_0_t - intgrl_0_tm1

        elseif regime == "Mt"
            return 256 / 273 / (15 * tan(π / 8))^0.25 * (
                                        Cbar * muPrime / Eprime)^0.25 * em * Vel^0.125 * dist^(21 / 8)

        elseif regime == "U" || regime == "U1"
            if Cbar == 0 && Kprime == 0 && !stagnant # if fully viscosity dominated
                return 0.7081526678 * (Vel * muPrime / Eprime)^(1 / 3) * em * dist^(8 / 3)
            end
            M0, M1 = MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime, regime)
            return em * (dist * M0 - M1)

        elseif regime == "MK"
            return (3.925544049000839e-9 * em * Kprime * (
            1.7320508075688772 * Kprime^9 * (Kprime^6 - 1872.0 * dist * Eprime^4 * muPrime^2 * Vel^2) + (
            1.0 + (31.17691453623979 * (dist)^0.5 * Eprime^2 * muPrime * Vel) / Kprime^3)^0.3333333333333333 * (
            -1.7320508075688772 * Kprime^15 + 18.0 * (
            dist)^0.5 * Eprime^2 * Kprime^12 * muPrime * Vel + 2868.2761373340604 * dist * Eprime^4 *
            Kprime^9 * muPrime^2 * Vel^2 - 24624.0 * dist^1.5 * Eprime^6 * Kprime^6 * muPrime^3 *
            Vel^3 + 464660.73424811783 * dist^2 * Eprime^8 * Kprime^3 * muPrime^4 * Vel^4 + 5.7316896e7
            * dist^2.5 * Eprime^10 * muPrime^5 * Vel^5))) / (Eprime^11 * muPrime^5 * Vel^5)

        elseif occursin("MDR", regime)
            density = 1000.0
            return (0.0885248 * dist^2.74074 * em * Vel^0.481481 * muPrime^0.259259 * density^0.111111
            ) / Eprime^0.37037
        
        elseif regime in ["HBF", "HBF_aprox"]
            args_HB = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, fluid_prop.T0)
            M0, M1 = MomentsTipAssymp_HBF_approx(dist, args_HB...)
            return em * (dist * M0 - M1)

        elseif regime == "HBF_num_quad"
            args_HB = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, fluid_prop.T0)
            return em * quadgk(s -> width_dist_product_HBF(s, args_HB...), 0, dist)[1]
        
        elseif regime in ["PLF", "PLF_aprox"]
            args_PLF = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, 0.0)
            M0, M1 = MomentsTipAssymp_HBF_approx(dist, args_PLF...)
            return em * (dist * M0 - M1)

        elseif regime == "PLF_num_quad"
            args_PLF = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, 0.0)
            return em * quadgk(s -> width_dist_product_HBF(s, args_PLF...), 0, dist)[1]

        elseif regime == "PLF_M"
            n = fluid_prop.n
            k = fluid_prop.k
            Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * k
            Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))
            
            return em * Bm * (Mprime * Vel^n / Eprime)^(1 / (2 + n)) * dist^((4 + n) / (2 + n)) * 
                    dist * (2 + n) * (1 / (4 + n) - 1 / (6 + 2 *n)) 

        else
            error("Unknown regime: $regime")
        end
    end

    """
        Area(dist, param...)

        Gives Area under the tip depending on the regime identifier ;  
        used in case of 0 or 90 degree angle; can be used for 1d case

        # Arguments
        - `dist::Float64`: distance
        - `param`: Variable arguments passed as tuple.

        # Returns
        - `Float64`: area
    """
    function Area(dist::Float64, param...)
        regime, fluid_prop, Kprime, Eprime, Cbar, Vel, stagnant, KIPrime, arrival_t, em, t_lstTS, dt, muPrime = param

        if stagnant
            regime = "U1"
        end

        if regime == "A"
            return dist

        elseif regime == "K"
            return 2 / 3 * Kprime / Eprime * dist^1.5

        elseif regime == "M"
            return 1.8884071141 * (Vel * muPrime / Eprime)^(1 / 3) * dist^(5 / 3)

        elseif regime == "Lk"
            t = t_lstTS + dt
            if Vel <= 0
                t_e = arrival_t
            else
                t_e = t - dist / Vel
            end

            intgrl_0_t = 2 / 3 * (t - t_e)^(3 / 2) * Vel
            if (t - t_e - dt) < 0
                intgrl_0_tm1 = 0.0
            else
                intgrl_0_tm1 = 2 / 3 * (t - t_e - dt)^(3 / 2) * Vel
            end

            return intgrl_0_t - intgrl_0_tm1

        elseif regime == "Mt"
            return 32 / 13 / (15 * tan(π / 8))^0.25 * (Cbar * muPrime / Eprime)^0.25 * Vel^0.125 * dist^(
            13 / 8)

        elseif regime == "U" || regime == "U1"
            if Cbar == 0 && Kprime == 0 && !stagnant  # if fully viscosity dominated
                return 1.8884071141 * (Vel * muPrime / Eprime)^(1 / 3) * dist^(5 / 3)
            end
            M0, M1 = MomentsTipAssympGeneral(dist, Kprime, Eprime, muPrime, Cbar, Vel, stagnant, KIPrime, regime)
            return M0

        elseif regime == "MK"
            return (7.348618459729571e-6 * Kprime * (-1.7320508075688772 * Kprime^9 +
                    (1.0 + (31.17691453623979 * (dist)^0.5 * Eprime^2 * muPrime * Vel) / Kprime^3)^0.3333333333333333 * (
                    1.7320508075688772 * Kprime^9 - 18.0 * (dist)^0.5 * Eprime^2 * Kprime^6 * muPrime * Vel + (
                    374.12297443487745 * dist * Eprime^4 * Kprime^3 * muPrime^2 * Vel^2) + (
                    81648.0 * dist^1.5 * Eprime^6 * muPrime^3 * Vel^3)))) / (
                    Eprime^7 * muPrime^3 * Vel^3)

        elseif occursin("MDR", regime)
            density = 1000.0
            return (0.242623 * dist^1.74074 * Vel^0.481481 * muPrime^0.259259 * density^0.111111
            ) / Eprime^0.37037
        
        elseif regime in ["HBF", "HBF_aprox"]
            args_HB = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, fluid_prop.T0)
            M0, M1 = MomentsTipAssymp_HBF_approx(dist, args_HB...)
            return M0
        
        elseif regime == "HBF_num_quad"
            args_HB = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, fluid_prop.T0)
            return quadgk(s -> width_HBF(s, args_HB...), 0, dist)[1]
        
        elseif regime in ["PLF", "PLF_aprox"]
            args_PLF = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, 0.0)
            M0, M1 = MomentsTipAssymp_HBF_approx(dist, args_PLF...)
            return M0
        
        elseif regime == "PLF_num_quad"
            args_PLF = (dist, Kprime, Eprime, muPrime, Cbar, Vel, fluid_prop.n, fluid_prop.k, 0.0)
            return quadgk(s -> width_HBF(s, args_PLF...), 0, dist)[1]
            
        elseif regime == "PLF_M"
            n = fluid_prop.n
            k = fluid_prop.k
            Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * k
            Bm = (2 * (2 + n)^2 / n * tan(π * n / (2 + n)))^(1 / (2 + n))
            
            return Bm * (Mprime * Vel^n / Eprime)^(1 / (2 + n)) * ((2 + n) * dist^((4 + n)/(2 + n)))/(4 + n)
            
        else
            error("Unknown regime: $regime")
        end
    end

    """
        Integral_over_cell(EltTip, alpha, l, mesh, function_name, frac=nothing, mat_prop=nothing, fluid_prop=nothing, Vel=nothing,
                        Kprime=nothing, Eprime=nothing, Cprime=nothing, stagnant=nothing, KIPrime=nothing, dt=nothing, arrival_t=nothing, projMethod=nothing)

        Calculate integral of the function specified by the argument function over the cell.

        # Arguments
        - `EltTip::Vector{Int}`: the tip cells over which the integral is to be evaluated
        - `alpha::Vector{Float64}`: the angle alpha of the perpendicular drawn on the front from the zero vertex.
        - `l::Vector{Float64}`: the length of the perpendicular drawn on the front from the zero vertex.
        - `mesh`: the mesh object.
        - `function_name::String`: the string specifying the type of function that is to be integreated.
        - `frac`: the fracture object.
        - `mat_prop`: the material properties object.
        - `fluid_prop`: the fluid properties object
        - `Vel::Union{Vector{Float64}, Nothing}`: the velocity of the front in the given tip cells.
        - `Kprime::Union{Vector{Float64}, Nothing}`: if provided, the toughness will be taken from the given array instead of taking it from the mat_prop object
        - `Eprime::Union{Vector{Float64}, Nothing}`: plain strain TI modulus for current iteration. if not given, the Eprime from the given material properties object will be used.
        - `Cprime::Union{Vector{Float64}, Nothing}`: the Carter's leak off coefficient multiplied by 2.
        - `stagnant::Union{Vector{Bool}, Nothing}`: list of tip cells where the front is not moving.
        - `KIPrime::Union{Vector{Float64}, Nothing}`: the stress intensity factor of the cells where the fracture front is not moving.
        - `dt::Union{Float64, Nothing}`: the time step, only used to calculate leak off.
        - `arrival_t::Union{Vector{Float64}, Nothing}`: the time at which the front passes the given point.
        - `projMethod::Union{String, Nothing}`: projection method.

        # Returns
        - `Vector{Float64}`: the integral of the specified function over the given tip cells.
    """
    function Integral_over_cell(EltTip::Vector{Int}, alpha::Vector{Float64}, l::Vector{Float64}, mesh, function_name::String, 
                            frac=nothing, mat_prop=nothing, fluid_prop=nothing, Vel=nothing,
                            Kprime=nothing, Eprime=nothing, Cprime=nothing, stagnant=nothing, 
                            KIPrime=nothing, dt=nothing, arrival_t=nothing, projMethod=nothing)

        dummy = fill(nothing, length(alpha))

        if stagnant === nothing
            stagnant = dummy
        end
        if KIPrime === nothing
            KIPrime = dummy
        end

        if Kprime === nothing && mat_prop !== nothing
            Kprime = mat_prop.Kprime[EltTip]
        end
        if Kprime === nothing && mat_prop === nothing
            Kprime = dummy
        end

        if Eprime === nothing && mat_prop !== nothing
            Eprime = fill(mat_prop.Eprime, length(alpha))
        end
        if Eprime === nothing && mat_prop === nothing
            Eprime = dummy
        end

        if Vel === nothing
            Vel = dummy
        end

        if mat_prop === nothing
            Cprime = dummy
        elseif Cprime === nothing
            Cprime = mat_prop.Cprime[EltTip]
        end

        if frac !== nothing
            t_lstTS = frac.time
        else
            t_lstTS = nothing
        end

        if arrival_t === nothing
            arrival_t = dummy
        end

        integral = zeros(Float64, length(l))
        i = 1
        while i <= length(l)
            if abs(alpha[i]) >= 1e-8 && abs(alpha[i] - π / 2) >= 1e-8
                m = 1 / (sin(alpha[i]) * cos(alpha[i]))  # the m parameter (see e.g. A. Pierce 2015)
            else 
                m = Inf
            end
            # packing parameters to pass
            # Check is viscosity constant or array
            _muPrime = nothing
            if fluid_prop !== nothing
                _muPrime = fluid_prop.muPrime 
            end
            
            param_pack = (function_name, fluid_prop, Kprime[i], Eprime[i], Cprime[i], Vel[i], stagnant[i], KIPrime[i],
                arrival_t[i], m, t_lstTS, dt, _muPrime)

            if abs(alpha[i]) < 1e-8
                # the angle inscribed by the perpendicular is zero
                if l[i] <= mesh.hx
                    # the front is within the cell.
                    integral[i] = Area(l[i], param_pack...) * mesh.hy
                else
                    # the front has surpassed this cell.
                    integral[i] = (Area(l[i], param_pack...) - Area(l[i] - mesh.hx, param_pack...)) * mesh.hy
                end

            elseif abs(alpha[i] - π / 2) < 1e-8
                # the angle inscribed by the perpendicular is 90 degrees
                if l[i] <= mesh.hy
                    # the front is within the cell.
                    integral[i] = Area(l[i], param_pack...) * mesh.hx
                else
                    # the front has surpassed this cell.
                    integral[i] = (Area(l[i], param_pack...) - Area(l[i] - mesh.hy, param_pack...)) * mesh.hx
                end
            else
                yIntrcpt = l[i] / cos(π / 2 - alpha[i]) # Y intercept of the front line
                grad = -1 / tan(alpha[i]) # gradient of the front line

                # integral of the triangle made by the front by intersecting the x and y directional lines of the cell
                TriVol = VolumeTriangle(l[i], param_pack...)

                # distance of the front from the upper left vertex of the grid cell
                lUp = Pdistance(0, mesh.hy, grad, yIntrcpt)

                if lUp > 0  # upper vertex of the triangle is higher than the grid cell height
                    UpTriVol = VolumeTriangle(lUp, param_pack...)
                else
                    UpTriVol = 0.0
                end

                # distance of the front from the lower right vertex of the grid cell
                lRt = Pdistance(mesh.hx, 0, grad, yIntrcpt)

                if lRt > 0  # right vertex of the triangle is wider than the grid cell width
                    RtTriVol = VolumeTriangle(lRt, param_pack...)
                else
                    RtTriVol = 0.0
                end

                # distance of the front from the upper right vertex of the grid cell
                IntrsctTriDist = Pdistance(mesh.hx, mesh.hy, grad, yIntrcpt)

                if IntrsctTriDist > 0  # front has passed the grid cell
                    IntrsctTri = VolumeTriangle(IntrsctTriDist, param_pack...)
                else
                    IntrsctTri = 0.0
                end

                integral[i] = TriVol - UpTriVol - RtTriVol + IntrsctTri
            end

            if projMethod == "LS_continousfront" && function_name == "A" && integral[i]/ mesh.EltArea > 1.0+1e-4
                @debug "Recomputing Integral over cell (filling fraction) --> if something else goes wrong the tip volume might be the problem" _group="JFrac.Integral_over_cell"
                if abs(alpha[i]) < π / 2 
                    alpha[i] = 0.0
                else 
                    alpha[i] = π / 2
                end
            else
                i = i + 1
            end
        end

        return integral
    end

    """
        FindBracket_w(dist, Kprime, Eprime, muPrime, Cprime, Vel, regime)

        This function finds the bracket to be used by the Universal tip asymptote root finder.

        # Arguments
        - `dist::Float64`: distance
        - `Kprime::Float64`: stress intensity factor
        - `Eprime::Float64`: plane strain modulus
        - `muPrime::Float64`: viscosity
        - `Cprime::Float64`: Carter's leak off coefficient multiplied by 2
        - `Vel::Float64`: velocity
        - `regime::String`: propagation regime

        # Returns
        - `(a, b)`: tuple of brackets
    """
    function FindBracket_w(dist::Float64, Kprime::Float64, Eprime::Float64, muPrime::Float64, 
                        Cprime::Float64, Vel::Float64, regime::String)

        res_func = regime == "U" ? TipAsym_UniversalW_zero_Res : TipAsym_UniversalW_delt_Res

        if dist == 0
            @warn "Zero distance!" _group="JFrac.FindBracket_w"
        end

        wk = dist^0.5 * Kprime / Eprime
        wmtld = 4 / (15^(1 / 4) * (2^0.5 - 1)^(1 / 4)) * 
                            (2 * Cprime * Vel^(1/2) * muPrime / Eprime)^(1/4) * 
                            dist^(5/8)
        wm = 2^(1 / 3) * 3^(5 / 6) * (Vel * muPrime / Eprime)^(1/3) * dist^(2/3)

        if nanmin([wk, wmtld, wm]) > eps()
            b = 0.95 * nanmin([wk, wmtld, wm])
            a = 1.05 * nanmax([wk, wmtld, wm])
        elseif nanmin([wmtld, wm]) > eps()
            b = 0.95 * nanmin([wmtld, wm])
            a = 1.05 * nanmax([wmtld, wm])
        elseif nanmin([wk, wm]) > eps()
            b = 0.95 * nanmin([wk, wm])
            a = 1.05 * nanmax([wk, wm])
        else
            b = 0.95 * nanmax([wk, wmtld, wm])
            a = 1.05 * nanmax([wk, wmtld, wm])
        end

        TipAsmptargs = (dist, Kprime, Eprime, muPrime, Cprime, Vel)

        cnt = 1
        Res_a = res_func(a, TipAsmptargs...)
        Res_b = res_func(b, TipAsmptargs)

        while (Res_a * Res_b > 0 || isnan(Res_a) || isnan(Res_b))
            a = 2 * a
            Res_a = res_func(a, TipAsmptargs...)

            b = 0.5 * b
            Res_b = res_func(b, TipAsmptargs...)

            cnt += 1
            if cnt >= 20
                a = NaN
                b = NaN
                @debug "Can't find bracket after 20 iterations" _group="JFrac.FindBracket_w"
                break
            end
        end

        return a, b
    end

    """
        FindBracket_w_HB(a, b, args...)

        This function finds the bracket to be used by the Universal tip asymptote root finder.

        # Arguments
        - `a::Float64`: initial lower bound
        - `b::Float64`: initial upper bound
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `(a, b)`: tuple of brackets
    """
    function FindBracket_w_HB(a::Float64, b::Float64, args...)

        (l, Kprime, Eprime, muPrime, Cbar, Vel, n, k, T0), dist = args

        Mprime = 2^(n + 1) * (2 * n + 1)^n / n^n * k
        ell = (Kprime^(n + 2) / Mprime / Vel^n / Eprime^(n + 1))^(2 / (2 - n))
        xt = sqrt(dist / ell)
        T0t = T0 * 2 * Eprime * ell / Kprime / Kprime
        alpha = -0.3107 * n + 1.9924
        a = Kprime * sqrt(dist) / Eprime * (1 + (sqrt(4 * π * T0t) * xt)^alpha)^(1 / alpha) + 10*eps()
        b = 1.0
        cnt = 1
        Res_a = TipAsym_res_Herschel_Bulkley_d_given(a, args...)
        Res_b = TipAsym_res_Herschel_Bulkley_d_given(b, args...)
        while Res_a * Res_b > 0
            b = 10^cnt * b
            Res_b = TipAsym_res_Herschel_Bulkley_d_given(b, args...)
            cnt += 1
            if cnt >= 12
                a = NaN
                b = NaN
                @debug "can't find bracket $(Res_a) $(Res_b)" _group="JFrac.FindBracket_w_HB"
            end
        end

        if isnan(Res_a) || isnan(Res_b)
            @debug "res is nan!" _group="JFrac.FindBracket_w_HB"
            a = NaN
            b = NaN
        end
            
        return a, b
    end

    """
        find_corresponding_ribbon_cell(tip_cells, alpha, zero_vertex, mesh)

        zero_vertex is the node index in the mesh.Connectivity
        The four vertices of an element have the following order
        ______ ______ ______
        |      |      |      |
        |   C  |  D   |  E   |
        |______3______2______|
        |      |      |      |
        |   B  |  i   |  F   |
        |______0______1______|
        |      |      |      |
        |   A  |  H   |  G   |
        |______|______|______|


        zero vertex =                0   1    2   3
        ______________________________________________
        case alpha = 0         ->    B   F    F   B
            alpha = pi/2      ->    H   H    D   D
            alpha = any other ->    A   G    E   C

        # Arguments
        - `tip_cells::Vector{Int}`: tip cells
        - `alpha::Vector{Float64}`: angles
        - `zero_vertex::Vector{Int}`: zero vertex indices
        - `mesh`: mesh object

        # Returns
        - `Vector{Int}`: corresponding ribbon cells
    """
    function find_corresponding_ribbon_cell(tip_cells::Vector{Int}, alpha::Vector{Float64}, 
                                        zero_vertex::Vector{Int}, mesh)
        #                         0     1      2      3
        #       NeiElements[i]->[left, right, bottom, up]
        #                         B     F      H      D
        corr_ribbon = Vector{Int}(undef, length(tip_cells))
        for i in 1:length(tip_cells)
            if alpha[i] == 0
                if zero_vertex[i] == 0 || zero_vertex[i] == 3
                    corr_ribbon[i] = mesh.NeiElements[tip_cells[i], 1]
                elseif zero_vertex[i] == 1 || zero_vertex[i] == 2
                    corr_ribbon[i] = mesh.NeiElements[tip_cells[i], 2]
                end
            elseif alpha[i] == π/2
                if zero_vertex[i] == 0 || zero_vertex[i] == 1
                    corr_ribbon[i] = mesh.NeiElements[tip_cells[i], 3]
                elseif zero_vertex[i] == 3 || zero_vertex[i] == 2
                    corr_ribbon[i] = mesh.NeiElements[tip_cells[i], 4]
                end
            else
                if zero_vertex[i] == 0
                    corr_ribbon[i] = mesh.NeiElements[mesh.NeiElements[tip_cells[i], 3], 1]
                elseif zero_vertex[i] == 1
                    corr_ribbon[i] = mesh.NeiElements[mesh.NeiElements[tip_cells[i], 3], 2]
                elseif zero_vertex[i] == 2
                    corr_ribbon[i] = mesh.NeiElements[mesh.NeiElements[tip_cells[i], 4], 2]
                elseif zero_vertex[i] == 3
                    corr_ribbon[i] = mesh.NeiElements[mesh.NeiElements[tip_cells[i], 4], 1]
                end
            end
        end

        return corr_ribbon
    end

    """
        leak_off_stagnant_tip(Elts, l, alpha, vrtx_arr_time, current_time, Cprime, time_step, mesh)

        This function evaluates leak-off in the tip cells with stagnant front. Its samples the leak-off midway from the
        zero vertex of the cell to the front and multiply it with the area of the fracture in the cell (filling fraction
        times the area of the cell).
        todo: can be more precise

        # Arguments
        - `Elts::Vector{Int}`: elements
        - `l::Vector{Float64}`: distances
        - `alpha::Vector{Float64}`: angles
        - `vrtx_arr_time::Vector{Float64}`: vertex arrival times
        - `current_time::Float64`: current time
        - `Cprime::Vector{Float64}`: Carter's leak off coefficient multiplied by 2
        - `time_step::Float64`: time step
        - `mesh`: mesh object

        # Returns
        - `Vector{Float64}`: leak-off
    """
    function leak_off_stagnant_tip(Elts::Vector{Int}, l::Vector{Float64}, alpha::Vector{Float64}, 
                                vrtx_arr_time::Vector{Float64}, current_time::Float64, 
                                Cprime::Vector{Float64}, time_step::Float64, mesh)
        arrival_time_mid = (current_time + vrtx_arr_time) / 2
        t_since_arrival = current_time - arrival_time_mid
        area = Integral_over_cell(Elts, alpha, l, mesh, "A")
        t_since_arrival_lstTS = t_since_arrival - time_step
        t_since_arrival_lstTS[t_since_arrival_lstTS .< 0] .= 0
        LkOff = 2 * Cprime[Elts] .* (t_since_arrival.^0.5 - t_since_arrival_lstTS.^0.5) .* area

        return LkOff
    end

    function nanmin(arr)
        valid_values = filter(!isnan, arr)
        return isempty(valid_values) ? NaN : minimum(valid_values)
    end

    function nanmax(arr)
        valid_values = filter(!isnan, arr)
        return isempty(valid_values) ? NaN : maximum(valid_values)
    end



end # module VolumeIntegral