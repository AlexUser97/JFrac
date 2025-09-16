# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac explicit_RKL module on Julia language.

"""

module FluidModel

    using SpecialFunctions
    using Roots

    export FF_YangJoseph_vector, FF_YangJoseph, FF_Yang_Dou_residual, FF_Yang_Dou, 
        friction_factor_lam_turb_rough, friction_factor_MDR, friction_factor_vector

    """
        FF_YangJoseph_vector(ReNum, rough)

        This function approximate the friction factor for the given Reynold's number and the relative roughness arrays with
        the Yang Joseph approximation (see Virtual Nikuradse Yang & Joseph 2009).

        # Arguments
        - `ReNum::Vector{Float64}`: Reynold's number
        - `rough::Vector{Float64}`: 1/relative roughness (w/roughness length scale)

        # Returns
        - `Vector{Float64}`: friction factor
    """
    function FF_YangJoseph_vector(ReNum::Vector{Float64}, rough::Vector{Float64})::Vector{Float64}
        ff = fill(Inf, length(ReNum))


        lam = findall(abs.(ReNum) .< 2100)
        if !isempty(lam)
            ff[lam] = 16.0 ./ ReNum[lam]
        end


        turb = findall(abs.(ReNum) .>= 2100)
        if !isempty(turb)
            ReNum_turb = ReNum[turb]
            rough_turb = rough[turb]
            

            term1 = (-64.0 ./ ReNum_turb + 0.000083 * ReNum_turb.^0.75) ./ sqrt.(1.0 + (2320.0^50) ./ ReNum_turb.^50)
            term2 = -term1 - 64.0 ./ ReNum_turb + 0.3164 ./ ReNum_turb.^0.25
            term3 = term2 ./ sqrt.(1.0 + (3810.0^15) ./ ReNum_turb.^15)
            term4 = -term3 - (-64.0 ./ ReNum_turb + 0.000083 * ReNum_turb.^0.75) ./ sqrt.(1.0 + (2320.0^50) ./ ReNum_turb.^50) - 64.0 ./ ReNum_turb + 0.1537 ./ ReNum_turb.^0.185
            term5 = term4 ./ sqrt.(1.0 + 1.6807e24 ./ ReNum_turb.^5)
            term6 = -term5 - term3 - (-64.0 ./ ReNum_turb + 0.000083 * ReNum_turb.^0.75) ./ sqrt.(1.0 + (2320.0^50) ./ ReNum_turb.^50) - 64.0 ./ ReNum_turb + 0.0753 ./ ReNum_turb.^0.136
            term7 = term6 ./ sqrt.(1.0 + 4.0e12 ./ ReNum_turb.^2)
            lamdaS = term7 + (-64.0 ./ ReNum_turb + 0.000083 * ReNum_turb.^0.75) ./ sqrt.(1.0 + (2320.0^50) ./ ReNum_turb.^50) + 64.0 ./ ReNum_turb


            term_a = ReNum_turb.^(-0.2032 + 7.348278 ./ rough_turb.^0.96433953)
            term_b = -0.022 + (-0.978 + 0.92820419 * rough_turb.^0.03569244 - 0.00255391 * rough_turb.^0.8353877) ./ 
                    sqrt.(1.0 + 2.6555068601372822e49 ./ rough_turb.^50) + 0.00255391 * rough_turb.^0.8353877
            term_c = -(term_a .* term_b) + 0.01105244 * ReNum_turb.^(-0.191 + 0.62935712 ./ rough_turb.^0.28022284) .* rough_turb.^0.23275646
            term_d = ReNum_turb.^(0.015 + 0.26827956 ./ rough_turb.^0.28852025) .* (0.0053 + 0.02166401 ./ rough_turb.^0.30702955)
            term_e = -0.01105244 * ReNum_turb.^(-0.191 + 0.62935712 ./ rough_turb.^0.28022284) .* rough_turb.^0.23275646
            term_f = ReNum_turb.^0.002 .* (0.011 + 0.18954211 ./ rough_turb.^0.510031)
            term_g = -ReNum_turb.^(0.015 + 0.26827956 ./ rough_turb.^0.28852025) .* (0.0053 + 0.02166401 ./ rough_turb.^0.30702955)
            term_h = (0.0098 - ReNum_turb.^0.002 .* (0.011 + 0.18954211 ./ rough_turb.^0.510031) + 0.17805185 ./ rough_turb.^0.46785053) ./ 
                    sqrt.(1.0 + 8.733801045300249e10 * rough_turb.^0.90870686 ./ ReNum_turb.^2)
            term_i = term_h ./ sqrt.(1.0 + 6.44205549308073e15 * rough_turb.^5.168887 ./ ReNum_turb.^5)
            term_j = term_i ./ sqrt.(1.0 + 1.1077593467238922e13 * rough_turb.^4.9771653 ./ ReNum_turb.^5)
            lamdaR = term_a .* term_b + term_c + term_d + term_e + term_f + term_g + term_j
            lamdaR ./= sqrt.(1.0 + 2.9505925619934144e14 * rough_turb.^3.7622822 ./ ReNum_turb.^5)


            denominator = 1.0 + (ReNum_turb ./ (45.196502 * rough_turb.^1.2369807 + 1891.0)).^(-5)
            ff[turb] = (lamdaS + (lamdaR - lamdaS) ./ sqrt.(denominator)) / 4.0
        end

        return ff
    end

    """
        FF_YangJoseph(ReNum, rough)

        This function approximate the friction factor for the given Reynold's number and the relative roughness float 
        with the Yang Joseph approximation (see Virtual Nikuradse Yang & Joseph 2009).

        # Arguments
        - `ReNum::Float64`: Reynold's number
        - `rough::Float64`: 1/relative roughness (w/roughness length scale)

        # Returns
        - `Float64`: friction factor
    """
    function FF_YangJoseph(ReNum::Float64, rough::Float64)::Float64
        if ReNum < 1e-8
            return 0.0
        elseif ReNum < 2100
            return 16.0 / ReNum
        else

            term1 = (-64.0 / ReNum + 0.000083 * ReNum^0.75) / sqrt(1.0 + (2320.0^50) / ReNum^50)
            term2 = -term1 - 64.0 / ReNum + 0.3164 / ReNum^0.25
            term3 = term2 / sqrt(1.0 + (3810.0^15) / ReNum^15)
            term4 = -term3 - (-64.0 / ReNum + 0.000083 * ReNum^0.75) / sqrt(1.0 + (2320.0^50) / ReNum^50) - 64.0 / ReNum + 0.1537 / ReNum^0.185
            term5 = term4 / sqrt(1.0 + 1.6807e24 / ReNum^5)
            term6 = -term5 - term3 - (-64.0 / ReNum + 0.000083 * ReNum^0.75) / sqrt(1.0 + (2320.0^50) / ReNum^50) - 64.0 / ReNum + 0.0753 / ReNum^0.136
            term7 = term6 / sqrt(1.0 + 4.0e12 / ReNum^2)
            lamdaS = term7 + (-64.0 / ReNum + 0.000083 * ReNum^0.75) / sqrt(1.0 + (2320.0^50) / ReNum^50) + 64.0 / ReNum


            term_a = ReNum^(-0.2032 + 7.348278 / rough^0.96433953)
            term_b = -0.022 + (-0.978 + 0.92820419 * rough^0.03569244 - 0.00255391 * rough^0.8353877) / 
                    sqrt(1.0 + 2.6555068601372822e49 / rough^50) + 0.00255391 * rough^0.8353877
            term_c = -(term_a * term_b) + 0.01105244 * ReNum^(-0.191 + 0.62935712 / rough^0.28022284) * rough^0.23275646
            term_d = ReNum^(0.015 + 0.26827956 / rough^0.28852025) * (0.0053 + 0.02166401 / rough^0.30702955)
            term_e = -0.01105244 * ReNum^(-0.191 + 0.62935712 / rough^0.28022284) * rough^0.23275646
            term_f = ReNum^0.002 * (0.011 + 0.18954211 / rough^0.510031)
            term_g = -ReNum^(0.015 + 0.26827956 / rough^0.28852025) * (0.0053 + 0.02166401 / rough^0.30702955)
            term_h = (0.0098 - ReNum^0.002 * (0.011 + 0.18954211 / rough^0.510031) + 0.17805185 / rough^0.46785053) / 
                    sqrt(1.0 + 8.733801045300249e10 * rough^0.90870686 / ReNum^2)
            term_i = term_h / sqrt(1.0 + 6.44205549308073e15 * rough^5.168887 / ReNum^5)
            term_j = term_i / sqrt(1.0 + 1.1077593467238922e13 * rough^4.9771653 / ReNum^5)
            lamdaR = term_a * term_b + term_c + term_d + term_e + term_f + term_g + term_j
            lamdaR /= sqrt(1.0 + 2.9505925619934144e14 * rough^3.7622822 / ReNum^5)


            denominator = 1.0 + (ReNum / (45.196502 * rough^1.2369807 + 1891.0))^(-5)
            return (lamdaS + (lamdaR - lamdaS) / sqrt(denominator)) / 4.0
        end
    end

    """
        FF_Yang_Dou_residual(vbyu, Re, rough)

        The Yang_Dou residual function; to be used by numerical root finder

        # Arguments
        - `vbyu::Float64`: velocity divided by friction velocity
        - `Re::Float64`: Reynolds number
        - `rough::Float64`: 1/relative roughness

        # Returns
        - `Float64`: residual value
    """
    function FF_Yang_Dou_residual(vbyu::Float64, Re::Float64, rough::Float64)::Float64
        Rstar = Re / (2.0 * vbyu * rough)
        theta = π * log(Rstar / 1.25) / log(100.0 / 1.25)
        alpha = (1.0 - cos(theta)) / 2.0
        beta = 1.0 - (1.0 - 0.107) * (alpha + theta/π) / 2.0
        R = Re / (2.0 * vbyu)

        rt = 1.0
        for i in 1:4
            rt = rt - 1.0 / exp(1) * (i / factorial(i) * (67.8 / R)^(2 * i))
        end

        term1 = (1.0 - rt) * R / 4.0
        term2 = rt * (2.5 * log(R) - 66.69 * R^(-0.72) + 1.8)
        term3 = 2.5 * log((1.0 + alpha * Rstar / 5.0) / (1.0 + alpha * beta * Rstar / 5.0))
        term4 = (5.8 + 1.25) * (alpha * Rstar / (5.0 + alpha * Rstar))^2
        term5 = 2.5 * (alpha * Rstar / (5.0 + alpha * Rstar))
        term6 = (5.8 + 1.25) * (alpha * beta * Rstar / (5.0 + alpha * beta * Rstar))^2
        term7 = 2.5 * (alpha * beta * Rstar / (5.0 + alpha * beta * Rstar))
        term8 = term3 + term4 + term5 - term6 - term7
        
        return vbyu - term1 - term2 - rt * term8
    end

    """
        FF_Yang_Dou(Re, rough)

        This function approximate the friction factor for the given Reynold's number and the relative roughness arrays with
        the Yang Dou approximation (see Yang, S. Dou, G. (2010). Turbulent drag reduction with polymer additive in rough 
        pipes. Journal of Fluid Mechanics, 642 279-294). The function is implicit and utilize a numerical root finder

        # Arguments
        - `Re::Float64`: Reynold's number
        - `rough::Float64`: 1/relative roughness (w/roughness length scale)

        # Returns
        - `Float64`: friction factor
    """
    function FF_Yang_Dou(Re::Float64, rough::Float64)::Float64
        # sol_vbyu = find_zero(x -> FF_Yang_Dou_residual(x, Re, rough), 15.0, Bisection())
        sol_vbyu = find_zero(x -> FF_Yang_Dou_residual(x, Re, rough), 15.0)
        
        ff_Yang_Dou = 2.0 / sol_vbyu^2
        Rplus = Re / (2.0 * sol_vbyu)
        ff_Man_Strkl = 0.143 / 4.0 / rough^(1.0/3.0)

        ff = ff_Yang_Dou
        if Rplus >= 100.0 * rough
            ff = ff_Man_Strkl
        end
        if rough < 32.0 && ff > ff_Man_Strkl
            ff = ff_Man_Strkl
        end

        return ff
    end

    """
        friction_factor_lam_turb_rough(Re, roughness)

        This function approximate the friction factor for the given Reynold's number and the relative roughness arrays. The
        analytical friction factor of 16/Re is returned in case of laminar flow, Yang Joseph (see Virtual Nikuradse Yang & 
        Joseph 2009) approximation is used for the turbulent flow cases where the 1/relative roughness is greater than 15, 
        and Yang Dou approximation (see Yang, S. Dou, G. (2010), Turbulent drag reduction with polymer additive in rough 
        pipes. Journal of Fluid Mechanics, 642 279-294) is used in the case of turbulent flow with 1/relative roughness 
        less than 15.

        # Arguments
        - `Re::Float64`: Reynold's number
        - `roughness::Float64`: 1/relative roughness (w/roughness length scale)

        # Returns
        - `Float64`: friction factor
    """
    function friction_factor_lam_turb_rough(Re::Float64, roughness::Float64)::Float64
        if Re < 1e-8
            return 0.0
        elseif Re < 2300.0
            return 16.0 / Re
        # elseif roughness >= 15.0
        #     return FF_YangJoseph(Re, roughness)
        else
            return FF_Yang_Dou(Re, roughness)
        end
    end

    """
        friction_factor_MDR(ReNum, roughness)

        This function approximate the friction factor for the given Reynold's number and the relative roughness arrays. The
        analytical friction factor of 16/Re is returned in case of laminar flow, and an explicit approximation of the
        maximum drag reduction asymptote is returned in case the Reynold's number is larger than 1760.

        # Arguments
        - `ReNum::Float64`: Reynold's number
        - `roughness::Float64`: relative roughness

        # Returns
        - `Float64`: friction factor
    """
    function friction_factor_MDR(ReNum::Float64, roughness::Float64)::Float64
        if ReNum < 1e-8
            return 0.0
        elseif ReNum < 1510.0 #18112100.0
            return 16.0 / ReNum
        else
            return 1.78 / ReNum^0.7 #1.1 / ReNum^0.65
        end
    end

    """
        friction_factor_vector(Re, roughness)

        Vector version of the friction_factor function (see the documentation of the friction_factor function)

        # Arguments
        - `Re::Vector{Float64}`: Reynold's number
        - `roughness::Vector{Float64}`: relative roughness

        # Returns
        - `Vector{Float64}`: friction factor
    """
    function friction_factor_vector(Re::Vector{Float64}, roughness::Vector{Float64})::Vector{Float64}
        ff = zeros(Float64, length(Re))
        for i in 1:length(Re)
            ff[i] = friction_factor_MDR(Re[i], roughness[i])
            # ff[i] = friction_factor_lam_turb_rough(Re[i], roughness[i])
        end
        return ff
    end

end # module FluidModel