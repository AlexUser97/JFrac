# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac elasticity module on Julia language.
"""

module ElastoHydrodynamicSolver

    include("fluid_model.jl")
    include("properties.jl")
    using .FluidModel: friction_factor_vector, friction_factor_MDR
    using .Properties: instrument_start, instrument_close

    using Logging
    using PyPlot
    using SparseArrays
    using Roots
    using LinearAlgebra


    corr_functions = Dict(
        "Amadei_Illangasekare" => (sigma, dm) -> dm * (1 / (1 + 0.6 * (sigma/dm)^1.2))^(1/3),
        "Patir_Cheng" => (sigma, dm) -> dm * cbrt(1 - 0.9 * exp(-0.56 * dm / sigma)),
        "Quadros" => (sigma, dm) -> dm * cbrt(1 / (1 + 20.5 * (sigma/2/dm)^1.5))
    )

    """
        The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
        equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated with the laminar flow assumption.

        Arguments:
            w (Vector{Float64}): The width of the trial fracture.
            EltCrack (Vector{Int}): The list of elements inside the fracture.
            muPrime (Float64): The scaled local viscosity of the injected fluid (12 * viscosity).
            Mesh (CartesianMesh): The mesh.
            InCrack (Vector{Int}): An array specifying whether elements are inside the fracture or not with 1 or 0 respectively.
            neiInCrack (Matrix{Int}): An ndarray giving indices of the neighbours of all the cells in the crack, in the EltCrack list.
            simProp (object): An object of the SimulationProperties class.

        Returns:
            FinDiffOprtr (Union{SparseMatrixCSC{Float64, Int}, Matrix{Float64}}): The finite difference matrix.
    """

    function finiteDiff_operator_laminar(w, EltCrack, muPrime, Mesh, InCrack, neiInCrack, simProp)
        
        # Note: corr_functions and roughness_func need to be implemented
        if simProp.roughness_model !== nothing && simProp.roughness_sigma !== nothing
            corr_function = corr_functions[simProp.roughness_model]
            roughness_func = (dm, sigma=simProp.roughness_sigma) -> corr_function(sigma, dm)
        end

        dx = Mesh.hx
        dy = Mesh.hy

        if simProp.solveSparse
            FinDiffOprtr = spzeros(Float64, length(EltCrack), length(EltCrack) + 1)
        else
            FinDiffOprtr = zeros(Float64, length(EltCrack), length(EltCrack) + 1)
        end

        # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
        wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 1]]
        wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 2]]
        wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 3]]
        wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 4]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 4]]

        # apply roughness correction
        if simProp.roughness_model !== nothing && simProp.roughness_sigma !== nothing
            # Note: map function in Julia works differently, applying function to each argument
            wLftEdge, wRgtEdge, wBtmEdge, wTopEdge = map(dm -> roughness_func(dm, simProp.roughness_sigma), 
                                                        [wLftEdge, wRgtEdge, wBtmEdge, wTopEdge])
        end

        indx_elts = 1:length(EltCrack)

        # Main diagonal
        FinDiffOprtr[indx_elts, indx_elts] = (-(wLftEdge.^3 + wRgtEdge.^3) / dx^2 - 
                                            (wBtmEdge.^3 + wTopEdge.^3) / dy^2) / muPrime
        
        # Off-diagonal elements
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = wLftEdge.^3 / dx^2 / muPrime
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = wRgtEdge.^3 / dx^2 / muPrime
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = wBtmEdge.^3 / dy^2 / muPrime
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 4]] = wTopEdge.^3 / dy^2 / muPrime

        return FinDiffOprtr
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function returns the gravity term (G in Zia and Lecampion 2019).

        Arguments:
            w (Vector{Float64}): The width of the trial fracture.
            EltCrack (Vector{Int}): The list of elements inside the fracture.
            fluidProp (object): FluidProperties class object giving the fluid properties.
            mesh (CartesianMesh): The mesh.
            InCrack (Vector{Int}): An array specifying whether elements are inside the fracture or not with 1 or 0 respectively.
            simProp (object): An object of the SimulationProperties class.
            cond (Matrix{Float64}): Condition matrix (needed for non-Newtonian fluids).

        Returns:
            G (Vector{Float64}): The matrix with the gravity terms.
    """
    function Gravity_term(w, EltCrack, fluidProp, mesh, InCrack, simProp, cond)
        
        G = zeros(Float64, mesh.NumberOfElts)

        if simProp.gravity
            if fluidProp.rheology == "Newtonian" && !fluidProp.turbulence
                # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
                wBtmEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 3]]) / 2 .* InCrack[mesh.NeiElements[EltCrack, 3]]
                wTopEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 4]]) / 2 .* InCrack[mesh.NeiElements[EltCrack, 4]]

                G[EltCrack] = fluidProp.density * 9.81 * (wTopEdge.^3 - wBtmEdge.^3) / mesh.hy / fluidProp.muPrime
            elseif fluidProp.rheology in ["Herschel-Bulkley", "HBF", "power law", "PLF"]
                G[EltCrack] = fluidProp.density * 9.81 * (cond[4, :] - cond[3, :]) / mesh.hy
            else
                throw(SystemExit("Effect of gravity is not supported with this fluid model!"))
            end
        end

        return G
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        The function evaluate the finite difference matrix, i.e. the A matrix in the ElastoHydrodynamic equations ( see e.g.
        Dontsov and Peirce 2008). The matrix is evaluated by taking turbulence into account.

        Arguments:
            w (Vector{Float64}): The width of the trial fracture.
            pf (Vector{Float64}): Pressure field.
            EltCrack (Vector{Int}): The list of elements inside the fracture.
            fluidProp (object): FluidProperties class object giving the fluid properties.
            matProp (object): An instance of the MaterialProperties class giving the material properties.
            simProp (object): An object of the SimulationProperties class.
            mesh (CartesianMesh): The mesh.
            InCrack (Vector{Int}): An array specifying whether elements are inside the fracture or not with 1 or 0 respectively.
            vkm1 (Matrix{Float64}): The velocity at cell edges from the previous iteration.
            to_solve (Vector{Int}): The channel elements to be solved.
            active (Vector{Int}): The channel elements where width constraint is active.
            to_impose (Vector{Int}): The tip elements to be imposed.

        Returns:
            - FinDiffOprtr (Union{SparseMatrixCSC{Float64, Int}, Matrix{Float64}}) -- the finite difference matrix.
            - vk (Matrix{Float64}) -- the velocity evaluated for current iteration.
            - cond (Matrix{Float64}) -- the conductivity matrix.
    """

    function FiniteDiff_operator_turbulent_implicit(w, pf, EltCrack, fluidProp, matProp, simProp, mesh, InCrack, vkm1, to_solve,
                                            active, to_impose)
        
        if simProp.roughness_model !== nothing && simProp.roughness_sigma !== nothing
            corr_function = corr_functions[simProp.roughness_model]
            roughness_func = (sigma, dm) -> corr_function(sigma, dm)
        end
        
        if simProp.solveSparse
            FinDiffOprtr = spzeros(Float64, length(w), length(w))
        else
            FinDiffOprtr = zeros(Float64, length(w), length(w))
        end

        dx = mesh.hx
        dy = mesh.hy

        # todo: can be evaluated at each cell edge
        rough = w[EltCrack] / matProp.grainSize
        rough[findall(rough .< 3.0)] .= 3.0

        # width on edges; evaluated by averaging the widths of adjacent cells
        wLftEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 1]]) / 2
        wRgtEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 2]]) / 2
        wBtmEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 3]]) / 2
        wTopEdge = (w[EltCrack] + w[mesh.NeiElements[EltCrack, 4]]) / 2

        if simProp.roughness_model !== nothing && simProp.roughness_sigma !== nothing
            w_results = map(dm -> roughness_func(simProp.roughness_sigma, dm), [wLftEdge, wRgtEdge, wBtmEdge, wTopEdge])
            wLftEdge, wRgtEdge, wBtmEdge, wTopEdge = w_results
        end

        # pressure gradient data structure. The rows store pressure gradient in the following order.
        # 0 - left edge in x-direction    # 1 - right edge in x-direction
        # 2 - bottom edge in y-direction  # 3 - top edge in y-direction
        # 4 - left edge in y-direction    # 5 - right edge in y-direction
        # 6 - bottom edge in x-direction  # 7 - top edge in x-direction

        dp = zeros(Float64, 8, mesh.NumberOfElts)
        dp[1, EltCrack] = (pf[EltCrack] - pf[mesh.NeiElements[EltCrack, 1]]) / dx
        dp[2, EltCrack] = (pf[mesh.NeiElements[EltCrack, 2]] - pf[EltCrack]) / dx
        dp[3, EltCrack] = (pf[EltCrack] - pf[mesh.NeiElements[EltCrack, 3]]) / dy
        dp[4, EltCrack] = (pf[mesh.NeiElements[EltCrack, 4]] - pf[EltCrack]) / dy
        # linear interpolation for pressure gradient on the edges where central difference not available
        dp[5, EltCrack] = (dp[3, mesh.NeiElements[EltCrack, 1]] + dp[4, mesh.NeiElements[EltCrack, 1]] + dp[3, EltCrack] + dp[4, EltCrack]) / 4
        dp[6, EltCrack] = (dp[3, mesh.NeiElements[EltCrack, 2]] + dp[4, mesh.NeiElements[EltCrack, 2]] + dp[3, EltCrack] + dp[4, EltCrack]) / 4
        dp[7, EltCrack] = (dp[1, mesh.NeiElements[EltCrack, 3]] + dp[2, mesh.NeiElements[EltCrack, 3]] + dp[1, EltCrack] + dp[2, EltCrack]) / 4
        dp[8, EltCrack] = (dp[1, mesh.NeiElements[EltCrack, 4]] + dp[2, mesh.NeiElements[EltCrack, 4]] + dp[1, EltCrack] + dp[2, EltCrack]) / 4

        # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
        dpLft = (dp[1, EltCrack].^2 + dp[5, EltCrack].^2).^0.5
        dpRgt = (dp[2, EltCrack].^2 + dp[6, EltCrack].^2).^0.5
        dpBtm = (dp[3, EltCrack].^2 + dp[7, EltCrack].^2).^0.5
        dpTop = (dp[4, EltCrack].^2 + dp[8, EltCrack].^2).^0.5

        vk = zeros(Float64, 8, mesh.NumberOfElts)
        # the factor to be multiplied to the velocity from last iteration to get the upper bracket
        upBracket_factor = 10

        # loop to calculate velocity on each cell edge implicitly
        for i in 1:length(EltCrack)
            viscosity = fluidProp.viscosity
            # todo !!! Hack. zero velocity if the pressure gradient is zero or very small width
            if dpLft[i] < 1e-8 || wLftEdge[i] < 1e-10
                vk[1, EltCrack[i]] = 0.0
            else
                arg = (wLftEdge[i], viscosity, fluidProp.density, dpLft[i], rough[i])
                # check if bracket gives residuals with opposite signs
                if Velocity_Residual(eps() * vkm1[1, EltCrack[i]], arg...) * Velocity_Residual(
                                upBracket_factor * vkm1[1, EltCrack[i]], arg...) > 0.0
                    # bracket not valid. finding suitable bracket
                    a, b = findBracket(Velocity_Residual, vkm1[1, EltCrack[i]], arg...)
                    vk[1, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), (a, b), Bisection())
                else
                    # find the root with brentq method.
                    vk[1, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), 
                                                (eps() * vkm1[1, EltCrack[i]], upBracket_factor * vkm1[1, EltCrack[i]]), 
                                                Bisection())
                end
            end

            if dpRgt[i] < 1e-8 || wRgtEdge[i] < 1e-10
                vk[2, EltCrack[i]] = 0.0
            else
                arg = (wRgtEdge[i], viscosity, fluidProp.density, dpRgt[i], rough[i])
                # check if bracket gives residuals with opposite signs
                if Velocity_Residual(eps() * vkm1[2, EltCrack[i]], arg...) * Velocity_Residual(
                                upBracket_factor * vkm1[2, EltCrack[i]], arg...) > 0.0
                    # bracket not valid. finding suitable bracket
                    a, b = findBracket(Velocity_Residual, vkm1[2, EltCrack[i]], arg...)
                    vk[2, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), (a, b), Bisection())
                else
                    # find the root with brentq method.
                    vk[2, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), 
                                                (eps() * vkm1[2, EltCrack[i]], upBracket_factor * vkm1[2, EltCrack[i]]), 
                                                Bisection())
                end
            end

            if dpBtm[i] < 1e-8 || wBtmEdge[i] < 1e-10
                vk[3, EltCrack[i]] = 0.0
            else
                arg = (wBtmEdge[i], viscosity, fluidProp.density, dpBtm[i], rough[i])
                # check if bracket gives residuals with opposite signs
                if Velocity_Residual(eps() * vkm1[3, EltCrack[i]], arg...) * Velocity_Residual(
                                upBracket_factor * vkm1[3, EltCrack[i]], arg...) > 0.0
                    # bracket not valid. finding suitable bracket
                    a, b = findBracket(Velocity_Residual, vkm1[3, EltCrack[i]], arg...)
                    vk[3, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), (a, b), Bisection())
                else
                    # find the root with brentq method.
                    vk[3, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), 
                                                (eps() * vkm1[3, EltCrack[i]], upBracket_factor * vkm1[3, EltCrack[i]]), 
                                                Bisection())
                end
            end

            if dpTop[i] < 1e-8 || wTopEdge[i] < 1e-10
                vk[4, EltCrack[i]] = 0.0
            else
                arg = (wTopEdge[i], viscosity, fluidProp.density, dpTop[i], rough[i])
                # check if bracket gives residuals with opposite signs
                if Velocity_Residual(eps() * vkm1[4, EltCrack[i]], arg...) * Velocity_Residual(
                                upBracket_factor * vkm1[4, EltCrack[i]], arg...) > 0.0
                    # bracket not valid. finding suitable bracket
                    a, b = findBracket(Velocity_Residual, vkm1[4, EltCrack[i]], arg...)
                    vk[4, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), (a, b), Bisection())
                else
                    # find the root with brentq method.
                    vk[4, EltCrack[i]] = find_zero(x -> Velocity_Residual(x, arg...), 
                                                (eps() * vkm1[4, EltCrack[i]], upBracket_factor * vkm1[4, EltCrack[i]]), 
                                                Bisection())
                end
            end
        end

        # calculating Reynold's number with the velocity
        viscosity = fluidProp.viscosity
        ReLftEdge = 4 / 3 * fluidProp.density * wLftEdge .* vk[1, EltCrack] / viscosity
        ReRgtEdge = 4 / 3 * fluidProp.density * wRgtEdge .* vk[2, EltCrack] / viscosity
        ReBtmEdge = 4 / 3 * fluidProp.density * wBtmEdge .* vk[3, EltCrack] / viscosity
        ReTopEdge = 4 / 3 * fluidProp.density * wTopEdge .* vk[4, EltCrack] / viscosity

        # non zeros Reynolds numbers
        ReLftEdge_nonZero = findall(ReLftEdge .> 0.0)
        ReRgtEdge_nonZero = findall(ReRgtEdge .> 0.0)
        ReBtmEdge_nonZero = findall(ReBtmEdge .> 0.0)
        ReTopEdge_nonZero = findall(ReTopEdge .> 0.0)

        # calculating friction factor with the Yang-Joseph explicit function
        ffLftEdge = zeros(Float64, length(EltCrack))
        ffRgtEdge = zeros(Float64, length(EltCrack))
        ffBtmEdge = zeros(Float64, length(EltCrack))
        ffTopEdge = zeros(Float64, length(EltCrack))
        ffLftEdge[ReLftEdge_nonZero] = friction_factor_vector(ReLftEdge[ReLftEdge_nonZero], rough[ReLftEdge_nonZero])
        ffRgtEdge[ReRgtEdge_nonZero] = friction_factor_vector(ReRgtEdge[ReRgtEdge_nonZero], rough[ReRgtEdge_nonZero])
        ffBtmEdge[ReBtmEdge_nonZero] = friction_factor_vector(ReBtmEdge[ReBtmEdge_nonZero], rough[ReBtmEdge_nonZero])
        ffTopEdge[ReTopEdge_nonZero] = friction_factor_vector(ReTopEdge[ReTopEdge_nonZero], rough[ReTopEdge_nonZero])

        # the conductivity matrix
        cond = zeros(Float64, 4, length(EltCrack))
        cond[1, ReLftEdge_nonZero] = wLftEdge[ReLftEdge_nonZero].^2 ./ (fluidProp.density * ffLftEdge[ReLftEdge_nonZero]
                                                                        .* vk[1, EltCrack[ReLftEdge_nonZero]])
        cond[2, ReRgtEdge_nonZero] = wRgtEdge[ReRgtEdge_nonZero].^2 ./ (fluidProp.density * ffRgtEdge[ReRgtEdge_nonZero]
                                                                        .* vk[2, EltCrack[ReRgtEdge_nonZero]])
        cond[3, ReBtmEdge_nonZero] = wBtmEdge[ReBtmEdge_nonZero].^2 ./ (fluidProp.density * ffBtmEdge[ReBtmEdge_nonZero]
                                                                        .* vk[3, EltCrack[ReBtmEdge_nonZero]])
        cond[4, ReTopEdge_nonZero] = wTopEdge[ReTopEdge_nonZero].^2 ./ (fluidProp.density * ffTopEdge[ReTopEdge_nonZero]
                                                                        .* vk[4, EltCrack[ReTopEdge_nonZero]])

        # assembling the finite difference matrix
        FinDiffOprtr[EltCrack, EltCrack] = -(cond[1, :] + cond[2, :]) / dx^2 - (cond[3, :] + cond[4, :]) / dy^2
        FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 1]] = cond[1, :] / dx^2
        FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 2]] = cond[2, :] / dx^2
        FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 3]] = cond[3, :] / dy^2
        FinDiffOprtr[EltCrack, mesh.NeiElements[EltCrack, 4]] = cond[4, :] / dy^2

        ch_indxs = 1:length(to_solve)
        act_indxs = length(to_solve) .+ (1:length(active))
        tip_indxs = length(to_solve) + length(active) .+ (1:length(to_impose))

        indx_elts = 1:length(EltCrack)
        
        if simProp.solveSparse
            FinDiffOprtr_csr = sparse(FinDiffOprtr)
            n_rows = length(EltCrack)
            n_cols = length(to_solve) + length(active) + length(to_impose)
            FD_compressed = spzeros(Float64, n_rows, n_cols)
            
            # Note: In Julia, we need to handle the indexing differently
            for (i, row_idx) in enumerate(EltCrack)
                for (j, col_idx) in enumerate(to_solve)
                    if row_idx <= size(FinDiffOprtr_csr, 1) && col_idx <= size(FinDiffOprtr_csr, 2)
                        FD_compressed[i, j] = FinDiffOprtr_csr[row_idx, col_idx]
                    end
                end
            end
            # Similar loops for active and to_impose...
        else
            FD_compressed = zeros(Float64, length(EltCrack), length(to_solve) + length(active) + length(to_impose))
            FD_compressed[indx_elts, ch_indxs] = FinDiffOprtr[EltCrack, to_solve]
            FD_compressed[indx_elts, act_indxs] = FinDiffOprtr[EltCrack, active]
            FD_compressed[indx_elts, tip_indxs] = FinDiffOprtr[EltCrack, to_impose]
        end

        if !simProp.gravity
            cond = nothing
        end

        return FD_compressed, vk, cond
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function gives the residual of the velocity equation. It is used by the root finder.

        Arguments:
            v (Float64): Current velocity guess.
            args (Tuple): A tuple consisting of the following:
                - w (Float64)          width at the given cell edge
                - mu (Float64)         viscosity at the given cell edge
                - rho (Float64)        density of the injected fluid
                - dp (Float64)         pressure gradient at the given cell edge
                - rough (Float64)      roughness (width / grain size) at the cell center

        Returns:
            Float64: Residual of the velocity equation.
    """

    function Velocity_Residual(v, args...)
        (w, mu, rho, dp, rough) = args

        # Reynolds number
        Re = 4/3 * rho * w * v / mu

        # friction factor using MDR approximation
        f = friction_factor_MDR(Re, rough)

        return v - w * dp / (v * rho * f)
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function can be used to find bracket for a root finding algorithm.

        Arguments:
            func (Function): The function giving the residual for which zero is to be found.
            guess (Float64): Starting guess.
            args (Tuple): Arguments passed to the function.

        Returns:
            - a (Float64): The lower bracket.
            - b (Float64): The higher bracket.
    """

    function findBracket(func, guess, args...)
        a = eps() * guess
        b = max(1000 * guess, 1.0)
        Res_a = func(a, args...)
        Res_b = func(b, args...)

        cnt = 0
        while Res_a * Res_b > 0
            b = 10 * b
            Res_b = func(b, args...)
            cnt += 1
            if cnt >= 60
                throw(SystemExit("Velocity bracket cannot be found"))
            end
        end

        return a, b
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
        equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated for Herschel-Bulkley fluid rheology.

        Arguments:
            w (Vector{Float64}): The width of the trial fracture.
            pf (Vector{Float64}): The fluid pressure.
            EltCrack (Vector{Int}): The list of elements inside the fracture.
            fluidProp (object): FluidProperties class object giving the fluid properties.
            Mesh (CartesianMesh): The mesh.
            InCrack (Vector{Int}): An array specifying whether elements are inside the fracture or not with 1 or 0 respectively.
            neiInCrack (Matrix{Int}): An ndarray giving indices of the neighbours of all the cells in the crack, in the EltCrack list.
            edgeInCrk_lst (Vector{Vector{Int}}): This list provides the indices of those cells in the EltCrack list whose neighbors are not outside the crack.
            simProp (object): An object of the SimulationProperties class.

        Returns:
            FinDiffOprtr (Union{SparseMatrixCSC{Float64, Int}, Matrix{Float64}}): The finite difference matrix.
            eff_mu (Union{Matrix{Float64}, Nothing}): Effective viscosity matrix.
            cond (Union{Matrix{Float64}, Nothing}): Conductivity matrix.
    """

    function finiteDiff_operator_power_law(w, pf, EltCrack, fluidProp, Mesh, InCrack, neiInCrack, edgeInCrk_lst, simProp)
        
        if simProp.solveSparse
            FinDiffOprtr = spzeros(Float64, length(w), length(w))
        else
            FinDiffOprtr = zeros(Float64, length(w), length(w))
        end

        dx = Mesh.hx
        dy = Mesh.hy

        # width on edges; evaluated by averaging the widths of adjacent cells
        wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2
        wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2
        wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2
        wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 4]]) / 2

        # pressure gradient data structure. The rows store pressure gradient in the following order.
        # 1 - left edge in x-direction    # 2 - right edge in x-direction
        # 3 - bottom edge in y-direction  # 4 - top edge in y-direction
        # 5 - left edge in y-direction    # 6 - right edge in y-direction
        # 7 - bottom edge in x-direction  # 8 - top edge in x-direction

        dp = zeros(Float64, 8, Mesh.NumberOfElts)
        dp[1, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 1]]) / dx
        dp[2, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 2]] - pf[EltCrack]) / dx
        dp[3, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 3]]) / dy
        dp[4, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 4]] - pf[EltCrack]) / dy
        # linear interpolation for pressure gradient on the edges where central difference not available
        dp[5, EltCrack] = (dp[3, Mesh.NeiElements[EltCrack, 1]] + dp[4, Mesh.NeiElements[EltCrack, 1]] + dp[3, EltCrack] +
                        dp[4, EltCrack]) / 4
        dp[6, EltCrack] = (dp[3, Mesh.NeiElements[EltCrack, 2]] + dp[4, Mesh.NeiElements[EltCrack, 2]] + dp[3, EltCrack] +
                        dp[4, EltCrack]) / 4
        dp[7, EltCrack] = (dp[1, Mesh.NeiElements[EltCrack, 3]] + dp[2, Mesh.NeiElements[EltCrack, 3]] + dp[1, EltCrack] +
                        dp[2, EltCrack]) / 4
        dp[8, EltCrack] = (dp[1, Mesh.NeiElements[EltCrack, 4]] + dp[2, Mesh.NeiElements[EltCrack, 4]] + dp[1, EltCrack] +
                        dp[2, EltCrack]) / 4

        # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
        dpLft = sqrt.(dp[1, EltCrack].^2 + dp[5, EltCrack].^2)
        dpRgt = sqrt.(dp[2, EltCrack].^2 + dp[6, EltCrack].^2)
        dpBtm = sqrt.(dp[3, EltCrack].^2 + dp[7, EltCrack].^2)
        dpTop = sqrt.(dp[4, EltCrack].^2 + dp[8, EltCrack].^2)

        cond = zeros(Float64, 4, length(EltCrack))
        cond[1, edgeInCrk_lst[1]] = (wLftEdge[edgeInCrk_lst[1]].^(2 * fluidProp.n + 1) .* dpLft[edgeInCrk_lst[1]] / 
                            fluidProp.Mprime).^(1 / fluidProp.n) ./ dpLft[edgeInCrk_lst[1]]
        cond[2, edgeInCrk_lst[2]] = (wRgtEdge[edgeInCrk_lst[2]].^(2 * fluidProp.n + 1) .* dpRgt[edgeInCrk_lst[2]] / 
                            fluidProp.Mprime).^(1 / fluidProp.n) ./ dpRgt[edgeInCrk_lst[2]]
        cond[3, edgeInCrk_lst[3]] = (wBtmEdge[edgeInCrk_lst[3]].^(2 * fluidProp.n + 1) .* dpBtm[edgeInCrk_lst[3]] / 
                            fluidProp.Mprime).^(1 / fluidProp.n) ./ dpBtm[edgeInCrk_lst[3]]
        cond[4, edgeInCrk_lst[4]] = (wTopEdge[edgeInCrk_lst[4]].^(2 * fluidProp.n + 1) .* dpTop[edgeInCrk_lst[4]] / 
                            fluidProp.Mprime).^(1 / fluidProp.n) ./ dpTop[edgeInCrk_lst[4]]

        indx_elts = 1:length(EltCrack)
        FinDiffOprtr[indx_elts, indx_elts] = -(cond[1, :] + cond[2, :]) / dx^2 - (cond[3, :] + cond[4, :]) / dy^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = cond[1, :] / dx^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = cond[2, :] / dx^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = cond[3, :] / dy^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 4]] = cond[4, :] / dy^2

        eff_mu = nothing
        if simProp.saveEffVisc
            eff_mu = zeros(Float64, 4, Mesh.NumberOfElts)
            eff_mu[1, EltCrack] = wLftEdge.^3 / (12 * cond[1, :])
            eff_mu[2, EltCrack] = wRgtEdge.^3 / (12 * cond[2, :])
            eff_mu[3, EltCrack] = wBtmEdge.^3 / (12 * cond[3, :])
            eff_mu[4, EltCrack] = wTopEdge.^3 / (12 * cond[4, :])
        end

        if !simProp.gravity
            cond = nothing
        end

        return FinDiffOprtr, eff_mu, cond
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        The function evaluate the finite difference 5 point stencil matrix, i.e. the A matrix in the ElastoHydrodynamic
        equations in e.g. Dontsov and Peirce 2008. The matrix is evaluated for Herschel-Bulkley fluid rheology.

        Arguments:
            w (Vector{Float64}): The width of the trial fracture.
            pf (Vector{Float64}): The fluid pressure.
            EltCrack (Vector{Int}): The list of elements inside the fracture.
            fluidProp (object): FluidProperties class object giving the fluid properties.
            Mesh (CartesianMesh): The mesh.
            InCrack (Vector{Int}): An array specifying whether elements are inside the fracture or not with 1 or 0 respectively.
            neiInCrack (Matrix{Int}): An ndarray giving indices of the neighbours of all the cells in the crack, in the EltCrack list.
            edgeInCrk_lst (Vector{Vector{Int}}): This list provides the indices of those cells in the EltCrack list whose neighbors are not outside the crack.
            simProp (object): An object of the SimulationProperties class.

        Returns:
            FinDiffOprtr (Union{SparseMatrixCSC{Float64, Int}, Matrix{Float64}}): The finite difference matrix.
            eff_mu (Union{Matrix{Float64}, Nothing}): Effective viscosity matrix.
            G (Union{Matrix{Float64}, Nothing}): G matrix.
            cond (Union{Matrix{Float64}, Nothing}): Conductivity matrix.
    """

    function finiteDiff_operator_Herschel_Bulkley(w, pf, EltCrack, fluidProp, Mesh, InCrack, neiInCrack, edgeInCrk_lst, simProp)
        
        if simProp.solveSparse
            FinDiffOprtr = spzeros(Float64, length(w), length(w))
        else
            FinDiffOprtr = zeros(Float64, length(w), length(w))
        end

        dx = Mesh.hx
        dy = Mesh.hy

        # width on edges; evaluated by averaging the widths of adjacent cells
        wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2
        wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2
        wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2
        wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 4]]) / 2

        # pressure gradient data structure. The rows store pressure gradient in the following order.
        # 1 - left edge in x-direction    # 2 - right edge in x-direction
        # 3 - bottom edge in y-direction  # 4 - top edge in y-direction
        # 5 - left edge in y-direction    # 6 - right edge in y-direction
        # 7 - bottom edge in x-direction  # 8 - top edge in x-direction

        dp = zeros(Float64, 8, Mesh.NumberOfElts)
        dp[1, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 1]]) / dx
        dp[2, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 2]] - pf[EltCrack]) / dx
        dp[3, EltCrack] = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 3]]) / dy
        dp[4, EltCrack] = (pf[Mesh.NeiElements[EltCrack, 4]] - pf[EltCrack]) / dy
        # linear interpolation for pressure gradient on the edges where central difference not available
        dp[5, EltCrack] = (dp[3, Mesh.NeiElements[EltCrack, 1]] + dp[4, Mesh.NeiElements[EltCrack, 1]] + dp[3, EltCrack] +
                        dp[4, EltCrack]) / 4
        dp[6, EltCrack] = (dp[3, Mesh.NeiElements[EltCrack, 2]] + dp[4, Mesh.NeiElements[EltCrack, 2]] + dp[3, EltCrack] +
                        dp[4, EltCrack]) / 4
        dp[7, EltCrack] = (dp[1, Mesh.NeiElements[EltCrack, 3]] + dp[2, Mesh.NeiElements[EltCrack, 3]] + dp[1, EltCrack] +
                        dp[2, EltCrack]) / 4
        dp[8, EltCrack] = (dp[1, Mesh.NeiElements[EltCrack, 4]] + dp[2, Mesh.NeiElements[EltCrack, 4]] + dp[1, EltCrack] +
                        dp[2, EltCrack]) / 4

        # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
        dpLft = (dp[1, EltCrack].^2 + dp[5, EltCrack].^2).^0.5
        dpRgt = (dp[2, EltCrack].^2 + dp[6, EltCrack].^2).^0.5
        dpBtm = (dp[3, EltCrack].^2 + dp[7, EltCrack].^2).^0.5
        dpTop = (dp[4, EltCrack].^2 + dp[8, EltCrack].^2).^0.5

        cond = zeros(Float64, 4, length(EltCrack))

        eps_val = simProp.HershBulkEpsilon
        G_min = simProp.HershBulkGmin

        phi = fluidProp.T0 / wLftEdge / dpLft
        phi_p = min.(phi, 0.5)
        G0 = max.((1.0 - 2.0 * phi_p).^fluidProp.var4 .* (1.0 + 2.0 * phi_p * fluidProp.var5),
                        eps_val * exp.(0.5 .- phi) + G_min)
        cond[1, edgeInCrk_lst[1]] = fluidProp.var1 * dpLft[edgeInCrk_lst[1]].^fluidProp.var2 * wLftEdge[edgeInCrk_lst[1]].^fluidProp.var3 * G0[edgeInCrk_lst[1]]

        phi = fluidProp.T0 / wRgtEdge / dpRgt
        phi_p = min.(phi, 0.5)
        G1 = max.((1.0 - 2.0 * phi_p).^fluidProp.var4 .* (1.0 + 2.0 * phi_p * fluidProp.var5),
                        eps_val * exp.(0.5 .- phi) + G_min)                            
        cond[2, edgeInCrk_lst[2]] = fluidProp.var1 * dpRgt[edgeInCrk_lst[2]].^fluidProp.var2 * wRgtEdge[edgeInCrk_lst[2]].^fluidProp.var3 * G1[edgeInCrk_lst[2]]
        
        phi = fluidProp.T0 / wBtmEdge / dpBtm
        phi_p = min.(phi, 0.5)
        G2 = max.((1.0 - 2.0 * phi_p).^fluidProp.var4 .* (1.0 + 2.0 * phi_p * fluidProp.var5),
                        eps_val * exp.(0.5 .- phi) + G_min)
        cond[3, edgeInCrk_lst[3]] = fluidProp.var1 * dpBtm[edgeInCrk_lst[3]].^fluidProp.var2 * wBtmEdge[edgeInCrk_lst[3]].^fluidProp.var3 * G2[edgeInCrk_lst[3]]

        phi = fluidProp.T0 / wTopEdge / dpTop
        phi_p = min.(phi, 0.5)
        G3 = max.((1.0 - 2.0 * phi_p).^fluidProp.var4 .* (1.0 + 2.0 * phi_p * fluidProp.var5),
                        eps_val * exp.(0.5 .- phi) + G_min)                             
        cond[4, edgeInCrk_lst[4]] = fluidProp.var1 * dpTop[edgeInCrk_lst[4]].^fluidProp.var2 * wTopEdge[edgeInCrk_lst[4]].^fluidProp.var3 * G3[edgeInCrk_lst[4]]

        indx_elts = 1:length(EltCrack)
        FinDiffOprtr[indx_elts, indx_elts] = -(cond[1, :] + cond[2, :]) / dx^2 - (cond[3, :] + cond[4, :]) / dy^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 1]] = cond[1, :] / dx^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 2]] = cond[2, :] / dx^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 3]] = cond[3, :] / dy^2
        FinDiffOprtr[indx_elts, neiInCrack[indx_elts, 4]] = cond[4, :] / dy^2

        eff_mu = nothing
        if simProp.saveEffVisc
            # Note: In Julia, we don't have np.errstate, but we can handle division by zero
            eff_mu = zeros(Float64, 4, Mesh.NumberOfElts)
            eff_mu[1, EltCrack[edgeInCrk_lst[1]]] = wLftEdge[edgeInCrk_lst[1]].^3 / (12 * cond[1, edgeInCrk_lst[1]])
            eff_mu[2, EltCrack[edgeInCrk_lst[2]]] = wRgtEdge[edgeInCrk_lst[2]].^3 / (12 * cond[2, edgeInCrk_lst[2]])
            eff_mu[3, EltCrack[edgeInCrk_lst[3]]] = wBtmEdge[edgeInCrk_lst[3]].^3 / (12 * cond[3, edgeInCrk_lst[3]])
            eff_mu[4, EltCrack[edgeInCrk_lst[4]]] = wTopEdge[edgeInCrk_lst[4]].^3 / (12 * cond[4, edgeInCrk_lst[4]])
        end

        G = nothing
        if simProp.saveG
            G = zeros(Float64, 4, Mesh.NumberOfElts)
            G[1, EltCrack] = G0
            G[2, EltCrack] = G1
            G[3, EltCrack] = G2
            G[4, EltCrack] = G3
        end

        if !simProp.gravity
            cond = nothing
        end

        return FinDiffOprtr, eff_mu, G, cond
    end
    #----------------------------------------------------------------------------------------------------------------------------------------

    """
        Get finite difference matrix for the fracture simulation.

        Arguments:
            wNplusOne (Vector{Float64}): Width at next time step.
            sol (Vector{Float64}): Solution vector.
            frac_n (Fracture): Fracture object at current time step.
            EltCrack (Vector{Int}): Elements in crack.
            neiInCrack (Matrix{Int}): Neighbor indices in crack.
            fluid_prop (object): Fluid properties.
            mat_prop (object): Material properties.
            sim_prop (object): Simulation properties.
            mesh (CartesianMesh): Mesh object.
            InCrack (Vector{Int}): In crack indicator.
            C (Matrix{Float64}): Elasticity matrix.
            interItr (Vector): Intermediate iteration data.
            to_solve (Vector{Int}): Elements to solve.
            to_impose (Vector{Int}): Elements to impose.
            active (Vector{Int}): Active elements.
            interItr_kp1 (Vector): Next iteration data.
            list_edgeInCrack (Vector{Vector{Int}}): Edge in crack list.

        Returns:
            FinDiffOprtr (Union{SparseMatrixCSC{Float64, Int}, Matrix{Float64}}): Finite difference operator.
            conductivity (Union{Matrix{Float64}, Nothing}): Conductivity matrix.
    """

    function get_finite_difference_matrix(wNplusOne, sol, frac_n, EltCrack, neiInCrack, fluid_prop, mat_prop, sim_prop, mesh,
                                    InCrack, C, interItr, to_solve, to_impose, active, interItr_kp1, list_edgeInCrack)

        if fluid_prop.rheology == "Newtonian" && !fluid_prop.turbulence
            FinDiffOprtr = finiteDiff_operator_laminar(wNplusOne,
                                                        EltCrack,
                                                        fluid_prop.muPrime,
                                                        mesh,
                                                        InCrack,
                                                        neiInCrack,
                                                        sim_prop)
            conductivity = nothing

        else
            pf = zeros(Float64, mesh.NumberOfElts)
            # pressure evaluated by dot product of width and elasticity matrix
            pf[to_solve] = C[to_solve, EltCrack] * wNplusOne[EltCrack] + mat_prop.SigmaO[to_solve]
            if sim_prop.solveDeltaP
                pf[active] = frac_n.pFluid[active] + sol[length(to_solve)+1:length(to_solve) + length(active)]
                pf[to_impose] = frac_n.pFluid[to_impose] + sol[length(to_solve) + length(active)+1:length(to_solve) + length(active) + length(to_impose)]
            else
                pf[active] = sol[length(to_solve)+1:length(to_solve) + length(active)]
                pf[to_impose] = sol[length(to_solve) + length(active)+1:length(to_solve) + length(active) + length(to_impose)]
            end

            if fluid_prop.turbulence
                FinDiffOprtr, interItr_kp1[1], conductivity = FiniteDiff_operator_turbulent_implicit(wNplusOne,
                                                            pf,
                                                            EltCrack,
                                                            fluid_prop,
                                                            mat_prop,
                                                            sim_prop,
                                                            mesh,
                                                            InCrack,
                                                            interItr[1],
                                                            to_solve,
                                                            active,
                                                            to_impose)
            elseif fluid_prop.rheology in ["Herschel-Bulkley", "HBF"]
                FinDiffOprtr, interItr_kp1[3], interItr_kp1[4], conductivity = finiteDiff_operator_Herschel_Bulkley(wNplusOne,
                                                            pf,
                                                            EltCrack,
                                                            fluid_prop,
                                                            mesh,
                                                            InCrack,
                                                            neiInCrack,
                                                            list_edgeInCrack,
                                                            sim_prop)

            elseif fluid_prop.rheology in ["power law", "PLF"]
                FinDiffOprtr, interItr_kp1[3], conductivity = finiteDiff_operator_power_law(wNplusOne,
                                                            pf,
                                                            EltCrack,
                                                            fluid_prop,
                                                            mesh,
                                                            InCrack,
                                                            neiInCrack,
                                                            list_edgeInCrack,
                                                            sim_prop)
            end
        end

        return FinDiffOprtr, conductivity
    end
    #--------------------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linearized system of equations to be solved by a linear system solver. The finite difference
        difference operator is saved as a sparse matrix. The system is assembled with the extended footprint (treating the
        channel and the extended tip elements distinctly; see description of the ILSA algorithm). The pressure in the tip
        cells and the cells where width constraint is active are solved separately. The pressure in the channel cells to be
        solved for change in width is substituted with width using the elasticity relation (see Zia and Lecamption 2019).

        Arguments:
            solk (Vector{Float64}): The trial change in width and pressure for the current iteration of fracture front.
            interItr (Vector): The information from the last iteration.
            args (Tuple): Arguments passed to the function.

        Returns:
            - A (Matrix{Float64}): The A matrix (in the system Ax=b) to be solved by a linear system solver.
            - S (Vector{Float64}): The b vector (in the system Ax=b) to be solved by a linear system solver.
            - interItr_kp1 (Vector): The information transferred between iterations.
            - indices (Vector{Vector{Int}}): The list containing 3 arrays giving indices of the cells.
    """

    function MakeEquationSystem_ViscousFluid_pressure_substituted_sparse(solk, interItr, args...)
        
        (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
        sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

        wNplusOne = copy(frac.w)
        wNplusOne[to_solve] .+= solk[1:length(to_solve)]
        wNplusOne[to_impose] = imposed_val
        if length(wc_to_impose) > 0
            wNplusOne[active] = wc_to_impose
        end

        below_wc = findall(wNplusOne[to_solve] .< mat_prop.wc)
        below_wc_km1 = interItr[2]
        below_wc = vcat(below_wc_km1, setdiff(below_wc, below_wc_km1))
        wNplusOne[to_solve[below_wc]] .= mat_prop.wc

        wcNplusHalf = (frac.w + wNplusOne) / 2

        interItr_kp1 = Vector{Any}(nothing, 4)
        FinDiffOprtr, conductivity = get_finite_difference_matrix(wNplusOne, solk, frac,
                                    EltCrack, neiInCrack, fluid_prop,
                                    mat_prop, sim_prop, frac.mesh,
                                    InCrack, C, interItr, to_solve,
                                    to_impose, active, interItr_kp1,
                                    lst_edgeInCrk)

        G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                        frac.mesh, InCrack, sim_prop,
                        conductivity)

        n_ch = length(to_solve)
        n_act = length(active)
        n_tip = length(imposed_val)
        n_total = n_ch + n_act + n_tip

        ch_indxs = 1:n_ch
        act_indxs = n_ch .+ (1:n_act)
        tip_indxs = n_ch + n_act .+ (1:n_tip)

        A = zeros(Float64, n_total, n_total)

        ch_AplusCf = dt * FinDiffOprtr[ch_indxs, ch_indxs] - spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[to_solve], n_ch))

        A[ch_indxs, ch_indxs] = - ch_AplusCf * C[to_solve, to_solve]
        A[ch_indxs, ch_indxs] += ones(Float64, length(ch_indxs))
        A[ch_indxs, tip_indxs] = -dt * FinDiffOprtr[ch_indxs, tip_indxs]
        A[ch_indxs, act_indxs] = -dt * FinDiffOprtr[ch_indxs, act_indxs]

        A[tip_indxs, ch_indxs] = - (dt * FinDiffOprtr[tip_indxs, ch_indxs] * C[to_solve, to_solve])
        A[tip_indxs, tip_indxs] = - dt * FinDiffOprtr[tip_indxs, tip_indxs] + spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[to_impose], n_tip))
        A[tip_indxs, act_indxs] = -dt * FinDiffOprtr[tip_indxs, act_indxs]

        A[act_indxs, ch_indxs] = - (dt * FinDiffOprtr[act_indxs, ch_indxs] * C[to_solve, to_solve])
        A[act_indxs, tip_indxs] = -dt * FinDiffOprtr[act_indxs, tip_indxs]
        A[act_indxs, act_indxs] = - dt * FinDiffOprtr[act_indxs, act_indxs] + spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[active], n_act))

        S = zeros(Float64, n_total)
        pf_ch_prime = C[to_solve, to_solve] * frac.w[to_solve] + 
                    C[to_solve, to_impose] * imposed_val + 
                    C[to_solve, active] * wNplusOne[active] + 
                    mat_prop.SigmaO[to_solve]

        S[ch_indxs] = ch_AplusCf * pf_ch_prime + 
                    dt * G[to_solve] + 
                    dt * Q[to_solve] / frac.mesh.EltArea - 
                    LeakOff[to_solve] / frac.mesh.EltArea + 
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]
        S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + 
                    dt * (FinDiffOprtr[tip_indxs, ch_indxs] * pf_ch_prime) + 
                    fluid_prop.compressibility * wcNplusHalf[to_impose] * frac.pFluid[to_impose] + 
                    dt * G[to_impose] + 
                    dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea
        S[act_indxs] = -(wc_to_impose - frac.w[active]) + 
                    dt * (FinDiffOprtr[act_indxs, ch_indxs] * pf_ch_prime) + 
                    fluid_prop.compressibility * wcNplusHalf[active] * frac.pFluid[active] + 
                    dt * G[active] + 
                    dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

        # indices of solved width, pressure and active width constraint in the solution
        indices = [ch_indxs, tip_indxs, act_indxs]

        interItr_kp1[2] = below_wc

        return A, S, interItr_kp1, indices
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linearized system of equations to be solved by a linear system solver. The system is
        assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
        description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
        active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
        with width using the elasticity relation (see Zia and Lecamption 2019). The finite difference difference operator
        is saved as a sparse matrix.

        Arguments:
            solk (Vector{Float64}): The trial change in width and pressure for the current iteration of fracture front.
            interItr (Vector): The information from the last iteration.
            args (Tuple): Arguments passed to the function.

        Returns:
            - A (Matrix{Float64}): The A matrix (in the system Ax=b) to be solved by a linear system solver.
            - S (Vector{Float64}): The b vector (in the system Ax=b) to be solved by a linear system solver.
            - interItr_kp1 (Vector): The information transferred between iterations.
            - indices (Vector{Vector{Int}}): The list containing 3 arrays giving indices of the cells.
    """

    function MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse(solk, interItr, args...)
        
        (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
        sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

        wNplusOne = copy(frac.w)
        wNplusOne[to_solve] .+= solk[1:length(to_solve)]
        wNplusOne[to_impose] = imposed_val
        if length(wc_to_impose) > 0
            wNplusOne[active] = wc_to_impose
        end

        below_wc = findall(wNplusOne[to_solve] .< mat_prop.wc)
        below_wc_km1 = interItr[2]
        below_wc = vcat(below_wc_km1, setdiff(below_wc, below_wc_km1))
        wNplusOne[to_solve[below_wc]] .= mat_prop.wc

        wcNplusHalf = (frac.w + wNplusOne) / 2

        interItr_kp1 = Vector{Any}(nothing, 4)
        FinDiffOprtr, conductivity = get_finite_difference_matrix(wNplusOne, solk, frac,
                                    EltCrack, neiInCrack, fluid_prop,
                                    mat_prop, sim_prop, frac.mesh,
                                    InCrack, C, interItr, to_solve,
                                    to_impose, active, interItr_kp1,
                                    lst_edgeInCrk)

        G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                        frac.mesh, InCrack, sim_prop,
                        conductivity)

        n_ch = length(to_solve)
        n_act = length(active)
        n_tip = length(imposed_val)
        n_total = n_ch + n_act + n_tip

        ch_indxs = 1:n_ch
        act_indxs = n_ch .+ (1:n_act)
        tip_indxs = n_ch + n_act .+ (1:n_tip)

        A = zeros(Float64, n_total, n_total)

        ch_AplusCf = dt * FinDiffOprtr[ch_indxs, ch_indxs] - spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[to_solve], n_ch))

        A[ch_indxs, ch_indxs] = - ch_AplusCf * C[to_solve, to_solve]
        A[ch_indxs, ch_indxs] += ones(Float64, length(ch_indxs))

        A[ch_indxs, tip_indxs] = -dt * FinDiffOprtr[ch_indxs, tip_indxs]
        A[ch_indxs, act_indxs] = -dt * FinDiffOprtr[ch_indxs, act_indxs]

        A[tip_indxs, ch_indxs] = - (dt * FinDiffOprtr[tip_indxs, ch_indxs] * C[to_solve, to_solve])
        A[tip_indxs, tip_indxs] = - dt * FinDiffOprtr[tip_indxs, tip_indxs] + spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[to_impose], n_tip))
        A[tip_indxs, act_indxs] = -dt * FinDiffOprtr[tip_indxs, act_indxs]

        A[act_indxs, ch_indxs] = - (dt * FinDiffOprtr[act_indxs, ch_indxs] * C[to_solve, to_solve])
        A[act_indxs, tip_indxs] = -dt * FinDiffOprtr[act_indxs, tip_indxs]
        A[act_indxs, act_indxs] = - dt * FinDiffOprtr[act_indxs, act_indxs] + spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[active], n_act))

        S = zeros(Float64, n_total)
        pf_ch_prime = C[to_solve, to_solve] * frac.w[to_solve] + 
                    C[to_solve, to_impose] * imposed_val + 
                    C[to_solve, active] * wNplusOne[active] + 
                    mat_prop.SigmaO[to_solve]

        S[ch_indxs] = ch_AplusCf * pf_ch_prime + 
                    dt * (FinDiffOprtr[ch_indxs, tip_indxs] * frac.pFluid[to_impose]) + 
                    dt * (FinDiffOprtr[ch_indxs, act_indxs] * frac.pFluid[active]) + 
                    dt * G[to_solve] + 
                    dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea + 
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]

        S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + 
                    dt * (FinDiffOprtr[tip_indxs, ch_indxs] * pf_ch_prime) + 
                    dt * (FinDiffOprtr[tip_indxs, tip_indxs] * frac.pFluid[to_impose]) + 
                    dt * (FinDiffOprtr[tip_indxs, act_indxs] * frac.pFluid[active]) + 
                    dt * G[to_impose] + 
                    dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea

        S[act_indxs] = -(wc_to_impose - frac.w[active]) + 
                    dt * (FinDiffOprtr[act_indxs, ch_indxs] * pf_ch_prime) + 
                    dt * (FinDiffOprtr[act_indxs, tip_indxs] * frac.pFluid[to_impose]) + 
                    dt * (FinDiffOprtr[act_indxs, act_indxs] * frac.pFluid[active]) + 
                    dt * G[active] + 
                    dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

        # indices of solved width, pressure and active width constraint in the solution
        indices = [ch_indxs, tip_indxs, act_indxs]

        interItr_kp1[2] = below_wc

        return A, S, interItr_kp1, indices
    end
    # -----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linearized system of equations to be solved by a linear system solver. The system is
        assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
        description of the ILSA algorithm). The pressure in the tip cells and the cells where width constraint is active
        are solved separately. The pressure in the channel cells to be solved for change in width is substituted with width
        using the elasticity relation (see Zia and Lecampion 2019).

        Arguments:
            solk (Vector{Float64}): The trial change in width and pressure for the current iteration of fracture front.
            interItr (Vector): The information from the last iteration.
            args (Tuple): Arguments passed to the function.

        Returns:
            - A (Matrix{Float64}): The A matrix (in the system Ax=b) to be solved by a linear system solver.
            - S (Vector{Float64}): The b vector (in the system Ax=b) to be solved by a linear system solver.
            - interItr_kp1 (Vector): The information transferred between iterations.
            - indices (Vector{Vector{Int}}): The list containing 3 arrays giving indices of the cells.
    """

    function MakeEquationSystem_ViscousFluid_pressure_substituted(solk, interItr, args...)
        
        (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
        sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

        wNplusOne = copy(frac.w)
        wNplusOne[to_solve] .+= solk[1:length(to_solve)]
        wNplusOne[to_impose] = imposed_val
        if length(wc_to_impose) > 0
            wNplusOne[active] = wc_to_impose
        end

        below_wc = findall(wNplusOne[to_solve] .< mat_prop.wc)
        below_wc_km1 = interItr[2]
        below_wc = vcat(below_wc_km1, setdiff(below_wc, below_wc_km1))
        wNplusOne[to_solve[below_wc]] .= mat_prop.wc

        wcNplusHalf = (frac.w + wNplusOne) / 2

        interItr_kp1 = Vector{Any}(nothing, 4)
        FinDiffOprtr, conductivity = get_finite_difference_matrix(wNplusOne, solk, frac,
                                    EltCrack, neiInCrack, fluid_prop,
                                    mat_prop, sim_prop, frac.mesh,
                                    InCrack, C, interItr, to_solve,
                                    to_impose, active, interItr_kp1,
                                    lst_edgeInCrk)

        G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                        frac.mesh, InCrack, sim_prop,
                        conductivity)

        n_ch = length(to_solve)
        n_act = length(active)
        n_tip = length(imposed_val)
        n_total = n_ch + n_act + n_tip

        ch_indxs = 1:n_ch
        act_indxs = n_ch .+ (1:n_act)
        tip_indxs = n_ch + n_act .+ (1:n_tip)

        A = zeros(Float64, n_total, n_total)

        ch_AplusCf = dt * FinDiffOprtr[ch_indxs, ch_indxs]
        ch_AplusCf[ch_indxs, ch_indxs] -= fluid_prop.compressibility * wcNplusHalf[to_solve]

        A[ch_indxs, ch_indxs] = - ch_AplusCf * C[to_solve, to_solve]
        A[ch_indxs, ch_indxs] += ones(Float64, length(ch_indxs))

        A[ch_indxs, tip_indxs] = -dt * FinDiffOprtr[ch_indxs, tip_indxs]
        A[ch_indxs, act_indxs] = -dt * FinDiffOprtr[ch_indxs, act_indxs]

        A[tip_indxs, ch_indxs] = - dt * FinDiffOprtr[tip_indxs, ch_indxs] * C[to_solve, to_solve]
        A[tip_indxs, tip_indxs] = - dt * FinDiffOprtr[tip_indxs, tip_indxs]
        A[tip_indxs, tip_indxs] += fluid_prop.compressibility * wcNplusHalf[to_impose]

        A[tip_indxs, act_indxs] = -dt * FinDiffOprtr[tip_indxs, act_indxs]

        A[act_indxs, ch_indxs] = - dt * FinDiffOprtr[act_indxs, ch_indxs] * C[to_solve, to_solve]
        A[act_indxs, tip_indxs] = -dt * FinDiffOprtr[act_indxs, tip_indxs]
        A[act_indxs, act_indxs] = - dt * FinDiffOprtr[act_indxs, act_indxs]
        A[act_indxs, act_indxs] += fluid_prop.compressibility * wcNplusHalf[active]

        S = zeros(Float64, n_total)
        pf_ch_prime = C[to_solve, to_solve] * frac.w[to_solve] + 
                    C[to_solve, to_impose] * imposed_val + 
                    C[to_solve, active] * wNplusOne[active] + 
                    mat_prop.SigmaO[to_solve]

        S[ch_indxs] = ch_AplusCf * pf_ch_prime + 
                    dt * G[to_solve] + 
                    dt * Q[to_solve] / frac.mesh.EltArea - 
                    LeakOff[to_solve] / frac.mesh.EltArea + 
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]
        S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + 
                    dt * (FinDiffOprtr[tip_indxs, ch_indxs] * pf_ch_prime) + 
                    fluid_prop.compressibility * wcNplusHalf[to_impose] * frac.pFluid[to_impose] + 
                    dt * G[to_impose] + 
                    dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea
        S[act_indxs] = -(wc_to_impose - frac.w[active]) + 
                    dt * (FinDiffOprtr[act_indxs, ch_indxs] * pf_ch_prime) + 
                    fluid_prop.compressibility * wcNplusHalf[active] * frac.pFluid[active] + 
                    dt * G[active] + 
                    dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

        # indices of solved width, pressure and active width constraint in the solution
        indices = [ch_indxs, tip_indxs, act_indxs]

        interItr_kp1[2] = below_wc

        return A, S, interItr_kp1, indices
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linearized system of equations to be solved by a linear system solver. The system is
        assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
        description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
        active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
        with width using the elasticity relation (see Zia and Lecamption 2019).

        Arguments:
            solk (Vector{Float64}): The trial change in width and pressure for the current iteration of fracture front.
            interItr (Vector): The information from the last iteration.
            args (Tuple): Arguments passed to the function.

        Returns:
            - A (Matrix{Float64}): The A matrix (in the system Ax=b) to be solved by a linear system solver.
            - S (Vector{Float64}): The b vector (in the system Ax=b) to be solved by a linear system solver.
            - interItr_kp1 (Vector): The information transferred between iterations.
            - indices (Vector{Vector{Int}}): The list containing 3 arrays giving indices of the cells.
    """

    function MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP(solk, interItr, args...)
        
        (EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
        sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk) = args

        wNplusOne = copy(frac.w)
        wNplusOne[to_solve] .+= solk[1:length(to_solve)]
        wNplusOne[to_impose] = imposed_val
        if length(wc_to_impose) > 0
            wNplusOne[active] = wc_to_impose
        end

        below_wc = findall(wNplusOne[to_solve] .< mat_prop.wc)
        below_wc_km1 = interItr[2]
        below_wc = vcat(below_wc_km1, setdiff(below_wc, below_wc_km1))
        wNplusOne[to_solve[below_wc]] .= mat_prop.wc

        wcNplusHalf = (frac.w + wNplusOne) / 2

        interItr_kp1 = Vector{Any}(nothing, 4)
        FinDiffOprtr, conductivity = get_finite_difference_matrix(wNplusOne, solk, frac,
                                    EltCrack, neiInCrack, fluid_prop,
                                    mat_prop, sim_prop, frac.mesh,
                                    InCrack, C, interItr, to_solve,
                                    to_impose, active, interItr_kp1,
                                    lst_edgeInCrk)

        G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                        frac.mesh, InCrack, sim_prop,
                        conductivity)

        n_ch = length(to_solve)
        n_act = length(active)
        n_tip = length(imposed_val)
        n_total = n_ch + n_act + n_tip

        ch_indxs = 1:n_ch
        act_indxs = n_ch .+ (1:n_act)
        tip_indxs = n_ch + n_act .+ (1:n_tip)

        A = zeros(Float64, n_total, n_total)

        ch_AplusCf = dt * FinDiffOprtr[ch_indxs, ch_indxs]
        ch_AplusCf[ch_indxs, ch_indxs] -= fluid_prop.compressibility * wcNplusHalf[to_solve]

        A[ch_indxs, ch_indxs] = - ch_AplusCf * C[to_solve, to_solve]
        A[ch_indxs, ch_indxs] += ones(Float64, length(ch_indxs))

        A[ch_indxs, tip_indxs] = - dt * FinDiffOprtr[ch_indxs, tip_indxs]
        A[ch_indxs, act_indxs] = - dt * FinDiffOprtr[ch_indxs, act_indxs]

        A[tip_indxs, ch_indxs] = - dt * FinDiffOprtr[tip_indxs, ch_indxs] * C[to_solve, to_solve]
        A[tip_indxs, tip_indxs] = - dt * FinDiffOprtr[tip_indxs, tip_indxs]
        A[tip_indxs, tip_indxs] += fluid_prop.compressibility * wcNplusHalf[to_impose]
        A[tip_indxs, act_indxs] = - dt * FinDiffOprtr[tip_indxs, act_indxs]

        A[act_indxs, ch_indxs] = - dt * FinDiffOprtr[act_indxs, ch_indxs] * C[to_solve, to_solve]
        A[act_indxs, tip_indxs] = - dt * FinDiffOprtr[act_indxs, tip_indxs]
        A[act_indxs, act_indxs] = - dt * FinDiffOprtr[act_indxs, act_indxs]
        A[act_indxs, act_indxs] += fluid_prop.compressibility * wcNplusHalf[active]

        S = zeros(Float64, n_total)
        pf_ch_prime = C[to_solve, to_solve] * frac.w[to_solve] + 
                    C[to_solve, to_impose] * imposed_val + 
                    C[to_solve, active] * wNplusOne[active] + 
                    mat_prop.SigmaO[to_solve]

        S[ch_indxs] = ch_AplusCf * pf_ch_prime + 
                    dt * FinDiffOprtr[ch_indxs, tip_indxs] * frac.pFluid[to_impose] + 
                    dt * FinDiffOprtr[ch_indxs, act_indxs] * frac.pFluid[active] + 
                    dt * G[to_solve] + 
                    dt * Q[to_solve] / frac.mesh.EltArea - LeakOff[to_solve] / frac.mesh.EltArea + 
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]

        S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + 
                    dt * FinDiffOprtr[tip_indxs, ch_indxs] * pf_ch_prime + 
                    dt * FinDiffOprtr[tip_indxs, tip_indxs] * frac.pFluid[to_impose] + 
                    dt * FinDiffOprtr[tip_indxs, act_indxs] * frac.pFluid[active] + 
                    dt * G[to_impose] + 
                    dt * Q[to_impose] / frac.mesh.EltArea - LeakOff[to_impose] / frac.mesh.EltArea

        S[act_indxs] = -(wc_to_impose - frac.w[active]) + 
                    dt * FinDiffOprtr[act_indxs, ch_indxs] * pf_ch_prime + 
                    dt * FinDiffOprtr[act_indxs, tip_indxs] * frac.pFluid[to_impose] + 
                    dt * FinDiffOprtr[act_indxs, act_indxs] * frac.pFluid[active] + 
                    dt * G[active] + 
                    dt * Q[active] / frac.mesh.EltArea - LeakOff[active] / frac.mesh.EltArea

        # indices of solved width, pressure and active width constraint in the solution
        indices = [ch_indxs, tip_indxs, act_indxs]
        
        interItr_kp1[2] = below_wc
        return A, S, interItr_kp1, indices
    end
    # -----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linearized system of equations to be solved by a linear system solver. The system is
        assembled with the extended footprint (treating the channel and the extended tip elements distinctly; see
        description of the ILSA algorithm). The change is pressure in the tip cells and the cells where width constraint is
        active are solved separately. The pressure in the channel cells to be solved for change in width is substituted
        with width using the elasticity relation (see Zia and Lecamption 2019). The finite difference difference operator
        is saved as a sparse matrix.

        Arguments:
            solk (Vector{Float64}): The trial change in width and pressure for the current iteration of fracture front.
            interItr (Vector): The information from the last iteration.
            args (Tuple): Arguments passed to the function.

        Returns:
            - A (Matrix{Float64}): The A matrix (in the system Ax=b) to be solved by a linear system solver.
            - S (Vector{Float64}): The b vector (in the system Ax=b) to be solved by a linear system solver.
            - interItr_kp1 (Vector): The information transferred between iterations.
            - indices (Vector{Vector{Int}}): The list containing 3 arrays giving indices of the cells.
    """

    function MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse_injection_line(solk, interItr, args...)
        
        ((EltCrack, to_solve, to_impose, imposed_val, wc_to_impose, frac, fluid_prop, mat_prop,
        sim_prop, dt, Q, C, InCrack, LeakOff, active, neiInCrack, lst_edgeInCrk), inj_prop,
        inj_ch, inj_act, sink_cells, p_il_0, inj_in_ch, inj_in_act, Q0, sink) = args

        wNplusOne = copy(frac.w)
        wNplusOne[to_solve] .+= solk[1:length(to_solve)]
        wNplusOne[to_impose] = imposed_val
        if length(wc_to_impose) > 0
            wNplusOne[active] = wc_to_impose
        end

        below_wc = findall(wNplusOne[to_solve] .< mat_prop.wc)
        below_wc_km1 = interItr[2]
        below_wc = vcat(below_wc_km1, setdiff(below_wc, below_wc_km1))
        wNplusOne[to_solve[below_wc]] .= mat_prop.wc

        wcNplusHalf = (frac.w + wNplusOne) / 2

        interItr_kp1 = Vector{Any}(nothing, 4)
        FinDiffOprtr, conductivity = get_finite_difference_matrix(wNplusOne, solk, frac,
                                                    EltCrack, neiInCrack, fluid_prop,
                                                    mat_prop, sim_prop, frac.mesh,
                                                    InCrack, C, interItr, to_solve,
                                                    to_impose, active, interItr_kp1,
                                                    lst_edgeInCrk)

        G = Gravity_term(wNplusOne, EltCrack, fluid_prop,
                        frac.mesh, InCrack, sim_prop,
                        conductivity)

        n_ch = length(to_solve)
        n_act = length(active)
        n_tip = length(imposed_val)
        n_inj_ch = length(inj_ch)
        n_inj_act = length(inj_act)
        n_total = n_ch + n_act + n_tip + n_inj_ch + n_inj_act + 1

        ch_indxs = 1:n_ch
        act_indxs = n_ch .+ (1:n_act)
        tip_indxs = n_ch + n_act .+ (1:n_tip)
        p_il_indxs = n_ch + n_act + n_tip + 1
        inj_ch_indx = n_ch + n_act + n_tip + 2:n_ch + n_act + n_tip + 1 + n_inj_ch
        inj_act_indx = n_ch + n_act + n_tip + 1 + n_inj_ch + 1:n_ch + n_act + n_tip + 1 + n_inj_ch + n_inj_act

        A = zeros(Float64, n_total, n_total)

        ch_AplusCf = dt * FinDiffOprtr[ch_indxs, ch_indxs] - spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[to_solve], n_ch))

        A[ch_indxs, ch_indxs] = - ch_AplusCf * C[to_solve, to_solve]
        A[ch_indxs, ch_indxs] += ones(Float64, length(ch_indxs))

        A[ch_indxs, tip_indxs] = -dt * FinDiffOprtr[ch_indxs, tip_indxs]
        A[ch_indxs, act_indxs] = -dt * FinDiffOprtr[ch_indxs, act_indxs]
        A[ch_indxs[inj_in_ch], inj_ch_indx] -= dt / frac.mesh.EltArea * ones(Float64, length(inj_ch_indx))

        A[tip_indxs, ch_indxs] = - (dt * FinDiffOprtr[tip_indxs, ch_indxs] * C[to_solve, to_solve])
        A[tip_indxs, tip_indxs] = - dt * FinDiffOprtr[tip_indxs, tip_indxs] + spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[to_impose], n_tip))
        A[tip_indxs, act_indxs] = -dt * FinDiffOprtr[tip_indxs, act_indxs]

        A[act_indxs, ch_indxs] = - (dt * FinDiffOprtr[act_indxs, ch_indxs] * C[to_solve, to_solve])
        A[act_indxs, tip_indxs] = -dt * FinDiffOprtr[act_indxs, tip_indxs]
        A[act_indxs, act_indxs] = - dt * FinDiffOprtr[act_indxs, act_indxs] + spdiagm(0 => fill(fluid_prop.compressibility * wcNplusHalf[active], n_act))
        A[act_indxs[inj_in_act], inj_act_indx] -= dt / frac.mesh.EltArea * ones(Float64, length(inj_act_indx))

        A[p_il_indxs, p_il_indxs] = inj_prop.ILVolume * inj_prop.ILCompressibility
        A[p_il_indxs, inj_ch_indx] = dt * ones(Float64, length(inj_ch_indx))
        A[p_il_indxs, inj_act_indx] = dt * ones(Float64, length(inj_act_indx))

        A[inj_ch_indx, ch_indxs] = -C[inj_ch, to_solve]
        A[inj_ch_indx, p_il_indxs] .= 1.0
        A[inj_ch_indx, inj_ch_indx] = -inj_prop.perforationFriction * abs.(solk[inj_ch_indx])

        A[inj_act_indx, act_indxs[inj_in_act]] = -ones(Float64, length(inj_in_act))
        A[inj_act_indx, p_il_indxs] .= 1.0
        A[inj_act_indx, inj_act_indx] = -inj_prop.perforationFriction * abs.(solk[inj_act_indx])

        S = zeros(Float64, n_total)
        pf_ch_prime = C[to_solve, to_solve] * frac.w[to_solve] + 
                    C[to_solve, to_impose] * imposed_val + 
                    C[to_solve, active] * wNplusOne[active] + 
                    mat_prop.SigmaO[to_solve]

        S[ch_indxs] = ch_AplusCf * pf_ch_prime + 
                    dt * (FinDiffOprtr[ch_indxs, tip_indxs] * frac.pFluid[to_impose]) + 
                    dt * (FinDiffOprtr[ch_indxs, act_indxs] * frac.pFluid[active]) + 
                    dt * G[to_solve] + 
                    -(LeakOff[to_solve] + dt * sink[to_solve]) / frac.mesh.EltArea + 
                    fluid_prop.compressibility * wcNplusHalf[to_solve] * frac.pFluid[to_solve]

        S[tip_indxs] = -(imposed_val - frac.w[to_impose]) + 
                    dt * (FinDiffOprtr[tip_indxs, ch_indxs] * pf_ch_prime) + 
                    dt * (FinDiffOprtr[tip_indxs, tip_indxs] * frac.pFluid[to_impose]) + 
                    dt * (FinDiffOprtr[tip_indxs, act_indxs] * frac.pFluid[active]) + 
                    dt * G[to_impose] + 
                    -(LeakOff[to_impose] + 0*dt * sink[to_impose]) / frac.mesh.EltArea

        S[act_indxs] = -(wc_to_impose - frac.w[active]) + 
                    dt * (FinDiffOprtr[act_indxs, ch_indxs] * pf_ch_prime) + 
                    dt * (FinDiffOprtr[act_indxs, tip_indxs] * frac.pFluid[to_impose]) + 
                    dt * (FinDiffOprtr[act_indxs, act_indxs] * frac.pFluid[active]) + 
                    dt * G[active] + 
                    -(LeakOff[active] + dt * sink[active]) / frac.mesh.EltArea

        S[p_il_indxs] = dt * Q0

        S[inj_ch_indx] = C[inj_ch, to_solve] * frac.w[to_solve] + 
                        C[inj_ch, active] * wc_to_impose + 
                        C[inj_ch, to_impose] * imposed_val + 
                        mat_prop.SigmaO[inj_ch] - p_il_0

        S[inj_act_indx] = frac.pFluid[inj_act] - p_il_0

        # indices of different solutions is the solution array
        indices = [ch_indxs, tip_indxs, act_indxs, [p_il_indxs], inj_ch_indx, inj_act_indx]

        interItr_kp1[2] = below_wc

        return A, S, interItr_kp1, indices
    end# -----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
        with the extended footprint (treating the channel and the extended tip elements distinctly). The given width is
        imposed on the given loaded elements.
    """

    function MakeEquationSystem_mechLoading(wTip, EltChannel, EltTip, C, EltLoaded, w_loaded)
        
        Ccc = C[EltChannel, EltChannel]
        Cct = C[EltChannel, EltTip]

        A = hcat(Ccc, -ones(Float64, length(EltChannel)))
        A = vcat(A, zeros(Float64, 1, length(EltChannel) + 1))
        loaded_idx = findfirst(EltChannel .== EltLoaded)
        if loaded_idx !== nothing
            A[end, loaded_idx] = 1.0
        end

        S = - Cct * wTip
        S = vcat(S, w_loaded)

        return A, S
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
        with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
        fracture is imposed to be equal to the fluid injected into the fracture.
    """

    function MakeEquationSystem_volumeControl_double_fracture(w_lst_tmstp, wTipFR0, wTipFR1, EltChannel0, EltChannel1,
                                                        EltTip0, EltTip1, sigma_o, C, dt, QFR0, QFR1, ElemArea, lkOff)
        
        wTip = vcat(wTipFR0, wTipFR1)
        EltChannel = vcat(EltChannel0, EltChannel1)
        EltTip = vcat(EltTip0, EltTip1)
        Ccc = C[EltChannel, EltChannel] # elasticity Channel Channel
        Cct = C[EltChannel, EltTip]

        varray0 = zeros(Float64, length(EltChannel))
        varray0[1:length(EltChannel0)] .= 1.0
        varray1 = zeros(Float64, length(EltChannel))
        varray1[length(EltChannel0)+1:end] .= 1.0

        A = hcat(Ccc, -varray0, -varray1)

        harray0 = zeros(Float64, 1, length(EltChannel) + 2)
        harray0[1, 1:length(EltChannel0)] .= 1.0
        harray1 = zeros(Float64, 1, length(EltChannel) + 2)
        harray1[1, length(EltChannel0)+1:length(EltChannel)] .= 1.0

        A = vcat(A, harray0, harray1)

        S = - sigma_o[EltChannel] - Ccc * w_lst_tmstp[EltChannel] - Cct * wTip
        S = vcat(S, sum(QFR0) * dt / ElemArea - (sum(wTipFR0) - sum(w_lst_tmstp[EltTip0])) -
                    sum(lkOff[vcat(EltChannel0, EltTip0)]))
        S = vcat(S, sum(QFR1) * dt / ElemArea - (sum(wTipFR1) - sum(w_lst_tmstp[EltTip1])) -
                    sum(lkOff[vcat(EltChannel1, EltTip1)]))

        return A, S
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
        with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
        fracture is imposed to be equal to the fluid injected into the fracture.
    """

    function MakeEquationSystem_volumeControl(w_lst_tmstp, wTip, EltChannel, EltTip, sigma_o, C, dt, Q, ElemArea, lkOff)
        
        Ccc = C[EltChannel, EltChannel]
        Cct = C[EltChannel, EltTip]

        A = hcat(Ccc, -ones(Float64, length(EltChannel)))
        A = vcat(A, ones(Float64, 1, length(EltChannel) + 1))
        A[end, end] = 0.0

        S = - sigma_o[EltChannel] - Ccc * w_lst_tmstp[EltChannel] - Cct * wTip
        S = vcat(S, sum(Q) * dt / ElemArea - (sum(wTip) - sum(w_lst_tmstp[EltTip])) - sum(lkOff))

        return A, S
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function gives the residual of the solution for the system of equations formed using the given function.
    """

    function Elastohydrodynamic_ResidualFun(solk, system_func, interItr, args...)
        A, S, interItr, indices = system_func(solk, interItr, args...)
        return A * solk - S, interItr, indices
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function gives the residual of the solution for the system of equations formed using the given function.
    """
    function Elastohydrodynamic_ResidualFun_nd(solk, system_func, interItr, InterItr_o, indices_o, args...)
        A, S, interItr, indices = system_func(solk, interItr, args...)
        if length(indices[4]) == 0
            Fx = A * solk - S
        else
            solk_red = solk[setdiff(1:length(solk), length(indices[1]) .+ indices[4])]
            Fx_red = A * solk_red - S
            Fx = populate_full(indices, Fx_red)
        end
        InterItr_o = interItr
        indices_o = indices
        return Fx
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function makes the linear system of equations to be solved by a linear system solver. The system is assembled
        with the extended footprint (treating the channel and the extended tip elements distinctly). The the volume of the
        fracture is imposed to be equal to the fluid injected into the fracture (see Zia and Lecampion 2018).
    """
    function MakeEquationSystem_volumeControl_symmetric(w_lst_tmstp, wTip_sym, EltChannel_sym, EltTip_sym, C_s, dt, Q, sigma_o,
                                                            ElemArea, LkOff, vol_weights, sym_elements, dwTip)
        
        Ccc = C_s[EltChannel_sym, EltChannel_sym]
        Cct = C_s[EltChannel_sym, EltTip_sym]

        A = hcat(Ccc, -ones(Float64, length(EltChannel_sym)))
        weights = vol_weights[EltChannel_sym]
        weights = vcat(weights, [0.0])
        A = vcat(A, weights')

        S = - sigma_o[EltChannel_sym] - Ccc * w_lst_tmstp[sym_elements[EltChannel_sym]] - Cct * wTip_sym
        S = vcat(S, sum(Q) * dt / ElemArea - sum(dwTip) - sum(LkOff))

        return A, S
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        Mixed Picard Newton solver for nonlinear systems.

        Arguments:
            Res_fun (Function): The function calculating the residual.
            sys_fun (Function): The function giving the system A, b for the Picard solver to solve the linear system.
            guess (Vector{Float64}): The initial guess.
            TypValue (Vector{Float64}): Typical value of the variable to estimate the Epsilon to calculate Jacobian.
            interItr_init (Vector): Initial value of the variable(s) exchanged between the iterations.
            sim_prop (SimulationProperties): The SimulationProperties object giving simulation parameters.
            args (Tuple): Arguments given to the residual and systems functions.
            PicardPerNewton (Int): For hybrid Picard/Newton solution. Number of picard iterations for every Newton iteration.
            perf_node (IterationProperties): The IterationProperties object passed to be populated with data.

        Returns:
            - solk (Vector{Float64}) -- solution at the end of iteration.
            - data (Tuple) -- any data to be returned
    """
    function Picard_Newton(Res_fun, sys_fun, guess, TypValue, interItr_init, sim_prop, args...;
                        PicardPerNewton=1000, perf_node=nothing)
        
        log = Logging.getlogger("PyFrac.Picard_Newton")
        relax = sim_prop.relaxation_factor
        solk = copy(guess)
        k = 0
        normlist = Float64[]
        interItr = interItr_init
        newton = 0
        converged = false

        while !converged #todo:check system change (AM)

            solkm1 = copy(solk)
            if (k + 1) % PicardPerNewton == 0
                Fx, interItr, indices = Elastohydrodynamic_ResidualFun(solk, sys_fun, interItr, args...)
                Jac = Jacobian(Elastohydrodynamic_ResidualFun, sys_fun, solk, TypValue, interItr, args...)
                dx = Jac \ (-Fx)
                solk = solkm1 + dx
                newton += 1
            else
                try
                    A, b, interItr, indices = sys_fun(solk, interItr, args...)
                    perfNode_linSolve = instrument_start("linear system solve", perf_node)
                    solk = relax * solkm1 + (1 - relax) * (A \ b)
                catch e
                    if isa(e, LinearAlgebra.SingularException)
                        @error log "singular matrix!"
                        solk = fill(NaN, length(solk))
                        if perf_node !== nothing
                            instrument_close(perf_node, perfNode_linSolve, nothing,
                                            length(b), false, "singular matrix", nothing)
                            push!(perf_node.linearSolve_data, perfNode_linSolve)
                        end
                        return solk, nothing
                    else
                        rethrow(e)
                    end
                end
            end

            converged, norm = check_convergence(solk, solkm1, indices, sim_prop.toleranceEHL)
            push!(normlist, norm)
            k = k + 1

            if perf_node !== nothing
                instrument_close(perf_node, perfNode_linSolve, norm, length(b), true, nothing, nothing)
                push!(perf_node.linearSolve_data, perfNode_linSolve)
            end

            if k == sim_prop.maxSolverItrs  # returns nan as solution if does not converge
                @warn log "Picard iteration not converged after $(sim_prop.maxSolverItrs) iterations, norm: $norm"
                solk = fill(NaN, length(solk))
                if perf_node !== nothing
                    perfNode_linSolve.failure_cause = "singular matrix"
                    perfNode_linSolve.status = "failed"
                end
                return solk, nothing
            end
        end

        @debug log "Converged after $k iterations"
        data = [interItr[1], interItr[3], interItr[4]]
        return solk, data
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function returns the Jacobian of the given function.
    """
    function Jacobian(Residual_function, sys_func, x, TypValue, interItr, args...)
        
        central = true
        Fx, interItr, indices = Residual_function(x, sys_func, interItr, args...)
        Jac = zeros(Float64, length(x), length(x))
        for i in 1:length(x)
            Epsilon = eps(Float64)^(0.5) * abs(max(x[i], TypValue[i]))
            if Epsilon == 0
                Epsilon = eps(Float64)^(0.5)
            end
            xip = copy(x)
            xip[i] = xip[i] + Epsilon
            if central
                xin = copy(x)
                xin[i] = xin[i] - Epsilon
                Jac[:, i] = (Residual_function(xip, sys_func, interItr, args...)[1] - Residual_function(
                    xin, sys_func, interItr, args...)[1]) / (2 * Epsilon)
                if any(isnan.(Jac[:, i]))
                    Jac[:, :] .= NaN
                    return Jac
                end
            else
                Fxi, interItr, indices = Residual_function(xip, sys_func, interItr, args...)
                Jac[:, i] = (Fxi - Fx) / Epsilon
            end
        end

        return Jac
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function checks for convergence of the solution

        Arguments:
            solk (Vector{Float64}) -- the evaluated solution on this iteration
            solkm1 (Vector{Float64}) -- the evaluated solution on last iteration
            indices (Vector{Vector{Int}}) -- the list containing arrays giving indices of the cells
            tol (Float64) -- tolerance

        Returns:
            - converged (Bool) -- True if converged
            - norm (Float64) -- the evaluated norm which is checked against tolerance
    """
    function check_convergence(solk, solkm1, indices, tol)
        
        cnt = 0
        w_normalization = norm(solkm1[indices[1]])
        if w_normalization > 0.0
            norm_w = norm(abs.(solk[indices[1]] - solkm1[indices[1]]) / w_normalization)
        else
            norm_w = norm(abs.(solk[indices[1]] - solkm1[indices[1]]))
        end
        cnt += 1

        p_normalization = norm(solkm1[indices[2]])
        norm_p = norm(abs.(solk[indices[2]] - solkm1[indices[2]]) / p_normalization)
        cnt += 1

        if length(indices[3]) > 0  # these are the cells with the active width constraints
            tr_normalization = norm(solkm1[indices[3]])
            if tr_normalization > 0.0
                norm_tr = norm(abs.(solk[indices[3]] - solkm1[indices[3]]) / tr_normalization)
            else
                norm_tr = norm(abs.(solk[indices[3]] - solkm1[indices[3]]))
            end
            cnt += 1
        else
            norm_tr = 0.0
        end

        if length(indices) > 3
            if length(indices[4]) > 0
                if abs(solkm1[indices[4][1]]) > 0
                    norm_pil = abs(solk[indices[4][1]] - solkm1[indices[4][1]]) / abs(solkm1[indices[4][1]])
                end
                cnt += 1
            else
                norm_pil = 0.0
            end

            if length(indices) > 4 && length(indices[5]) > 0  # these are the cells with the active width constraints
                Q_ch_normalization = norm(solkm1[indices[5]])
                if Q_ch_normalization > 0.0
                    norm_Q_ch = norm(abs.(solk[indices[5]] - solkm1[indices[5]]) / Q_ch_normalization)
                else
                    norm_Q_ch = norm(abs.(solk[indices[5]] - solkm1[indices[5]]))
                end
                cnt += 1
            else
                norm_Q_ch = 0.0
            end

            if length(indices) > 5 && length(indices[6]) > 0  # these are the cells with the active width constraints
                Q_act_normalization = norm(solkm1[indices[6]])
                if Q_act_normalization > 0.0
                    norm_Q_act = norm(abs.(solk[indices[6]] - solkm1[indices[6]]) / Q_act_normalization)
                else
                    norm_Q_act = norm(abs.(solk[indices[6]] - solkm1[indices[6]]))
                end
                cnt += 1
            else
                norm_Q_act = 0.0
            end
        else
            norm_pil = 0.0
            norm_Q_ch = 0.0
            norm_Q_act = 0.0
        end

        norm = (norm_w + norm_p + norm_tr + norm_pil + norm_Q_ch + norm_Q_act) / cnt
        # println("w $norm_w p $norm_p act $norm_tr pil $norm_pil Qch $norm_Q_ch Qact $norm_Q_act")

        converged = (norm_w <= tol && norm_p <= tol && norm_tr <= tol && norm_pil <= tol &&
                    norm_Q_ch <= tol && norm_Q_act < tol)

        return converged, norm
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function gives the velocity at the cell edges evaluated using the Poiseuille flow assumption.
    """
    function velocity(w, EltCrack, Mesh, InCrack, muPrime, C, sigma0)
        
        dpdxLft, dpdxRgt, dpdyBtm, dpdyTop = pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)

        # velocity at the edges in the following order (x-left edge, x-right edge, y-bottom edge, y-top edge, y-left edge,
        #                                               y-right edge, x-bottom edge, x-top edge)
        vel = zeros(Float64, 8, Mesh.NumberOfElts)
        vel[1, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2).^2 / muPrime .* dpdxLft
        vel[2, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2).^2 / muPrime .* dpdxRgt
        vel[3, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2).^2 / muPrime .* dpdyBtm
        vel[4, EltCrack] = -((w[EltCrack] + w[Mesh.NeiElements[EltCrack, 4]]) / 2).^2 / muPrime .* dpdyTop

        vel[5, EltCrack] = (vel[3, Mesh.NeiElements[EltCrack, 1]] + vel[4, Mesh.NeiElements[EltCrack, 1]] + vel[
            3, EltCrack] + vel[4, EltCrack]) / 4
        vel[6, EltCrack] = (vel[3, Mesh.NeiElements[EltCrack, 2]] + vel[4, Mesh.NeiElements[EltCrack, 2]] + vel[
            3, EltCrack] + vel[4, EltCrack]) / 4
        vel[7, EltCrack] = (vel[1, Mesh.NeiElements[EltCrack, 3]] + vel[2, Mesh.NeiElements[EltCrack, 3]] + vel[
            1, EltCrack] + vel[2, EltCrack]) / 4
        vel[8, EltCrack] = (vel[1, Mesh.NeiElements[EltCrack, 4]] + vel[2, Mesh.NeiElements[EltCrack, 4]] + vel[
            1, EltCrack] + vel[2, EltCrack]) / 4

        vel_magnitude = zeros(Float64, 4, Mesh.NumberOfElts)
        vel_magnitude[1, :] = (vel[1, :].^2 + vel[5, :].^2).^0.5
        vel_magnitude[2, :] = (vel[2, :].^2 + vel[6, :].^2).^0.5
        vel_magnitude[3, :] = (vel[3, :].^2 + vel[7, :].^2).^0.5
        vel_magnitude[4, :] = (vel[4, :].^2 + vel[8, :].^2).^0.5

        return vel_magnitude
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function gives the pressure gradient at the cell edges evaluated with the pressure calculated from the
        elasticity relation for the given fracture width.
    """
    function pressure_gradient(w, C, sigma0, Mesh, EltCrack, InCrack)
        
        pf = zeros(Float64, Mesh.NumberOfElts)
        pf[EltCrack] = C[EltCrack, EltCrack] * w[EltCrack] + sigma0[EltCrack]

        dpdxLft = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 1]]) .* InCrack[Mesh.NeiElements[EltCrack, 1]]
        dpdxRgt = (pf[Mesh.NeiElements[EltCrack, 2]] - pf[EltCrack]) .* InCrack[Mesh.NeiElements[EltCrack, 2]]
        dpdyBtm = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 3]]) .* InCrack[Mesh.NeiElements[EltCrack, 3]]
        dpdyTop = (pf[Mesh.NeiElements[EltCrack, 4]] - pf[EltCrack]) .* InCrack[Mesh.NeiElements[EltCrack, 4]]

        return dpdxLft, dpdxRgt, dpdyBtm, dpdyTop
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function gives the pressure gradient at the cell edges evaluated with the pressure
    """
    function pressure_gradient_form_pressure(pf, Mesh, EltCrack, InCrack)
        
        dpdxLft = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 1]]) .* InCrack[Mesh.NeiElements[EltCrack, 1]] / Mesh.hx
        dpdxRgt = (pf[Mesh.NeiElements[EltCrack, 2]] - pf[EltCrack]) .* InCrack[Mesh.NeiElements[EltCrack, 2]] / Mesh.hx
        dpdyBtm = (pf[EltCrack] - pf[Mesh.NeiElements[EltCrack, 3]]) .* InCrack[Mesh.NeiElements[EltCrack, 3]] / Mesh.hy
        dpdyTop = (pf[Mesh.NeiElements[EltCrack, 4]] - pf[EltCrack]) .* InCrack[Mesh.NeiElements[EltCrack, 4]] / Mesh.hy

        return dpdxLft, dpdxRgt, dpdyBtm, dpdyTop
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function calculate fluid flux and velocity at the cell edges evaluated with the pressure calculated from the
        elasticity relation for the given fracture width and the poisoille's Law.
    """
    function calculate_fluid_flow_characteristics_laminar(w, pf, sigma0, Mesh, EltCrack, InCrack, muPrime, density)
        
        # here doesn't matter type of muPrime
        if minimum(muPrime) != 0 
            dp = zeros(Float64, 8, Mesh.NumberOfElts)
            dpdxLft, dpdxRgt, dpdyBtm, dpdyTop = pressure_gradient_form_pressure(pf, Mesh, EltCrack, InCrack)
            # dp = [dpdxLft , dpdxRgt, dpdyBtm, dpdyTop, dpdyLft, dpdyRgt, dpdxBtm, dpdxTop]
            dp[1, EltCrack] = dpdxLft
            dp[2, EltCrack] = dpdxRgt
            dp[3, EltCrack] = dpdyBtm
            dp[4, EltCrack] = dpdyTop
            # linear interpolation for pressure gradient on the edges where central difference not available
            dp[5, EltCrack] = (dp[3, Mesh.NeiElements[EltCrack, 1]] + dp[4, Mesh.NeiElements[EltCrack, 1]] + dp[3, EltCrack] +
                            dp[4, EltCrack]) / 4
            dp[6, EltCrack] = (dp[3, Mesh.NeiElements[EltCrack, 2]] + dp[4, Mesh.NeiElements[EltCrack, 2]] + dp[3, EltCrack] +
                            dp[4, EltCrack]) / 4
            dp[7, EltCrack] = (dp[1, Mesh.NeiElements[EltCrack, 3]] + dp[2, Mesh.NeiElements[EltCrack, 3]] + dp[1, EltCrack] +
                            dp[2, EltCrack]) / 4
            dp[8, EltCrack] = (dp[1, Mesh.NeiElements[EltCrack, 4]] + dp[2, Mesh.NeiElements[EltCrack, 4]] + dp[1, EltCrack] +
                            dp[2, EltCrack]) / 4

            # magnitude of pressure gradient vector on the cell edges. Used to calculate the friction factor
            dpLft = (dp[1, EltCrack].^2 + dp[5, EltCrack].^2).^0.5
            dpRgt = (dp[2, EltCrack].^2 + dp[6, EltCrack].^2).^0.5
            dpBtm = (dp[3, EltCrack].^2 + dp[7, EltCrack].^2).^0.5
            dpTop = (dp[4, EltCrack].^2 + dp[8, EltCrack].^2).^0.5

            # width at the cell edges evaluated by averaging. Zero if the edge is outside fracture
            wLftEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 1]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 1]]
            wRgtEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 2]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 2]]
            wBtmEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 3]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 3]]
            wTopEdge = (w[EltCrack] + w[Mesh.NeiElements[EltCrack, 4]]) / 2 .* InCrack[Mesh.NeiElements[EltCrack, 4]]

            fluid_flux = vcat(-wLftEdge.^3 .* dpLft / muPrime, -wRgtEdge.^3 .* dpRgt / muPrime)
            fluid_flux = vcat(fluid_flux, -wBtmEdge.^3 .* dpBtm / muPrime)
            fluid_flux = vcat(fluid_flux, -wTopEdge.^3 .* dpTop / muPrime)

            #          1    ,    2   ,     3  ,    4   ,    5   ,    6   ,    7   ,    8
            # dp = [dpdxLft , dpdxRgt, dpdyBtm, dpdyTop, dpdyLft, dpdyRgt, dpdxBtm, dpdxTop]

            # fluid_flux_components = [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge, fy bottom edge, fx top edge, fy top edge]
            #                                                      fx left edge          ,              fy left edge
            fluid_flux_components = vcat(-wLftEdge.^3 .* dp[1, EltCrack] / muPrime, -wLftEdge.^3 .* dp[5, EltCrack] / muPrime)
            #                                                      fx right edge
            fluid_flux_components = vcat(fluid_flux_components, -wRgtEdge.^3 .* dp[2, EltCrack] / muPrime)
            #                                                      fy right edge
            fluid_flux_components = vcat(fluid_flux_components, -wRgtEdge.^3 .* dp[6, EltCrack] / muPrime)
            #                                                      fx bottom edge
            fluid_flux_components = vcat(fluid_flux_components, -wBtmEdge.^3 .* dp[7, EltCrack] / muPrime)
            #                                                      fy bottom edge
            fluid_flux_components = vcat(fluid_flux_components, -wBtmEdge.^3 .* dp[3, EltCrack] / muPrime)
            #                                                      fx top edge
            fluid_flux_components = vcat(fluid_flux_components, -wTopEdge.^3 .* dp[8, EltCrack] / muPrime)
            #                                                      fy top edge
            fluid_flux_components = vcat(fluid_flux_components, -wTopEdge.^3 .* dp[4, EltCrack] / muPrime)

            fluid_vel = copy(fluid_flux)
            wEdges = [wLftEdge, wRgtEdge, wBtmEdge, wTopEdge]
            for i in 1:4
                local_nonzero_indexes = findall(!iszero, fluid_vel[i, :])
                fluid_vel[i, local_nonzero_indexes] ./= wEdges[i][local_nonzero_indexes]
            end

            fluid_vel_components = copy(fluid_flux_components)
            for i in 1:8
                local_nonzero_indexes = findall(!iszero, fluid_vel_components[i, :])
                fluid_vel_components[i, local_nonzero_indexes] ./= wEdges[Int(floor((i-1)/2))+1][local_nonzero_indexes]
            end

            Rey_number = abs.(4 / 3 * density * fluid_flux / muPrime * 12) #doesn't matter in a case of constant

            return abs.(fluid_flux), abs.(fluid_vel), Rey_number, fluid_flux_components, fluid_vel_components
        else
            throw(SystemExit("ERROR: if the fluid viscosity is equal to 0 does not make sense to compute the fluid velocity or the fluid flux"))
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        Anderson solver for non linear system.

        Arguments:
            sys_fun (Function): The function giving the system A, b for the Anderson solver to solve the linear system.
            guess (Vector{Float64}): The initial guess.
            interItr_init (Vector): Initial value of the variable(s) exchanged between the iterations.
            sim_prop (SimulationProperties): The SimulationProperties object giving simulation parameters.
            args (Tuple): Arguments given to the residual and systems functions.
            perf_node (IterationProperties): The IterationProperties object passed to be populated with data.

        Returns:
            - Xks[mk+1] (Vector{Float64}) -- final solution at the end of the iterations.
            - data (Tuple) -- any data to be returned
    """
    function Anderson(sys_fun, guess, interItr_init, sim_prop, args...; perf_node=nothing)
        
        log = Logging.getlogger("PyFrac.Anderson")
        m_Anderson = sim_prop.Anderson_parameter
        relax = sim_prop.relaxation_factor

        ## Initialization of solution vectors
        xks = zeros(Float64, m_Anderson+2, length(guess))
        Fks = zeros(Float64, m_Anderson+1, length(guess))
        Gks = zeros(Float64, m_Anderson+1, length(guess))

        ## Initialization of iteration parameters
        k = 0
        normlist = Float64[]
        interItr = interItr_init
        converged = false
        try
            perfNode_linSolve = instrument_start("linear system solve", perf_node)
            # First iteration
            xks[1, :] = guess                                       # xo
            A, b, interItr, indices = sys_fun(xks[1, :], interItr, args...)     # assembling A and b

            Gks[1, :] = A \ b
            Fks[1, :] = Gks[1, :] - xks[1, :]
            xks[2, :] = Gks[1, :]                                               # x1
        catch e
            if isa(e, LinearAlgebra.SingularException)
                @error log "singular matrix!"
                solk = fill(NaN, length(xks[1, :]))
                if perf_node !== nothing
                    instrument_close(perf_node, perfNode_linSolve, nothing,
                                    length(b), false, "singular matrix", nothing)
                    push!(perf_node.linearSolve_data, perfNode_linSolve)
                end
                return solk, nothing
            else
                rethrow(e)
            end
        end

        while !converged

            try
                mk = min(k, m_Anderson-1)  # Assess the amount of solutions available for the least square problem
                if k >= m_Anderson
                    A, b, interItr, indices = sys_fun(xks[mk + 2, :], interItr, args...)
                    Gks = circshift(Gks, (-1, 0))
                    Fks = circshift(Fks, (-1, 0))
                else
                    A, b, interItr, indices = sys_fun(xks[mk + 1, :], interItr, args...)
                end
                perfNode_linSolve = instrument_start("linear system solve", perf_node)

                Gks[mk + 1, :] = A \ b
                Fks[mk + 1, :] = Gks[mk + 1, :] - xks[mk + 1, :]

                ## Setting up the Least square problem of Anderson
                A_Anderson = (Fks[1:mk+1, :] .- Fks[mk+1, :]')'
                b_Anderson = -Fks[mk+1, :]

                # Solving the least square problem for the coefficients
                # Note: lsq_linear needs to be implemented or imported
                # omega_s = lsq_linear(A_Anderson, b_Anderson, bounds=(0, 1/(mk+2)), lsmr_tol='auto').x
                omega_s = pinv(A_Anderson) * b_Anderson  # Simplified version
                omega_s = vcat(omega_s, 1.0 - sum(omega_s))

                ## Updating xk in a relaxed version
                if k >= m_Anderson
                    xks = circshift(xks, (-1, 0))
                end

                xks[mk + 2, :] = (1-relax) * sum(xks[1:mk+2, :] .* omega_s, dims=1) +
                    relax * sum(Gks[1:mk+2, :] .* omega_s, dims=1)

            catch e
                if isa(e, LinearAlgebra.SingularException)
                    @error log "singular matrix!"
                    solk = fill(NaN, length(xks[mk, :]))
                    if perf_node !== nothing
                        instrument_close(perf_node, perfNode_linSolve, nothing,
                                        length(b), false, "singular matrix", nothing)
                        push!(perf_node.linearSolve_data, perfNode_linSolve)
                    end
                    return solk, nothing
                else
                    rethrow(e)
                end
            end

            ## Check for convergence of the solution

            converged, norm = check_convergence(xks[mk + 1, :], xks[mk + 2, :], indices, sim_prop.toleranceEHL)
            push!(normlist, norm)
            k = k + 1

            if perf_node !== nothing
                instrument_close(perf_node, perfNode_linSolve, norm, length(b), true, nothing, nothing)
                push!(perf_node.linearSolve_data, perfNode_linSolve)
            end

            if k == sim_prop.maxSolverItrs  # returns nan as solution if does not converge
                @warn log "Anderson iteration not converged after $(sim_prop.maxSolverItrs) iterations, norm: $norm"
                solk = fill(NaN, size(xks[1, :], 2))
                if perf_node !== nothing
                    perfNode_linSolve.failure_cause = "singular matrix"
                    perfNode_linSolve.status = "failed"
                end
                return solk, nothing
            end
        end

        @debug log "Converged after $k iterations"

        data = [interItr[1], interItr[3], interItr[4]]
        return xks[mk + 2, :], data
    end
end # module ElastoHydrodynamicSolver