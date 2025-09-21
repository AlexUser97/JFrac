# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac explicit_RKL module on Julia language.

"""

module ExplicitRKL

    using LinearAlgebra
    using SparseArrays
    using Logging
    using Base.Threads

    include("elastohydrodynamic_solver.jl")
    using .ElastoHydrodynamicSolver: finiteDiff_operator_laminar, FiniteDiff_operator_turbulent_implicit, Gravity_term
    include("properties.jl")
    using .Properties: instrument_start, instrument_close

    export solve_width_pressure_RKL2, RKL_substep_neg, pardot_matrix_vector

    const s_max = 1000
    const a = zeros(Float64, s_max)
    const b = zeros(Float64, s_max)
    const mu = zeros(Float64, s_max)
    const nu = zeros(Float64, s_max)

    b[1:2] .= 1 / 3
    a[1:2] = 1 .- b[1:2]
    for j in 3:s_max
        b[j] = (j * j + j - 2) / (2 * j * (j + 1))
        a[j] = 1 - b[j]
        mu[j] = (2 * j - 1) * b[j] / (j * b[j - 1])
        nu[j] = - (j - 1) * b[j] / (j * b[j - 2])
    end

    """
        solve_width_pressure_RKL2(Eprime, GPU, n_threads, perf_node, args...)

        Solve width and pressure using RKL2 scheme.

        # Arguments
        - `Eprime::Float64`: Plane strain modulus.
        - `GPU::Bool`: Flag to use GPU.
        - `n_threads::Int`: Number of threads to use.
        - `perf_node`: Performance node for profiling.
        - `args`: Variable arguments passed as tuple.

        # Returns
        - `(sol, s)`: Solution and number of sub-steps.
    """
    function solve_width_pressure_RKL2(Eprime::Float64, GPU::Bool, n_threads::Int, perf_node, args...)
        log = Logging.current_logger()
        # perfNode_RKL = instrument_start("linear system solve", perf_node)

        to_solve = args[1]
        to_impose = args[2]
        wLastTS = args[3]
        pfLastTS = args[4]
        imposed_val = args[5]
        EltCrack = args[6]
        Mesh = args[7]
        dt = args[8]
        Q = args[9]
        C = args[10]
        muPrime = args[11]
        rho = args[12]
        InCrack = args[13]
        LeakOff = args[14]
        sigma0 = args[15]
        turb = args[16]
        dgrain = args[17]
        gravity = args[18]
        active = args[19]
        wc_to_impose = args[20]
        wc = args[21]
        cf = args[22]
        neiInCrack = args[23]

        viscosity = muPrime / 12
        dt_CFL = 5 * mean(viscosity) * min(Mesh.hx, Mesh.hy)^3 / (Eprime * maximum(wLastTS)^3)
        s = ceil(Int, -0.5 + sqrt(8 + 16 * dt / dt_CFL) / 2)
        @info "no. of sub-steps = $s" _group="JFrac.solve_width_pressure_RKL2"

        delt_wTip = imposed_val - wLastTS[to_impose]
        tip_delw_step = delt_wTip / s

        act_tip_val = vcat(wc_to_impose, tip_delw_step)
        n_ch = length(to_solve)
        ch_indxs = 1:n_ch

        if turb
            error("RKL scheme with turbulence is not yet implemented!")
        else
            cond_0_lil = finiteDiff_operator_laminar(wLastTS,
                                                EltCrack,
                                                muPrime,
                                                Mesh,
                                                InCrack,
                                                neiInCrack,
                                                sparse_flag=true)
        end
        
        cond_0 = sparse(cond_0_lil)
        mu_t_1 = 4 / (3 * (s * s + s - 2))

        if gravity
            w_0 = zeros(Float64, Mesh.NumberOfElts)
            w_0[EltCrack] = wLastTS[EltCrack]
            G = Gravity_term(w_0,
                            EltCrack,
                            muPrime,
                            Mesh,
                            InCrack,
                            rho)[EltCrack]
        else
            G = zeros(Float64, length(EltCrack))
        end

        if GPU
            В Julia для GPU обычно используется CUDA.jl
            if !isdefined(Main, :CuArray)
                using CUDA
            end
            C_red = CuArray(C[to_solve, EltCrack])
            # error("GPU support not implemented yet")
        else
            C_red = C[to_solve, EltCrack]
        end

        Lk_rate = LeakOff / dt
        W_0 = wLastTS[EltCrack]
        pf_0 = Vector{Float64}(undef, length(EltCrack))
        pf_0[ch_indxs] = C[to_solve, EltCrack] * wLastTS[EltCrack] + sigma0[to_solve]
        
        tip_operator = dt * mu_t_1 * cond_0[n_ch+1:end, n_ch+1:end-1] # sparse matrix
        tip_rhs_terms = dt * mu_t_1 * (cond_0[n_ch+1:end, 1:n_ch] * pf_0[1:n_ch] +
                                    G[n_ch+1:end] + (Q[EltCrack[n_ch+1:end]] - Lk_rate[EltCrack[n_ch+1:end]]) / Mesh.EltArea)
        pf_0[n_ch+1:end] = tip_operator \ (act_tip_val - tip_rhs_terms)

        M_0 = cond_0[:, 1:end-1] * pf_0 + (Q[EltCrack] - Lk_rate[EltCrack]) / Mesh.EltArea + G
        W_1 = wLastTS[EltCrack] + dt * mu_t_1 * M_0
        W_jm1 = copy(W_1)
        W_jm2 = copy(W_0)
        tau_M0 = dt * M_0
        param_pack = (muPrime, Mesh, InCrack, neiInCrack)

        Lk_rate_cr = Lk_rate[EltCrack]
        Q_ = Q[EltCrack]
        
        for j in 2:s
            W_j = RKL_substep_neg(j, s, W_jm1, W_jm2, W_0, EltCrack, n_ch,
                                tip_delw_step, param_pack, C_red, dt,
                                tau_M0, Q_, to_solve, sigma0, act_tip_val,
                                gravity, rho, Lk_rate_cr, GPU, n_threads,
                                turb)
            W_jm2 = W_jm1
            W_jm1 = W_j
        end

        sol = W_j - wLastTS[EltCrack]

        # Если profiling реализован:
        # if perf_node !== nothing
        #     instrument_close(perf_node, perfNode_RKL, nothing, length(W_j), true, false, nothing)
        #     perfNode_RKL.iterations = s
        #     push!(perf_node.RKL_data, perfNode_RKL)
        # end

        return sol, s
    end

    """
        RKL_substep_neg(j, s, W_jm1, W_jm2, W_0, crack, n_channel, tip_delw_step, param_pack, C, tau, tau_M0, Qin,
                        EltChannel, sigmaO, imposed_value, gravity, rho, LeakOff, GPU_flag, n_threads, turb)

        Perform a single RKL substep.

        # Arguments
        - `j::Int`: Current substep index.
        - `s::Int`: Total number of substeps.
        - `W_jm1::Vector{Float64}`: Solution from previous step.
        - `W_jm2::Vector{Float64}`: Solution from two steps ago.
        - `W_0::Vector{Float64}`: Initial solution.
        - `crack::Vector{Int}`: Crack elements.
        - `n_channel::Int`: Number of channel elements.
        - `tip_delw_step::Float64`: Tip width step.
        - `param_pack::Tuple`: Parameter pack (muPrime, Mesh, InCrack, neiInCrack).
        - `C`: Reduced elasticity matrix.
        - `tau::Float64`: Time step.
        - `tau_M0::Vector{Float64}`: Initial M0 term.
        - `Qin::Vector{Float64}`: Injection rate.
        - `EltChannel::Vector{Int}`: Channel elements.
        - `sigmaO::Vector{Float64}`: Confining stress.
        - `imposed_value::Vector{Float64}`: Imposed values.
        - `gravity::Bool`: Gravity flag.
        - `rho::Float64`: Fluid density.
        - `LeakOff::Vector{Float64}`: Leak-off rate.
        - `GPU_flag::Bool`: GPU flag.
        - `n_threads::Int`: Number of threads.
        - `turb::Bool`: Turbulence flag.

        # Returns
        - `W_j::Vector{Float64}`: Solution at current step.
    """
    function RKL_substep_neg(j::Int, s::Int, W_jm1::Vector{Float64}, W_jm2::Vector{Float64}, W_0::Vector{Float64}, 
                            crack::Vector{Int}, n_channel::Int, tip_delw_step::Float64, param_pack::Tuple, C, tau::Float64, 
                            tau_M0::Vector{Float64}, Qin::Vector{Float64}, EltChannel::Vector{Int}, sigmaO::Vector{Float64}, 
                            imposed_value::Vector{Float64}, gravity::Bool, rho::Float64, LeakOff::Vector{Float64}, 
                            GPU_flag::Bool, n_threads::Int, turb::Bool)

        muPrime, Mesh, InCrack, neiInCrack = param_pack
        w_jm1 = zeros(Float64, Mesh.NumberOfElts)
        cp_W_jm1 = copy(W_jm1)
        cp_W_jm1[W_jm1 .< 1e-6] .= 1e-6
        w_jm1[crack] = cp_W_jm1

        if turb
            error("RKL scheme with turbulence is not yet implemented!")
        else
            cond_lil = finiteDiff_operator_laminar(w_jm1,
                                                crack,
                                                muPrime,
                                                Mesh,
                                                InCrack,
                                                neiInCrack,
                                                sparse_flag=true)
        end

        cond = sparse(cond_lil)
        mu_t = 4 * (2 * j - 1) * b[j] / (j * (s * s + s - 2) * b[j - 1])
        gamma_t = -a[j - 1] * mu_t
        
        if gravity
            G = Gravity_term(w_jm1,
                            crack,
                            muPrime,
                            Mesh,
                            InCrack,
                            rho)[crack]
        else
            G = zeros(Float64, length(crack))
        end

        pf = Vector{Float64}(undef, length(crack))

        if GPU_flag
            W_jm1_cp = CuArray(W_jm1)
            pn = C * W_jm1_cp
            pf[1:n_channel] = Array(pn) + sigmaO[EltChannel]
            # error("GPU support not implemented yet")
        else
            pf[1:n_channel] = pardot_matrix_vector(C, W_jm1, n_threads) + sigmaO[EltChannel]
        end

        imposed_value[end-length(tip_delw_step)+1:end] = j * tip_delw_step
        

        tip_cond_part = cond[n_channel+1:end, 1:n_channel] # sparse matrix part
        tip_pf_part = pf[1:n_channel]
        tip_other_terms = G[n_channel+1:end] + (Qin[n_channel+1:end] - LeakOff[n_channel+1:end]) / Mesh.EltArea
        M_jm1_tip = tip_cond_part * tip_pf_part + tip_other_terms
        

        S = imposed_value - mu[j] * W_jm1[n_channel+1:end] - nu[j] * W_jm2[n_channel+1:end] + 
            (mu[j] + nu[j]) * W_0[n_channel+1:end] - gamma_t * tau_M0[n_channel+1:end] - mu_t * tau * M_jm1_tip
        

        A = mu_t * tau * cond[n_channel+1:end, n_channel+1:end-1] # sparse matrix
        pf[n_channel+1:end] = A \ S


        M_jm1 = cond[:, 1:end-1] * pf + G + (Qin - LeakOff) / Mesh.EltArea

        W_j = mu[j] * W_jm1 + nu[j] * W_jm2 + (1 - mu[j] - nu[j]) * W_0 + mu_t * tau * M_jm1 + gamma_t * tau_M0

        return W_j
    end

    """
        pardot_matrix_vector(a, b, nblocks, dot_func=do_dot)

        Perform parallel matrix-vector multiplication.

        # Arguments
        - `a::Matrix`: Matrix.
        - `b::Vector`: Vector.
        - `nblocks::Int`: Number of blocks.
        - `dot_func`: Dot product function.

        # Returns
        - `out::Vector`: Result vector.
    """
    function pardot_matrix_vector(a::Matrix, b::Vector, nblocks::Int, dot_func=do_dot)
        n_jobs = nblocks
        # log = Logging.current_logger()
        @info "running $n_jobs jobs in parallel" _group="JFrac.pardot_matrix_vector"

        out = Vector{eltype(a)}(undef, size(a, 1))

        n_in_block = size(a, 1) รท nblocks
        @threads for i in 1:nblocks
            start_idx = (i-1) * n_in_block + 1
            end_idx = i == nblocks ? size(a, 1) : i * n_in_block
            out[start_idx:end_idx] = a[start_idx:end_idx, :] * b
        end


        return out
    end


    function do_dot(a::Matrix, b::Vector, out::Vector)
        out[:] = a * b
    end

end # module ExplicitRKL