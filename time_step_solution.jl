# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""
module TimeStepSolution

using Logging
using LinearAlgebra

using .VolumeIntegral: leak_off_stagnant_tip, find_corresponding_ribbon_cell
using .Symmetry: get_symetric_elements, self_influence
using .TipInversion: TipAsymInversion, StressIntensityFactor
using .ElastohydrodynamicSolver: 
using .LevelSet: SolveFMM, reconstruct_front, reconstruct_front_LS_gradient, UpdateLists
using .ContinuousFrontReconstruction: reconstruct_front_continuous, UpdateListsFromContinuousFrontRec, you_advance_more_than_2_cells
using .Properties: IterationProperties, instrument_start, instrument_close
using .Anisotropy:
using .Labels: TS_errorMessages
using .ExplicitRKL: solve_width_pressure_RKL2
using .PostprocessFracture: append_to_json_file

export attempt_time_step, injection_same_footprint, injection_extended_footprint,
       solve_width_pressure, turbulence_check_tip, time_step_explicit_front


"""
    attempt_time_step(Frac, C, mat_properties, fluid_properties, sim_properties, inj_properties,
                      timeStep, perfNode=nothing)

This function attempts to propagate fracture with the given time step. The function injects fluid and propagates
the fracture front according to the front advancing scheme given in the simulation properties.

# Arguments
- `Frac`: fracture object from the last time step.
- `C::Matrix`: the elasticity matrix.
- `mat_properties`: material properties.
- `fluid_properties`: fluid properties.
- `sim_properties`: simulation parameters.
- `inj_properties`: injection properties.
- `timeStep::Float64`: time step.
- `perfNode::Union{IterationProperties, Nothing}`: a performance node to store performance data.

# Returns
- `Tuple{Int, Fracture}`: (exitstatus, Fr_k) - see documentation for possible values and fracture after advancing time step.
"""
function attempt_time_step(Frac, C, mat_properties, fluid_properties, sim_properties, inj_properties,
                          timeStep::Float64, perfNode::Union{IterationProperties, Nothing}=nothing)

    @debug "attempt_time_step called" _group="JFrac.attempt_time_step"
    
    Qin = inj_properties.get_injection_rate(Frac.time, Frac)
    
    if inj_properties.sinkLocFunc !== nothing
        Qin[inj_properties.sinkElem] -= inj_properties.sinkVel * Frac.mesh.EltArea
    end

    if inj_properties.delayed_second_injpoint_elem !== nothing
        if inj_properties.rate_delayed_inj_pt_func === nothing
            if Frac.time >= inj_properties.injectionTime_delayed_second_injpoint
                Qin[inj_properties.delayed_second_injpoint_elem] = inj_properties.injectionRate_delayed_second_injpoint/length(inj_properties.delayed_second_injpoint_elem)
            else
                Qin[inj_properties.delayed_second_injpoint_elem] = inj_properties.init_rate_delayed_second_injpoint/length(inj_properties.delayed_second_injpoint_elem)
            end
        else
            Qin[inj_properties.delayed_second_injpoint_elem] = inj_properties.rate_delayed_inj_pt_func(Frac.time)/length(inj_properties.delayed_second_injpoint_elem)
        end
        @debug "max value of the array Q(x,y) = $(maximum(Qin))" _group="JFrac.attempt_time_step"
        @debug "Q at the delayed inj point = $(Qin[inj_properties.delayed_second_injpoint_elem])" _group="JFrac.attempt_time_step"
    end

    if sim_properties.frontAdvancing == "explicit"
        perfNode_explFront = instrument_start("extended front", perfNode)
        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    inj_properties,
                                                    perfNode_explFront)

        if perfNode_explFront !== nothing
            instrument_close(perfNode, perfNode_explFront, nothing,
                             length(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus+1], Frac.time)
            push!(perfNode.extendedFront_data, perfNode_explFront)
        end

        # check if we advanced more than two cells
        if exitstatus == 1
            if you_advance_more_than_2_cells(Fr_k.fully_traversed, Frac.EltTip, Frac.mesh.NeiElements, Frac.Ffront, Fr_k.Ffront, Fr_k.mesh) && 
                sim_properties.limitAdancementTo2cells
                exitstatus = 17
                return exitstatus, Frac
            end
        end

        return exitstatus, Fr_k

    elseif sim_properties.frontAdvancing == "predictor-corrector"
        @debug "Advancing front with velocity from last time-step..." _group="JFrac.attempt_time_step"

        perfNode_explFront = instrument_start("extended front", perfNode)
        exitstatus, Fr_k = time_step_explicit_front(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    inj_properties,
                                                    perfNode_explFront)

        if perfNode_explFront !== nothing
            instrument_close(perfNode, perfNode_explFront, nothing,
                             length(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus+1], Frac.time)
            push!(perfNode.extendedFront_data, perfNode_explFront)
        end

    elseif sim_properties.frontAdvancing == "implicit"
        @debug "Solving ElastoHydrodynamic equations with same footprint..." _group="JFrac.attempt_time_step"

        perfNode_sameFP = instrument_start("same front", perfNode)

        # width by injecting the fracture with the same footprint (balloon like inflation)
        exitstatus, Fr_k = injection_same_footprint(Frac,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    mat_properties,
                                                    fluid_properties,
                                                    sim_properties,
                                                    inj_properties,
                                                    perfNode_sameFP)
        
        if perfNode_sameFP !== nothing
            instrument_close(perfNode, perfNode_sameFP, nothing,
                             length(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus+1], Frac.time)
            push!(perfNode.sameFront_data, perfNode_sameFP)
        end

    else
        error("Provided front advancing type not supported")
    end

    if exitstatus != 1
        return exitstatus, Fr_k
    end

    # Check for the propagation condition with the new width. If the all of the front is stagnant, return fracture as
    # final without front iteration.
    stagnant_crt = falses(length(Fr_k.EltRibbon))
    # stagnant cells where propagation criteria is not met
    # В Julia нужно быть осторожным с индексацией и операциями над массивами
    prop_condition = mat_properties.Kprime[Fr_k.EltRibbon] .* (-Fr_k.sgndDist[Fr_k.EltRibbon]) .^ 0.5 ./ 
                     (mat_properties.Eprime * Fr_k.w[Fr_k.EltRibbon])
    stagnant_indices = findall(prop_condition .> 1)
    stagnant_crt[stagnant_indices] .= true
    
    # stagnant cells where fracture is closed
    stagnant_closed = falses(length(Fr_k.EltRibbon))
    for i in 1:length(Fr_k.EltRibbon)
        if Fr_k.EltRibbon[i] in Fr_k.closed
            stagnant_closed[i] = true
        end
    end
    stagnant = stagnant_closed .| stagnant_crt

    if all(stagnant)
        return 1, Fr_k
    end

    @debug "Starting Fracture Front loop..." _group="JFrac.attempt_time_step"

    norm = 10.0
    k = 0
    previous_norm = 100.0 # initially set with a big value

    # Fracture front loop to find the correct front location
    while norm > sim_properties.tolFractFront
        k = k + 1
        @debug " " _group="JFrac.attempt_time_step"
        @debug "Iteration $k" _group="JFrac.attempt_time_step"
        fill_frac_last = copy(Fr_k.FillF)

        perfNode_extFront = instrument_start("extended front", perfNode)
        # find the new footprint and solve the elastohydrodynamic equations to to get the new fracture
        (exitstatus, Fr_k) = injection_extended_footprint(Fr_k.w,
                                                          Frac,
                                                          C,
                                                          timeStep,
                                                          Qin,
                                                          mat_properties,
                                                          fluid_properties,
                                                          sim_properties,
                                                          inj_properties,
                                                          perfNode_extFront)

        if exitstatus == 1
            # norm is evaluated by dividing the difference in the area of the tip cells between two successive
            # iterations with the number of tip cells.
            norm = abs((sum(Fr_k.FillF) - sum(fill_frac_last)) / length(Fr_k.FillF))
        else
            norm = NaN
        end

        if perfNode_extFront !== nothing
            instrument_close(perfNode, perfNode_extFront, norm,
                             length(Frac.EltCrack), exitstatus == 1,
                             TS_errorMessages[exitstatus+1], Frac.time) # +1 из-за индексации с 0 в Python
            push!(perfNode.extendedFront_data, perfNode_extFront)
        end

        if exitstatus != 1
            return exitstatus, Fr_k
        end

        @debug "Norm of subsequent filling fraction estimates = $norm" _group="JFrac.attempt_time_step"

        # sometimes the code is going to fail because of the max number of iterations due to the lack of
        # improvement of the norm
        if !isnan(norm)
            if abs((previous_norm-norm)/norm) < 0.001
                @debug "Norm of subsequent Norms of subsequent filling fraction estimates = $(abs((previous_norm-norm)/norm)) < 0.001" _group="JFrac.attempt_time_step"
                exitstatus = 15
                return exitstatus, nothing
            else
                previous_norm = norm
            end
        end

        if k == sim_properties.maxFrontItrs
            exitstatus = 6
            return exitstatus, nothing
        end
    end

    # check if we advanced more than two cells
    if exitstatus == 1
        if you_advance_more_than_2_cells(Fr_k.fully_traversed, Frac.EltTip, Frac.mesh.NeiElements, Frac.Ffront, Fr_k.Ffront, Fr_k.mesh) && 
                sim_properties.limitAdancementTo2cells
            exitstatus = 17
            return exitstatus, Frac
        end
    end

    @debug "Fracture front converged after $k iterations with norm = $norm" _group="JFrac.attempt_time_step"

    return exitstatus, Fr_k
end


# ----------------------------------------------------------------------------------------------------------------------

"""
    injection_same_footprint(Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
                             inj_properties, perfNode=nothing)

This function solves the ElastoHydrodynamic equations to get the fracture width. The fracture footprint is taken
to be the same as in the fracture from the last time step.

# Arguments
- `Fr_lstTmStp`: fracture object from the last time step.
- `C::Matrix`: the elasticity matrix.
- `timeStep::Float64`: time step.
- `Qin::Vector{Float64}`: current injection rate.
- `mat_properties`: material properties.
- `fluid_properties`: fluid properties.
- `sim_properties`: simulation parameters.
- `inj_properties`: injection properties.
- `perfNode::Union{IterationProperties, Nothing}`: a performance node to store performance data.

# Returns
- `Tuple{Int, Fracture}`: (exitstatus, Fr_kplus1) - exit status and the fracture after injection with the same footprint.
"""
function injection_same_footprint(Fr_lstTmStp, C, timeStep::Float64, Qin::Vector{Float64}, mat_properties, 
                                 fluid_properties, sim_properties, inj_properties, 
                                 perfNode::Union{IterationProperties, Nothing}=nothing)

    @debug "injection_same_footprint called" _group="JFrac.injection_same_footprint"

    if sum(Fr_lstTmStp.InCrack .== 1) > sim_properties.maxElementIn && sim_properties.meshReductionPossible
        exitstatus = 16
        return exitstatus, Fr_lstTmStp
    end

    LkOff = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
    if sum(mat_properties.Cprime[Fr_lstTmStp.EltCrack]) > 0.0
        # the tip cells are assumed to be stagnant in same footprint evaluation
        LkOff[Fr_lstTmStp.EltTip] = leak_off_stagnant_tip(Fr_lstTmStp.EltTip,
                                                          Fr_lstTmStp.l,
                                                          Fr_lstTmStp.alpha,
                                                          Fr_lstTmStp.TarrvlZrVrtx[Fr_lstTmStp.EltTip],
                                                          Fr_lstTmStp.time + timeStep,
                                                          mat_properties.Cprime,
                                                          timeStep,
                                                          Fr_lstTmStp.mesh)

        # Calculate leak-off term for the channel cell
        t_lst_min_t0 = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_lst_min_t0[t_lst_min_t0 .< 0.0] .= 0.0
        t_min_t0 = t_lst_min_t0 + timeStep
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * (t_min_t0 .^ 0.5 -
                                                                                             t_lst_min_t0 .^ 0.5) * Fr_lstTmStp.mesh.EltArea
    end

    LkOff[Fr_lstTmStp.pFluid .<= mat_properties.porePressure] .= 0.0

    if any(isnan.(LkOff[Fr_lstTmStp.EltCrack]))
        exitstatus = 13
        return exitstatus, nothing
    end

    # solve for width. All of the fracture cells are solved (tip values imposed from the last time step)
    empty = Int[]
    
    # Обработка doublefracturedictionary
    doublefracturedictionary = Dict{String, Any}()
    if sim_properties.doublefracture && Fr_lstTmStp.fronts_dictionary["number_of_fronts"] == 2
        # here we save the cells in the two cracks
        doublefracturedictionary["number_of_fronts"] = Fr_lstTmStp.fronts_dictionary["number_of_fronts"]
        doublefracturedictionary["crackcells_0"] = Fr_lstTmStp.fronts_dictionary["crackcells_0"]
        doublefracturedictionary["crackcells_1"] = Fr_lstTmStp.fronts_dictionary["crackcells_1"]
    elseif sim_properties.projMethod != "LS_continousfront"
        doublefracturedictionary["number_of_fronts"] = 1
    else
        doublefracturedictionary["number_of_fronts"] = Fr_lstTmStp.fronts_dictionary["number_of_fronts"]
    end

    # todo: make tip correction while injecting in the same footprint
    ##########################################################################################
    #                                                                                        #
    #  when we inject on the same footprint we should make the tip correction at the tip     #
    #  this is not done, but it will not affect the accuracy, only the speed of convergence  #
    #  since we are never accessing the diagonal of the elasticity matrix when iterating on  #
    #  the position of the front.                                                            #
    #                                                                                        #
    ##########################################################################################
    
    w_k, p_k, return_data = solve_width_pressure(Fr_lstTmStp,
                                         sim_properties,
                                         fluid_properties,
                                         mat_properties,
                                         inj_properties,
                                         empty,  # EltTip
                                         empty,  # partlyFilledTip
                                         C,
                                         Fr_lstTmStp.FillF[empty],  # FillFrac
                                         Fr_lstTmStp.EltCrack,
                                         Fr_lstTmStp.InCrack,
                                         LkOff,
                                         empty,  # wTip
                                         timeStep,
                                         Qin,
                                         perfNode,
                                         empty,  # Vel
                                         empty,  # corr_ribbon
                                         doublefracturedictionary=doublefracturedictionary)

    # check if the solution is valid
    if any(isnan.(w_k)) || any(isnan.(p_k))
        exitstatus = 5
        return exitstatus, nothing
    end

    if any(w_k .< 0)
        @warn "Neg width encountered!" _group="JFrac.injection_same_footprint"
    end

    Fr_kplus1 = deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w = w_k
    Fr_kplus1.pFluid = p_k
    Fr_kplus1.pNet = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
    Fr_kplus1.pNet[Fr_lstTmStp.EltCrack] = p_k[Fr_lstTmStp.EltCrack] - mat_properties.SigmaO[Fr_lstTmStp.EltCrack]
    Fr_kplus1.closed = return_data[2]
    Fr_kplus1.v = zeros(Float64, length(Fr_kplus1.EltTip))
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.FractureVolume = sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += LkOff
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - sum(Fr_kplus1.LkOffTotal[Fr_kplus1.EltCrack])) / Fr_kplus1.injectedVol
    Fr_kplus1.effVisc = return_data[1][2]
    Fr_kplus1.G = return_data[1][3]
    fluidVel = return_data[1][1] 

    if length(return_data) > 3
        Fr_kplus1.injectionRate = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
        Fr_kplus1.pInjLine = Fr_lstTmStp.pInjLine + return_data[4]
        Fr_kplus1.injectionRate = return_data[5]
        Fr_kplus1.source = findall(Fr_kplus1.injectionRate .> 0)
        Fr_kplus1.sink = findall(Fr_kplus1.injectionRate .< 0)
    else
        Fr_kplus1.source = Fr_lstTmStp.EltCrack[findall(Qin[Fr_lstTmStp.EltCrack] .> 0)]
        Fr_kplus1.sink = Fr_lstTmStp.EltCrack[findall(Qin[Fr_lstTmStp.EltCrack] .< 0)]
    end

    
    if fluid_properties.turbulence
        if sim_properties.saveReynNumb || sim_properties.saveFluidFlux
            ReNumb, check = turbulence_check_tip(fluidVel, Fr_kplus1, fluid_properties, return_ReyNumb=true)
            if sim_properties.saveReynNumb
                Fr_kplus1.ReynoldsNumber = ReNumb
            end
            if sim_properties.saveFluidFlux
                Fr_kplus1.fluidFlux = ReNumb * 3 / 4 / fluid_properties.density * mean(fluid_properties.viscosity)
            end
        end
        if sim_properties.saveFluidVel
            Fr_kplus1.fluidVelocity = fluidVel
        end
        if sim_properties.saveFluidVelAsVector
            error("saveFluidVelAsVector Not yet implemented")
        end
        if sim_properties.saveFluidFluxAsVector
            error("saveFluidFluxAsVector Not yet implemented")
        end
    else
        if sim_properties.saveFluidFlux || sim_properties.saveFluidVel || sim_properties.saveReynNumb || 
           sim_properties.saveFluidFluxAsVector || sim_properties.saveFluidVelAsVector
            ###todo: re-evaluating these parameters is highly inefficient. They have to be stored if necessary when
            # the solution is evaluated.
            fluid_results = calculate_fluid_flow_characteristics_laminar(Fr_kplus1.w,
                              Fr_kplus1.pFluid,
                              mat_properties.SigmaO,
                              Fr_kplus1.mesh,
                              Fr_kplus1.EltCrack,
                              Fr_kplus1.InCrack,
                              fluid_properties.muPrime,
                              fluid_properties.density)
            
            fluid_flux = fluid_results[1]
            fluid_vel = fluid_results[2]
            Rey_num = fluid_results[3]
            fluid_flux_components = fluid_results[4]
            fluid_vel_components = fluid_results[5]

            if sim_properties.saveFluidFlux
                fflux = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                fflux[:, Fr_kplus1.EltCrack] = fluid_flux
                Fr_kplus1.fluidFlux = fflux
            end

            if sim_properties.saveFluidFluxAsVector
                fflux_components = zeros(Float32, 8, Fr_kplus1.mesh.NumberOfElts)
                fflux_components[:, Fr_kplus1.EltCrack] = fluid_flux_components
                Fr_kplus1.fluidFlux_components = fflux_components
            end

            if sim_properties.saveFluidVel
                fvel = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                fvel[:, Fr_kplus1.EltCrack] = fluid_vel
                Fr_kplus1.fluidVelocity = fvel
            end

            if sim_properties.saveFluidVelAsVector
                fvel_components = zeros(Float32, 8, Fr_kplus1.mesh.NumberOfElts)
                fvel_components[:, Fr_kplus1.EltCrack] = fluid_vel_components
                Fr_kplus1.fluidVelocity_components = fvel_components
            end

            if sim_properties.saveReynNumb
                Rnum = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                Rnum[:, Fr_kplus1.EltCrack] = Rey_num
                Fr_kplus1.ReynoldsNumber = Rnum
            end
        end
    end

    Fr_lstTmStp.closed = return_data[2]

    # check if the solution is valid
    if return_data[3]
        return 14, Fr_kplus1
    end

    exitstatus = 1
    return exitstatus, Fr_kplus1
end


# -----------------------------------------------------------------------------------------------------------------------


"""
    injection_extended_footprint(w_k, Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties,
                                 sim_properties, inj_properties, perfNode=nothing)

This function takes the fracture width from the last iteration of the fracture front loop, calculates the level set
(fracture front position) by inverting the tip asymptote and then solves the ElastoHydrodynamic equations to obtain
the new fracture width.

# Arguments
- `w_k::Vector{Float64}`: the width from last iteration of fracture front.
- `Fr_lstTmStp`: fracture object from the last time step.
- `C::Matrix`: the elasticity matrix.
- `timeStep::Float64`: time step.
- `Qin::Vector{Float64}`: current injection rate.
- `mat_properties`: material properties.
- `fluid_properties`: fluid properties.
- `sim_properties`: simulation parameters.
- `inj_properties`: injection properties.
- `perfNode::Union{IterationProperties, Nothing}`: the IterationProperties object passed to be populated with data.

# Returns
- `Tuple{Int, Union{Fracture, Nothing}}`: (exitstatus, Fracture) - see documentation for possible values.
"""
function injection_extended_footprint(w_k::Vector{Float64}, Fr_lstTmStp, C::Matrix, timeStep::Float64, 
                                     Qin::Vector{Float64}, mat_properties, fluid_properties,
                                     sim_properties, inj_properties, 
                                     perfNode::Union{IterationProperties, Nothing}=nothing)

    @debug "injection_extended_footprint called" _group="JFrac.injection_extended_footprint"
    
    itr = 0
    sgndDist_k = copy(Fr_lstTmStp.sgndDist)

    # toughness iteration loop
    while itr < sim_properties.maxProjItrs
        if sim_properties.paramFromTip || mat_properties.anisotropic_K1c || mat_properties.TI_elasticity
            projection_method = nothing
            if sim_properties.projMethod == "ILSA_orig"
                projection_method = projection_from_ribbon
            elseif sim_properties.projMethod == "LS_grad"
                projection_method = projection_from_ribbon_LS_gradient
            elseif sim_properties.projMethod == "LS_continousfront" #todo: test this case!!!
                projection_method = projection_from_ribbon_LS_gradient
            end
            
            if itr == 0
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   Fr_lstTmStp.EltChannel,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k)
                alpha_ribbon_km1 = zeros(Float64, length(Fr_lstTmStp.EltRibbon))
            else
                alpha_ribbon_k = 0.3 * alpha_ribbon_k + 0.7 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                Fr_lstTmStp.EltChannel,
                                                                                Fr_lstTmStp.mesh,
                                                                                sgndDist_k)
            end
            
            if any(isnan.(alpha_ribbon_k))
                exitstatus = 11
                return exitstatus, nothing
            end
        end

        Kprime_k = nothing
        if sim_properties.paramFromTip || mat_properties.anisotropic_K1c
            Kprime_k = get_toughness_from_cellCenter(alpha_ribbon_k,
                                                     sgndDist_k,
                                                     Fr_lstTmStp.EltRibbon,
                                                     mat_properties,
                                                     Fr_lstTmStp.mesh) * (32 / π)^0.5

            if any(isnan.(Kprime_k))
                exitstatus = 11
                return exitstatus, nothing
            end
        end

        Eprime_k = nothing
        if mat_properties.TI_elasticity
            Eprime_k = TI_plain_strain_modulus(alpha_ribbon_k,
                                               mat_properties.Cij)
            if any(isnan.(Eprime_k))
                exitstatus = 11
                return exitstatus, nothing
            end
        end

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e50 * ones(Float64, Fr_lstTmStp.mesh.NumberOfElts)  # Initializing the cells with extremely


        perfNode_tipInv = instrument_start("tip inversion", perfNode)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(w_k,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               fluid_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k,
                                                               Eprime_k=Eprime_k,
                                                               perfNode=perfNode_tipInv)

        status = true
        fail_cause = nothing
        # if tip inversion returns nan
        if any(isnan.(sgndDist_k[Fr_lstTmStp.EltRibbon]))
            status = false
            fail_cause = "tip inversion failed"
            exitstatus = 7
        end

        if perfNode_tipInv !== nothing
            instrument_close(perfNode, perfNode_tipInv, nothing, length(Fr_lstTmStp.EltRibbon),
                             status, fail_cause, Fr_lstTmStp.time)
            push!(perfNode.tipInv_data, perfNode_tipInv)
        end

        if !status
            return exitstatus, nothing
        end

        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = min.(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                 Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
        front_region = findall(abs.(Fr_lstTmStp.sgndDist) .< current_prefactor * 12.66 * 
                              (Fr_lstTmStp.mesh.hx^2 + Fr_lstTmStp.mesh.hy^2)^0.5)
        
        # the search region outwards from the front position at last time step
        pstv_region = findall(Fr_lstTmStp.sgndDist[front_region] .>= -(Fr_lstTmStp.mesh.hx^2 +
                                                                      Fr_lstTmStp.mesh.hy^2)^0.5)
        # the search region inwards from the front position at last time step
        ngtv_region = findall(Fr_lstTmStp.sgndDist[front_region] .< 0)

        # SOLVE EIKONAL eq via Fast Marching Method to get the distance from tip for each cell.
        SolveFMM(sgndDist_k,
                 Fr_lstTmStp.EltRibbon,
                 Fr_lstTmStp.EltChannel,
                 Fr_lstTmStp.mesh,
                 front_region[pstv_region],
                 front_region[ngtv_region])

        # do it only once if not anisotropic
        if !(sim_properties.paramFromTip || mat_properties.anisotropic_K1c ||
             mat_properties.TI_elasticity) || sim_properties.explicitProjection
            break
        end

        norm = norm(abs.(alpha_ribbon_k - alpha_ribbon_km1) / π * 2)
        if norm < sim_properties.toleranceProjection
            @debug "projection iteration converged after $(itr - 1) iterations; exiting norm $norm" _group="JFrac.injection_extended_footprint"
            break
        end

        alpha_ribbon_km1 = copy(alpha_ribbon_k)
        @debug "iterating on projection... norm $norm" _group="JFrac.injection_extended_footprint"
        itr += 1
    end

    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    EltsTipNew = nothing
    l_k = nothing
    alpha_k = nothing
    CellStatus = nothing
    listofTIPcellsONLY = nothing
    newRibbon = nothing
    zrVertx_k_with_fully_traversed = nothing
    zrVertx_k_without_fully_traversed = nothing
    correct_size_of_pstv_region = nothing
    sgndDist_k_temp = nothing
    Ffront = nothing
    number_of_fronts = nothing
    fronts_dictionary = nothing
    
    if sim_properties.projMethod == "ILSA_orig"
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front(sgndDist_k,
                                                                 front_region,
                                                                 Fr_lstTmStp.EltChannel,
                                                                 Fr_lstTmStp.mesh)
    elseif sim_properties.projMethod == "LS_grad"
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front_LS_gradient(sgndDist_k,
                                                                             front_region,
                                                                             Fr_lstTmStp.EltChannel,
                                                                             Fr_lstTmStp.mesh)
    elseif sim_properties.projMethod == "LS_continousfront"
        correct_size_of_pstv_region = [false, false, false]
        recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = false
        while !correct_size_of_pstv_region[1]
            result = reconstruct_front_continuous(sgndDist_k,
                                                  front_region[pstv_region],
                                                  Fr_lstTmStp.EltRibbon,
                                                  Fr_lstTmStp.EltChannel,
                                                  Fr_lstTmStp.mesh,
                                                  recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                  lstTmStp_EltCrack0=Fr_lstTmStp.fronts_dictionary["crackcells_0"], 
                                                  oldfront=Fr_lstTmStp.Ffront)
            
            EltsTipNew = result[1]
            listofTIPcellsONLY = result[2]
            l_k = result[3]
            alpha_k = result[4]
            CellStatus = result[5]
            newRibbon = result[6]
            zrVertx_k_with_fully_traversed = result[7]
            zrVertx_k_without_fully_traversed = result[8]
            correct_size_of_pstv_region = result[9]
            sgndDist_k_temp = result[10]
            Ffront = result[11]
            number_of_fronts = result[12]
            fronts_dictionary = result[13]
            
            if correct_size_of_pstv_region[3] # correct_size_of_pstv_region[2] в Python, но индексация с 1
                exitstatus = 7 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, nothing
            end

            if correct_size_of_pstv_region[2] # correct_size_of_pstv_region[1] в Python
                Fr_kplus1 = deepcopy(Fr_lstTmStp)
                Fr_kplus1.EltTipBefore = Fr_lstTmStp.EltTip
                Fr_kplus1.EltTip = EltsTipNew  # !!! EltsTipNew are the intersection between the fictitius cells and the frontlist as tip in order to decide the direction of remeshing
                # (in case of anisotropic remeshing)
                exitstatus = 12 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, Fr_kplus1
            end

            if !correct_size_of_pstv_region[1]
                # Expand the
                # - front region by 1 cell tickness
                # - pstv_region by 1 cell tickness
                # - ngtv_region by 1 cell tickness

                front_region = unique(vec(Fr_lstTmStp.mesh.NeiElements[front_region, :]))

                # the search region outwards from the front position at last time step
                pstv_region = findall(Fr_lstTmStp.sgndDist[front_region] .>= -(Fr_lstTmStp.mesh.hx^2 +
                                                                               Fr_lstTmStp.mesh.hy^2)^0.5)
                # the search region inwards from the front position at last time step
                ngtv_region = findall(Fr_lstTmStp.sgndDist[front_region] .< 0)

                # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
                SolveFMM(sgndDist_k,
                         Fr_lstTmStp.EltTip,
                         Fr_lstTmStp.EltCrack,
                         Fr_lstTmStp.mesh,
                         front_region[pstv_region],
                         front_region[ngtv_region])
            end
        end
        sgndDist_k = sgndDist_k_temp
    else
        error("projection method not supported")
    end

    if !any(in.(EltsTipNew, Ref(front_region)))
        error("The tip elements are not in the band. Increase the size of the band for FMM to evaluate level set.")
    end

    # If the angle and length of the perpendicular are not correct
    nan = (isnan.(alpha_k) .| isnan.(l_k))
    if any(nan) || any(l_k .< 0) || any(alpha_k .< 0) || any(alpha_k .> π / 2)
        exitstatus = 3
        return exitstatus, nothing
    end

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    if length(intersect(Fr_lstTmStp.mesh.Frontlist, EltsTipNew)) > 0
        Fr_lstTmStp.EltTipBefore = Fr_lstTmStp.EltTip
        Fr_lstTmStp.EltTip = EltsTipNew
        exitstatus = 12
        return exitstatus, Fr_lstTmStp
    end

    # generate the InCrack array for the current front position
    InCrack_k = zeros(Int8, Fr_lstTmStp.mesh.NumberOfElts)
    InCrack_k[Fr_lstTmStp.EltChannel] .= 1
    InCrack_k[EltsTipNew] .= 1  #EltsTipNew is new tip + fully traversed

    if sum(InCrack_k .== 1) > sim_properties.maxElementIn && sim_properties.meshReductionPossible
        exitstatus = 16
        return exitstatus, Fr_lstTmStp
    end

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = Integral_over_cell(EltsTipNew,
                                    alpha_k,
                                    l_k,
                                    Fr_lstTmStp.mesh,
                                    "A",
                                    projMethod=sim_properties.projMethod) / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-4 (up to 4 figures)
    fill_frac_condition = (FillFrac_k .> 1.0) .& (FillFrac_k .< 1 + 1e-4)
    FillFrac_k[fill_frac_condition] .= 1.0

    # if filling fraction is below zero or above 1+1e-4
    if any(FillFrac_k .> 1.0) || any(FillFrac_k .< 0.0 - eps())
        exitstatus = 9
        return exitstatus, nothing
    end

    # Evaluate the element lists for the trial fracture front
    EltChannel_k = nothing
    EltTip_k = nothing
    EltCrack_k = nothing
    EltRibbon_k = nothing
    zrVertx_k = nothing
    CellStatus_k = nothing
    fully_traversed_k = nothing
    
    if sim_properties.projMethod != "LS_continousfront"
        # todo: some of the list are redundant to calculate on each iteration
        # Evaluate the element lists for the trial fracture front
        result = UpdateLists(Fr_lstTmStp.EltChannel,
                             EltsTipNew,
                             FillFrac_k,
                             sgndDist_k,
                             Fr_lstTmStp.mesh)
        EltChannel_k = result[1]
        EltTip_k = result[2]
        EltCrack_k = result[3]
        EltRibbon_k = result[4]
        zrVertx_k = result[5]
        CellStatus_k = result[6]
        fully_traversed_k = result[7]
    elseif sim_properties.projMethod == "LS_continousfront"
        zrVertx_k = zrVertx_k_without_fully_traversed
        result = UpdateListsFromContinuousFrontRec(newRibbon,
                                                   sgndDist_k,
                                                   Fr_lstTmStp.EltChannel,
                                                   EltsTipNew,
                                                   listofTIPcellsONLY,
                                                   Fr_lstTmStp.mesh)

        EltChannel_k = result[1]
        EltTip_k = result[2]
        EltCrack_k = result[3]
        EltRibbon_k = result[4]
        CellStatus_k = result[5]
        fully_traversed_k = result[6]

        if any(isnan.(EltChannel_k))
            exitstatus = 3
            return exitstatus, nothing
        end
    end

    # EletsTipNew may contain fully filled elements also. Identifying only the partially filled elements
    partlyFilledTip = findall(in.(EltsTipNew, Ref(EltTip_k)))
    @debug "Solving the EHL system with the new trial footprint" _group="JFrac.injection_extended_footprint"

    zrVrtx_newTip = nothing
    if sim_properties.projMethod != "LS_continousfront"
        # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
    else 
        zrVrtx_newTip = transpose(zrVertx_k_with_fully_traversed)
    end

    # finding ribbon cells corresponding to tip cells
    corr_ribbon = find_corresponding_ribbon_cell(EltsTipNew,
                                                 alpha_k,
                                                 zrVrtx_newTip,
                                                 Fr_lstTmStp.mesh)
    Cprime_tip = mat_properties.Cprime[corr_ribbon]

    # Calculating toughness at tip to be used to calculate the volume integral in the tip cells
    Kprime_tip = nothing
    if sim_properties.paramFromTip || mat_properties.anisotropic_K1c
        if sim_properties.projMethod != "LS_continousfront"
            # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
            zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
        else 
            zrVrtx_newTip = transpose(zrVertx_k_with_fully_traversed)
        end

        # get toughness from tip in case of anisotropic or
        Kprime_tip = (32 / π)^0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                  Fr_lstTmStp.mesh,
                                                                  mat_properties,
                                                                  alpha_k,
                                                                  l_k,
                                                                  zrVrtx_newTip)
    else
        Kprime_tip = mat_properties.Kprime[corr_ribbon]
    end

    Eprime_tip = nothing
    if mat_properties.TI_elasticity
        Eprime_tip = TI_plain_strain_modulus(alpha_k,
                                             mat_properties.Cij)
    else
        Eprime_tip = fill(mat_properties.Eprime, length(EltsTipNew))
    end

    if perfNode !== nothing
        perfNode_wTip = instrument_start("nonlinear system solve", perfNode)
    end

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = (-(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) /
                (Fr_lstTmStp.mesh.hx^2 + Fr_lstTmStp.mesh.hy^2)^0.5 .< sim_properties.toleranceVStagnant)
    
    if perfNode !== nothing
        perfNode_tipWidth = instrument_start("tip width", perfNode)
        #todo close tip width instrumentation
    end

    wTip = nothing
    if any(stagnant)
        # if any tip cell with stagnant front calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(w_k,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        Eprime=Eprime_tip)

        # todo: Find the right cause of failure
        # if the stress Intensity factor cannot be found. The most common reason is wiggles in the front resulting
        # in isolated tip cells.
        if any(isnan.(KIPrime))
            exitstatus = 8
            return exitstatus, nothing
        end

        # Calculate average width in the tip cells by integrating tip asymptote. Width of stagnant cells are calculated
        # using the stress intensity factor (see Dontsov and Peirce, JFM RAPIDS, 2017)
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  stagnant=stagnant,
                                  KIPrime=KIPrime,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip) / Fr_lstTmStp.mesh.EltArea
    else
        # Calculate average width in the tip cells by integrating tip asymptote
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  Kprime=Kprime_tip,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip,
                                  stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea
    end

    # check if the tip volume has gone into negative
    smallNgtvWTip = findall((wTip .< 0) .& (wTip .> -1e-4 * mean(wTip)))
    if length(smallNgtvWTip) > 0
        #  warnings.warn("Small negative volume integral(s) received, ignoring "..repr(wTip[smallngtvwTip])..' ...')
        wTip[smallNgtvWTip] = abs.(wTip[smallNgtvWTip])
    end

    if any(wTip .< 0) || sum(wTip) == 0.0
        exitstatus = 4
        return exitstatus, nothing
    end

    if perfNode !== nothing
        pass
        # todo close tip width instrumentation
    end

    LkOff = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
    if sum(mat_properties.Cprime[EltsTipNew]) > 0
        # Calculate leak-off term for the tip cell
        LkOff[EltsTipNew] = 2 * mat_properties.Cprime[EltsTipNew] * Integral_over_cell(EltsTipNew,
                                                                                       alpha_k,
                                                                                       l_k,
                                                                                       Fr_lstTmStp.mesh,
                                                                                       "Lk",
                                                                                       mat_prop=mat_properties,
                                                                                       frac=Fr_lstTmStp,
                                                                                       Vel=Vel_k,
                                                                                       dt=timeStep,
                                                                                       arrival_t=
                                                                                       Fr_lstTmStp.TarrvlZrVrtx[
                                                                                           EltsTipNew])
    end

    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0
        # todo: no need to evaluate on each iteration. Need to decide. Evaluating here for now for better readability
        t_lst_min_t0 = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_lst_min_t0[t_lst_min_t0 .< 0.0] .= 0.0
        t_min_t0 = t_lst_min_t0 + timeStep
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * (
                t_min_t0 .^ 0.5 - t_lst_min_t0 .^ 0.5) * Fr_lstTmStp.mesh.EltArea
        if any(stagnant)
            LkOff[EltsTipNew[stagnant]] = leak_off_stagnant_tip(EltsTipNew[stagnant],
                                                                l_k[stagnant],
                                                                alpha_k[stagnant],
                                                                Fr_lstTmStp.TarrvlZrVrtx[EltsTipNew[stagnant]],
                                                                Fr_lstTmStp.time + timeStep,
                                                                mat_properties.Cprime,
                                                                timeStep,
                                                                Fr_lstTmStp.mesh)
        end
    end

    # set leak off to zero if pressure below pore pressure
    LkOff[Fr_lstTmStp.pFluid .<= mat_properties.porePressure] .= 0.0

    if any(isnan.(LkOff[EltsTipNew]))
        exitstatus = 13
        return exitstatus, nothing
    end
    
    doublefracturedictionary = Dict{String, Any}()
    if sim_properties.doublefracture && fronts_dictionary["number_of_fronts"] == 2
        doublefracturedictionary["number_of_fronts"] = fronts_dictionary["number_of_fronts"]
        doublefracturedictionary["crackcells_0"] = fronts_dictionary["crackcells_0"]
        doublefracturedictionary["crackcells_1"] = fronts_dictionary["crackcells_1"]
        doublefracturedictionary["TIPcellsANDfullytrav_0"] = fronts_dictionary["TIPcellsANDfullytrav_0"]
        doublefracturedictionary["TIPcellsANDfullytrav_1"] = fronts_dictionary["TIPcellsANDfullytrav_1"]
    elseif sim_properties.projMethod != "LS_continousfront"
        doublefracturedictionary["number_of_fronts"] = 1
    else
        doublefracturedictionary["number_of_fronts"] = fronts_dictionary["number_of_fronts"]
    end

    result = solve_width_pressure(Fr_lstTmStp,
                                  sim_properties,
                                  fluid_properties,
                                  mat_properties,
                                  inj_properties,
                                  EltsTipNew,
                                  partlyFilledTip,
                                  C,
                                  FillFrac_k,
                                  EltCrack_k,
                                  InCrack_k,
                                  LkOff,
                                  wTip,
                                  timeStep,
                                  Qin,
                                  perfNode,
                                  Vel_k,
                                  corr_ribbon,
                                  doublefracturedictionary=doublefracturedictionary)

    w_n_plus1 = result[1]
    pf_n_plus1 = result[2]
    data = result[3]

    # check if the new width is valid
    if any(isnan.(w_n_plus1))
        exitstatus = 5
        return exitstatus, nothing
    end

    fluidVel = data[1][1]
    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = copy(Fr_lstTmStp.Tarrival)
    max_Tarrival = nanmax(Tarrival_k)
    nc = setdiff(EltChannel_k, Fr_lstTmStp.EltChannel)
    new_channel = Int[]
    for i in nc
        append!(new_channel, findall(EltsTipNew .== i))
    end
    
    if !isempty(new_channel)
        t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] ./ Vel_k[new_channel]
        max_l = Fr_lstTmStp.mesh.hx * cos.(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * sin.(alpha_k[new_channel])
        t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) ./ Vel_k[new_channel]
        Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave) / 2
        to_correct = findall(Tarrival_k[EltsTipNew[new_channel]] .< max_Tarrival)
        Tarrival_k[EltsTipNew[new_channel[to_correct]]] = max_Tarrival
    end

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = deepcopy(Fr_lstTmStp)
    Fr_kplus1.time += timeStep
    Fr_kplus1.w = w_n_plus1
    Fr_kplus1.pFluid = pf_n_plus1
    Fr_kplus1.pNet = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
    Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k]
    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.fully_traversed = fully_traversed_k
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.v = Vel_k[partlyFilledTip]
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    Fr_kplus1.InCrack = InCrack_k
    
    if sim_properties.projMethod != "LS_continousfront"
        Fr_kplus1.process_fracture_front()
    else
        Fr_kplus1.fronts_dictionary = fronts_dictionary
        Fr_kplus1.Ffront = Ffront
        Fr_kplus1.number_of_fronts = number_of_fronts
        if sim_properties.saveToDisk && sim_properties.saveStatisticsPostCoalescence && Fr_lstTmStp.number_of_fronts != Fr_kplus1.number_of_fronts
            myJsonName = sim_properties.set_outputFolder * "_mesh_study.json"
            append_to_json_file(myJsonName, Fr_kplus1.mesh.nx, "append2keyAND2list", key="nx")
            append_to_json_file(myJsonName, Fr_kplus1.mesh.ny, "append2keyAND2list", key="ny")
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hx, "append2keyAND2list", key="hx")
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hy, "append2keyAND2list", key="hy")
            append_to_json_file(myJsonName, length(Fr_kplus1.EltCrack), "append2keyAND2list", key="elements_in_crack")
            append_to_json_file(myJsonName, length(Fr_kplus1.EltTip), "append2keyAND2list", key="elements_in_tip")
            append_to_json_file(myJsonName, Fr_kplus1.time, "append2keyAND2list", key="coalescence_time")
        end
    end

    Fr_kplus1.FractureVolume = sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.Tarrival = Tarrival_k
    new_tip = findall(isnan.(Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip]))
    if !isempty(new_tip)
        Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip[new_tip]] = Fr_kplus1.time - Fr_kplus1.l[new_tip] ./ Fr_kplus1.v[new_tip]
    end
    Fr_kplus1.wHist = max.(Fr_kplus1.w, Fr_lstTmStp.wHist)
    Fr_kplus1.closed = data[2]
    tip_neg_rib = Int[]
    # adding tip cells with closed corresponding ribbon cells to the list of closed cells
    for i in 1:length(Fr_kplus1.EltTip)
        elem = Fr_kplus1.EltTip[i]
        if corr_ribbon[i] in Fr_kplus1.closed && !(elem in Fr_kplus1.closed)
            push!(tip_neg_rib, elem)
        end
    end
    Fr_kplus1.closed = vcat(Fr_kplus1.closed, tip_neg_rib)
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - Fr_kplus1.LkOffTotal) / Fr_kplus1.injectedVol

    if sim_properties.saveRegime
        Fr_kplus1.update_tip_regime(mat_properties, fluid_properties, timeStep)
    end

    Fr_kplus1.effVisc = data[1][2]
    Fr_kplus1.G = data[1][3]

    if length(data) > 3
        Fr_kplus1.injectionRate = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
        Fr_kplus1.pInjLine = Fr_lstTmStp.pInjLine + data[4]
        Fr_kplus1.injectionRate = data[5]
        Fr_kplus1.source = findall(Fr_kplus1.injectionRate .> 0)
        Fr_kplus1.sink = findall(Fr_kplus1.injectionRate .< 0)
    else
        Fr_kplus1.source = Fr_lstTmStp.EltCrack[findall(Qin[Fr_lstTmStp.EltCrack] .> 0)]
        Fr_kplus1.sink = Fr_lstTmStp.EltCrack[findall(Qin[Fr_lstTmStp.EltCrack] .< 0)]
    end

    if fluid_properties.turbulence
        if sim_properties.saveReynNumb || sim_properties.saveFluidFlux
            ReNumb, check = turbulence_check_tip(fluidVel, Fr_kplus1, fluid_properties, return_ReyNumb=true)
            if sim_properties.saveReynNumb
                Fr_kplus1.ReynoldsNumber = ReNumb
            end
            if sim_properties.saveFluidFlux
                Fr_kplus1.fluidFlux = ReNumb * 3 / 4 / fluid_properties.density * mean(fluid_properties.viscosity)
            end
        end
        if sim_properties.saveFluidVel
            Fr_kplus1.fluidVelocity = fluidVel
        end
        if sim_properties.saveFluidVelAsVector
            error("saveFluidVelAsVector Not yet implemented")
        end
        if sim_properties.saveFluidFluxAsVector
            error("saveFluidFluxAsVector Not yet implemented")
        end
    else
        if sim_properties.saveFluidFlux || sim_properties.saveFluidVel || sim_properties.saveReynNumb || sim_properties.saveFluidFluxAsVector || sim_properties.saveFluidVelAsVector
            ###todo: re-evaluating these parameters is highly inefficient. They have to be stored if neccessary when
            # the solution is evaluated.
            fluid_results = calculate_fluid_flow_characteristics_laminar(Fr_kplus1.w,
                                                                        Fr_kplus1.pFluid,
                                                                        mat_properties.SigmaO,
                                                                        Fr_kplus1.mesh,
                                                                        Fr_kplus1.EltCrack,
                                                                        Fr_kplus1.InCrack,
                                                                        fluid_properties.muPrime,
                                                                        fluid_properties.density)

            fluid_flux = fluid_results[1]
            fluid_vel = fluid_results[2]
            Rey_num = fluid_results[3]
            fluid_flux_components = fluid_results[4]
            fluid_vel_components = fluid_results[5]

            if sim_properties.saveFluidFlux
                fflux = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                fflux[:, Fr_kplus1.EltCrack] = fluid_flux
                Fr_kplus1.fluidFlux = fflux
            end

            if sim_properties.saveFluidFluxAsVector
                fflux_components = zeros(Float32, 8, Fr_kplus1.mesh.NumberOfElts)
                fflux_components[:, Fr_kplus1.EltCrack] = fluid_flux_components
                Fr_kplus1.fluidFlux_components = fflux_components
            end

            if sim_properties.saveFluidVel
                fvel = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                fvel[:, Fr_kplus1.EltCrack] = fluid_vel
                Fr_kplus1.fluidVelocity = fvel
            end

            if sim_properties.saveFluidVelAsVector
                fvel_components = zeros(Float32, 8, Fr_kplus1.mesh.NumberOfElts)
                fvel_components[:, Fr_kplus1.EltCrack] = fluid_vel_components
                Fr_kplus1.fluidVelocity_components = fvel_components
            end

            if sim_properties.saveReynNumb
                Rnum = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                Rnum[:, Fr_kplus1.EltCrack] = Rey_num
                Fr_kplus1.ReynoldsNumber = Rnum
            end
        end
    end

    if data[3]
        return 14, Fr_kplus1
    end

    exitstatus = 1
    return exitstatus, Fr_kplus1
end

function nanmax(arr)
    valid_values = filter(!isnan, arr)
    return isempty(valid_values) ? NaN : maximum(valid_values)
end


# ----------------------------------------------------------------------------------------------------------------------


"""
    solve_width_pressure(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, inj_properties, EltTip,
                         partlyFilledTip, C, FillFrac, EltCrack, InCrack, LkOff, wTip, timeStep, Qin, perfNode,
                         Vel, corr_ribbon, doublefracturedictionary=nothing)

This function evaluates the width and pressure by constructing and solving the coupled elasticity and fluid flow
equations. The system of equations are formed according to the type of solver given in the simulation properties.

# Arguments
- `Fr_lstTmStp`: fracture object from the last time step.
- `sim_properties`: simulation properties.
- `fluid_properties`: fluid properties.
- `mat_properties`: material properties.
- `inj_properties`: injection properties.
- `EltTip::Vector{Int}`: tip elements.
- `partlyFilledTip::Vector{Int}`: partly filled tip elements.
- `C::Matrix`: elasticity matrix.
- `FillFrac::Vector{Float64}`: filling fraction.
- `EltCrack::Vector{Int}`: crack elements.
- `InCrack::Vector{Int8}`: in-crack indicator.
- `LkOff::Vector{Float64}`: leak-off.
- `wTip::Vector{Float64}`: tip width.
- `timeStep::Float64`: time step.
- `Qin::Vector{Float64}`: injection rate.
- `perfNode`: performance node.
- `Vel::Vector{Float64}`: velocity.
- `corr_ribbon::Vector{Int}`: corresponding ribbon.
- `doublefracturedictionary::Union{Dict, Nothing}`: double fracture dictionary.

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}, Tuple}`: (width, pressure, return_data)
"""
function solve_width_pressure(Fr_lstTmStp, sim_properties, fluid_properties, mat_properties, inj_properties, EltTip::Vector{Int},
                             partlyFilledTip::Vector{Int}, C::Matrix, FillFrac::Vector{Float64}, EltCrack::Vector{Int}, 
                             InCrack::Vector{Int8}, LkOff::Vector{Float64}, wTip::Vector{Float64}, timeStep::Float64, 
                             Qin::Vector{Float64}, perfNode, Vel::Vector{Float64}, corr_ribbon::Vector{Int}, 
                             doublefracturedictionary::Union{Dict, Nothing}=nothing)

    @debug "solve_width_pressure called" _group="JFrac.solve_width_pressure"
    
    if sim_properties.get_volumeControl()
        if sim_properties.symmetric && !sim_properties.useBlockToeplizCompression
            try
                if !hasfield(typeof(Fr_lstTmStp.mesh), :corresponding)
                    error("Symmetric fracture needs symmetric mesh. Set symmetric flag to True\nwhile initializing the mesh")
                end
                Fr_lstTmStp.mesh.corresponding[Fr_lstTmStp.EltChannel]
            catch e
                error("Symmetric fracture needs symmetric mesh. Set symmetric flag to True\nwhile initializing the mesh")
            end

            EltChannel_sym = unique(Fr_lstTmStp.mesh.corresponding[Fr_lstTmStp.EltChannel])
            EltTip_sym = unique(Fr_lstTmStp.mesh.corresponding[EltTip])

            wTip_sym = zeros(Float64, length(EltTip_sym))
            wTip_sym_elts = Fr_lstTmStp.mesh.activeSymtrc[EltTip_sym]
            
            for i in 1:length(EltTip_sym)
                if sum(EltTip .== wTip_sym_elts[i]) != 1
                    other_corr = get_symetric_elements(Fr_lstTmStp.mesh, [wTip_sym_elts[i]])
                    found = false
                    for j in 1:4
                        in_tip = findall(EltTip .== other_corr[1][j])
                        if length(in_tip) > 0
                            wTip_sym[i] = wTip[in_tip[1]]
                            found = true
                            break
                        end
                    end
                    if !found
                        wTip_sym[i] = 0.0
                    end
                else
                    wTip_sym[i] = wTip[findall(EltTip .== wTip_sym_elts[i])[1]]
                end
            end

            dwTip = wTip - Fr_lstTmStp.w[EltTip]
            A, b = MakeEquationSystem_volumeControl_symmetric(Fr_lstTmStp.w,
                                                              wTip_sym,
                                                              EltChannel_sym,
                                                              EltTip_sym,
                                                              C,
                                                              timeStep,
                                                              Qin,
                                                              mat_properties.SigmaO,
                                                              Fr_lstTmStp.mesh.EltArea,
                                                              LkOff,
                                                              Fr_lstTmStp.mesh.volWeights,
                                                              Fr_lstTmStp.mesh.activeSymtrc,
                                                              dwTip)
        else
            if sim_properties.doublefracture && doublefracturedictionary["number_of_fronts"] == 2
                # compute the channel from the last time step for the two fractures
                EltChannelFracture0 = setdiff(Fr_lstTmStp.fronts_dictionary["crackcells_0"], Fr_lstTmStp.fronts_dictionary["TIPcellsONLY_0"])
                EltChannelFracture1 = setdiff(Fr_lstTmStp.fronts_dictionary["crackcells_1"], Fr_lstTmStp.fronts_dictionary["TIPcellsONLY_1"])
                
                if length(EltTip) == 0
                    EltTipFracture0 = EltTip
                    EltTipFracture1 = EltTip
                else
                    EltTipFracture0 = doublefracturedictionary["TIPcellsANDfullytrav_0"]
                    EltTipFracture1 = doublefracturedictionary["TIPcellsANDfullytrav_1"]
                end
                
                wtipindexFR0 = findall(in.(EltTip, Ref(EltTipFracture0)))
                wtipindexFR1 = findall(in.(EltTip, Ref(EltTipFracture1)))
                wTipFR0 = wTip[wtipindexFR0]
                wTipFR1 = wTip[wtipindexFR1]
                QinFR0 = Qin[EltChannelFracture0]
                QinFR1 = Qin[EltChannelFracture1]

                # CARLO: I check if can be possible to have a Channel to be tip
                if any(in.(vcat(EltChannelFracture0, EltChannelFracture1), Ref(vcat(EltTipFracture0, EltTipFracture1))))
                    error("Some of the tip cells are also channel cells. This was not expected. If you allow that you should implement the tip filling fraction correction for element in the tip region")
                end

                A, b = MakeEquationSystem_volumeControl_double_fracture(Fr_lstTmStp.w,
                                                                        wTipFR0,
                                                                        wTipFR1,
                                                                        EltChannelFracture0,
                                                                        EltChannelFracture1,
                                                                        EltTipFracture0,
                                                                        EltTipFracture1,
                                                                        mat_properties.SigmaO,
                                                                        C,
                                                                        timeStep,
                                                                        QinFR0,
                                                                        QinFR1,
                                                                        Fr_lstTmStp.mesh.EltArea,
                                                                        LkOff)
            else
                # CARLO: I check if can be possible to have a Channel to be tip
                if any(in.(Fr_lstTmStp.EltChannel, Ref(EltTip)))
                    error("Some of the tip cells are also channel cells. This was not expected. If you allow that you should implement the tip filling fraction correction for element in the tip region")
                end
                
                A, b = MakeEquationSystem_volumeControl(Fr_lstTmStp.w,
                                                    wTip,
                                                    Fr_lstTmStp.EltChannel,
                                                    EltTip,
                                                    mat_properties.SigmaO,
                                                    C,
                                                    timeStep,
                                                    Qin,
                                                    Fr_lstTmStp.mesh.EltArea,
                                                    LkOff)
            end
        end

        perfNode_nonLinSys = instrument_start("nonlinear system solve", perfNode)
        perfNode_widthConstrItr = instrument_start("width constraint iteration", perfNode_nonLinSys)
        perfNode_linSys = instrument_start("linear system solve", perfNode_widthConstrItr)
        
        status = true
        fail_cause = nothing
        sol = nothing
        
        try
            sol = A \ b
        catch e
            if isa(e, LinearAlgebra.SingularException)
                status = false
                fail_cause = "singular matrix"
            else
                rethrow(e)
            end
        end

        if perfNode !== nothing
            instrument_close(perfNode_widthConstrItr, perfNode_linSys, nothing,
                             length(b), status, fail_cause, Fr_lstTmStp.time)
            push!(perfNode_widthConstrItr.linearSolve_data, perfNode_linSys)

            instrument_close(perfNode_nonLinSys, perfNode_widthConstrItr, nothing,
                             length(b), status, fail_cause, Fr_lstTmStp.time)
            push!(perfNode_nonLinSys.widthConstraintItr_data, perfNode_widthConstrItr)

            instrument_close(perfNode, perfNode_nonLinSys, nothing, length(b), status, fail_cause, Fr_lstTmStp.time)
            push!(perfNode.nonLinSolve_data, perfNode_nonLinSys)
        end

        # equate other three quadrants to the evaluated quadrant
        if sim_properties.symmetric
            del_w = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
            for i in 1:(length(sol) - 1)
                del_w[Fr_lstTmStp.mesh.symmetricElts[Fr_lstTmStp.mesh.activeSymtrc[EltChannel_sym[i]]]] = sol[i]
            end
            w = copy(Fr_lstTmStp.w)
            w[Fr_lstTmStp.EltChannel] += del_w[Fr_lstTmStp.EltChannel]
            for i in 1:length(wTip_sym_elts)
                w[Fr_lstTmStp.mesh.symmetricElts[wTip_sym_elts[i]]] = wTip_sym[i]
            end
        else
            w = copy(Fr_lstTmStp.w)
            if sim_properties.doublefracture && doublefracturedictionary["number_of_fronts"] == 2
                w[EltChannelFracture0] += sol[1:length(EltChannelFracture0)]
                w[EltChannelFracture1] += sol[(length(EltChannelFracture0)+1):(length(EltChannelFracture0)+length(EltChannelFracture1))]
                w[EltTipFracture0] = wTipFR0
                w[EltTipFracture1] = wTipFR1
            else
                w[Fr_lstTmStp.EltChannel] += sol[1:length(Fr_lstTmStp.EltChannel)]
                w[EltTip] = wTip
            end
        end

        p = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
        if sim_properties.doublefracture && doublefracturedictionary["number_of_fronts"] == 2
            p[doublefracturedictionary["crackcells_0"]] = sol[end-1]
            p[doublefracturedictionary["crackcells_1"]] = sol[end]
        else
            p[EltCrack] = sol[end]
        end

        return_data_solve = (nothing, nothing, nothing)
        return_data = (return_data_solve, Int[], Int[])
        return w, p, return_data
    end

    if sim_properties.get_viscousInjection()
        # velocity at the cell edges evaluated with the guess width. Used as guess
        # values for the implicit velocity solver.
        vk = zeros(Float64, 4, Fr_lstTmStp.mesh.NumberOfElts)
        if fluid_properties.turbulence
            wguess = copy(Fr_lstTmStp.w)
            wguess[EltTip] = wTip

            vk = velocity(wguess,
                          EltCrack,
                          Fr_lstTmStp.mesh,
                          InCrack,
                          Fr_lstTmStp.muPrime,
                          C,
                          mat_properties.SigmaO)
        end

        perfNode_nonLinSys = instrument_start("nonlinear system solve", perfNode)

        neg = Fr_lstTmStp.closed
        wc_to_impose = Fr_lstTmStp.w[neg]
        
        new_neg = Int[]
        active_contraint = true
        to_solve = setdiff(EltCrack, EltTip)  # only taking channel elements to solve

        # adding stagnant tip cells to the cells which are solved. This adds stability as the elasticity is also
        # solved for the stagnant tip cells as compared to tip cells which are moving.
        if sim_properties.solveStagnantTip
            stagnant_tip = findall(Vel .< 1e-10)
        else
            stagnant_tip = Int[]
        end
        to_impose = setdiff(EltTip, EltTip[stagnant_tip])
        imposed_val = wTip[setdiff(1:length(wTip), stagnant_tip)]
        to_solve = vcat(to_solve, EltTip[stagnant_tip])

        fully_closed = false
        to_open_cumm = Int[]
        
        # Making and solving the system of equations. The width constraint is checked. If active, system is remade with
        # the constraint imposed and is resolved.

        while active_contraint
            perfNode_widthConstrItr = instrument_start("width constraint iteration", perfNode_nonLinSys)

            to_solve_k = setdiff(to_solve, neg)
            comm_result = intersect(neg, to_impose, sorted=true)
            comm_neg = comm_result[2]
            comm_to_impose = comm_result[3]
            to_impose_k = deleteat!(copy(to_impose), comm_to_impose)
            imposed_val_k = deleteat!(copy(imposed_val), comm_to_impose)

            EltCrack_k = vcat(to_solve_k, neg)
            EltCrack_k = vcat(EltCrack_k, to_impose_k)

            # The code below finds the indices(in the EltCrack list) of the neighbours of all the cells in the crack.
            # This is done to avoid costly slicing of the large numpy arrays while making the linear system during the
            # fixed point iterations. For neighbors that are outside the fracture, len(EltCrack) + 1 is returned.
            corr_nei = fill(length(EltCrack_k), length(EltCrack_k), 4)
            for (i, elem) in enumerate(EltCrack_k)
                corresponding = findall(EltCrack_k .== Fr_lstTmStp.mesh.NeiElements[elem, 1]) # left
                if length(corresponding) > 0
                    corr_nei[i, 1] = corresponding[1]
                end
                corresponding = findall(EltCrack_k .== Fr_lstTmStp.mesh.NeiElements[elem, 2]) # right
                if length(corresponding) > 0
                    corr_nei[i, 2] = corresponding[1]
                end
                corresponding = findall(EltCrack_k .== Fr_lstTmStp.mesh.NeiElements[elem, 3]) # bottom
                if length(corresponding) > 0
                    corr_nei[i, 3] = corresponding[1]
                end
                corresponding = findall(EltCrack_k .== Fr_lstTmStp.mesh.NeiElements[elem, 4]) # up
                if length(corresponding) > 0
                    corr_nei[i, 4] = corresponding[1]
                end
            end

            lst_edgeInCrk = nothing
            if fluid_properties.rheology in ["Herschel-Bulkley", "HBF", "power law", "PLF"]
                lst_edgeInCrk = [
                    findall(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 1]]), # left
                    findall(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 2]]), # right
                    findall(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 3]]), # bottom
                    findall(InCrack[Fr_lstTmStp.mesh.NeiElements[EltCrack_k, 4]])  # up
                ]
            end

            arg = (
                EltCrack_k,
                to_solve_k,
                to_impose_k,
                imposed_val_k,
                wc_to_impose,
                Fr_lstTmStp,
                fluid_properties,
                mat_properties,
                sim_properties,
                timeStep,
                Qin,
                C,
                InCrack,
                LkOff,
                neg,
                corr_nei,
                lst_edgeInCrk)

            w_guess = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
            avg_dw = (sum(Qin) * timeStep / Fr_lstTmStp.mesh.EltArea - sum(
                    imposed_val_k - Fr_lstTmStp.w[to_impose_k])) / length(to_solve_k)
            w_guess[to_solve_k] = Fr_lstTmStp.w[to_solve_k] #+ avg_dw
            w_guess[to_impose_k] = imposed_val_k
            pf_guess_neg = C[neg, EltCrack_k] * w_guess[EltCrack_k] + mat_properties.SigmaO[neg]
            pf_guess_tip = C[to_impose_k, EltCrack_k] * w_guess[EltCrack_k] + mat_properties.SigmaO[to_impose_k]

            sys_fun = nothing
            guess = nothing
            
            if sim_properties.elastohydrSolver == "implicit_Picard" || sim_properties.elastohydrSolver == "implicit_Anderson"
                if inj_properties.modelInjLine
                    sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse_injection_line
                    guess = vcat(
                        fill(avg_dw, length(to_solve_k)),
                        pf_guess_neg - Fr_lstTmStp.pFluid[neg],
                        pf_guess_tip - Fr_lstTmStp.pFluid[to_impose_k])
                elseif sim_properties.solveDeltaP
                    if sim_properties.solveSparse
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP_sparse
                    else
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_deltaP
                    end
                    guess = vcat(
                        fill(avg_dw, length(to_solve_k)),
                        pf_guess_neg - Fr_lstTmStp.pFluid[neg],
                        pf_guess_tip - Fr_lstTmStp.pFluid[to_impose_k])
                else
                    if sim_properties.solveSparse
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted_sparse
                    else
                        sys_fun = MakeEquationSystem_ViscousFluid_pressure_substituted
                    end
                    guess = vcat(
                        fill(avg_dw, length(to_solve_k)),
                        pf_guess_neg,
                        pf_guess_tip)
                end

                inter_itr_init = [vk, Int[], nothing]

                if inj_properties.modelInjLine
                    inj_cells = intersect(inj_properties.sourceElem, Fr_lstTmStp.EltChannel)
                    sink_cells = intersect(Array{Int}(inj_properties.sinkElem), Fr_lstTmStp.EltCrack)

                    sink = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
                    if !isempty(inj_properties.sinkElem)
                        sink[inj_properties.sinkElem] = inj_properties.sinkVel * Fr_lstTmStp.mesh.EltArea
                    end

                    indxCurTime = maximum(findall(Fr_lstTmStp.time + timeStep .>= inj_properties.injectionRate[1, :]))
                    currentRate = inj_properties.injectionRate[2, indxCurTime]  # current injection rate

                    inj_ch = intersect(inj_cells, to_solve_k)
                    inj_act = intersect(inj_cells, neg)
                    inj_in_ch = Int[]
                    inj_in_act = Int[]
                    for m in inj_ch
                        push!(inj_in_ch, findfirst(to_solve_k .== m)[1])
                    end
                    for m in inj_act
                        push!(inj_in_act, findfirst(neg .== m)[1])
                    end

                    arg = (arg, inj_properties, inj_ch, inj_act, sink_cells, Fr_lstTmStp.pInjLine,
                           inj_in_ch, inj_in_act, currentRate, sink)

                    guess_il = zeros(Float64, length(inj_cells) + 1)
                    guess = vcat(guess, guess_il)
                end

                sol = nothing
                data_nonLinSolve = nothing
                
                if sim_properties.elastohydrSolver == "implicit_Picard"
                    typValue = copy(guess)
                    sol, data_nonLinSolve = Picard_Newton(nothing,
                                           sys_fun,
                                           guess,
                                           typValue,
                                           inter_itr_init,
                                           sim_properties,
                                           arg...,
                                           perf_node=perfNode_widthConstrItr)
                else
                    sol, data_nonLinSolve = Anderson(sys_fun,
                                             guess,
                                             inter_itr_init,
                                             sim_properties,
                                             arg...,
                                             perf_node=perfNode_widthConstrItr)
                end

            elseif sim_properties.elastohydrSolver == "RKL2"
                sol, data_nonLinSolve = solve_width_pressure_RKL2(mat_properties.Eprime,
                                                          sim_properties.enableGPU,
                                                          sim_properties.nThreads,
                                                          perfNode_widthConstrItr,
                                                          arg...)
            else
                error("The given elasto-hydrodynamic solver is not supported!")
            end

            failed_sol = any(isnan.(sol))

            if perfNode_widthConstrItr !== nothing
                fail_cause = nothing
                norm = nothing
                if length(neg) > 0
                    norm = length(new_neg) / length(neg)
                end
                if failed_sol
                    if length(perfNode_widthConstrItr.linearSolve_data) >= sim_properties.maxSolverItrs
                        fail_cause = "did not converge after max iterations"
                    else
                        fail_cause = "singular matrix"
                    end
                end

                instrument_close(perfNode_nonLinSys, perfNode_widthConstrItr, norm, length(sol),
                                 !failed_sol, fail_cause, Fr_lstTmStp.time)
                push!(perfNode_nonLinSys.widthConstraintItr_data, perfNode_widthConstrItr)
            end

            if failed_sol
                if perfNode_nonLinSys !== nothing
                    instrument_close(perfNode, perfNode_nonLinSys, nothing, length(sol), !failed_sol,
                                     fail_cause, Fr_lstTmStp.time)
                    push!(perfNode.nonLinSolve_data, perfNode_nonLinSys)
                end
                return NaN, NaN, (NaN, NaN)
            end

            w = copy(Fr_lstTmStp.w)
            w[to_solve_k] += sol[1:length(to_solve_k)]
            w[to_impose_k] = imposed_val_k
            w[neg] = wc_to_impose

            neg_km1 = copy(neg)
            wc_km1 = copy(wc_to_impose)
            below_wc_k = findall(w[to_solve_k] .< mat_properties.wc)

            if length(below_wc_k) > 0
                # for cells where max width in w history is greater than wc
                wHst_above_wc = findall(Fr_lstTmStp.wHist[to_solve_k] .>= mat_properties.wc)
                impose_wc_at = intersect(wHst_above_wc, below_wc_k)

                # for cells with max width in w history less than wc
                wHst_below_wc = findall(Fr_lstTmStp.wHist[to_solve_k] .< mat_properties.wc)
                dwdt_neg = findall(w[to_solve_k] .<= Fr_lstTmStp.w[to_solve_k])
                impose_wHist_at = intersect(wHst_below_wc, dwdt_neg)

                neg_k = to_solve_k[vcat(impose_wc_at, impose_wHist_at)]
                # the corresponding values of width to be imposed in cells where width constraint is active
                wc_k = vcat(
                    fill(mat_properties.wc, length(impose_wc_at)),
                    Fr_lstTmStp.wHist[to_solve_k[impose_wHist_at]]
                )

                new_neg = setdiff(neg_k, neg)
                if length(new_neg) == 0
                    active_contraint = false
                else
                    # cumulatively add the cells with active width constraint
                    neg = vcat(neg_km1, new_neg)
                    new_wc = Float64[]
                    for i in new_neg
                        push!(new_wc, wc_k[findfirst(neg_k .== i)[1]])
                    end
                    wc_to_impose = vcat(wc_km1, new_wc)
                    @debug "Iterating on cells with active width constraint..." _group="JFrac.solve_width_pressure"
                end
            else
                active_contraint = false
            end

            # pressure is evaluated to remove cells from the active contraint list where the fluid pressure is 
            # larger than the confining stress
            pf_k = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
            # pressure evaluated by dot product of width and elasticity matrix
            pf_k[to_solve_k] = C[to_solve_k, EltCrack] * w[EltCrack] + mat_properties.SigmaO[to_solve_k]
            
            if sim_properties.solveDeltaP
                pf_k[neg_km1] = Fr_lstTmStp.pFluid[neg_km1] + sol[length(to_solve_k)+1:length(to_solve_k)+length(neg_km1)]
                pf_k[to_impose_k] = Fr_lstTmStp.pFluid[to_impose_k] + sol[length(to_solve_k)+length(neg_km1)+1:length(to_solve_k)+length(neg_km1)+length(to_impose_k)]
            else
                pf_k[neg_km1] = sol[length(to_solve_k)+1:length(to_solve_k)+length(neg_km1)]
                pf_k[to_impose_k] = sol[length(to_solve_k)+length(neg_km1)+1:length(to_solve_k)+length(neg_km1)+length(to_impose_k)]
            end
            
            # removing cells where the fluid pressure is greater than the confining stress. If the cells that
            # are removed once re-appear in the set of cells where the width is less then the minimum residual
            # width, they are not removed. In other words, the cells are removed once.
            to_open = findall(pf_k[neg] .> mat_properties.SigmaO[neg])
            to_open = setdiff(to_open, to_open_cumm)
            to_open_cumm = vcat(to_open_cumm, to_open)

            if length(to_open) > 0
                neg = deleteat!(neg, to_open)
                wc_to_impose = deleteat!(wc_to_impose, to_open)
                active_contraint = true
                @debug "removing cells from the active width constraint..." _group="JFrac.solve_width_pressure"
            end
        end

        if perfNode_nonLinSys !== nothing
            instrument_close(perfNode, perfNode_nonLinSys, nothing, length(sol), true, nothing, Fr_lstTmStp.time)
            push!(perfNode.nonLinSolve_data, perfNode_nonLinSys)
        end

        if length(neg) == length(to_solve)
            fully_closed = true
        end

        if inj_properties.modelInjLine
            pil_indx = length(to_solve_k) + length(neg_km1) + length(to_impose_k)
            dp_il = sol[pil_indx]
            Q = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
            Q[inj_ch] = sol[pil_indx + 1: pil_indx + length(inj_ch)]
            Q[inj_act] = sol[pil_indx + 1 + length(inj_ch): pil_indx + 1 + length(inj_ch) + length(inj_act)]
            Q[sink_cells] = Q[sink_cells] - sink[sink_cells]
            
            return_data = (data_nonLinSolve, neg_km1, fully_closed, dp_il, Q)
        else
            p_il = nothing
            Q = nothing
            return_data = (data_nonLinSolve, neg_km1, fully_closed)
        end

        return w, pf_k, return_data
    end
    return NaN, NaN, (NaN, NaN)
end

# -----------------------------------------------------------------------------------------------------------------------

"""
    turbulence_check_tip(vel, Fr, fluid, return_ReyNumb=false)

This function calculate the Reynolds number at the cell edges and check if any to the edge between the ribbon cells
and the tip cells are turbulent (i.e. the Reynolds number is greater than 2100).

# Arguments
- `vel::Matrix{Float64}`: the array giving velocity of each edge of the cells in domain
- `Fr`: the fracture object to be checked
- `fluid`: fluid properties object
- `return_ReyNumb::Bool`: if true, Reynolds number at all cell edges will also be returned

# Returns
- `Union{Tuple{Matrix{Float64}, Bool}, Bool}`: Reynolds number of all the cells in the domain and boolean, or just boolean
"""
function turbulence_check_tip(vel::Matrix{Float64}, Fr, fluid, return_ReyNumb::Bool=false)
    # width at the edges by averaging
    wLftEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 1]]) / 2
    wRgtEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 2]]) / 2
    wBtmEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 3]]) / 2
    wTopEdge = (Fr.w[Fr.EltRibbon] + Fr.w[Fr.mesh.NeiElements[Fr.EltRibbon, 4]]) / 2

    Re = zeros(Float64, 4, length(Fr.EltRibbon))

    Re[1, :] = 4 / 3 * fluid.density * wLftEdge .* vel[1, Fr.EltRibbon] / fluid.viscosity
    Re[2, :] = 4 / 3 * fluid.density * wRgtEdge .* vel[2, Fr.EltRibbon] / fluid.viscosity
    Re[3, :] = 4 / 3 * fluid.density * wBtmEdge .* vel[3, Fr.EltRibbon] / fluid.viscosity
    Re[4, :] = 4 / 3 * fluid.density * wTopEdge .* vel[4, Fr.EltRibbon] / fluid.viscosity

    ReNum_Ribbon = Float64[]
    # adding Reynolds number of the edges between the ribbon and tip cells to a list
    for i in 1:length(Fr.EltRibbon)
        for j in 1:4
            # if the current neighbor (j) of the ribbon cells is in the tip elements list
            if length(findall(Fr.mesh.NeiElements[Fr.EltRibbon[i], j] .== Fr.EltTip)) > 0
                push!(ReNum_Ribbon, Re[j, i])
            end
        end
    end

    if return_ReyNumb
        wLftEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 1]]) / 2
        wRgtEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 2]]) / 2
        wBtmEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 3]]) / 2
        wTopEdge = (Fr.w[Fr.EltCrack] + Fr.w[Fr.mesh.NeiElements[Fr.EltCrack, 4]]) / 2

        Re = zeros(Float64, 4, Fr.mesh.NumberOfElts)
        Re[1, Fr.EltCrack] = 4 / 3 * fluid.density * wLftEdge .* vel[1, Fr.EltCrack] / fluid.viscosity
        Re[2, Fr.EltCrack] = 4 / 3 * fluid.density * wRgtEdge .* vel[2, Fr.EltCrack] / fluid.viscosity
        Re[3, Fr.EltCrack] = 4 / 3 * fluid.density * wBtmEdge .* vel[3, Fr.EltCrack] / fluid.viscosity
        Re[4, Fr.EltCrack] = 4 / 3 * fluid.density * wTopEdge .* vel[4, Fr.EltCrack] / fluid.viscosity

        return Re, any(ReNum_Ribbon .> 2100.0)
    else
        return any(ReNum_Ribbon .> 2100.0)
    end
end


# -----------------------------------------------------------------------------------------------------------------------
"""
    time_step_explicit_front(Fr_lstTmStp, C, timeStep, Qin, mat_properties, fluid_properties, sim_properties,
                             inj_properties, perfNode=nothing)

This function advances the fracture front in an explicit manner by propagating it with the velocity from the last
time step (see Zia and Lecampion 2019 for details).

# Arguments
- `Fr_lstTmStp`: fracture object from the last time step.
- `C::Matrix`: the elasticity matrix.
- `timeStep::Float64`: time step.
- `Qin::Vector{Float64}`: current injection rate.
- `mat_properties`: material properties.
- `fluid_properties`: fluid properties.
- `sim_properties`: simulation parameters.
- `inj_properties`: injection properties.
- `perfNode::Union{IterationProperties, Nothing}`: a performance node to store performance data.

# Returns
- `Tuple{Int, Union{Fracture, Nothing}}`: (exitstatus, Fracture) - see documentation for possible values.
"""
function time_step_explicit_front(Fr_lstTmStp, C::Matrix, timeStep::Float64, Qin::Vector{Float64}, 
                                 mat_properties, fluid_properties, sim_properties, inj_properties, 
                                 perfNode::Union{IterationProperties, Nothing}=nothing)

    @debug "time_step_explicit_front called" _group="JFrac.time_step_explicit_front"
    
    sgndDist_k = 1e50 * ones(Float64, Fr_lstTmStp.mesh.NumberOfElts)  # Initializing the cells with maximum
                                                                        # float value. (algorithm requires inf)
    sgndDist_k[Fr_lstTmStp.EltChannel] .= 0.0  # for cells inside the fracture

    sgndDist_k[Fr_lstTmStp.EltTip] = Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltTip] - (timeStep *
                                                                                 Fr_lstTmStp.v)
    
    current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
    cell_diag = (Fr_lstTmStp.mesh.hx^2 + Fr_lstTmStp.mesh.hy^2)^0.5
    expected_range = max(current_prefactor * 12.66 * cell_diag, 1.5 * cell_diag) # expected range of possible propagation
    front_region = findall(abs.(Fr_lstTmStp.sgndDist) .< expected_range)

    # the search region outwards from the front position at last time step
    pstv_region = findall(Fr_lstTmStp.sgndDist[front_region] .>= -(Fr_lstTmStp.mesh.hx^2 +
                                                                   Fr_lstTmStp.mesh.hy^2)^0.5)
    # the search region inwards from the front position at last time step
    ngtv_region = findall(Fr_lstTmStp.sgndDist[front_region] .< 0)

    # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
    SolveFMM(sgndDist_k,
             Fr_lstTmStp.EltTip,
             Fr_lstTmStp.EltCrack,
             Fr_lstTmStp.mesh,
             front_region[pstv_region],
             front_region[ngtv_region])

    # gets the new tip elements, along with the length and angle of the perpendiculars drawn on front (also containing
    # the elements which are fully filled after the front is moved outward)
    EltsTipNew = nothing
    l_k = nothing
    alpha_k = nothing
    CellStatus = nothing
    listofTIPcellsONLY = nothing
    newRibbon = nothing
    zrVertx_k_with_fully_traversed = nothing
    zrVertx_k_without_fully_traversed = nothing
    correct_size_of_pstv_region = nothing
    sgndDist_k_temp = nothing
    Ffront = nothing
    number_of_fronts = nothing
    fronts_dictionary = nothing
    
    if sim_properties.projMethod == "ILSA_orig"
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front(sgndDist_k,
                                                                 front_region,
                                                                 Fr_lstTmStp.EltChannel,
                                                                 Fr_lstTmStp.mesh)
    elseif sim_properties.projMethod == "LS_grad"
        EltsTipNew, l_k, alpha_k, CellStatus = reconstruct_front_LS_gradient(sgndDist_k,
                                                                             front_region,
                                                                             Fr_lstTmStp.EltChannel,
                                                                             Fr_lstTmStp.mesh)
    elseif sim_properties.projMethod == "LS_continousfront"
        correct_size_of_pstv_region = [false, false, false]
        recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = false
        while !correct_size_of_pstv_region[1]
            result = reconstruct_front_continuous(sgndDist_k,
                                                  front_region[pstv_region],
                                                  Fr_lstTmStp.EltRibbon,
                                                  Fr_lstTmStp.EltChannel,
                                                  Fr_lstTmStp.mesh,
                                                  recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                  lstTmStp_EltCrack0=Fr_lstTmStp.fronts_dictionary["crackcells_0"], 
                                                  oldfront=Fr_lstTmStp.Ffront)
            
            EltsTipNew = result[1]
            listofTIPcellsONLY = result[2]
            l_k = result[3]
            alpha_k = result[4]
            CellStatus = result[5]
            newRibbon = result[6]
            zrVertx_k_with_fully_traversed = result[7]
            zrVertx_k_without_fully_traversed = result[8]
            correct_size_of_pstv_region = result[9]
            sgndDist_k_temp = result[10]
            Ffront = result[11]
            number_of_fronts = result[12]
            fronts_dictionary = result[13]
            
            if correct_size_of_pstv_region[3] # correct_size_of_pstv_region[2] в Python
                exitstatus = 7 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, nothing
            end

            if correct_size_of_pstv_region[2] # correct_size_of_pstv_region[1] в Python
                Fr_kplus1 = deepcopy(Fr_lstTmStp)
                Fr_kplus1.EltTipBefore = Fr_lstTmStp.EltTip
                Fr_kplus1.EltTip = EltsTipNew  # !!! EltsTipNew are the intersection between the fictitius cells and the frontlist as tip in order to decide the direction of remeshing
                # (in case of anisotropic remeshing)
                exitstatus = 12 # You are here because the level set has negative values until the end of the mesh
                                # or because a fictitius cell has intersected the mesh.frontlist
                return exitstatus, Fr_kplus1
            end

            if !correct_size_of_pstv_region[1]
                # Expand the
                # - front region by 1 cell tickness
                # - pstv_region by 1 cell tickness
                # - ngtv_region by 1 cell tickness

                front_region = unique(vec(Fr_lstTmStp.mesh.NeiElements[front_region, :]))

                # the search region outwards from the front position at last time step
                pstv_region = findall(Fr_lstTmStp.sgndDist[front_region] .>= -(Fr_lstTmStp.mesh.hx^2 +
                                                                               Fr_lstTmStp.mesh.hy^2)^0.5)
                # the search region inwards from the front position at last time step
                ngtv_region = findall(Fr_lstTmStp.sgndDist[front_region] .< 0)

                # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
                SolveFMM(sgndDist_k,
                         Fr_lstTmStp.EltTip,
                         Fr_lstTmStp.EltCrack,
                         Fr_lstTmStp.mesh,
                         front_region[pstv_region],
                         front_region[ngtv_region])
            end
        end
        
        sgndDist_k = sgndDist_k_temp
    else
        error("projection method not supported")
    end

    if !any(in.(EltsTipNew, Ref(front_region)))
        error("The tip elements are not in the band. Increase the size of the band for FMM to evaluate level set.")
    end

    # If the angle and length of the perpendicular are not correct
    nan = (isnan.(alpha_k) .| isnan.(l_k))
    if any(nan) || any(l_k .< 0) || any(alpha_k .< 0) || any(alpha_k .> π / 2)
        exitstatus = 3
        return exitstatus, nothing
    end

    # check if any of the tip cells has a neighbor outside the grid, i.e. fracture has reached the end of the grid.
    if length(intersect(Fr_lstTmStp.mesh.Frontlist, EltsTipNew)) > 0
        Fr_lstTmStp.EltTipBefore = Fr_lstTmStp.EltTip
        Fr_lstTmStp.EltTip = EltsTipNew
        exitstatus = 12
        return exitstatus, Fr_lstTmStp
    end

    # generate the InCrack array for the current front position
    InCrack_k = zeros(Int8, Fr_lstTmStp.mesh.NumberOfElts)
    InCrack_k[Fr_lstTmStp.EltChannel] = 1
    InCrack_k[EltsTipNew] = 1

    if sum(InCrack_k .== 1) > sim_properties.maxElementIn && sim_properties.meshReductionPossible
        exitstatus = 16
        return exitstatus, Fr_lstTmStp
    end

    # Calculate filling fraction of the tip cells for the current fracture position
    FillFrac_k = Integral_over_cell(EltsTipNew,
                                    alpha_k,
                                    l_k,
                                    Fr_lstTmStp.mesh,
                                    "A",
                                    projMethod=sim_properties.projMethod) / Fr_lstTmStp.mesh.EltArea

    # todo !!! Hack: This check rounds the filling fraction to 1 if it is not bigger than 1 + 1e-4 (up to 4 figures)
    fill_frac_condition = (FillFrac_k .> 1.0) .& (FillFrac_k .< 1.0 + 1e-4)
    FillFrac_k[fill_frac_condition] .= 1.0

    # if filling fraction is below zero or above 1+1e-6
    if any(FillFrac_k .> 1.0) || any(FillFrac_k .< 0.0 - eps())
        exitstatus = 9
        return exitstatus, nothing
    end

    # Evaluate the element lists for the trial fracture front
    EltChannel_k = nothing
    EltTip_k = nothing
    EltCrack_k = nothing
    EltRibbon_k = nothing
    zrVertx_k = nothing
    CellStatus_k = nothing
    fully_traversed_k = nothing
    
    if sim_properties.projMethod != "LS_continousfront"
        # todo: some of the list are redundant to calculate on each iteration
        # Evaluate the element lists for the trial fracture front
        # new tip elements contain only the partially filled elements
        result = UpdateLists(Fr_lstTmStp.EltChannel,
                             EltsTipNew,
                             FillFrac_k,
                             sgndDist_k,
                             Fr_lstTmStp.mesh)
        EltChannel_k = result[1]
        EltTip_k = result[2]
        EltCrack_k = result[3]
        EltRibbon_k = result[4]
        zrVertx_k = result[5]
        CellStatus_k = result[6]
        fully_traversed_k = result[7]
    elseif sim_properties.projMethod == "LS_continousfront"
        # new tip elements contain only the partially filled elements
        zrVertx_k = zrVertx_k_without_fully_traversed
        result = UpdateListsFromContinuousFrontRec(newRibbon,
                                                   sgndDist_k,
                                                   Fr_lstTmStp.EltChannel,
                                                   EltsTipNew,
                                                   listofTIPcellsONLY,
                                                   Fr_lstTmStp.mesh)
        EltChannel_k = result[1]
        EltTip_k = result[2]
        EltCrack_k = result[3]
        EltRibbon_k = result[4]
        CellStatus_k = result[5]
        fully_traversed_k = result[6]
        
        if any(isnan.(EltChannel_k))
            exitstatus = 3
            return exitstatus, nothing
        end
    end

    # EletsTipNew may contain fully filled elements also. Identifying only the partially filled elements
    partlyFilledTip = findall(in.(EltsTipNew, Ref(EltTip_k)))
    @debug "Solving the EHL system with the new trial footprint" _group="JFrac.time_step_explicit_front"

    zrVrtx_newTip = nothing
    if sim_properties.projMethod != "LS_continousfront"
        # Calculating Carter's coefficient at tip to be used to calculate the volume integral in the tip cells
        zrVrtx_newTip = find_zero_vertex(EltsTipNew, sgndDist_k, Fr_lstTmStp.mesh)
    else 
        zrVrtx_newTip = transpose(zrVertx_k_with_fully_traversed)
    end
    
    # finding ribbon cells corresponding to tip cells
    corr_ribbon = find_corresponding_ribbon_cell(EltsTipNew,
                                                 alpha_k,
                                                 zrVrtx_newTip,
                                                 Fr_lstTmStp.mesh)
    Cprime_tip = mat_properties.Cprime[corr_ribbon]

    Kprime_tip = nothing
    if sim_properties.paramFromTip || mat_properties.anisotropic_K1c
        Kprime_tip = (32 / π)^0.5 * get_toughness_from_zeroVertex(EltsTipNew,
                                                                  Fr_lstTmStp.mesh,
                                                                  mat_properties,
                                                                  alpha_k,
                                                                  l_k,
                                                                  zrVrtx_newTip)
    else
        Kprime_tip = mat_properties.Kprime[corr_ribbon]
    end

    Eprime_tip = nothing
    if mat_properties.TI_elasticity
        Eprime_tip = TI_plain_strain_modulus(alpha_k,
                                             mat_properties.Cij)
    else
        Eprime_tip = fill(mat_properties.Eprime, length(EltsTipNew))
    end

    # the velocity of the front for the current front position
    # todo: not accurate on the first iteration. needed to be checked
    Vel_k = -(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) / timeStep

    if perfNode !== nothing
        perfNode_tipWidth = instrument_start("tip width", perfNode)
        # todo close tip width instrumentation
    end

    # stagnant tip cells i.e. the tip cells whose distance from front has not changed.
    stagnant = (-(sgndDist_k[EltsTipNew] - Fr_lstTmStp.sgndDist[EltsTipNew]) /
                (Fr_lstTmStp.mesh.hx^2 + Fr_lstTmStp.mesh.hy^2)^0.5 .< sim_properties.toleranceVStagnant)
    
    # we need to remove it:
    # if any(stagnant) && !((sim_properties.get_tipAsymptote() == "U") || (sim_properties.get_tipAsymptote() == "U1"))
    #     @warn "Stagnant front is only supported with universal tip asymptote. Continuing..." _group="JFrac.time_step_explicit_front"
    #     stagnant = falses(length(EltsTipNew))
    # end

    wTip = nothing
    if any(stagnant)
        # if any tip cell with stagnant front calculate stress intensity factor for stagnant cells
        KIPrime = StressIntensityFactor(Fr_lstTmStp.w,
                                        sgndDist_k,
                                        EltsTipNew,
                                        EltRibbon_k,
                                        stagnant,
                                        Fr_lstTmStp.mesh,
                                        Eprime_tip)

        # todo: Find the right cause of failure
        # if the stress Intensity factor cannot be found. The most common reason is wiggles in the front resulting
        # in isolated tip cells.
        if any(isnan.(KIPrime))
            exitstatus = 8
            return exitstatus, nothing
        end

        # Calculate average width in the tip cells by integrating tip asymptote. Width of stagnant cells are calculated
        # using the stress intensity factor (see Dontsov and Peirce, JFM RAPIDS, 2017)
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  stagnant=stagnant,
                                  KIPrime=KIPrime,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip) / Fr_lstTmStp.mesh.EltArea
    else
        # Calculate average width in the tip cells by integrating tip asymptote
        wTip = Integral_over_cell(EltsTipNew,
                                  alpha_k,
                                  l_k,
                                  Fr_lstTmStp.mesh,
                                  sim_properties.get_tipAsymptote(),
                                  frac=Fr_lstTmStp,
                                  mat_prop=mat_properties,
                                  fluid_prop=fluid_properties,
                                  Vel=Vel_k,
                                  Kprime=Kprime_tip,
                                  Eprime=Eprime_tip,
                                  Cprime=Cprime_tip,
                                  stagnant=stagnant) / Fr_lstTmStp.mesh.EltArea
    end

    # check if the tip volume has gone into negative
    smallNgtvWTip = findall((wTip .< 0) .& (wTip .> -1e-4 * mean(wTip)))
    if length(smallNgtvWTip) > 0
        wTip[smallNgtvWTip] = abs.(wTip[smallNgtvWTip])
    end

    if any(wTip .< 0) || sum(wTip) == 0.0
        exitstatus = 4
        return exitstatus, nothing
    end

    if perfNode !== nothing
        pass
        # todo close tip width instrumentation
    end

    LkOff = zeros(Float64, length(CellStatus))
    if sum(mat_properties.Cprime[EltsTipNew]) > 0
        # Calculate leak-off term for the tip cell
        LkOff[EltsTipNew] = 2 * mat_properties.Cprime[EltsTipNew] * Integral_over_cell(EltsTipNew,
                                                                                       alpha_k,
                                                                                       l_k,
                                                                                       Fr_lstTmStp.mesh,
                                                                                       "Lk",
                                                                                       mat_prop=mat_properties,
                                                                                       frac=Fr_lstTmStp,
                                                                                       Vel=Vel_k,
                                                                                       dt=timeStep,
                                                                                       arrival_t=
                                                                                       Fr_lstTmStp.TarrvlZrVrtx[
                                                                                           EltsTipNew])
        if any(isnan.(LkOff[EltsTipNew]))
            exitstatus = 13
            return exitstatus, nothing
        end
    end

    if sum(mat_properties.Cprime[Fr_lstTmStp.EltChannel]) > 0
        t_since_arrival = Fr_lstTmStp.time - Fr_lstTmStp.Tarrival[Fr_lstTmStp.EltChannel]
        t_since_arrival[t_since_arrival .< 0.0] .= 0.0
        LkOff[Fr_lstTmStp.EltChannel] = 2 * mat_properties.Cprime[Fr_lstTmStp.EltChannel] * ((t_since_arrival
                                                                                              + timeStep) .^ 0.5 - t_since_arrival .^ 0.5) * Fr_lstTmStp.mesh.EltArea
        if any(isnan.(LkOff[Fr_lstTmStp.EltChannel]))
            exitstatus = 13
            return exitstatus, nothing
        end

        if any(stagnant)
            LkOff[EltsTipNew[stagnant]] = leak_off_stagnant_tip(EltsTipNew[stagnant],
                                                                l_k[stagnant],
                                                                alpha_k[stagnant],
                                                                Fr_lstTmStp.TarrvlZrVrtx[EltsTipNew[stagnant]],
                                                                Fr_lstTmStp.time + timeStep,
                                                                mat_properties.Cprime,
                                                                timeStep,
                                                                Fr_lstTmStp.mesh)
        end
    end

    # set leak off to zero if pressure below pore pressure
    LkOff[Fr_lstTmStp.pFluid .<= mat_properties.porePressure] .= 0.0
    
    doublefracturedictionary = Dict{String, Any}()
    if sim_properties.doublefracture && fronts_dictionary["number_of_fronts"] == 2
        doublefracturedictionary["number_of_fronts"] = fronts_dictionary["number_of_fronts"]
        doublefracturedictionary["crackcells_0"] = fronts_dictionary["crackcells_0"]
        doublefracturedictionary["crackcells_1"] = fronts_dictionary["crackcells_1"]
        doublefracturedictionary["TIPcellsANDfullytrav_0"] = fronts_dictionary["TIPcellsANDfullytrav_0"]
        doublefracturedictionary["TIPcellsANDfullytrav_1"] = fronts_dictionary["TIPcellsANDfullytrav_1"]
    elseif sim_properties.projMethod != "LS_continousfront"
        doublefracturedictionary["number_of_fronts"] = 1
    else
        doublefracturedictionary["number_of_fronts"] = fronts_dictionary["number_of_fronts"]
    end
    
    result = solve_width_pressure(Fr_lstTmStp,
                                  sim_properties,
                                  fluid_properties,
                                  mat_properties,
                                  inj_properties,
                                  EltsTipNew,
                                  partlyFilledTip,
                                  C,
                                  FillFrac_k,
                                  EltCrack_k,
                                  InCrack_k,
                                  LkOff,
                                  wTip,
                                  timeStep,
                                  Qin,
                                  perfNode,
                                  Vel_k,
                                  corr_ribbon,
                                  doublefracturedictionary)

    w_n_plus1 = result[1]
    pf_n_plus1 = result[2]
    data = result[3]

    # check if the new width is valid
    if any(isnan.(w_n_plus1))
        exitstatus = 5
        return exitstatus, nothing
    end

    fluidVel = data[1][1]
    # setting arrival time for fully traversed tip elements (new channel elements)
    Tarrival_k = copy(Fr_lstTmStp.Tarrival)
    max_Tarrival = nanmax(Tarrival_k)
    nc = setdiff(EltChannel_k, Fr_lstTmStp.EltChannel)
    new_channel = Int[]
    for i in nc
        append!(new_channel, findall(EltsTipNew .== i))
    end
    
    if any(Vel_k[new_channel] .== 0)
        @debug "why we have zeros?" _group="JFrac.time_step_explicit_front"
    end
    
    if !isempty(new_channel)
        t_enter = Fr_lstTmStp.time + timeStep - l_k[new_channel] ./ Vel_k[new_channel]
        max_l = Fr_lstTmStp.mesh.hx * cos.(alpha_k[new_channel]) + Fr_lstTmStp.mesh.hy * sin.(alpha_k[new_channel])
        t_leave = Fr_lstTmStp.time + timeStep - (l_k[new_channel] - max_l) ./ Vel_k[new_channel]
        Tarrival_k[EltsTipNew[new_channel]] = (t_enter + t_leave) / 2
        to_correct = findall(Tarrival_k[EltsTipNew[new_channel]] .< max_Tarrival)
        Tarrival_k[EltsTipNew[new_channel[to_correct]]] = max_Tarrival
    end

    # the fracture to be returned for k plus 1 iteration
    Fr_kplus1 = deepcopy(Fr_lstTmStp)
    Fr_kplus1.w = w_n_plus1
    Fr_kplus1.pFluid = pf_n_plus1
    Fr_kplus1.pNet = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
    Fr_kplus1.pNet[EltCrack_k] = pf_n_plus1[EltCrack_k] - mat_properties.SigmaO[EltCrack_k]
    Fr_kplus1.time += timeStep
    Fr_kplus1.closed = data[2]
    Fr_kplus1.FillF = FillFrac_k[partlyFilledTip]
    Fr_kplus1.fully_traversed = fully_traversed_k
    Fr_kplus1.EltChannel = EltChannel_k
    Fr_kplus1.EltTip = EltTip_k
    Fr_kplus1.EltCrack = EltCrack_k
    Fr_kplus1.EltRibbon = EltRibbon_k
    Fr_kplus1.ZeroVertex = zrVertx_k
    Fr_kplus1.alpha = alpha_k[partlyFilledTip]
    Fr_kplus1.l = l_k[partlyFilledTip]
    Fr_kplus1.InCrack = InCrack_k
    
    if sim_properties.projMethod != "LS_continousfront"
        Fr_kplus1.process_fracture_front()
    else
        Fr_kplus1.fronts_dictionary = fronts_dictionary
        Fr_kplus1.Ffront = Ffront
        Fr_kplus1.number_of_fronts = number_of_fronts
        if sim_properties.saveToDisk && sim_properties.saveStatisticsPostCoalescence && Fr_lstTmStp.number_of_fronts != Fr_kplus1.number_of_fronts
            myJsonName = sim_properties.set_outputFolder * "_mesh_study.json"
            append_to_json_file(myJsonName, Fr_kplus1.mesh.nx, "append2keyAND2list", key="nx")
            append_to_json_file(myJsonName, Fr_kplus1.mesh.ny, "append2keyAND2list", key="ny")
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hx, "append2keyAND2list", key="hx")
            append_to_json_file(myJsonName, Fr_kplus1.mesh.hy, "append2keyAND2list", key="hy")
            append_to_json_file(myJsonName, length(Fr_kplus1.EltCrack), "append2keyAND2list", key="elements_in_crack")
            append_to_json_file(myJsonName, length(Fr_kplus1.EltTip), "append2keyAND2list", key="elements_in_tip")
            append_to_json_file(myJsonName, Fr_kplus1.time, "append2keyAND2list", key="coalescence_time")
        end
    end
    
    Fr_kplus1.FractureVolume = sum(Fr_kplus1.w) * Fr_kplus1.mesh.EltArea
    Fr_kplus1.Tarrival = Tarrival_k
    Fr_kplus1.wHist = max.(Fr_kplus1.w, Fr_lstTmStp.wHist)
    Fr_kplus1.effVisc = data[1][2]
    Fr_kplus1.G = data[1][3]

    if length(data) > 3
        Fr_kplus1.injectionRate = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
        Fr_kplus1.pInjLine = Fr_lstTmStp.pInjLine + data[4]
        Fr_kplus1.injectionRate = data[5]
        Fr_kplus1.source = findall(Fr_kplus1.injectionRate .> 0)
        Fr_kplus1.sink = findall(Fr_kplus1.injectionRate .< 0)
    else
        Fr_kplus1.source = Fr_lstTmStp.EltCrack[findall(Qin[Fr_lstTmStp.EltCrack] .> 0)]
        Fr_kplus1.sink = Fr_lstTmStp.EltCrack[findall(Qin[Fr_lstTmStp.EltCrack] .< 0)]
    end

    @debug "Solved..." _group="JFrac.time_step_explicit_front"
    @debug "Finding velocity of front..." _group="JFrac.time_step_explicit_front"

    itr = 0
    # toughness iteration loop
    alpha_ribbon_k = nothing
    alpha_ribbon_km1 = nothing
    Kprime_k = nothing
    Eprime_k = nothing
    
    while itr < sim_properties.maxProjItrs
        projection_method = nothing
        if sim_properties.paramFromTip || mat_properties.anisotropic_K1c || mat_properties.TI_elasticity
            if sim_properties.projMethod == "ILSA_orig"
                projection_method = projection_from_ribbon
            elseif sim_properties.projMethod == "LS_grad"
                projection_method = projection_from_ribbon_LS_gradient
            elseif sim_properties.projMethod == "LS_continousfront"
                projection_method = projection_from_ribbon_LS_gradient #todo: test this case!!!
            end

            if itr == 0
                # first iteration
                alpha_ribbon_k = projection_method(Fr_lstTmStp.EltRibbon,
                                                   Fr_lstTmStp.EltChannel,
                                                   Fr_lstTmStp.mesh,
                                                   sgndDist_k)
                alpha_ribbon_km1 = zeros(Float64, length(Fr_lstTmStp.EltRibbon))
            else
                alpha_ribbon_k = 0.3 * alpha_ribbon_k + 0.7 * projection_method(Fr_lstTmStp.EltRibbon,
                                                                                Fr_lstTmStp.EltChannel,
                                                                                Fr_lstTmStp.mesh,
                                                                                sgndDist_k)
            end
            
            if any(isnan.(alpha_ribbon_k))
                exitstatus = 11
                return exitstatus, nothing
            end
        end

        if sim_properties.paramFromTip || mat_properties.anisotropic_K1c
            Kprime_k = get_toughness_from_cellCenter(alpha_ribbon_k,
                                                     sgndDist_k,
                                                     Fr_lstTmStp.EltRibbon,
                                                     mat_properties,
                                                     Fr_lstTmStp.mesh) * (32 / π)^0.5

            if any(isnan.(Kprime_k))
                exitstatus = 11
                return exitstatus, nothing
            end
        else
            Kprime_k = nothing
        end

        if mat_properties.TI_elasticity
            Eprime_k = TI_plain_strain_modulus(alpha_ribbon_k,
                                               mat_properties.Cij)
            if any(isnan.(Eprime_k))
                exitstatus = 11
                return exitstatus, nothing
            end
        else
            Eprime_k = nothing
        end

        # Initialization of the signed distance in the ribbon element - by inverting the tip asymptotics
        sgndDist_k = 1e50 * ones(Float64, Fr_lstTmStp.mesh.NumberOfElts)  # Initializing the cells with extremely
        # large float value. (algorithm requires inf)

        perfNode_tipInv = instrument_start("tip inversion", perfNode)

        sgndDist_k[Fr_lstTmStp.EltRibbon] = - TipAsymInversion(Fr_kplus1.w,
                                                               Fr_lstTmStp,
                                                               mat_properties,
                                                               fluid_properties,
                                                               sim_properties,
                                                               timeStep,
                                                               Kprime_k=Kprime_k,
                                                               Eprime_k=Eprime_k)

        status, fail_cause = true, nothing
        # if tip inversion returns nan
        if any(isnan.(sgndDist_k[Fr_lstTmStp.EltRibbon]))
            status = false
            fail_cause = "tip inversion failed"
            exitstatus = 7
        end

        if perfNode_tipInv !== nothing
            instrument_close(perfNode, perfNode_tipInv, nothing, length(Fr_lstTmStp.EltRibbon),
                             status, fail_cause, Fr_lstTmStp.time)
            push!(perfNode.tipInv_data, perfNode_tipInv)
        end

        if !status
            return exitstatus, nothing
        end

        # Check if the front is receding
        sgndDist_k[Fr_lstTmStp.EltRibbon] = min.(sgndDist_k[Fr_lstTmStp.EltRibbon],
                                                 Fr_lstTmStp.sgndDist[Fr_lstTmStp.EltRibbon])

        # region expected to have the front after propagation. The signed distance of the cells only in this region will
        # evaluated with the fast marching method to avoid unnecessary computation cost
        current_prefactor = sim_properties.get_time_step_prefactor(Fr_lstTmStp.time + timeStep)
        front_region = findall(abs.(Fr_lstTmStp.sgndDist) .< current_prefactor * 12.66 * 
                              (Fr_lstTmStp.mesh.hx^2 + Fr_lstTmStp.mesh.hy^2)^0.5)

        if !any(in.(Fr_kplus1.EltTip, Ref(front_region)))
            error("The tip elements are not in the band. Increase the size of the band for FMM to evaluate level set.")
        end
        
        # the search region outwards from the front position at last time step
        pstv_region = findall(Fr_lstTmStp.sgndDist[front_region] .>= -(Fr_lstTmStp.mesh.hx^2 +
                                                                       Fr_lstTmStp.mesh.hy^2)^0.5)
        # the search region inwards from the front position at last time step
        ngtv_region = findall(Fr_lstTmStp.sgndDist[front_region] .< 0)

        # SOLVE EIKONAL eq via Fast Marching Method starting to get the distance from tip for each cell.
        SolveFMM(sgndDist_k,
                 Fr_lstTmStp.EltRibbon,
                 Fr_lstTmStp.EltChannel,
                 Fr_lstTmStp.mesh,
                 front_region[pstv_region],
                 front_region[ngtv_region])

        # do it only once if not anisotropic
        if !(sim_properties.paramFromTip || mat_properties.anisotropic_K1c ||
             mat_properties.TI_elasticity) || sim_properties.explicitProjection
            break
        end

        norm = norm(abs.(alpha_ribbon_k - alpha_ribbon_km1) / π * 2)
        if norm < sim_properties.toleranceProjection
            @debug "Projection iteration converged after $(itr - 1) iterations; exiting norm $norm" _group="JFrac.time_step_explicit_front"
            break
        end
        
        alpha_ribbon_km1 = copy(alpha_ribbon_k)
        @debug "iterating on projection... norm = $norm" _group="JFrac.time_step_explicit_front"
        itr += 1
    end

    # todo Hack!!! keep going if projection does not converge
    # if itr == sim_properties.maxProjItrs
    #     exitstatus = 10
    #     return exitstatus, nothing
    # end

    Fr_kplus1.v = -(sgndDist_k[Fr_kplus1.EltTip] - Fr_lstTmStp.sgndDist[Fr_kplus1.EltTip]) / timeStep
    Fr_kplus1.sgndDist = sgndDist_k
    Fr_kplus1.sgndDist_last = Fr_lstTmStp.sgndDist
    Fr_kplus1.timeStep_last = timeStep
    new_tip = findall(isnan.(Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip]))
    if !isempty(new_tip)
        Fr_kplus1.TarrvlZrVrtx[Fr_kplus1.EltTip[new_tip]] = Fr_kplus1.time - Fr_kplus1.l[new_tip] ./ Fr_kplus1.v[new_tip]
    end
    Fr_kplus1.LkOff = LkOff
    Fr_kplus1.LkOffTotal += sum(LkOff)
    Fr_kplus1.injectedVol += sum(Qin) * timeStep
    Fr_kplus1.efficiency = (Fr_kplus1.injectedVol - Fr_kplus1.LkOffTotal) / Fr_kplus1.injectedVol

    if sim_properties.saveRegime
        Fr_kplus1.update_tip_regime(mat_properties, fluid_properties, timeStep)
    end

    if fluid_properties.turbulence
        if sim_properties.saveReynNumb || sim_properties.saveFluidFlux
            ReNumb, check = turbulence_check_tip(fluidVel, Fr_kplus1, fluid_properties, return_ReyNumb=true)
            if sim_properties.saveReynNumb
                Fr_kplus1.ReynoldsNumber = ReNumb
            end
            if sim_properties.saveFluidFlux
                Fr_kplus1.fluidFlux = ReNumb * 3 / 4 / fluid_properties.density * mean(fluid_properties.viscosity)
            end
        end
        if sim_properties.saveFluidVel
            Fr_kplus1.fluidVelocity = fluidVel
        end
        if sim_properties.saveFluidVelAsVector
            error("saveFluidVelAsVector Not yet implemented")
        end
        if sim_properties.saveFluidFluxAsVector
            error("saveFluidFluxAsVector Not yet implemented")
        end
    else
        if sim_properties.saveFluidFlux || sim_properties.saveFluidVel || sim_properties.saveReynNumb || sim_properties.saveFluidFluxAsVector || sim_properties.saveFluidVelAsVector
            ###todo: re-evaluating these parameters is highly inefficient. They have to be stored if neccessary when
            # the solution is evaluated.
            fluid_results = calculate_fluid_flow_characteristics_laminar(Fr_kplus1.w,
                                                                        Fr_kplus1.pFluid,
                                                                        mat_properties.SigmaO,
                                                                        Fr_kplus1.mesh,
                                                                        Fr_kplus1.EltCrack,
                                                                        Fr_kplus1.InCrack,
                                                                        fluid_properties.muPrime,
                                                                        fluid_properties.density)

            fluid_flux = fluid_results[1]
            fluid_vel = fluid_results[2]
            Rey_num = fluid_results[3]
            fluid_flux_components = fluid_results[4]
            fluid_vel_components = fluid_results[5]

            if sim_properties.saveFluidFlux
                fflux = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                fflux[:, Fr_kplus1.EltCrack] = fluid_flux
                Fr_kplus1.fluidFlux = fflux
            end

            if sim_properties.saveFluidFluxAsVector
                fflux_components = zeros(Float32, 8, Fr_kplus1.mesh.NumberOfElts)
                fflux_components[:, Fr_kplus1.EltCrack] = fluid_flux_components
                Fr_kplus1.fluidFlux_components = fflux_components
            end

            if sim_properties.saveFluidVel
                fvel = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                fvel[:, Fr_kplus1.EltCrack] = fluid_vel
                Fr_kplus1.fluidVelocity = fvel
            end

            if sim_properties.saveFluidVelAsVector
                fvel_components = zeros(Float32, 8, Fr_kplus1.mesh.NumberOfElts)
                fvel_components[:, Fr_kplus1.EltCrack] = fluid_vel_components
                Fr_kplus1.fluidVelocity_components = fvel_components
            end

            if sim_properties.saveReynNumb
                Rnum = zeros(Float32, 4, Fr_kplus1.mesh.NumberOfElts)
                Rnum[:, Fr_kplus1.EltCrack] = Rey_num
                Fr_kplus1.ReynoldsNumber = Rnum
            end
        end
    end

    if data[3]
        return 14, Fr_kplus1
    end

    exitstatus = 1
    return exitstatus, Fr_kplus1
end

end # module TimeStepSolution