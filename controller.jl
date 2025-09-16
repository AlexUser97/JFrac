# -*- coding: utf-8 -*-
"""
This file is a part of JFrac.
Realization of Pyfrac on Julia language.

"""

module Controller

    using Logging
    using LinearAlgebra
    using PyPlot
    using Dates
    using JLD2
    using Printf

    using .Properties: LabelProperties, IterationProperties, PlotProperties, instrument_start, instrument_close
    using .Elasticity: load_isotropic_elasticity_matrix, load_TI_elasticity_matrix, mapping_old_indexes,
                    load_isotropic_elasticity_matrix_toepliz, load_isotropic_elasticity_matrix_symmetric,
                    symmetric_elasticity_matrix_from_full
    using .Mesh: CartesianMesh
    using .TimeStepSolution: attempt_time_step
    using .Visualization: plot_footprint_analytical, plot_analytical_solution, plot_injection_source, get_elements
    using .Symmetry: 
    using .Labels: TS_errorMessages, supported_projections, suitable_elements

    export Controller, run, advance_time_step, output, get_time_step, remesh, extend_isotropic_elasticity_matrix



    """
        Controller
        This class describes the controller which takes the given material, fluid, injection and loading properties and
        advances a given fracture according to the provided simulation properties.
    """
    mutable struct Controller
        const errorMessages::Dict
        fracture::Fracture
        solid_prop::MaterialProperties
        fluid_prop::FluidProperties
        injection_prop::InjectionProperties
        sim_prop::SimulationProperties
        load_prop::Union{Nothing, Any}
        C::Union{Nothing, Array{Float64}}
        fr_queue::Vector{Union{Nothing, Any}}
        stepsFromChckPnt::Int
        tmStpPrefactor_copy::Any
        stagnant_TS::Union{Nothing, Float64}
        perfData::Vector{Any} 
        lastSavedFile::Int
        lastSavedTime::Float64
        lastPlotTime::Float64
        TmStpCount::Int
        chkPntReattmpts::Int
        TmStpReductions::Int
        delta_w::Union{Nothing, Any}
        lstTmStp::Union{Nothing, Any}
        solveDetlaP_cp::Bool
        PstvInjJmp::Union{Nothing, Bool}
        fullyClosed::Bool
        setFigPos::Bool
        lastSuccessfulTS::Float64
        maxTmStp::Float64
        Figures::Vector{Union{Nothing, Any}} # Vector{Union{Nothing, PyPlot.Figure}}
        timeToHit::Union{Nothing, Vector{Float64}}
        remeshings::Int
        successfulTimeSteps::Int
        failedTimeSteps::Int
        frontAdvancing::String
        logAddress::String
        """
        Constructor for the Controller class.

        # Arguments
        - `Fracture`:                     -- the fracture to be propagated.
        - `Solid_prop`:                   -- the MaterialProperties object giving the material properties.
        - `Fluid_prop`:                   -- the FluidProperties object giving the fluid properties.
        - `Injection_prop`:               -- the InjectionProperties object giving the injection properties.
        - `Sim_prop`:                     -- the SimulationProperties object giving the numerical parameters.
        - `Load_prop`:                    -- the LoadingProperties object specifying mechanical loading.
        - `C`:                            -- the elasticity matrix.
        """
        function Controller(Fracture, Solid_prop, Fluid_prop, Injection_prop, Sim_prop, Load_prop=nothing, C=nothing)

            obj = new()
            obj.errorMessages = TS_errorMessages

            obj.fracture = Fracture
            obj.solid_prop = Solid_prop
            obj.fluid_prop = Fluid_prop
            obj.injection_prop = Injection_prop
            obj.sim_prop = Sim_prop
            obj.load_prop = Load_prop
            obj.C = C
            obj.fr_queue = [nothing, nothing, nothing, nothing, nothing]
            obj.stepsFromChckPnt = 0
            observedbj.tmStpPrefactor_copy = copy(Sim_prop.tmStpPrefactor)
            obj.stagnant_TS = nothing
            obj.perfData = []
            obj.lastSavedFile = 0
            obj.lastSavedTime = -Inf
            obj.lastPlotTime = -Inf
            obj.TmStpCount = 0
            obj.chkPntReattmpts = 0
            obj.TmStpReductions = 0
            obj.delta_w = nothing
            obj.lstTmStp = nothing
            obj.solveDetlaP_cp = Sim_prop.solveDeltaP
            obj.PstvInjJmp = nothing
            obj.fullyClosed = false
            obj.setFigPos = true
            obj.lastSuccessfulTS = Fracture.time
            obj.maxTmStp = 0.0

            # make a list of Nones with the size of the number of variables to plot during simulation
            obj.Figures = Any[nothing for _ in 1:length(Sim_prop.plotVar)] # Явное указание типа для массива Any

            param_change_at = Float64[] # np.array([], dtype=np.float64)
            
            if size(Injection_prop.injectionRate, 2) > 1
            param_change_at = vcat(param_change_at, Injection_prop.injectionRate[1, :]) # Предполагаем, что injectionRate 2D
            end
            if isa(Sim_prop.fixedTmStp, Array) && ndims(Sim_prop.fixedTmStp) == 2
            param_change_at = vcat(param_change_at, Sim_prop.fixedTmStp[1, :])
            end
            if isa(Sim_prop.tmStpPrefactor, Array) && ndims(Sim_prop.tmStpPrefactor) == 2
            param_change_at = vcat(param_change_at, Sim_prop.tmStpPrefactor[1, :])
            end

            if length(param_change_at) > 0
                if Sim_prop.get_solTimeSeries() !== nothing
                    # add the times where any parameter changes to the required solution time series
                    sol_time_srs = vcat(Sim_prop.get_solTimeSeries(), param_change_at)
                else
                    sol_time_srs = param_change_at
                end
                sol_time_srs = unique(sol_time_srs)
                if sol_time_srs[1] == 0.0
                    sol_time_srs = sol_time_srs[2:end]
                end
            else
            sol_time_srs = Sim_prop.get_solTimeSeries()
            end
            obj.timeToHit = sol_time_srs

            if Sim_prop.finalTime === nothing
            if Sim_prop.get_solTimeSeries() === nothing
                ## Not necessarily an error
                    throw(ArgumentError("The final time to stop the simulation is not provided!"))
            else
                Sim_prop.finalTime = maximum(Sim_prop.get_solTimeSeries())
            end
            else
                if obj.timeToHit !== nothing
                    greater_finalTime = findall(obj.timeToHit .> Sim_prop.finalTime) # np.where(self.timeToHit > self.sim_prop.finalTime)[0]
                    if !isempty(greater_finalTime)
                        obj.timeToHit = obj.timeToHit[setdiff(1:length(obj.timeToHit), greater_finalTime)] # np.delete(self.timeToHit, greater_finalTime)
                    end
                end
            end

            # Setting to volume control solver if viscosity is zero
            if obj.fluid_prop.rheology == "Newtonian"
                if isa(obj.fluid_prop.viscosity, Array) && minimum(obj.fluid_prop.viscosity) < 1e-15
                    println("Fluid viscosity is zero. Setting solver to volume control...")
                    obj.sim_prop.set_volumeControl(true)
                elseif (isa(obj.fluid_prop.viscosity, Number)) && obj.fluid_prop.viscosity < 1e-15
                    println("Fluid viscosity is zero. Setting solver to volume control...")
                    obj.sim_prop.set_volumeControl(true)
                end
            end

            if obj.injection_prop.sourceLocFunc === nothing
                if !all(in(obj.fracture.EltChannel), Injection_prop.sourceElem) # Проверка, все ли элементы источника находятся в канале
                    message = """INJECTION LOCATION ERROR: injection points are located outisde of the fracture footprints"""
                    throw(ErrorException(message))
                end
            end

            # Setting whether sparse matrix is used to make fluid conductivity matrix
            if Sim_prop.solveSparse === nothing
            # if Fracture.mesh.NumberOfElts > 2500 or self.injection_prop.modelInjLine:
            if Fracture.mesh.NumberOfElts > 2500 || obj.injection_prop.modelInjLine
                obj.sim_prop.solveSparse = true
            else
                obj.sim_prop.solveSparse = false
            end
            end

            # basic performance data
            obj.remeshings = 0
            obj.successfulTimeSteps = 0
            obj.failedTimeSteps = 0

            # setting front advancing scheme to implicit if velocity is not available for the first time step.
            obj.frontAdvancing = Sim_prop.frontAdvancing
            if Sim_prop.frontAdvancing in ["explicit", "predictor-corrector"]
                # if np.nanmax(Fracture.v) <= 0 or np.isnan(Fracture.v).any():
                try
                    if maximum(Fracture.v) <= 0.0 || any(isnan, Fracture.v)
                        obj.sim_prop.frontAdvancing = "implicit"
                    end
                catch e
                    if isa(e, ArgumentError) && occursin("NaN", string(e))
                        if all(isnan, Fracture.v) || maximum(filter(!isnan, Fracture.v)) <= 0.0
                            obj.sim_prop.frontAdvancing = "implicit"
                        end
                    else
                        rethrow(e)
                    end
                end
            end

            if obj.sim_prop.saveToDisk
                obj.logAddress = Sim_prop.get_outputFolder()
            else
                obj.logAddress = "./"
            end

            # setting up tip asymptote
            if obj.fluid_prop.rheology in ["Herschel-Bulkley", "HBF"]
                if !(obj.sim_prop.get_tipAsymptote() in ["HBF", "HBF_aprox", "HBF_num_quad"])
                    @warn "Fluid rheology and tip asymptote does not match. Setting tip asymptote to 'HBF'"
                    obj.sim_prop.set_tipAsymptote("HBF")
                end
            elseif obj.fluid_prop.rheology in ["power-law", "PLF"]
                if !(obj.sim_prop.get_tipAsymptote() in ["PLF", "PLF_aprox", "PLF_num_quad", "PLF_M"])
                    @warn "Fluid rheology and tip asymptote does not match. Setting tip asymptote to 'PLF'"
                    obj.sim_prop.set_tipAsymptote("PLF")
                end
            elseif obj.fluid_prop.rheology == "Newtonian"
                if !(obj.sim_prop.get_tipAsymptote() in ["K", "M", "Mt", "U", "U1", "MK", "MDR", "M_MDR"])
                    @warn "Fluid rheology and tip asymptote does not match. Setting tip asymptote to 'U'"
                    obj.sim_prop.set_tipAsymptote("U1")
                end
            end

            if obj.fluid_prop.rheology != "Newtonian"
                obj.sim_prop.saveRegime = false
            end

            # if you set the code to advance max 1 cell then remove the SimulProp.timeStepLimit
            if obj.sim_prop.timeStepLimit !== nothing && obj.sim_prop.limitAdancementTo2cells == true
                if obj.sim_prop.forceTmStpLmtANDLmtAdvTo2cells == false
                    @warn "You have set sim_prop.limitAdancementTo2cells = True. This imply that sim_prop.timeStepLimit will be deactivated."
                    obj.sim_prop.timeStepLimit = nothing
                else
                    @warn "You have forced <limitAdancementTo2cells> to be True and set <timeStepLimit> - the first one might be uneffective onto the second one until the prefactor has been reduced to produce a time step < timeStepLimit"
                end
            end

            return obj
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        run(self)

        This function runs the simulation according to the parameters given in the properties classes. See especially
        the documentation of the SimulationProperties class to get details of the parameters controlling the simulation run.

        # Arguments
        - `self::Controller`: the controller object

        # Returns
        - `Bool`: True if simulation completed successfully
    """
    function run(self::Controller)::Bool
        # log = get_logger("JFrac.controller.run")
        # log_only_to_logfile = get_logger("JFrac_LF.controller.run") # Placeholder for specific file logger

        # output initial fracture
        if self.sim_prop.saveToDisk
            # save properties
            if !isdir(self.sim_prop.get_outputFolder()) # Check if directory exists
                mkpath(self.sim_prop.get_outputFolder()) # Create directory and parents if needed
            end

            prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)

            jldsave(joinpath(self.sim_prop.get_outputFolder(), "properties.jld2"); prop)
        end

        if self.sim_prop.plotFigure || self.sim_prop.saveToDisk
            # save or plot fracture
            output(self, self.fracture) # Assuming output is a function or method
            self.lastSavedTime = self.fracture.time
        end

        if self.sim_prop.log2file
            set_logging_to_file(self.sim_prop, self.logAddress) # Assuming this function exists
        end

        # deactivate the block_toepliz_compression functions
        # DO THIS CHECK BEFORE COMPUTING C!
        # if self.C is not None: # in the case C is provided
        if self.C !== nothing # in the case C is provided
            # self.sim_prop.useBlockToeplizCompression = False
            self.sim_prop.useBlockToeplizCompression = false
        elseif self.solid_prop.TI_elasticity # in case of TI_elasticity
            self.sim_prop.useBlockToeplizCompression = false
        elseif !self.solid_prop.TI_elasticity && self.sim_prop.symmetric  # in case you save 1/4 of the elasticity due to domain symmetry
            self.sim_prop.useBlockToeplizCompression = false
        end

        # load elasticity matrix
        if self.C === nothing
            # log.info("Making elasticity matrix...")
            @info "Making elasticity matrix..."
            # if self.sim_prop.symmetric:
            if self.sim_prop.symmetric
                # if not self.sim_prop.get_volumeControl():
                if !self.sim_prop.get_volumeControl()
                    # raise ValueError("Symmetric fracture is only supported for inviscid fluid yet!")
                    throw(ArgumentError("Symmetric fracture is only supported for inviscid fluid yet!"))
                end
            end

            # if not self.solid_prop.TI_elasticity:
            if !self.solid_prop.TI_elasticity
                # if self.sim_prop.symmetric:
                if self.sim_prop.symmetric
                    # self.C = load_isotropic_elasticity_matrix_symmetric(self.fracture.mesh,
                    #                                                     self.solid_prop.Eprime)
                    self.C = load_isotropic_elasticity_matrix_symmetric(self.fracture.mesh,
                                                                        self.solid_prop.Eprime)
                else
                    # if not self.sim_prop.useBlockToeplizCompression:
                    if !self.sim_prop.useBlockToeplizCompression
                        # self.C = load_isotropic_elasticity_matrix(self.fracture.mesh,
                        #                                             self.solid_prop.Eprime)
                        self.C = load_isotropic_elasticity_matrix(self.fracture.mesh,
                                                                    self.solid_prop.Eprime)
                    else
                        # self.C = load_isotropic_elasticity_matrix_toepliz(self.fracture.mesh,
                        #                                                     self.solid_prop.Eprime)
                        self.C = load_isotropic_elasticity_matrix_toepliz(self.fracture.mesh,
                                                                            self.solid_prop.Eprime)
                    end
                end
            else
                # C = load_TI_elasticity_matrix(self.fracture.mesh,
                #                                     self.solid_prop,
                #                                     self.sim_prop)
                C = load_TI_elasticity_matrix(self.fracture.mesh,
                                                    self.solid_prop,
                                                    self.sim_prop)
                # compressing the elasticity matrix for symmetric fracture
                # if self.sim_prop.symmetric:
                if self.sim_prop.symmetric
                    # self.C = symmetric_elasticity_matrix_from_full(C, self.fracture.mesh)
                    self.C = symmetric_elasticity_matrix_from_full(C, self.fracture.mesh)
                else
                    # self.C = C
                    self.C = C
                end
            end
            # log.info('Done!')
            @info "Done!"
        end

        @info "Starting time = $(self.fracture.time)"
        # starting time stepping loop
        while self.fracture.time < 0.999 * self.sim_prop.finalTime && self.TmStpCount < self.sim_prop.maxTimeSteps

            timeStep = get_time_step(self) # Assuming get_time_step is a function

            tmStp_perf = nothing
            if self.sim_prop.collectPerfData
                tmStp_perf = Dict("itr_type" => "time step", "CpuTime_end" => 0.0, "status" => false,
                                "failure_cause" => "", "time" => 0.0, "NumbOfElts" => 0)
            else
                tmStp_perf = nothing
            end

            # Assuming advance_time_step is a function that takes self as first argument
            status, Fr_n_pls1 = advance_time_step(self, self.fracture,
                                                    self.C,
                                                    timeStep,
                                                    tmStp_perf)

            # if self.sim_prop.collectPerfData:
            if self.sim_prop.collectPerfData
                # tmStp_perf.CpuTime_end = time.time()
                # Assuming time() function from Dates or similar is available
                tmStp_perf["CpuTime_end"] = time() # Update Dict placeholder
                tmStp_perf["status"] = status == 1
                tmStp_perf["failure_cause"] = get(self.errorMessages, status, "Unknown error")
                tmStp_perf["time"] = self.fracture.time
                tmStp_perf["NumbOfElts"] = length(self.fracture.EltCrack)
                push!(self.perfData, tmStp_perf)
            end

            if status == 1
                # Successful time step
                @info "Time step successful!"
                @debug "Element in the crack: $(length(Fr_n_pls1.EltCrack))"
                @debug "Nx: $(Fr_n_pls1.mesh.nx)"
                @debug "Ny: $(Fr_n_pls1.mesh.ny)"
                @debug "hx: $(Fr_n_pls1.mesh.hx)"
                @debug "hy: $(Fr_n_pls1.mesh.hy)"
                self.delta_w = Fr_n_pls1.w - self.fracture.w
                self.lstTmStp = Fr_n_pls1.time - self.fracture.time
                # output
                if self.sim_prop.plotFigure || self.sim_prop.saveToDisk
                    if Fr_n_pls1.time > self.lastSavedTime
                        output(self, Fr_n_pls1) # Assuming output is a function
                    end
                end

                # add the advanced fracture to the last five fractures list
                self.fracture = deepcopy(Fr_n_pls1) # deepcopy from Base
                self.fr_queue[(self.successfulTimeSteps % 5) + 1] = deepcopy(Fr_n_pls1)

                # if self.fracture.time > self.lastSuccessfulTS:
                if self.fracture.time > self.lastSuccessfulTS
                    # self.lastSuccessfulTS = self.fracture.time
                    self.lastSuccessfulTS = self.fracture.time
                end
                # if self.maxTmStp < self.lstTmStp:
                if self.maxTmStp < self.lstTmStp
                    # self.maxTmStp = self.lstTmStp
                    self.maxTmStp = self.lstTmStp
                end
                # put check point reattempts to zero if the simulation has advanced past the time where it failed
                # if Fr_n_pls1.time > self.lastSuccessfulTS + 2 * self.maxTmStp:
                if Fr_n_pls1.time > self.lastSuccessfulTS + 2 * self.maxTmStp
                    # self.chkPntReattmpts = 0
                    self.chkPntReattmpts = 0
                    # set the prefactor to the original value after four time steps (after the 5 time steps back jump)
                    # self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                end
                # self.successfulTimeSteps += 1
                self.successfulTimeSteps += 1
                # set to 0 the counter of time step reductions
                # if self.TmStpReductions > 0:
                if self.TmStpReductions > 0
                    # self.TmStpReductions = 0
                    self.TmStpReductions = 0
                    # self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                    self.sim_prop.tmStpPrefactor = self.tmStpPrefactor_copy
                end
                # resetting the parameters for closure
                # if self.fullyClosed:
                if self.fullyClosed
                    # set to solve for pressure if the fracture was fully closed in last time step and is open now
                    # self.sim_prop.solveDeltaP = False
                    self.sim_prop.solveDeltaP = false
                else
                    # self.sim_prop.solveDeltaP = self.solveDetlaP_cp
                    self.sim_prop.solveDeltaP = self.solveDetlaP_cp
                end
                # self.PstvInjJmp = None
                self.PstvInjJmp = nothing
                # self.fullyClosed = False
                self.fullyClosed = false

                # set front advancing back as set in simulation properties originally if velocity becomes available.
                # if np.max(Fr_n_pls1.v) > 0 or not np.isnan(Fr_n_pls1.v).any():
                # This requires careful handling of potential NaN in maximum. Julia's maximum can throw ArgumentError.
                try
                    # if maximum(Fr_n_pls1.v) > 0 || !any(isnan, Fr_n_pls1.v)
                    if maximum(Fr_n_pls1.v) > 0.0 || !any(isnan, Fr_n_pls1.v)
                        # self.sim_prop.frontAdvancing = copy.copy(self.frontAdvancing)
                        # For primitive types like String, assignment copies the value in Julia
                        self.sim_prop.frontAdvancing = self.frontAdvancing
                    else
                        # self.sim_prop.frontAdvancing = 'implicit'
                        self.sim_prop.frontAdvancing = "implicit"
                    end
                catch e
                    # Handle case where maximum fails due to all NaNs or other issues
                    if isa(e, ArgumentError) && occursin("NaN", string(e))
                        # If maximum fails because of NaN, check if there are non-NaN values > 0
                        # Filter out NaNs and check maximum of remaining values
                        non_nan_vals = filter(!isnan, Fr_n_pls1.v)
                        if !isempty(non_nan_vals) && maximum(non_nan_vals) > 0.0
                            self.sim_prop.frontAdvancing = self.frontAdvancing
                        else
                            self.sim_prop.frontAdvancing = "implicit"
                        end
                    else
                        rethrow(e) # Re-throw if it's a different kind of error
                    end
                end

                # if self.TmStpCount == self.sim_prop.maxTimeSteps:
                if self.TmStpCount == self.sim_prop.maxTimeSteps
                    # log.warning("Max time steps reached!")
                    @warn "Max time steps reached!"
                end

            # elif status == 12 or status == 16:
            elseif status == 12 || status == 16
                # re-meshing required
                # if self.sim_prop.enableRemeshing:
                if self.sim_prop.enableRemeshing
                    # the following update is required because Fr_n_pls1.EltTip contains the intersection between the cells at the boundary of the mesh and
                    # the reconstructed front. For that reason in case of mesh extension
                    # if hasattr(Fr_n_pls1, 'EltTipBefore'):
                    # In Julia, we can check if a field exists using isdefined
                    # if isdefined(Fr_n_pls1, :EltTipBefore)
                    if isdefined(typeof(Fr_n_pls1), :EltTipBefore) || hasfield(typeof(Fr_n_pls1), :EltTipBefore)
                        # self.fracture.EltTipBefore = Fr_n_pls1.EltTipBefore
                        self.fracture.EltTipBefore = Fr_n_pls1.EltTipBefore
                    end
                    # we need to decide which remeshings are to be considered
                    # compress = False
                    compress = false
                    # if status == 16:
                    if status == 16
                        # we reached cell number limit so we adapt by compressing the domain accordingly

                        # calculate the new number of cells
                        # new_elems = [int((self.fracture.mesh.nx + np.round(self.sim_prop.meshReductionFactor, 0))
                        #                     / self.sim_prop.meshReductionFactor),
                        #                 int((self.fracture.mesh.ny + np.round(self.sim_prop.meshReductionFactor, 0))
                        #                     / self.sim_prop.meshReductionFactor)]
                        # Using Julia's round and Int conversion
                        # new_elems = [
                        #     Int(round((self.fracture.mesh.nx + round(self.sim_prop.meshReductionFactor)) / self.sim_prop.meshReductionFactor)),
                        #     Int(round((self.fracture.mesh.ny + round(self.sim_prop.meshReductionFactor)) / self.sim_prop.meshReductionFactor))
                        # ]
                        # Corrected for potential Float64 return from round
                        new_elems = [
                            Int(floor((self.fracture.mesh.nx + round(self.sim_prop.meshReductionFactor)) / self.sim_prop.meshReductionFactor)),
                            Int(floor((self.fracture.mesh.ny + round(self.sim_prop.meshReductionFactor)) / self.sim_prop.meshReductionFactor))
                        ]
                        # if new_elems[0] % 2 == 0:
                        #     new_elems[0] = new_elems[0] + 1
                        # if new_elems[1] % 2 == 0:
                        #     new_elems[1] = new_elems[1] + 1
                        if new_elems[1] % 2 == 0 # 1-based indexing for new_elems
                            new_elems[1] = new_elems[1] + 1
                        end
                        if new_elems[2] % 2 == 0 # 1-based indexing for new_elems
                            new_elems[2] = new_elems[2] + 1
                        end

                        # Decide if we still can reduce the number of elements
                        # if (2 * self.fracture.mesh.Lx / new_elems[0] > self.sim_prop.maxCellSize) or (2 *
                        #     self.fracture.mesh.Ly / new_elems[1] > self.fracture.mesh.hy / self.fracture.mesh.hx *
                        #     self.sim_prop.maxCellSize):
                        # Correcting for 1-based indexing in new_elems and potential division by zero
                        # Also correcting the order of operations for the second condition
                        cond1 = (2 * self.fracture.mesh.Lx / new_elems[1]) > self.sim_prop.maxCellSize
                        cond2 = (2 * self.fracture.mesh.Ly / new_elems[2]) > (self.fracture.mesh.hy / self.fracture.mesh.hx) * self.sim_prop.maxCellSize
                        # if cond1 || cond2
                        if cond1 || cond2
                            # log.warning("Reduction of cells not possible as minimal cell size would be violated!")
                            @warn "Reduction of cells not possible as minimal cell size would be violated!"
                            # self.sim_prop.meshReductionPossible = False
                            self.sim_prop.meshReductionPossible = false
                        else
                            # log.info("Reducing cell number...")
                            @info "Reducing cell number..."

                            # We need to make sure the injection point stays where it is. We also do this for two points
                            # on same x or y
                            # Initialize variables that might be used later
                            index = 1 # Default initialization
                            cent_point = [0.0, 0.0] # Default initialization
                            reduction_factor = self.sim_prop.meshReductionFactor # Default value

                            # if len(self.fracture.source) == 1:
                            if length(self.fracture.source) == 1
                                # index = self.fracture.source[0]
                                index = self.fracture.source[1] # 1-based indexing
                                # cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]
                                # Assuming CenterCoor is a Matrix{Float64} with rows for elements and columns for x,y
                                cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[1], :] # 1-based indexing

                                # reduction_factor = self.sim_prop.meshReductionFactor
                                reduction_factor = self.sim_prop.meshReductionFactor
                            # elif len(self.fracture.source) == 2:
                            elseif length(self.fracture.source) == 2
                                # index = self.fracture.source[0]
                                index = self.fracture.source[1] # 1-based indexing
                                # cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[0]]
                                cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[1], :] # 1-based indexing

                                # if self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] == \
                                #         self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]:
                                # 1-based indexing for CenterCoor (assuming Matrix{Float64}) and source (Vector{Int})
                                if self.fracture.mesh.CenterCoor[self.fracture.source[1], 1] == self.fracture.mesh.CenterCoor[self.fracture.source[2], 1] # 1-based indexing
                                    # elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] -
                                    #     self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]) / \
                                    #                 self.fracture.mesh.hy)
                                    elems_inter = Int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[1], 2] - # 1-based indexing
                                        self.fracture.mesh.CenterCoor[self.fracture.source[2], 2]) / # 1-based indexing
                                                    self.fracture.mesh.hy)
                                    # new_inter = int(np.ceil(elems_inter/self.sim_prop.meshReductionFactor))
                                    new_inter = Int(ceil(elems_inter/self.sim_prop.meshReductionFactor))
                                    # reduction_factor = elems_inter / new_inter
                                    reduction_factor = elems_inter / new_inter

                                # elif self.fracture.mesh.CenterCoor[self.fracture.source[0]][1] == \
                                #         self.fracture.mesh.CenterCoor[self.fracture.source[1]][1]:
                                elseif self.fracture.mesh.CenterCoor[self.fracture.source[1], 2] == self.fracture.mesh.CenterCoor[self.fracture.source[2], 2] # 1-based indexing
                                    # elems_inter = int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[0]][0] -
                                    #     self.fracture.mesh.CenterCoor[self.fracture.source[1]][0]) / \
                                    #                 self.fracture.mesh.hx)
                                    elems_inter = Int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[1], 1] - # 1-based indexing
                                        self.fracture.mesh.CenterCoor[self.fracture.source[2], 1]) / # 1-based indexing
                                                    self.fracture.mesh.hx)
                                    # new_inter = int(np.ceil(elems_inter / self.sim_prop.meshReductionFactor))
                                    new_inter = Int(ceil(elems_inter / self.sim_prop.meshReductionFactor))
                                    # reduction_factor = elems_inter / new_inter
                                    reduction_factor = elems_inter / new_inter

                                else
                                    # reduction_factor = self.sim_prop.meshReductionFactor
                                    reduction_factor = self.sim_prop.meshReductionFactor
                                end
                                # log.info("The real reduction factor used is " + repr(reduction_factor))
                                @info "The real reduction factor used is $(reduction_factor)"

                            else
                                # index = self.fracture.mesh.locate_element(0., 0.)[0]
                                # Assuming locate_element returns a tuple and we need the first element (1-based index in Julia)
                                # index = self.fracture.mesh.locate_element(0., 0.)[1] # 1-based indexing for tuple
                                # Use findfirst or similar if locate_element is not available
                                # For now, assume a default index
                                # index = 1 # Placeholder
                                # cent_point = np.asarray([0., 0.])
                                # cent_point = [0., 0.]
                                # cent_point = [0.0, 0.0]

                                # reduction_factor = self.sim_prop.meshReductionFactor
                                # reduction_factor = self.sim_prop.meshReductionFactor
                                # These were already initialized above
                            end

                            # row = int(index/self.fracture.mesh.nx)
                            # column = index - self.fracture.mesh.nx * row
                            # Correcting for 1-based indexing:
                            # In Python: row = index // nx, column = index % nx (assuming 0-based index)
                            # In Julia: row = ceil(Int, index / nx), column = index - (row-1)*nx (for 1-based index)
                            # row = Int(index/self.fracture.mesh.nx) # This is float division, not integer division
                            # row = Int(ceil(index / self.fracture.mesh.nx)) # 1-based row
                            # column = index - (row - 1) * self.fracture.mesh.nx # 1-based column within row
                            # Actually, let's re-evaluate this logic based on how mesh indices are stored.
                            # Often, for a mesh with nx columns and ny rows:
                            # 0-based: element i -> row = i // nx, col = i % nx
                            # 1-based: element i -> row = ceil(i / nx), col = mod1(i, nx) (or i - (row-1)*nx)
                            row = Int(ceil(index / self.fracture.mesh.nx)) # 1-based row
                            column = mod1(index, self.fracture.mesh.nx) # 1-based column using mod1

                            # row_frac = (self.fracture.mesh.ny - (row + 1))/row
                            # col_frac = column/(self.fracture.mesh.nx - (column + 1))
                            # Correcting for 1-based indexing and avoiding division by zero
                            # row_frac = (self.fracture.mesh.ny - (row + 1)) / row # Original Python, 0-based row logic
                            # col_frac = column / (self.fracture.mesh.nx - (column + 1)) # Original Python, 0-based col logic
                            # For 1-based indexing:
                            # row_frac = (self.fracture.mesh.ny - row) / (row - 1 + eps()) # Add eps() to avoid div by zero if row=1
                            # if row == 1; row_frac = Inf; end # Handle special case
                            # col_frac = (column - 1) / (self.fracture.mesh.nx - column + eps()) # Add eps() to avoid div by zero if column=nx
                            # if column == self.fracture.mesh.nx; col_frac = Inf; end # Handle special case
                            # Let's simplify and follow the original logic more closely, but adjust for 1-based:
                            # The original seems to be calculating ratios for mesh positioning.
                            # Let's assume row and column are 1-based indices of the element.
                            # row_frac = (ny - row) / (row - 1) # Fraction of rows below to rows above
                            # col_frac = (column - 1) / (nx - column) # Fraction of cols left to cols right
                            # Need to handle cases where denominator is 0.
                            row_frac = (self.fracture.mesh.ny - row) / (row - 1 + eps()) # Avoid division by zero
                            if row == 1
                                row_frac = Inf # Or handle specially if row is 1
                            end
                            col_frac = (column - 1) / (self.fracture.mesh.nx - column + eps()) # Avoid division by zero
                            if column == self.fracture.mesh.nx
                                col_frac = Inf # Or handle specially if column is at the edge
                            end

                            # calculate the new number of cells
                            # new_elems = [int((self.fracture.mesh.nx + np.round(reduction_factor, 0))
                            #                     / reduction_factor),
                            #                 int((self.fracture.mesh.ny + np.round(reduction_factor, 0))
                            #                     / reduction_factor)]
                            # new_elems = [
                            #     Int(round((self.fracture.mesh.nx + round(reduction_factor)) / reduction_factor)),
                            #     Int(round((self.fracture.mesh.ny + round(reduction_factor)) / reduction_factor))
                            # ]
                            # Corrected for potential Float64 return from round
                            new_elems = [
                                Int(floor((self.fracture.mesh.nx + round(reduction_factor)) / reduction_factor)),
                                Int(floor((self.fracture.mesh.ny + round(reduction_factor)) / reduction_factor))
                            ]
                            # if new_elems[0] % 2 == 0:
                            #     new_elems[0] = new_elems[0] + 1
                            # if new_elems[1] % 2 == 0:
                            if new_elems[1] % 2 == 0 # 1-based indexing
                                new_elems[1] = new_elems[1] + 1
                            end
                            if new_elems[2] % 2 == 0 # 1-based indexing
                                new_elems[2] = new_elems[2] + 1
                            end

                            # We calculate the new dimension of the meshed area
                            # Handle potential Inf in col_frac and row_frac
                            col_term = 1.0
                            row_term = 1.0
                            if !isinf(col_frac) && !isnan(col_frac)
                                col_term = 1 / col_frac + 1
                            end
                            if !isinf(row_frac) && !isnan(row_frac)
                                row_term = row_frac + 1
                            end

                            col_rounded = round((new_elems[1] - 1) / col_term) # 1-based indexing for new_elems
                            row_rounded = round((new_elems[2] - 1) / row_term) # 1-based indexing for new_elems

                            new_limits = [
                                [
                                    cent_point[1] - col_rounded * self.fracture.mesh.hx * reduction_factor,
                                    cent_point[1] + (new_elems[1] - col_rounded - 1) * self.fracture.mesh.hx * reduction_factor
                                ],
                                [
                                    cent_point[2] - row_rounded * self.fracture.mesh.hy * reduction_factor,
                                    cent_point[2] + (new_elems[2] - row_rounded - 1) * self.fracture.mesh.hy * reduction_factor
                                ]
                            ]

                            # elems = new_elems
                            elems = new_elems
                            # direction = 'reduce'
                            direction = "reduce"
                            # self.remesh(new_limits, elems, direction)
                            # Assuming remesh is a method of Controller or a function
                            remesh(self, new_limits, elems, direction=direction) # Pass as keyword argument

                            # set all other to zero
                            # side_bools = [False, False, False, False]
                            side_bools = [false, false, false, false]
                        end # if reduction possible

                    # elif status == 12:
                    elseif status == 12
                        # if self.sim_prop.meshExtensionAllDir:
                        if self.sim_prop.meshExtensionAllDir
                            set_mesh_extension_direction(self.sim_prop, ["all"]) # Pass as Vector{String}
                        end
                        front_indices = findall(in(self.fracture.mesh.Frontlist), Fr_n_pls1.EltTip) # 1-based indices

                        bool1 = any(front_indices .<= Fr_n_pls1.mesh.nx - 3)
                        # 2. (front_indices[front_indices > Fr_n_pls1.mesh.nx - 3] <= 2 * (Fr_n_pls1.mesh.nx - 3) + 1).any()
                        #    Filter front_indices > (nx-3), then check if any of those are <= 2*(nx-3)+1
                        filtered_indices = front_indices[front_indices .> Fr_n_pls1.mesh.nx - 3]
                        bool2 = any(filtered_indices .<= 2 * (Fr_n_pls1.mesh.nx - 3) + 1)
                        # 3. (front_indices[front_indices >= 2 * (Fr_n_pls1.mesh.nx - 2)] % 2 == 0).any()
                        #    Filter front_indices >= 2*(nx-2), then check if any of those indices are even
                        filtered_indices2 = front_indices[front_indices .>= 2 * (Fr_n_pls1.mesh.nx - 2)]
                        bool3 = any(filtered_indices2 .% 2 .== 0)
                        # 4. (front_indices[front_indices >= 2 * (Fr_n_pls1.mesh.nx - 2)] % 2 != 0).any()
                        #    Same filter, check if any are odd
                        bool4 = any(filtered_indices2 .% 2 .!= 0)
                        # side_bools = [bool1, bool2, bool3, bool4]
                        side_bools = [bool1, bool2, bool3, bool4]
                        # side_bools is a set of booleans telling us which sides are touched by the remeshing.
                        # First boolean represents bottom, top, left, right

                        # if not self.sim_prop.meshExtensionAllDir:
                        if !self.sim_prop.meshExtensionAllDir
                            # Simplifying:
                            mesh_ext_dir = falses(4)
                            if isa(self.sim_prop.meshExtension, Vector) && length(self.sim_prop.meshExtension) >= 4
                                mesh_ext_dir = self.sim_prop.meshExtension[1:4] # Assuming first 4 elements correspond to directions
                            end
                            side_bools_log = falses(4)
                            if length(side_bools) >= 4
                                side_bools_log = side_bools[1:4]
                            end
                            compress = !any(mesh_ext_dir .& side_bools_log) || (sum(side_bools_log) > 3)
                        end # if not meshExtensionAllDir
                    end # if status == 16 elseif status == 12

                    # This is the classical remeshing where the sides of the elements are multiplied by a constant.
                    # if compress:
                    if compress
                        @info "Remeshing by compressing the domain..."

                        # We need to make sure the injection point stays where it is. We also do this for two points
                        # on same x or y
                        # Initialize variables
                        index = 1 # Default
                        cent_point = [0.0, 0.0] # Default
                        compression_factor = self.sim_prop.remeshFactor # Default

                        # if len(self.fracture.source) == 1:
                        if length(self.fracture.source) == 1

                            index = self.fracture.source[1] # 1-based
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[1], :] # 1-based

                            compression_factor = self.sim_prop.remeshFactor
                        elseif length(self.fracture.source) == 2
                            index = self.fracture.source[1] # 1-based
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[1], :] # 1-based


                            if self.fracture.mesh.CenterCoor[self.fracture.source[1], 1] == self.fracture.mesh.CenterCoor[self.fracture.source[2], 1] # 1-based

                                elems_inter = Int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[1], 2] - # 1-based
                                                        self.fracture.mesh.CenterCoor[self.fracture.source[2], 2]) / # 1-based
                                                    self.fracture.mesh.hy)
                                # new_inter = int(np.ceil(elems_inter / self.sim_prop.remeshFactor))
                                new_inter = Int(ceil(elems_inter / self.sim_prop.remeshFactor))
                                # compression_factor = elems_inter / new_inter
                                compression_factor = elems_inter / new_inter

                            elseif self.fracture.mesh.CenterCoor[self.fracture.source[1], 2] == self.fracture.mesh.CenterCoor[self.fracture.source[2], 2] # 1-based

                                elems_inter = Int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[1], 1] - # 1-based
                                                        self.fracture.mesh.CenterCoor[self.fracture.source[2], 1]) / # 1-based
                                                    self.fracture.mesh.hx)
                                new_inter = Int(ceil(elems_inter / self.sim_prop.remeshFactor))
                                # compression_factor = elems_inter / new_inter
                                compression_factor = elems_inter / new_inter

                            else
                                # compression_factor = self.sim_prop.remeshFactor
                                compression_factor = self.sim_prop.remeshFactor
                            end
                            # log.info("The real reduction factor used is " + repr(compression_factor))
                            @info "The real reduction factor used is $(compression_factor)"


                        end
                        row = Int(ceil(index / self.fracture.mesh.nx)) # 1-based row
                        column = mod1(index, self.fracture.mesh.nx) # 1-based column using mod1

                        row_frac = (self.fracture.mesh.ny - row) / (row - 1 + eps()) # Avoid division by zero
                        if row == 1
                            row_frac = Inf
                        end
                        col_frac = (column - 1) / (self.fracture.mesh.nx - column + eps()) # Avoid division by zero
                        if column == self.fracture.mesh.nx
                            col_frac = Inf
                        end

                        # We calculate the new dimension of the meshed area
                        # Handle potential Inf in col_frac and row_frac
                        col_term = 1.0
                        row_term = 1.0
                        if !isinf(col_frac) && !isnan(col_frac)
                            col_term = 1 / col_frac + 1
                        end
                        if !isinf(row_frac) && !isnan(row_frac)
                            row_term = row_frac + 1
                        end

                        col_rounded = round((self.fracture.mesh.nx - 1) / col_term)
                        row_rounded = round((self.fracture.mesh.ny - 1) / row_term)

                        new_limits = [
                            [
                                cent_point[1] - col_rounded * self.fracture.mesh.hx * compression_factor,
                                cent_point[1] + (self.fracture.mesh.nx - col_rounded - 1) * self.fracture.mesh.hx * compression_factor
                            ],
                            [
                                cent_point[2] - row_rounded * self.fracture.mesh.hy * compression_factor,
                                cent_point[2] + (self.fracture.mesh.ny - row_rounded - 1) * self.fracture.mesh.hy * compression_factor
                            ]
                        ]
                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]

                        if isempty(intersect(self.fracture.mesh.CenterElts, [index])) # intersect needs vectors
                            compression_factor = 10
                        end

                        remesh(self, new_limits, elems, rem_factor=compression_factor)

                        side_bools = [false, false, false, false]

                    else
                        nx_init = self.fracture.mesh.nx
                        ny_init = self.fracture.mesh.ny
                        # Create a copy of side_bools to modify inside loop if needed
                        side_bools_copy = copy(side_bools)
                        mesh_ext_dir = trues(4)
                        if isa(self.sim_prop.meshExtension, Vector) && length(self.sim_prop.meshExtension) >= 4
                            mesh_ext_dir = self.sim_prop.meshExtension[1:4]
                        end

                        for side in 1:4
                            # if np.asarray(np.asarray(self.sim_prop.meshExtension) * np.asarray(side_bools))[side]:
                            # if mesh_ext_dir[side] && side_bools[side] # Check if extension is allowed for this side and it's touched
                            if mesh_ext_dir[side] && side_bools_copy[side] # Use the copy to check
                                if side == 1 # Corresponds to Python's side == 0 (bottom)
                                    elems_add = Int(nx_init * (self.sim_prop.meshExtensionFactor[side] - 1)) # nx_init for side 1 (bottom/top)
                                    if elems_add % 2 != 0
                                        elems_add = elems_add + 1
                                    end

                                    # if not self.sim_prop.symmetric:
                                    if !self.sim_prop.symmetric
                                        # log.info("Remeshing by extending towards negative y...")
                                        @info "Remeshing by extending towards negative y..."
                                        # Assuming domainLimits is [y_min, y_max, x_min, x_max] in Python (0-based indices)
                                        # In Julia, it would be [y_min, y_max, x_min, x_max] -> indices 1,2,3,4
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[3], self.fracture.mesh.domainLimits[4]], # x limits
                                            [self.fracture.mesh.domainLimits[1] - elems_add * self.fracture.mesh.hy, self.fracture.mesh.domainLimits[2]] # y limits
                                        ]
                                    else
                                        # log.info("Remeshing by extending in vertical direction to keep symmetry...")
                                        @info "Remeshing by extending in vertical direction to keep symmetry..."
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[3], self.fracture.mesh.domainLimits[4]],
                                            [
                                                self.fracture.mesh.domainLimits[1] - elems_add * self.fracture.mesh.hy/2,
                                                self.fracture.mesh.domainLimits[2] + elems_add * self.fracture.mesh.hy/2
                                            ]
                                        ]
                                        # side_bools[2] = false # Disable opposite side extension (Python 1 -> Julia 2)
                                        side_bools_copy[2] = false # Modify the copy
                                    end

                                    # direction = 'bottom'
                                    direction = "bottom"

                                    # elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]
                                    elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]


                                end
                                # if side == 1: # This should be side == 2 in 1-based indexing
                                if side == 2 # Corresponds to Python's side == 1 (top)

                                    # elems_add = int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1))
                                    elems_add = Int(nx_init * (self.sim_prop.meshExtensionFactor[side] - 1)) # nx_init for side 2 (top)
                                    # if elems_add % 2 != 0:
                                    #     elems_add = elems_add + 1
                                    if elems_add % 2 != 0
                                        elems_add = elems_add + 1
                                    end

                                    # if not self.sim_prop.symmetric:
                                    if !self.sim_prop.symmetric
                                        # log.info("Remeshing by extending towards positive y...")
                                        @info "Remeshing by extending towards positive y..."
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[3], self.fracture.mesh.domainLimits[4]],
                                            [self.fracture.mesh.domainLimits[1], self.fracture.mesh.domainLimits[2] + elems_add * self.fracture.mesh.hy]
                                        ]
                                    else
                                        # log.info("Remeshing by extending in vertical direction to keep symmetry...")
                                        @info "Remeshing by extending in vertical direction to keep symmetry..."
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[3], self.fracture.mesh.domainLimits[4]],
                                            [
                                                self.fracture.mesh.domainLimits[1] - elems_add * self.fracture.mesh.hy/2,
                                                self.fracture.mesh.domainLimits[2] + elems_add * self.fracture.mesh.hy/2
                                            ]
                                        ]
                                        # side_bools[1] = false # Disable opposite side extension (Python 0 -> Julia 1)
                                        side_bools_copy[1] = false # Modify the copy
                                    end

                                    direction = "top"

                                    elems = [self.fracture.mesh.nx, self.fracture.mesh.ny + elems_add]
                                end
                                # if side == 2: # This should be side == 3 in 1-based indexing
                                if side == 3 # Corresponds to Python's side == 2 (left)
                                    elems_add = Int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1)) # ny_init for side 3 (left/right)
                                    if elems_add % 2 != 0
                                        elems_add = elems_add + 1
                                    end

                                    # if not self.sim_prop.symmetric:
                                    if !self.sim_prop.symmetric
                                        # log.info("Remeshing by extending towards negative x...")
                                        @info "Remeshing by extending towards negative x..."
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[3] - elems_add * self.fracture.mesh.hx, self.fracture.mesh.domainLimits[4]],
                                            [self.fracture.mesh.domainLimits[1], self.fracture.mesh.domainLimits[2]]
                                        ]
                                    else
                                        
                                        @info "Remeshing by extending in horizontal direction to keep symmetry..."
                                        new_limits = [
                                            [
                                                self.fracture.mesh.domainLimits[3] - elems_add * self.fracture.mesh.hx/2,
                                                self.fracture.mesh.domainLimits[4] + elems_add * self.fracture.mesh.hx/2
                                            ],
                                            [self.fracture.mesh.domainLimits[1], self.fracture.mesh.domainLimits[2]]
                                        ]
                                        if length(side_bools_copy) >= 4
                                            side_bools_copy[4] = false # Modify the copy
                                        end
                                    end

                                    direction = "left"
                                    elems = [self.fracture.mesh.nx + elems_add, self.fracture.mesh.ny]

                                end
                                if side == 4 # Corresponds to Python's side == 3 (right)

                                    elems_add = Int(ny_init * (self.sim_prop.meshExtensionFactor[side] - 1)) # ny_init for side 4 (right)
                                    if elems_add % 2 != 0
                                        elems_add = elems_add + 1
                                    end

                                    # if not self.sim_prop.symmetric:
                                    if !self.sim_prop.symmetric
                                        @info "Remeshing by extending towards positive x..."
                                        new_limits = [
                                            [self.fracture.mesh.domainLimits[3], self.fracture.mesh.domainLimits[4] + elems_add * self.fracture.mesh.hx],
                                            [self.fracture.mesh.domainLimits[1], self.fracture.mesh.domainLimits[2]]
                                        ]
                                    else
                                        # log.info("Remeshing by extending in horizontal direction to keep symmetry...")
                                        @info "Remeshing by extending in horizontal direction to keep symmetry..."
                                        new_limits = [
                                            [
                                                self.fracture.mesh.domainLimits[3] - elems_add * self.fracture.mesh.hx/2,
                                                self.fracture.mesh.domainLimits[4] + elems_add * self.fracture.mesh.hx/2
                                            ],
                                            [self.fracture.mesh.domainLimits[1], self.fracture.mesh.domainLimits[2]]
                                        ]
                                        if length(side_bools_copy) >= 3
                                            side_bools_copy[3] = false # Modify the copy
                                        end
                                    end
                                    direction = "right"
                                    elems = [self.fracture.mesh.nx + elems_add, self.fracture.mesh.ny]

                                end

                                remesh(self, new_limits, elems, direction=direction)
                                # side_bools[side] = False # After remeshing this side, disable it
                                side_bools_copy[side] = false # Modify the copy
                            end # if mesh extension allowed and side touched
                        end # for side in 1:4
                    end # if compress else

                    # Handle any remaining sides that need remeshing (fallback to compression)
                    # if np.asarray(side_bools).any():
                    # if any(side_bools) # Check the original side_bools
                    if any(side_bools) # Check the original side_bools vector
                        @info "Remeshing by compressing the domain (fallback)..."

                        # Re-calculate compression parameters (similar to earlier)
                        # Initialize variables
                        index = 1 # Default
                        cent_point = [0.0, 0.0] # Default
                        compression_factor = self.sim_prop.remeshFactor # Default

                        if length(self.fracture.source) == 1
                            index = self.fracture.source[1] # 1-based
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[1], :] # 1-based

                            # compression_factor = self.sim_prop.remeshFactor
                            compression_factor = self.sim_prop.remeshFactor
                        # elif len(self.fracture.source) == 2:
                        elseif length(self.fracture.source) == 2
                            index = self.fracture.source[1]
                            cent_point = self.fracture.mesh.CenterCoor[self.fracture.source[1], :] # 1-based

                            if self.fracture.mesh.CenterCoor[self.fracture.source[1], 1] == self.fracture.mesh.CenterCoor[self.fracture.source[2], 1] # 1-based
                                elems_inter = Int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[1], 2] - # 1-based
                                                        self.fracture.mesh.CenterCoor[self.fracture.source[2], 2]) / # 1-based
                                                    self.fracture.mesh.hy)
                                new_inter = Int(ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            elseif self.fracture.mesh.CenterCoor[self.fracture.source[1], 2] == self.fracture.mesh.CenterCoor[self.fracture.source[2], 2] # 1-based
                                elems_inter = Int(abs(self.fracture.mesh.CenterCoor[self.fracture.source[1], 1] - # 1-based
                                                        self.fracture.mesh.CenterCoor[self.fracture.source[2], 1]) / # 1-based
                                                    self.fracture.mesh.hx)
                                new_inter = Int(ceil(elems_inter / self.sim_prop.remeshFactor))
                                compression_factor = elems_inter / new_inter

                            else
                                # compression_factor = self.sim_prop.remeshFactor
                                compression_factor = self.sim_prop.remeshFactor
                            end
                            # log.info("The real reduction factor used is " + repr(compression_factor))
                            @info "The real reduction factor used is $(compression_factor)"


                        end
                        row = Int(ceil(index / self.fracture.mesh.nx)) # 1-based row
                        column = mod1(index, self.fracture.mesh.nx) # 1-based column using mod1

                        row_frac = (self.fracture.mesh.ny - row) / (row - 1 + eps()) # Avoid division by zero
                        if row == 1
                            row_frac = Inf
                        end
                        col_frac = (column - 1) / (self.fracture.mesh.nx - column + eps()) # Avoid division by zero
                        if column == self.fracture.mesh.nx
                            col_frac = Inf
                        end

                        # We calculate the new dimension of the meshed area
                        # Handle potential Inf in col_frac and row_frac
                        col_term = 1.0
                        row_term = 1.0
                        if !isinf(col_frac) && !isnan(col_frac)
                            col_term = 1 / col_frac + 1
                        end
                        if !isinf(row_frac) && !isnan(row_frac)
                            row_term = row_frac + 1
                        end

                        col_rounded = round((self.fracture.mesh.nx - 1) / col_term)
                        row_rounded = round((self.fracture.mesh.ny - 1) / row_term)

                        new_limits = [
                            [
                                cent_point[1] - col_rounded * self.fracture.mesh.hx * compression_factor,
                                cent_point[1] + (self.fracture.mesh.nx - col_rounded - 1) * self.fracture.mesh.hx * compression_factor
                            ],
                            [
                                cent_point[2] - row_rounded * self.fracture.mesh.hy * compression_factor,
                                cent_point[2] + (self.fracture.mesh.ny - row_rounded - 1) * self.fracture.mesh.hy * compression_factor
                            ]
                        ]
                        elems = [self.fracture.mesh.nx, self.fracture.mesh.ny]

                        if isempty(intersect(self.fracture.mesh.CenterElts, [index])) # intersect needs vectors
                            # compression_factor = 10
                            compression_factor = 10
                        end
                        remesh(self, new_limits, elems, rem_factor=compression_factor)
                    end # if any(side_bools) fallback

                    @info "\nRemeshed at $(self.fracture.time)"

                else
                    @info "Reached end of the domain. Exiting..."
                    # break
                    break
                end # if enableRemeshing

            elseif status == 14
                # fracture fully closed
                output(self, Fr_n_pls1) # Assuming output is a function
                if self.PstvInjJmp === nothing
                    print("Fracture is fully closed.\n\nDo you want to jump to the time of next positive injection? [y/n]")
                    inp = readline()
                    t0 = time() # Assuming time() from Dates
                    while !(inp in ["y", "Y", "n", "N"]) && (time() - t0) < 600
                        # inp = input("Press y or n")
                        print("Press y or n: ")
                        inp = readline()
                    end
                    if inp in ["y", "Y"] || (time() - t0) >= 600
                        self.PstvInjJmp = true
                    else
                        self.PstvInjJmp = false
                    end
                end

                if self.PstvInjJmp
                    self.sim_prop.solveDeltaP = false
                    time_larger = findall(Fr_n_pls1.time .<= self.injection_prop.injectionRate[1, :]) # 1-based indexing for rows
                    pos_inj = findall(self.injection_prop.injectionRate[2, :] .> 0) # 1-based indexing for rows
                    Qact = get_injection_rate(self.injection_prop, self.fracture.time, self.fracture) # Assuming function
                    after_time = intersect(time_larger, pos_inj)
                    if length(after_time) == 0 && maximum(Qact) == 0.0
                        @warn "Positive injection not found!"
                        break
                    elseif length(after_time) == 0
                        jump_to = self.fracture.time + self.fracture.time * 0.1
                    else
                        jump_to = minimum(self.injection_prop.injectionRate[1, after_time]) # 1-based indexing for rows and columns
                    end
                    Fr_n_pls1.time = jump_to
                elseif inp in ["n", "N"]
                    self.sim_prop.solveDeltaP = true
                end
                self.fullyClosed = true
                self.fracture = deepcopy(Fr_n_pls1)
            elseif status == 17
                @info "The fracture is advancing more than two cells in a row at time $(self.fracture.time)"

                if self.TmStpReductions == self.sim_prop.maxReattemptsFracAdvMore2Cells
                    @warn "We can not reduce the time step more than that"
                    if self.sim_prop.collectPerfData
                        local file_address
                        if self.sim_prop.saveToDisk
                            file_address = joinpath(self.sim_prop.get_outputFolder(), "perf_data.dat")
                        else
                            file_address = "./perf_data.dat"
                        end
                        jldsave(file_address; perfData=self.perfData)
                    end
                    @info "\n\n---Simulation failed---"
                    error("Simulation failed.")
                else
                    @info "- limiting the time step - "
                    if isa(self.sim_prop.tmStpPrefactor, Matrix) # Assuming it's a 2D array like in Python
                        times_le_current = findall(self.sim_prop.tmStpPrefactor[1, :] .<= self.fracture.time) # 1-based indexing for rows
                        if !isempty(times_le_current)
                            indxCurTime = maximum(times_le_current)
                            self.sim_prop.tmStpPrefactor[2, indxCurTime] *= 0.5^self.TmStpReductions # 1-based indexing for rows
                        end
                    else
                        self.sim_prop.tmStpPrefactor *= 0.5^self.TmStpReductions
                    end
                    self.TmStpReductions += 1
                end
            else
                # time step failed
                @warn "\n$(get(self.errorMessages, status, "Unknown error status: $status"))" # Safe get from dict
                @warn "\nTime step failed at = $(self.fracture.time)"
                queue_element = self.fr_queue[(self.successfulTimeSteps % 5) + 1]
                if queue_element === nothing || self.chkPntReattmpts == 4
                    if self.sim_prop.collectPerfData
                        local file_address
                        if self.sim_prop.saveToDisk
                            file_address = joinpath(self.sim_prop.get_outputFolder(), "perf_data.dat")
                        else
                            file_address = "./perf_data.dat"
                        end
                        jldsave(file_address; perfData=self.perfData)
                    end

                    @info "\n\n---Simulation failed---"
                    error("Simulation failed.")
                else
                    # decrease time step pre-factor before taking the next fracture in the queue having last
                    # five time steps
                    current_PreFctr = 0.0 # Initialize
                    if isa(self.sim_prop.tmStpPrefactor, Matrix) # Assuming it's a 2D array like in Python
                        times_le_current = findall(self.sim_prop.tmStpPrefactor[1, :] .<= self.fracture.time) # 1-based indexing for rows
                        indxCurTime = 1 # Default
                        if !isempty(times_le_current)
                            indxCurTime = maximum(times_le_current)
                            self.sim_prop.tmStpPrefactor[2, indxCurTime] *= 0.8 # 1-based indexing for rows
                            current_PreFctr = self.sim_prop.tmStpPrefactor[2, indxCurTime] # 1-based indexing for rows
                        end
                    else
                        self.sim_prop.tmStpPrefactor *= 0.8
                        current_PreFctr = self.sim_prop.tmStpPrefactor
                    end

                    # self.chkPntReattmpts += 1
                    self.chkPntReattmpts += 1

                    idx_to_get = mod(self.successfulTimeSteps - self.chkPntReattmpts, 5) + 1
                    # Ensure idx_to_get is within valid range [1, 5]
                    if idx_to_get < 1
                        idx_to_get += 5
                    end
                    # Now get the fracture object
                    fracture_to_restore = self.fr_queue[idx_to_get]
                    # Check if it's nothing
                    if fracture_to_restore === nothing
                        # This shouldn't happen if chkPntReattmpts <= 4 and the queue is properly initialized.
                        # But let's handle it gracefully.
                        @error "Attempted to restore from an uninitialized checkpoint in fr_queue[$idx_to_get]"
                        error("Simulation failed due to uninitialized checkpoint.")
                    end

                    remesh(self.solid_prop, fracture_to_restore.mesh) # Assuming remesh function for solid_prop
                    remesh(self.injection_prop, fracture_to_restore.mesh, self.fracture.mesh) # Assuming remesh function for injection_prop

                    self.fracture = deepcopy(fracture_to_restore)
                    @warn "Time step have failed despite of reattempts with slightly smaller/bigger time steps...\nGoing $(5 - self.chkPntReattmpts) time steps back and re-attempting with the time step pre-factor of $(current_PreFctr)"

                    # self.failedTimeSteps += 1
                    self.failedTimeSteps += 1
                end # if queue element is nothing or max reattempts
            end # if status == 1 elseif status in [12,16] elseif status == 14 elseif status == 17 else

            self.TmStpCount += 1
        end # while loop

        println("\n")
        @info "Final time = $(self.fracture.time)"
        @info "-----Simulation finished------"
        @info "number of time steps = $(self.successfulTimeSteps)"
        @info "failed time steps = $(self.failedTimeSteps)"
        @info "number of remeshings = $(self.remeshings)"

        # Assuming PyPlot is used
        plt.show(block=false) # PyPlot
        plt.close("all") # PyPlot

        if self.sim_prop.collectPerfData
            local file_address
            if self.sim_prop.saveToDisk
                file_address = joinpath(self.sim_prop.get_outputFolder(), "perf_data.dat")
            else
                file_address = "./perf_data.dat"
            end
            mkpath(dirname(file_address))
            jldsave(file_address; perfData=self.perfData)
        end
        return true
    end # function run
    #-----------------------------------------------------------------------------------------------------------------------

    """
        advance_time_step(self, Frac, C, timeStep, perfNode=nothing)

        This function advances the fracture by the given time step. In case of failure, reattempts are made with smaller
        time steps.

        # Arguments
        - `self::Controller`: the controller object
        - `Frac::Any`: fracture object from the last time step
        - `C::Union{Nothing, Array{Float64}}`: the elasticity matrix
        - `timeStep::Float64`: time step to be attempted
        - `perfNode::Union{Nothing, Any}`: An IterationProperties instance to store performance data

        # Returns
        - `Tuple{Int, Any}`: (exitstatus, Fr) -- see documentation for possible values of exitstatus and Fr is the fracture after advancing time step.
    """
    function advance_time_step(self::Controller, Frac::Any, C::Union{Nothing, Array{Float64}}, timeStep::Float64, perfNode::Union{Nothing, Any}=nothing)::Tuple{Int, Any}
        # loop for reattempting time stepping in case of failure.
        for i in 1:(self.sim_prop.maxReattempts)
            # smaller time step to reattempt time stepping; equal to the given time step on first iteration
            tmStp_to_attempt = timeStep * self.sim_prop.reAttemptFactor ^ i

            # try larger prefactor
            if i > self.sim_prop.maxReattempts/2 - 1
                tmStp_to_attempt = timeStep * (1/self.sim_prop.reAttemptFactor)^(i+1 - self.sim_prop.maxReattempts/2)
            end

            # check for final time
            if Frac.time + tmStp_to_attempt > 1.01 * self.sim_prop.finalTime
                @info "$(Frac.time + tmStp_to_attempt)"
                return -99, Frac 
            end
            println('\n')
            @info "Evaluating solution at time = $(Frac.time+tmStp_to_attempt) ..."
            @debug "Attempting time step of $(tmStp_to_attempt) sec..."

            perfNode_TmStpAtmpt = instrument_start("time step attempt", perfNode) # Assuming instrument_start is defined

            self.attmptedTimeStep = tmStp_to_attempt
            status, Fr = attempt_time_step(Frac,
                                            C,
                                            self.solid_prop,
                                            self.fluid_prop,
                                            self.sim_prop,
                                            self.injection_prop,
                                            tmStp_to_attempt,
                                            perfNode_TmStpAtmpt) # Assuming attempt_time_step is defined

            if perfNode_TmStpAtmpt !== nothing
                instrument_close(perfNode, perfNode_TmStpAtmpt,
                                    nothing, length(Frac.EltCrack), status == 1,
                                    get(self.errorMessages, status, "Unknown error"), Frac.time) # Assuming instrument_close is defined
                push!(perfNode.attempts_data, perfNode_TmStpAtmpt) # Assuming attempts_data is a Vector
            end
            if status in [1, 12, 14, 16, 17]
                break
            else
                @warn "$(get(self.errorMessages, status, "Unknown error status: $status"))" # Safe get from dict
                @warn "Time step failed..."
            end
        end
        return status, Fr
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        output(self, Fr_advanced)

        This function plot the fracture footprint and/or save file to disk according to the parameters set in the
        simulation properties. See documentation of SimulationProperties class to get the details of parameters which
        determines when and how the output is made.

        # Arguments
        - `self::Controller`: the controller object
        - `Fr_advanced::Any`: fracture after time step is advanced.

        # Returns
        - nothing
    """
    function output(self::Controller, Fr_advanced::Any)::Nothing
        in_req_TSrs = false
        # current time in the time series given at which the solution is to be evaluated
        if self.sim_prop.get_solTimeSeries() !== nothing && self.sim_prop.plotATsolTimeSeries
            if Fr_advanced.time in self.sim_prop.get_solTimeSeries()
                in_req_TSrs = true
            end
        end

        # if the time is final time
        if Fr_advanced.time >= self.sim_prop.finalTime
            in_req_TSrs = true
        end

        if self.sim_prop.saveToDisk

            save_TP_exceeded = false
            save_TS_exceeded = false

            # check if save time period is exceeded since last save
            if self.sim_prop.saveTimePeriod !== nothing
                if Fr_advanced.time >= self.lastSavedTime + self.sim_prop.saveTimePeriod
                    save_TP_exceeded = true
                end
            end

            # check if the number of time steps since last save exceeded
            if self.sim_prop.saveTSJump !== nothing
                if self.successfulTimeSteps % self.sim_prop.saveTSJump == 0
                    save_TS_exceeded = true
                end
            end
            if save_TP_exceeded || in_req_TSrs || save_TS_exceeded

                # save fracture to disk
                @info "Saving solution at $(Fr_advanced.time)..."
                Fr_advanced.SaveFracture(joinpath(self.sim_prop.get_outputFolder(), 
                                                self.sim_prop.get_simulation_name() * "_file_" * string(self.lastSavedFile)))
                self.lastSavedFile += 1
                @info "Done! "

                self.lastSavedTime = Fr_advanced.time
            end
        end

        # plot fracture variables
        if self.sim_prop.plotFigure

            plot_TP_exceeded = false
            plot_TS_exceeded = false

            # check if plot time period is exceeded since last plot
            if self.sim_prop.plotTimePeriod !== nothing
                if Fr_advanced.time >= self.lastPlotTime + self.sim_prop.plotTimePeriod
                    plot_TP_exceeded = true
                end
            end

            # check if the number of time steps since last plot exceeded
            if self.sim_prop.plotTSJump !== nothing
                if self.successfulTimeSteps % self.sim_prop.plotTSJump == 0
                    plot_TS_exceeded = true
                end
            end

            if plot_TP_exceeded || in_req_TSrs || plot_TS_exceeded
                for (index, plt_var) in enumerate(self.sim_prop.plotVar)
                    @info "Plotting solution at $(Fr_advanced.time)..."
                    plot_prop = PlotProperties() # Assuming PlotProperties is defined

                    if self.Figures[index] !== nothing
                        axes = self.Figures[index].get_axes() # Assuming PyPlot object
                        plt.figure(self.Figures[index].number) # Assuming PyPlot
                        plt.clf() # Assuming PyPlot
                        self.Figures[index].add_axes(axes[1]) # 1-based indexing for axes, assuming PyPlot
                    end

                    if plt_var == "footprint"
                        # footprint is plotted if variable to plot is not given
                        plot_prop.lineColor = "b"
                        if self.sim_prop.plotAnalytical
                            self.Figures[index] = plot_footprint_analytical(self.sim_prop.analyticalSol,
                                                                        self.solid_prop,
                                                                        self.injection_prop,
                                                                        self.fluid_prop,
                                                                        [Fr_advanced.time],
                                                                        fig=self.Figures[index],
                                                                        h=self.sim_prop.height,
                                                                        samp_cell=nothing,
                                                                        plot_prop=plot_prop,
                                                                        gamma=self.sim_prop.aspectRatio,
                                                                        inj_point=self.injection_prop.sourceCoordinates)
                        end

                        self.Figures[index] = Fr_advanced.plot_fracture(variable="mesh",
                                                                        mat_properties=self.solid_prop,
                                                                        projection="2D",
                                                                        backGround_param=self.sim_prop.bckColor,
                                                                        fig=self.Figures[index],
                                                                        plot_prop=plot_prop)

                        plot_prop.lineColor = "k"
                        self.Figures[index] = Fr_advanced.plot_fracture(variable="footprint",
                                                                        projection="2D",
                                                                        fig=self.Figures[index],
                                                                        plot_prop=plot_prop)

                    
                    elseif plt_var in ("fluid velocity as vector field", "fvvf", "fluid flux as vector field", "ffvf")
                        
                        if (isa(self.fluid_prop.viscosity, Number)) && self.fluid_prop.viscosity == 0.0
                            error("ERROR: if the fluid viscosity is equal to 0 does not make sense to ask a plot of the fluid velocity or fluid flux")
                        end
                        if isa(self.fluid_prop.viscosity, Array) && minimum(self.fluid_prop.viscosity) == 0.0
                            error("ERROR: if the fluid viscosity is equal to 0 does not make sense to ask a plot of the fluid velocity or fluid flux")
                        elseif self.sim_prop._SimulationProperties__tipAsymptote == "K" # Assuming this private attribute exists
                            error("ERROR: if tipAsymptote == K, does not make sense to ask a plot of the fluid velocity or fluid flux")
                        end
                        self.Figures[index] = Fr_advanced.plot_fracture(variable="mesh",
                                                                        mat_properties=self.solid_prop,
                                                                        projection="2D",
                                                                        backGround_param=self.sim_prop.bckColor,
                                                                        fig=self.Figures[index],
                                                                        plot_prop=plot_prop)

                        plot_prop.lineColor = "k"
                        self.Figures[index] = Fr_advanced.plot_fracture(variable="footprint",
                                                                        projection="2D",
                                                                        fig=self.Figures[index],
                                                                        plot_prop=plot_prop)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable=plt_var,
                                                                        projection="2D_vectorfield",
                                                                        mat_properties=self.solid_prop,
                                                                        fig=self.Figures[index])
                    else
                        if self.sim_prop.plotAnalytical
                            proj = supported_projections[plt_var][1]
                            self.Figures[index] = plot_analytical_solution(regime=self.sim_prop.analyticalSol,
                                                                        variable=plt_var,
                                                                        mat_prop=self.solid_prop,
                                                                        inj_prop=self.injection_prop,
                                                                        fluid_prop=self.fluid_prop,
                                                                        projection=proj,
                                                                        time_srs=[Fr_advanced.time],
                                                                        h=self.sim_prop.height,
                                                                        gamma=self.sim_prop.aspectRatio)
                        end

                        fig_labels = LabelProperties(plt_var, "whole mesh", "2D") # Assuming LabelProperties is defined
                        fig_labels.figLabel = ""
                        self.Figures[index] = Fr_advanced.plot_fracture(variable="footprint",
                                                                        projection="2D",
                                                                        fig=self.Figures[index],
                                                                        labels=fig_labels)

                        self.Figures[index] = Fr_advanced.plot_fracture(variable=plt_var,
                                                                        projection="2D_clrmap",
                                                                        mat_properties=self.solid_prop,
                                                                        fig=self.Figures[index],
                                                                        elements=get_elements(suitable_elements[plt_var], Fr_advanced)) # Assuming get_elements is defined
                        # plotting source elements
                        self.Figures[index] = plot_injection_source(Fr_advanced,
                                                fig=self.Figures[index]) # Assuming plot_injection_source is defined
                    end

                    # plotting closed cells
                    if length(Fr_advanced.closed) > 0
                        plot_prop.lineColor = "orangered"
                        self.Figures[index] = Fr_advanced.mesh.identify_elements(Fr_advanced.closed,
                                                                                fig=self.Figures[index],
                                                                                plot_prop=plot_prop,
                                                                                plot_mesh=false,
                                                                                print_number=false) # Assuming identify_elements is a method
                    end
                    plt.ion() # Assuming PyPlot
                    plt.pause(0.4) # Assuming PyPlot
                end # for loop over plot variables
                
                # set figure position
                if self.setFigPos
                    for i in 1:length(self.sim_prop.plotVar)
                        plt.figure(i + 1) # Assuming PyPlot
                        mngr = plt.get_current_fig_manager() # Assuming PyPlot
                        x_offset = 650 * (i - 1)d
                        y_ofset = 50
                        if i >= 3
                            x_offset = (i - 3) * 650
                            y_ofset = 500
                        end
                    end
                    self.setFigPos = false
                end

                @info "Done! "
                if self.sim_prop.blockFigure
                    println("click on the window to continue...")
                    plt.waitforbuttonpress() # Assuming PyPlot
                end
                self.lastPlotTime = Fr_advanced.time
            end
        end
        return nothing
    end


    #------------------------------------------------------------------------------------------------------------------
    """
        get_time_step(self)

        This function calculates the appropriate time step. It takes minimum of the time steps evaluated according to
        the following:

            - time step evaluated with the current front velocity to limit the increase in length compared to a cell \
                length
            - time step evaluated with the current front velocity to limit the increase in length compared to the \
                current fracture length
            - time step evaluated with the injection rate in the coming time step
            - time step evaluated to limit the change in total volume of the fracture
        In addition, the limit on the time step and the times at which the solution is required are also taken in
        account to get the appropriate time step.

        # Arguments
        - `self::Controller`: the controller object

        # Returns
        - `Float64`: the appropriate time step.
    """
    function get_time_step(self::Controller)::Float64
        time_step_given = false
        time_step = 0.0 # Initialize time_step
        
        if self.sim_prop.fixedTmStp !== nothing
            # fixed time step
            if isa(self.sim_prop.fixedTmStp, Number) # Number includes Float64, Int, etc. in Julia
                time_step = self.sim_prop.fixedTmStp
                time_step_given = true
            elseif isa(self.sim_prop.fixedTmStp, Array) && ndims(self.sim_prop.fixedTmStp) == 2 && size(self.sim_prop.fixedTmStp, 1) == 2
                # fixed time step list is given
                times_past = findall(self.sim_prop.fixedTmStp[1, :] .<= self.fracture.time) # 1-based indexing, <= for where condition
                if length(times_past) > 0
                    indxCurTime = maximum(times_past)
                    if self.sim_prop.fixedTmStp[2, indxCurTime] !== nothing # 1-based indexing
                        # time step is not given as None.
                        time_step = self.sim_prop.fixedTmStp[2, indxCurTime]
                        time_step_given = true
                    else
                        time_step_given = false
                    end
                else
                    # time step is given as None. In this case time step will be evaluated with current state
                    time_step_given = false
                end
            else
                throw(ArgumentError("Fixed time step can be a float or an ndarray with two rows giving the time and"
                                    * " corresponding time steps."))
            end
        end

        if !time_step_given
            delta_x = minimum([self.fracture.mesh.hx, self.fracture.mesh.hy])
            if any(isnan, self.fracture.v)
                @warn "you should not get nan velocities"
            end
            non_zero_v = findall(self.fracture.v .> 0) # 1-based indexing
            # time step is calculated with the current propagation velocity
            TS_cell_length = Inf
            TS_fracture_length = Inf
            if length(non_zero_v) > 0
                if length(self.injection_prop.sourceElem) < 4
                    vertex_indices = [self.fracture.mesh.Connectivity[self.fracture.EltTip[i], self.fracture.ZeroVertex[i]] for i in 1:length(self.fracture.EltTip)]
                    tipVrtxCoord = self.fracture.mesh.VertexCoor[vertex_indices, :] # Matrix{Float64}
                    
                    # the distance of tip from the injection point in each of the tip cell
                    dist_x_diff = tipVrtxCoord[:, 1] .- self.injection_prop.sourceCoordinates[1] # 1-based indexing for columns
                    dist_y_diff = tipVrtxCoord[:, 2] .- self.injection_prop.sourceCoordinates[2] # 1-based indexing for columns
                    dist_Inj_pnt = sqrt.(dist_x_diff.^2 .+ dist_y_diff.^2) .+ self.fracture.l

                    # the time step evaluated by restricting the fracture to propagate not more than 20 percent of the
                    # current maximum length
                    TS_fracture_length_vals = abs.(0.2 .* dist_Inj_pnt[non_zero_v] ./ self.fracture.v[non_zero_v])
                    if !isempty(TS_fracture_length_vals)
                        TS_fracture_length = minimum(TS_fracture_length_vals)
                    else
                        TS_fracture_length = Inf
                    end
                else
                    TS_fracture_length = Inf
                end

                # the time step evaluated by restricting the fraction of the cell that would be traversed in the time
                # step. e.g., if the pre-factor is 0.5, the tip in the cell with the largest velocity will progress half
                # of the cell width in either x or y direction depending on which is smaller.
                TS_cell_length = delta_x / maximum(self.fracture.v)

            else
                TS_cell_length = Inf
                TS_fracture_length = Inf
            end

            # index of current time in the time series (first row) of the injection rate array
            indx_cur_time = maximum(findall(self.injection_prop.injectionRate[1, :] .<= self.fracture.time))
            current_rate = self.injection_prop.injectionRate[2, indx_cur_time]
            TS_inj_cell = Inf
            if current_rate < 0
                vel_injection = current_rate / (2 * (self.fracture.mesh.hx + self.fracture.mesh.hy) *
                                    self.fracture.w[self.fracture.mesh.CenterElts]) # Assuming CenterElts is Vector{Int}
                TS_inj_cell = 10 * delta_x / abs(vel_injection[1]) # 1-based indexing
            elseif current_rate > 0
                # for positive injection, use the increase in total fracture volume criteria
                TS_inj_cell = 0.1 * sum(self.fracture.w) * self.fracture.mesh.EltArea / current_rate
            else
                TS_inj_cell = Inf
            end

            TS_delta_vol = Inf
            if self.delta_w !== nothing
                delta_vol = sum(self.delta_w) / sum(self.fracture.w)
                if delta_vol < 0
                    TS_delta_vol = self.lstTmStp / abs(delta_vol) * 0.05
                else
                    TS_delta_vol = self.lstTmStp / abs(delta_vol) * 0.12
                end
            end

            # getting pre-factor for current time
            current_prefactor = get_time_step_prefactor(self.sim_prop, self.fracture.time) # Assuming external function
            time_step = current_prefactor * minimum([TS_cell_length, TS_fracture_length, TS_inj_cell, TS_delta_vol])

            # limit time step to be max 2 * last time step
            if (self.lstTmStp !== nothing && !isinf(time_step)) && time_step > 2 * self.lstTmStp
                time_step = 2 * self.lstTmStp
            end

            # limit the time step to be at max 15% of the actual time
            if time_step > 0.15 * self.fracture.time
                time_step = 0.15 * self.fracture.time
            end
        end # if !time_step_given

        # in case of fracture not propagating
        if time_step <= 0 || isinf(time_step)
            if self.stagnant_TS !== nothing
                time_step = self.stagnant_TS
                self.stagnant_TS = time_step * 1.2
            else
                TS_obtained = false
                @warn "The fracture front is stagnant and there is no injection. In these conditions, " *
                    "there is no criterion to calculate time step size."
                while !TS_obtained
                    try
                        print("Enter the time step size(seconds) you would like to try: ")
                        inp = readline()
                        time_step = parse(Float64, inp)
                        TS_obtained = true
                    catch e
                        # Handle parsing error silently and retry
                        if isa(e, ArgumentError)
                        else
                            rethrow(e) # Re-throw unexpected errors
                        end
                    end
                end
            end
        end

        # to get the solution at the times given in time series, any change in parameters or final time
        next_in_TS = self.sim_prop.finalTime

        if self.timeToHit !== nothing
            larger_in_TS = findall(self.timeToHit .> self.fracture.time)
            if length(larger_in_TS) > 0
                next_in_TS = minimum(self.timeToHit[larger_in_TS])
            end
        end

        if next_in_TS < self.fracture.time
            error("The minimum time required in the given time series or the end time" *
                " is less than initial time.")
        end

        # check if time step would step over the next time in required time series
        if self.fracture.time + time_step > next_in_TS
            time_step = next_in_TS - self.fracture.time
        # check if the current time is very close the next time to hit. If yes, set it to the next time to avoid
        # very small time step in the next time step advance.
        elseif next_in_TS - self.fracture.time < 1.05 * time_step
            time_step = next_in_TS - self.fracture.time
        end

        # checking if the time step is above the limit
        if self.sim_prop.timeStepLimit !== nothing && time_step > self.sim_prop.timeStepLimit
            @warn "Evaluated/given time step is more than the time step limit! Limiting time step..."
            time_step = self.sim_prop.timeStepLimit
        end

        return time_step
    end

    # ------------------------------------------------------------------------------------------------------------------

    """
        remesh(self, new_limits, elems, direction=nothing, rem_factor=10)

        # Arguments
        - `new_limits`: limits of the new mesh
        - `elems`: number of elements in the new mesh
        - `direction`: direction of remeshing
        - `rem_factor`: remeshing factor
    """
    function remesh(self::Controller, new_limits::Vector{Vector{Float64}}, elems::Vector{Int}, direction::Union{Nothing, String}=nothing, rem_factor::Number=10)

        # Generating the new mesh (with new limits but same number of elements)
        coarse_mesh = CartesianMesh(new_limits[1],
                                    new_limits[2],
                                    elems[1],
                                    elems[2],
                                    symmetric=self.sim_prop.symmetric)

        # Finalizing the transfer of information from the fine to the coarse mesh
        remesh(self.solid_prop, coarse_mesh) # Assuming remesh is a function
        remesh(self.injection_prop, coarse_mesh, self.fracture.mesh) # Assuming remesh is a function

        # We adapt the elasticity matrix
        if !self.sim_prop.useBlockToeplizCompression
            if direction === nothing
                if rem_factor == self.sim_prop.remeshFactor
                    self.C *= 1 / self.sim_prop.remeshFactor
                else
                    if !self.sim_prop.symmetric
                        self.C = load_isotropic_elasticity_matrix(coarse_mesh, self.solid_prop.Eprime)
                    else
                        self.C = load_isotropic_elasticity_matrix_symmetric(coarse_mesh, self.solid_prop.Eprime)
                    end
                end
            elseif direction == "reduce"
                if !self.sim_prop.symmetric
                    self.C = load_isotropic_elasticity_matrix(coarse_mesh, self.solid_prop.Eprime)
                else
                    self.C = load_isotropic_elasticity_matrix_symmetric(coarse_mesh, self.solid_prop.Eprime)
                end
            else
                @info "Extending the elasticity matrix..."
                extend_isotropic_elasticity_matrix(self, coarse_mesh, direction=direction)
            end
        else
            reload(self.C, coarse_mesh) # Assuming reload is a method or function
        end

        self.fracture = self.fracture.remesh(rem_factor,
                                                self.C,
                                                coarse_mesh,
                                                self.solid_prop,
                                                self.fluid_prop,
                                                self.injection_prop,
                                                self.sim_prop,
                                                direction) # Assuming remesh is a method

        self.fracture.mesh = coarse_mesh

        if self.sim_prop.saveToDisk
            if isfile(joinpath(self.sim_prop.get_outputFolder(), "properties.jld2")) # Check for JLD2 file
                rm(joinpath(self.sim_prop.get_outputFolder(), "properties.jld2")) # Remove JLD2 file
            end
            prop = (self.solid_prop, self.fluid_prop, self.injection_prop, self.sim_prop)
            jldsave(joinpath(self.sim_prop.get_outputFolder(), "properties.jld2"); prop)
        end
        self.remeshings += 1

        @info "Done!"
        return nothing
    end

    # -----------------------------------------------------------------------------------------------------------------------

    """
        extend_isotropic_elasticity_matrix(self, new_mesh, direction=nothing)

        In the case of extension of the mesh we don't need to recalculate the entire elasticity matrix. All we need to do is
        to map all the elements to their new index and calculate what lasts

        # Arguments
        - `new_mesh::CartesianMesh`:    -- a mesh object describing the domain.
        - `direction::Union{Nothing, String}=nothing`: -- direction of extension
    """
    function extend_isotropic_elasticity_matrix(self::Controller, new_mesh::CartesianMesh, direction::Union{Nothing, String}=nothing)
            a = new_mesh.hx / 2.0
        b = new_mesh.hy / 2.0
        Ne = new_mesh.NumberOfElts
        Ne_old = self.fracture.mesh.NumberOfElts

        new_indexes = mapping_old_indexes(new_mesh, self.fracture.mesh, direction)

        if size(self.C, 1) != Ne && !self.sim_prop.symmetric
            # Create new matrix with extended size
            new_C = zeros(Float64, Ne, Ne)
            # Copy old matrix block
            new_C[1:Ne_old, 1:Ne_old] = self.C
            # Reorder old elements according to new indexing
            new_C[new_indexes, new_indexes] = self.C[1:Ne_old, 1:Ne_old]
            
            # Find added elements
            add_el = setdiff(1:Ne, new_indexes)
            
            # Calculate values for new elements
            for i in add_el
                x = new_mesh.CenterCoor[i, 1] .- new_mesh.CenterCoor[:, 1]
                y = new_mesh.CenterCoor[i, 2] .- new_mesh.CenterCoor[:, 2]

                new_C[i, :] = (self.solid_prop.Eprime / (8.0 * π)) .* (
                    sqrt.((a .- x).^2 .+ (b .- y).^2) ./ ((a .- x) .* (b .- y)) .+ 
                    sqrt.((a .+ x).^2 .+ (b .- y).^2) ./ ((a .+ x) .* (b .- y)) .+ 
                    sqrt.((a .- x).^2 .+ (b .+ y).^2) ./ ((a .- x) .* (b .+ y)) .+ 
                    sqrt.((a .+ x).^2 .+ (b .+ y).^2) ./ ((a .+ x) .* (b .+ y))
                )
            end
            
            # Transpose appropriate block
            new_C[new_indexes, add_el] = new_C[add_el, new_indexes]'
            
            self.C = new_C
        elseif !self.sim_prop.symmetric
            self.C = load_isotropic_elasticity_matrix(new_mesh, self.solid_prop.Eprime)
        else
            self.C = load_isotropic_elasticity_matrix_symmetric(new_mesh, self.solid_prop.Eprime)
        end
        
        return nothing
    end

end # module Controller