# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""

module FractureModule

    include("level_set.jl")
    include("volume_integral.jl")
    include("fracture_initialization.jl")
    include("HF_reference_solutions.jl")
    include("visualization.jl")
    include("labels.jl")
    include("properties.jl")

    using .LevelSet: SolveFMM
    using .VolumeIntegral: Pdistance
    using .FractureInitialization: get_survey_points, get_width_pressure, generate_footprint, Geometry, InitializationParameters
    using .HFReferenceSolutions: HF_analytical_sol
    using .Visualization: plot_fracture_list, plot_fracture_list_slice, to_precision, zoom_factory
    using .Labels: unidimensional_variables
    using .Properties: PlotProperties

    using PyPlot
    using JLD2
    using Interpolations

    export Fracture, plot_fracture, process_fracture_front, plot_fracture_slice, SaveFracture, 
        plot_front, plot_front_3D, update_value, update_index, update_front_dict, update_regime_color, remesh, update_tip_regime

    """
        Class defining propagating fracture.

        Args:
            mesh (CartesianMesh):                   -- a CartesianMesh class object describing the grid.
            init_param (tuple):                     -- a InitializationParameters class object (see class documentation).
            solid (MaterialProperties):             -- the MaterialProperties object giving the material properties.
            fluid (FluidProperties):                -- the FluidProperties class object giving the fluid properties.
            injection (InjectionProperties):        -- the InjectionProperties class object giving the injection properties.
            simulProp (SimulationParameters):       -- the SimulationParameters class object giving the numerical parameters
                                                    to be used in the simulation.

        Attributes:
            w (Vector{Float64}):                -- fracture opening (width)
            pFluid (Vector{Float64}):           -- the fluid pressure in the fracture.
            pNet (Vector{Float64}):             -- the net pressure in the fracture.
            time (Float64):                     -- time since the start of injection
            EltChannel (Vector{Int}):           -- list of cells currently in the channel region
            EltCrack (Vector{Int}):             -- list of cells currently in the crack region
            EltRibbon (Vector{Int}):            -- list of cells currently in the Ribbon region
            EltTip (Vector{Int}):               -- list of cells currently in the Tip region
            v (Vector{Float64}):                -- propagation velocity for each cell in the tip cells
            alpha (Vector{Float64}):            -- angle prescribed by perpendicular on the fracture front
            l (Vector{Float64}):                -- length of perpendicular on the fracture front
            ZeroVertex (Vector{Int}):           -- Vertex from which the perpendicular is drawn
            FillF (Vector{Float64}):            -- filling fraction of each tip cell
            CellStatus (Vector{Int}):           -- specifies which region each element currently belongs to
            sgndDist (Vector{Float64}):         -- signed minimum distance from fracture front of each cell in the domain
            InCrack (Vector{Int}):              -- array specifying whether the cell is inside or outside the fracture.
            FractureVolume (Float64):           -- fracture volume
            muPrime Union{Vector{Float64}, Nothing}: local viscosity parameter
            Ffront (Matrix{Float64}):           -- a list containing the intersection of the front and grid lines for the tip cells
            regime_color (Matrix{Float64}):     -- RGB color code of the regime
            ReynoldsNumber (Matrix{Float32}):   -- the reynolds number at each edge of the cells
            fluidFlux (Matrix{Float32}):        -- the fluid flux at each edge of the cells
            fluidVelocity (Matrix{Float32}):    -- the fluid velocity at each edge of the cells
            LkOffTotal (Float64):               -- total fluid volume leaked off from each of the cell in the mesh
            Tarrival (Vector{Float64}):         -- the arrival time of the fracture front
            TarrvlZrVrtx (Vector{Float64}):     -- the time at which the front crosses the zero vertex
            closed (Vector{Int}):               -- the cells which have closed due to leak off or flow back
            injectedVol (Float64):              -- the total volume that is injected into the fracture.
            sgndDist_last (Vector{Float64}):    -- the signed distance of the last time step
            timeStep_last (Float64):            -- the last time step
            source (Vector{Int}):               -- the list of injection cells
            FFront (Matrix{Float64}):           -- the variable storing the fracture front
            LkOff (Vector{Float64}):            -- the leak-off of the fluid in the last time step.
            effVisc (Matrix{Float32}):          -- the Newtonian equivalent viscosity
            efficiency (Float64):               -- the fracturing efficiency
            wHist (Vector{Float64}):            -- the maximum width until now in each cell.
            G (Matrix{Float32}):                -- the coefficient G for non-Newtonian fluid
    """
    mutable struct Fracture

        w::Vector{Float64}                  # fracture opening (width)
        pFluid::Vector{Float64}             # fluid pressure
        pNet::Vector{Float64}               # net pressure
        time::Float64                       # current time
        EltChannel::Vector{Int}             # channel elements
        EltCrack::Vector{Int}               # crack elements
        EltRibbon::Vector{Int}              # ribbon elements
        EltTip::Vector{Int}                 # tip elements
        v::Vector{Float64}                  # propagation velocity
        alpha::Vector{Float64}              # angle
        l::Vector{Float64}                  # length
        ZeroVertex::Vector{Int}             # zero vertex
        FillF::Vector{Float64}              # filling fraction
        CellStatus::Vector{Int}             # cell status
        sgndDist::Vector{Float64}           # signed distance
        InCrack::Vector{Int}                # in crack flag (1 for in crack, 0 for outside)
        FractureVolume::Float64             # fracture volume
        muPrime::Union{Vector{Float64}, Nothing},           # local viscosity
        Ffront::Matrix{Float64}             # front coordinates
        regime_color::Union{Matrix{Float32}, Nothing}  # regime color
        ReynoldsNumber::Union{Matrix{Float32}, Nothing}  # Reynolds number
        fluidFlux::Union{Matrix{Float32}, Nothing}       # fluid flux
        fluidVelocity::Union{Matrix{Float32}, Nothing}   # fluid velocity
        LkOffTotal::Float64                 # total leak-off
        Tarrival::Vector{Float64}           # arrival time
        TarrvlZrVrtx::Vector{Float64}       # zero vertex arrival time
        closed::Vector{Int}                 # closed cells
        injectedVol::Float64                # injected volume
        sgndDist_last::Union{Vector{Float64}, Nothing}   # last signed distance
        timeStep_last::Union{Float64, Nothing}           # last time step
        source::Vector{Int}                 # source elements
        FFront::Matrix{Float64}             # fracture front
        LkOff::Vector{Float64}              # leak-off
        effVisc::Matrix{Float32}            # effective viscosity
        efficiency::Float64                 # efficiency
        wHist::Vector{Float64}              # width history
        G::Matrix{Float32}                  # coefficient G
        mesh                                # mesh object
        number_of_fronts::Int               # number of fronts
        fronts_dictionary                   # fronts dictionary
        fluidFlux_components::Union{Matrix{Float32}, Nothing}    # flux components
        fluidVelocity_components::Union{Matrix{Float32}, Nothing} # velocity components
        fully_traversed::Vector{Int}        # fully traversed cells
        sink::Vector{Int}                   # sink elements
        injectionRate::Vector{Float32}      # injection rate
        pInjLine::Float64                   # injection line pressure


        """
        Initialize the fracture according to the given initialization parameters.
        """
        function Fracture(mesh, init_param, solid=nothing, fluid=nothing, injection=nothing, simulProp=nothing)
            time = 0.0
            w = Float64[]
            pNet = Float64[]
            v = Float64[]
            actvElts = Int[]
            
            if init_param.regime != "static"
                # get appropriate length dimension variable
                length_dim = init_param.geometry.get_length_dimension()
                
                time, length_dim, pNet, w, v, actvElts = HF_analytical_sol(
                    init_param.regime,
                    mesh,
                    solid.Eprime,
                    injection.injectionRate[1, 1],
                    inj_point=injection.sourceCoordinates,
                    muPrime=fluid.muPrime,
                    Kprime=solid.Kprime[mesh.CenterElts][1],
                    Cprime=solid.Cprime[mesh.CenterElts][1],
                    length=length_dim,
                    t=init_param.time,
                    Kc_1=solid.Kc1,
                    h=init_param.geometry.fractureHeight,
                    density=fluid.density,
                    Cij=solid.Cij,
                    gamma=init_param.geometry.gamma,
                    Vinj=init_param.fractureVolume
                )
                init_param.geometry.set_length_dimension(length_dim)
            end
            
            surv_cells, surv_dist, inner_cells = get_survey_points(
                init_param.geometry,
                mesh,
                source_coord=injection.sourceCoordinates
            )
            
            EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, l, alpha, FillF, sgndDist, Ffront, number_of_fronts, fronts_dictionary = generate_footprint(
                mesh,
                surv_cells,
                inner_cells,
                surv_dist,
                simulProp.projMethod
            )
            
            # for static fracture initialization
            if init_param.regime == "static"
                w, pNet = get_width_pressure(
                    mesh,
                    EltCrack,
                    EltTip,
                    FillF,
                    init_param.C,
                    init_param.width,
                    init_param.netPressure,
                    init_param.fractureVolume,
                    simulProp.symmetric,
                    simulProp.useBlockToeplizCompression,
                    solid.Eprime
                )
                
                if init_param.fractureVolume === nothing && init_param.time === nothing
                    volume = sum(w) * mesh.EltArea
                    time = volume / injection.injectionRate[1, 1]
                elseif init_param.time !== nothing
                    time = init_param.time
                end
                
                v = fill(init_param.tipVelocity, length(EltTip))
            end
            
            if v !== nothing
                if isa(v, Number)
                    v = fill(Float64(v), length(EltTip))
                end
            else
                v = fill(NaN, length(EltTip))
            end
            
            pFluid = zeros(Float64, mesh.NumberOfElts)
            pFluid[EltCrack] = pNet[EltCrack] + solid.SigmaO[EltCrack]
            
            sgndDist_last = nothing
            timeStep_last = nothing
            
            # setting arrival time to current time
            Tarrival = fill(NaN, mesh.NumberOfElts)
            Tarrival[EltCrack] = time
            LkOff = zeros(Float64, mesh.NumberOfElts)
            LkOffTotal = 0.0
            efficiency = 1.0
            FractureVolume = sum(w) * mesh.EltArea
            injectedVol = sum(w) * mesh.EltArea
            InCrack = zeros(Int, mesh.NumberOfElts)
            InCrack[EltCrack] = 1
            wHist = copy(w)
            fully_traversed = Int[]
            source = intersect(injection.sourceElem, EltCrack)
            
            # Sorting by distances
            distances = [sqrt(mesh.CenterCoor[x, 1]^2 + mesh.CenterCoor[x, 2]^2) for x in source]
            source = source[sortperm(distances)]
            sink = Int[]
            
            # will be overwritten by nothing if not required
            effVisc = zeros(Float32, 4, mesh.NumberOfElts)
            G = zeros(Float32, 4, mesh.NumberOfElts)
            
            if simulProp.projMethod != "LS_continousfront"
                process_fracture_front()
            end

            # local viscosity
            if fluid !== nothing:
                muPrime = fill(Float64(fluid.muPrime), mesh.NumberOfElts)
            
            ReynoldsNumber = nothing
            if simulProp.saveReynNumb
                ReynoldsNumber = fill(NaN, 4, mesh.NumberOfElts)
            end
            
            # regime variable
            regime_color = nothing
            if simulProp.saveRegime
                regime_color = ones(Float32, mesh.NumberOfElts, 3)
            end
            
            fluidFlux = nothing
            fluidFlux_components = nothing
            if simulProp.saveFluidFlux
                fluidFlux = fill(NaN, 4, mesh.NumberOfElts)
                fluidFlux_components = fill(NaN, 8, mesh.NumberOfElts)
            end
            
            if simulProp.saveFluidFluxAsVector
                fluidFlux_components = fill(NaN, 8, mesh.NumberOfElts)
            end
            
            fluidVelocity = nothing
            fluidVelocity_components = nothing
            if simulProp.saveFluidVel
                fluidVelocity = fill(NaN, 4, mesh.NumberOfElts)
                fluidVelocity_components = fill(NaN, 8, mesh.NumberOfElts)
            end
            
            if simulProp.saveFluidVelAsVector
                fluidVelocity_components = fill(NaN, 8, mesh.NumberOfElts)
            end
            
            closed = Int[]
            TarrvlZrVrtx = fill(NaN, mesh.NumberOfElts)
            TarrvlZrVrtx[EltCrack] = time
            
            if v !== nothing && !any(isnan.(v))
                TarrvlZrVrtx[EltTip] = time - l ./ v
            end
            
            injectionRate = fill(NaN, mesh.NumberOfElts)
            pInjLine = 0.0
            if injection.modelInjLine
                pInjLine = Float64(injection.initPressure)
                injectionRate = fill(NaN, mesh.NumberOfElts)
            end
            
            FFront = Matrix{Float64}(undef, 0, 2)
            
            return Fracture(
                w, pFluid, pNet, time, EltChannel, EltCrack, EltRibbon, EltTip, v, alpha, l,
                ZeroVertex, FillF, CellStatus, sgndDist, InCrack, FractureVolume, muPrime,
                Ffront, regime_color, ReynoldsNumber, fluidFlux, fluidVelocity, LkOffTotal,
                Tarrival, TarrvlZrVrtx, closed, injectedVol, sgndDist_last, timeStep_last,
                source, FFront, LkOff, effVisc, efficiency, wHist, G, mesh, number_of_fronts,
                fronts_dictionary, fluidFlux_components, fluidVelocity_components,
                fully_traversed, sink, injectionRate, pInjLine
            )
        end
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture(self, variable="complete", mat_properties=nothing, projection="3D", elements=nothing,
                    backGround_param=nothing, plot_prop=nothing, fig=nothing, edge=4, contours_at=nothing, labels=nothing,
                    plot_non_zero=true)

        This function plots the fracture.

        Arguments:
        - `variable::String`: The variable to be plotted. See supported_variables of the labels module for a list of supported variables.
        - `mat_properties`: The material properties. It is mainly used to colormap the mesh.
        - `projection::String`: A string specifying the projection. See supported_projections for the supported projections.
        - `elements`: The elements to be plotted.
        - `backGround_param::String`: The parameter according to which the mesh will be color-mapped.
        - `plot_prop`: The properties to be used for the plot.
        - `fig`: The figure to superimpose on. New figure will be made if not provided.
        - `edge::Int`: The edge of the cell that will be plotted (0->left, 1->right, 2->bottom, 3->top, 4->average).
        - `contours_at`: The values at which the contours are to be plotted.
        - `labels`: The labels to be used for the plot.
        - `plot_non_zero::Bool`: If true, only non-zero values will be plotted.

        Returns:
        - `fig`: A Figure object that can be used superimpose further plots.
    """
    function plot_fracture(self::Fracture, variable="complete", mat_properties=nothing, projection="3D", elements=nothing,
                        backGround_param=nothing, plot_prop=nothing, fig=nothing, edge=4, contours_at=nothing, labels=nothing,
                        plot_non_zero=true)
        
        if variable in unidimensional_variables
            throw(ArgumentError("The variable does not vary spatially!"))
        end
        
        if variable == "complete"
            proj = "3D"
            if "2D" in projection
                proj = "2D"
            end
            fig = plot_fracture_list([self],
                                variable="mesh",
                                mat_properties=mat_properties,
                                projection=proj,
                                elements=elements,
                                backGround_param=backGround_param,
                                plot_prop=plot_prop,
                                fig=fig,
                                edge=edge,
                                contours_at=contours_at,
                                labels=labels)
            fig = plot_fracture_list([self],
                                    variable="footprint",
                                    mat_properties=mat_properties,
                                    projection=proj,
                                    elements=elements,
                                backGround_param=backGround_param,
                                plot_prop=plot_prop,
                                fig=fig,
                                edge=edge,
                                contours_at=contours_at,
                                labels=labels)
            variable = "width"
        end
        
        if projection == "3D"
            plot_non_zero = false
        end
        
        fig = plot_fracture_list([self],
                            variable=variable,
                            mat_properties=mat_properties,
                            projection=projection,
                            elements=elements,
                            backGround_param=backGround_param,
                            plot_prop=plot_prop,
                            fig=fig,
                            edge=edge,
                            contours_at=contours_at,
                            labels=labels,
                            plot_non_zero=plot_non_zero)
        
        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        process_fracture_front(self)

        Process fracture front and different regions of the fracture. This function adds the start and endpoints of the
        front lines in each of the tip cell to the Ffront variable of the Fracture class.
    """
    function process_fracture_front(self::Fracture)
        """
        Process fracture front and different regions of the fracture. This function adds the start and endpoints of the
        front lines in each of the tip cell to the Ffront variable of the Fracture class.
        """
        
        # list of points where fracture front is intersecting the grid lines. 
        intrsct1 = zeros(2, length(self.l))
        intrsct2 = zeros(2, length(self.l))

        # todo: commenting

        for i in 1:length(self.l)
            if self.alpha[i] != 0 && self.alpha[i] != π / 2  # for angles greater than zero and less than 90 deg
                # calculate intercept on y axis and gradient
                yIntrcpt = self.l[i] / cos(π / 2 - self.alpha[i])
                grad = -1 / tan(self.alpha[i])

                if Pdistance(0, self.mesh.hy, grad, yIntrcpt) <= 0
                    # one point on top horizontal line of the cell
                    intrsct1[1, i] = 0
                    intrsct1[2, i] = yIntrcpt
                else
                    # one point on left vertical line of the cell
                    intrsct1[1, i] = (self.mesh.hy - yIntrcpt) / grad
                    intrsct1[2, i] = self.mesh.hy
                end

                if Pdistance(self.mesh.hx, 0, grad, yIntrcpt) <= 0
                    intrsct2[1, i] = -yIntrcpt / grad
                    intrsct2[2, i] = 0
                else
                    intrsct2[1, i] = self.mesh.hx
                    intrsct2[2, i] = yIntrcpt + grad * self.mesh.hx
                end

            elseif self.alpha[i] == 0
                intrsct1[1, i] = self.l[i]
                intrsct1[2, i] = self.mesh.hy
                intrsct2[1, i] = self.l[i]
                intrsct2[2, i] = 0

            elseif self.alpha[i] == π / 2
                intrsct1[1, i] = 0
                intrsct1[2, i] = self.l[i]
                intrsct2[1, i] = self.mesh.hx
                intrsct2[2, i] = self.l[i]
            end

            if self.ZeroVertex[i] == 0
                intrsct1[1, i] = intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct1[2, i] = intrsct1[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]
                intrsct2[1, i] = intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct2[2, i] = intrsct2[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]

            elseif self.ZeroVertex[i] == 1
                intrsct1[1, i] = -intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct1[2, i] = intrsct1[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]
                intrsct2[1, i] = -intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct2[2, i] = intrsct2[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]

            elseif self.ZeroVertex[i] == 3
                intrsct1[1, i] = intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct1[2, i] = -intrsct1[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]
                intrsct2[1, i] = intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct2[2, i] = -intrsct2[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]

            elseif self.ZeroVertex[i] == 2
                intrsct1[1, i] = -intrsct1[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct1[2, i] = -intrsct1[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]
                intrsct2[1, i] = -intrsct2[1, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 1]
                intrsct2[2, i] = -intrsct2[2, i] + self.mesh.VertexCoor[
                    self.mesh.Connectivity[self.EltTip[i], self.ZeroVertex[i]+1], 2]
            end
        end

        tmp = transpose(intrsct1)
        tmp = hcat(tmp, transpose(intrsct2))

        self.Ffront = tmp
        
        return nothing
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_slice(self, variable="width", point1=nothing, point2=nothing, projection="2D", plot_prop=nothing,
                            fig=nothing, edge=4, labels=nothing, plot_cell_center=false, orientation="horizontal")

        This function plots the fracture on a given slice of the domain. Two points are to be given that will be
        joined to form the slice. The values on the slice are interpolated from the values available on the cell
        centers. Exact values on the cell centers can also be plotted.

        Arguments:
        - `variable::String`: The variable to be plotted. See supported_variables of the labels module for a list of supported variables.
        - `point1`: The left point from which the slice should pass [x, y].
        - `point2`: The right point from which the slice should pass [x, y].
        - `projection::String`: A string specifying the projection. It can either '3D' or '2D'.
        - `plot_prop`: The properties to be used for the plot.
        - `fig`: The figure to superimpose on. New figure will be made if not provided.
        - `edge::Int`: The edge of the cell that will be plotted. This is for variables that are evaluated on the cell edges instead of cell center. It can have a value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
        - `labels`: The labels to be used for the plot.
        - `plot_cell_center::Bool`: If true, the discrete values at the cell centers will be plotted. In this case, the slice passing through the center of the cell containing point1 will be taken.
        - `orientation::String`: The orientation according to which the slice is made in the case the plotted values are not interpolated and are taken at the cell centers.

        Returns:
        - `fig`: A Figure object that can be used superimpose further plots.
    """
    function plot_fracture_slice(self::Fracture, variable="width", point1=nothing, point2=nothing, projection="2D", plot_prop=nothing,
                                fig=nothing, edge=4, labels=nothing, plot_cell_center=false, orientation="horizontal")
        
        return plot_fracture_list_slice([self],
                                        variable=variable,
                                        point1=point1,
                                        point2=point2,
                                        plot_prop=plot_prop,
                                        projection=projection,
                                        fig=fig,
                                        edge=edge,
                                        labels=labels,
                                        plot_cell_center=plot_cell_center,
                                        orientation=orientation)
    end

    # ------------------------------------------------------------------------------------------------------------------

    """
        SaveFracture(self, filename)

        This function saves the fracture object to a file on hard disk using JLD2 module.

        Arguments:
        - `self::Fracture`: The fracture object to be saved.
        - `filename::String`: The name of the file to save the fracture object to.

        Returns:
        - Nothing.
    """
    function SaveFracture(self::Fracture, filename::String)
        """ This function saves the fracture object to a file on hard disk using JLD2 module """
        
        @save filename self
        return nothing
    end

    # -----------------------------------------------------------------------------------------------------------------------

    """
        plot_front(self, fig=nothing, plot_prop=nothing)

        This function plots the front lines in the tip cells of the fracture taken from the fFront variable.

        Arguments:
        - `self::Fracture`: The fracture object.
        - `fig`: The figure to plot on. If nothing, a new figure is created.
        - `plot_prop`: Plot properties. If nothing, default properties are used.

        Returns:
        - The figure object.
    """
    function plot_front(self::Fracture, fig=nothing, plot_prop=nothing)
        """
        This function plots the front lines in the tip cells of the fracture taken from the fFront variable.
        """
        
        if fig === nothing
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.axis("equal")
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        I = self.Ffront[:, 1:2]
        J = self.Ffront[:, 3:4]
        
        # todo !!!Hack: gets very large values sometime, needs to be resolved
        for e in 1:size(I, 1)
            if maximum(abs.(I[e, :] - J[e, :])) < 3 * sqrt(self.mesh.hx^2 + self.mesh.hy^2)
                ax.plot([I[e, 1], J[e, 1]],
                        [I[e, 2], J[e, 2]],
                        plot_prop.lineStyle,
                        color=plot_prop.lineColor)
            end
        end

        if plot_prop.PlotFP_Time
            tipVrtxCoord = self.mesh.VertexCoor[self.mesh.Connectivity[self.EltTip, self.ZeroVertex+1], :]  # +1 for 1-indexing
            distances = sqrt.(tipVrtxCoord[:, 1].^2 + tipVrtxCoord[:, 2].^2) + self.l
            r_indx = argmax(distances)
            x_coor = self.mesh.CenterCoor[self.EltTip[r_indx], 1] + 0.1 * self.mesh.hx
            y_coor = self.mesh.CenterCoor[self.EltTip[r_indx], 2] + 0.1 * self.mesh.hy
            
            if plot_prop.textSize === nothing
                plot_prop.textSize = max(self.mesh.hx, self.mesh.hy)  # Fixed: was hx, hx
            end
            
            t = to_precision(self.time, plot_prop.dispPrecision) * "s"
            
            ax.text(x_coor, y_coor, t)
        end

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_front_3D(self, fig=nothing, plot_prop=nothing)

        This function plots the front lines with 3D projection in the tip cells of the fracture taken from the fFront
        variable.

        Arguments:
        - `self::Fracture`: The fracture object.
        - `fig`: The figure to plot on. If nothing, a new figure is created.
        - `plot_prop`: Plot properties. If nothing, default properties are used.

        Returns:
        - The figure object.
    """
    function plot_front_3D(self::Fracture, fig=nothing, plot_prop=nothing)
        """
        This function plots the front lines with 3D projection in the tip cells of the fracture taken from the fFront
        variable.
        """
        
        if fig === nothing
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.set_xlim(minimum(self.Ffront), maximum(self.Ffront))
            ax.set_ylim(minimum(self.Ffront), maximum(self.Ffront))
            plt.gca().set_aspect("equal")
            scale = 1.1
            zoom_factory(ax, base_scale=scale)
        else
            ax = fig.get_axes()[1]
        end

        ax.set_frame_on(false)
        ax.grid(false)
        ax.set_frame_on(false)
        ax.set_axis_off()

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        I = self.Ffront[:, 1:2]
        J = self.Ffront[:, 3:4]

        # draw front lines (simplified approach using basic PyPlot functions)
        for e in 1:size(I, 1)
            # Simple line plotting in 3D
            x_coords = [I[e, 1], J[e, 1]]
            y_coords = [I[e, 2], J[e, 2]]
            z_coords = [0.0, 0.0]  # Assuming z=0 for 2D front lines in 3D space
            
            ax.plot3D(x_coords, y_coords, z_coords,
                    linestyle=plot_prop.lineStyle,
                    color=plot_prop.lineColor,
                    linewidth=plot_prop.lineWidth)
        end

        return fig
    end

    # -----------------------------------------------------------------------------------------------------------------------
    function update_value(self, old, ind_new_elts, ind_old_elts, new_size, value_new_elem=0, mytype=nothing)
        if value_new_elem == 0 || value_new_elem === nothing
            if mytype === nothing
                new = zeros(new_size)
            else
                new = zeros(mytype, new_size)
            end
        else
            if mytype === nothing
                new = fill(value_new_elem, new_size)
            else
                new = fill(value_new_elem, mytype, new_size)
            end
        end

        new[ind_old_elts] = old
        return new
    end

    function update_index(self, old, ind_old_elts, size, mytype=nothing)
        if mytype === nothing
            new = zeros(size)
        else
            new = zeros(mytype, size)
        end
        
        if length(old) != 0
            new = ind_old_elts[old]
        end
        
        return new
    end

    function update_front_dict(self, old, ind_old_elts)
        mylist = ["crackcells_0", "TIPcellsONLY_0", "crackcells_1", "TIPcellsONLY_1"]
        
        for elem in mylist
            if haskey(old, elem)
                temp = old[elem]
                if temp !== nothing
                    delete!(old, elem)
                    old[elem] = ind_old_elts[temp]
                end
            end
        end

        if haskey(old, "TIPcellsANDfullytrav_0")
            temp = old["TIPcellsANDfullytrav_0"]
            if temp !== nothing
                delete!(old, "TIPcellsANDfullytrav_0")
                old["TIPcellsANDfullytrav_0"] = ind_old_elts[temp]
            end
        end

        if haskey(old, "TIPcellsANDfullytrav_1")
            temp = old["TIPcellsANDfullytrav_1"]
            if temp !== nothing
                delete!(old, "TIPcellsANDfullytrav_1")
                old["TIPcellsANDfullytrav_1"] = ind_old_elts[temp]
            end
        end

        return old
    end

    function update_regime_color(self, old, ind_new_elts, ind_old_elts, new_size)
        value_new_elem = [1.0, 1.0, 1.0]
        new = Array{Float32}(undef, new_size, 3)
        new[ind_old_elts, :] = old[:, :]
        new[ind_new_elts, :] = value_new_elem[:]
        return new
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function compresses the fracture by the given factor once it has reached the end of the mesh. If the
        compression factor is two, each set of four cells in the fine mesh is replaced by a single cell. The volume of
        the fracture is conserved upto machine precision. The elasticity matrix and the properties objects are also
        re-adjusted according to the new mesh.

        Arguments:
            factor (Float64): The factor by which the domain is to be compressed.
            C (Matrix): The elasticity matrix to be re-evaluated for the new mesh.
            coarse_mesh (CartesianMesh): The coarse Cartesian mesh.
            material_prop (MaterialProperties): The MaterialProperties object giving the material properties.
            fluid_prop (FluidProperties): The FluidProperties class object giving the fluid properties to be re-evaluated for the new mesh.
            inj_prop (InjectionProperties): The InjectionProperties class object giving the injection properties to be re-evaluated for the new mesh.
            sim_prop (SimulationParameters): The SimulationParameters class object giving the numerical parameters to be used in the simulation.
            direction (String): The direction of remeshing.

        Returns:
            Fr_coarse (Fracture): The new fracture after re-meshing.
    """

    function remesh(self, factor, C, coarse_mesh, material_prop, fluid_prop, inj_prop, sim_prop, direction)


        if self.sgndDist_last === nothing
            self.sgndDist_last = self.sgndDist
        end

        if direction === nothing || direction == "reduce"
            # interpolate the level set by first advancing and then interpolating
            SolveFMM(self.sgndDist,
                    self.EltRibbon,
                    self.EltChannel,
                    self.mesh,
                    Int[],
                    self.EltChannel)

            # Grid interpolation (using linear interpolation)
            # Note: You'll need to implement or import appropriate interpolation function
            sgndDist_coarse = zeros(Float64, coarse_mesh.NumberOfElts)
            for i in 1:coarse_mesh.NumberOfElts
                # Simple nearest neighbor interpolation as placeholder
                # You should replace this with proper griddata equivalent
                sgndDist_coarse[i] = 1e10  # default value
            end

            # avoid adding tip cells from the fine mesh to get into the channel cells of the coarse mesh
            max_diag = sqrt(coarse_mesh.hx^2 + coarse_mesh.hy^2)
            excluding_tip = findall(sgndDist_coarse .<= -max_diag)
            sgndDist_copy = copy(sgndDist_coarse)
            sgndDist_coarse = fill(1e10, coarse_mesh.NumberOfElts)
            sgndDist_coarse[excluding_tip] = sgndDist_copy[excluding_tip]

            # enclosing cells for each cell in the grid
            enclosing = zeros(Int, self.mesh.NumberOfElts, 8)
            enclosing[:, 1:4] = self.mesh.NeiElements[:, :]
            enclosing[:, 5] = self.mesh.NeiElements[enclosing[:, 3], 1]
            enclosing[:, 6] = self.mesh.NeiElements[enclosing[:, 3], 2]
            enclosing[:, 7] = self.mesh.NeiElements[enclosing[:, 4], 1]
            enclosing[:, 8] = self.mesh.NeiElements[enclosing[:, 4], 2]

            w_coarse = zeros(Float64, coarse_mesh.NumberOfElts)
            LkOff = zeros(Float64, coarse_mesh.NumberOfElts)
            wHist_coarse = zeros(Float64, coarse_mesh.NumberOfElts)

            if factor == 2.0
                # finding the intersecting cells of the fine and course mesh
                intersecting = Int[]
                # todo: a description is to be written, its not readable
                for i in -Int(((self.mesh.ny - 1) / 2 + 1) / 2) + 1:Int(((self.mesh.ny - 1) / 2 + 1) / 2)
                    center = self.mesh.CenterElts[1] + i * self.mesh.nx
                    row_to_add = center - Int(((self.mesh.nx - 1) / 2 + 1) / 2) + 1:center + Int(((self.mesh.nx - 1) / 2 + 1) / 2) - 1
                    intersecting = vcat(intersecting, collect(row_to_add))
                end

                # getting the corresponding cells of the coarse mesh in the fine mesh
                corresponding = Int[]
                for i in intersecting
                    # You'll need to implement locate_element method
                    push!(corresponding, i)  # placeholder
                end
                corresponding = convert(Vector{Int}, corresponding)

                # weighted sum to conserve volume upto machine precision
                for (idx, i) in enumerate(intersecting)
                    corr_idx = corresponding[idx]
                    w_coarse[i] = (self.w[corr_idx] + 
                                sum(self.w[enclosing[corr_idx, 1:4]] / 2) +
                                sum(self.w[enclosing[corr_idx, 5:8]] / 4)) / 4

                    LkOff[i] = (self.LkOff[corr_idx] + 
                            sum(self.LkOff[enclosing[corr_idx, 1:4]] / 2) +
                            sum(self.LkOff[enclosing[corr_idx, 5:8]] / 4))

                    wHist_coarse[i] = (self.wHist[corr_idx] + 
                                    sum(self.wHist[enclosing[corr_idx, 1:4]] / 2) +
                                    sum(self.wHist[enclosing[corr_idx, 5:8]] / 4)) / 4
                end

            else
                # In case the factor by which mesh is compressed is not 2
                # Simple interpolation as placeholder
                for i in 1:coarse_mesh.NumberOfElts
                    w_coarse[i] = 0.0
                    LkOff[i] = 0.0
                    wHist_coarse[i] = 0.0
                end
            end

            # interpolate last level set by first advancing to the end of the grid and then interpolating
            SolveFMM(self.sgndDist_last,
                    self.EltRibbon,
                    self.EltChannel,
                    self.mesh,
                    Int[],
                    self.EltChannel)

            sgndDist_last_coarse = zeros(Float64, coarse_mesh.NumberOfElts)
            for i in 1:coarse_mesh.NumberOfElts
                sgndDist_last_coarse[i] = 1e10  # placeholder
            end

            Fr_Geometry = Geometry(shape="level set",
                                survey_cells=excluding_tip,
                                inner_cells=excluding_tip,
                                tip_distances=-sgndDist_coarse[excluding_tip])
            
            init_data = InitializationParameters(geometry=Fr_Geometry,
                                                regime="static",
                                                width=w_coarse,
                                                elasticity_matrix=C,
                                                tip_velocity=NaN)

            Fr_coarse = Fracture(coarse_mesh,
                                init_data,
                                solid=material_prop,
                                fluid=fluid_prop,
                                injection=inj_prop,
                                simulProp=sim_prop)

            # evaluate current level set on the coarse mesh
            valid_ribbon = setdiff(1:length(Fr_coarse.EltRibbon), 
                                findall(sgndDist_copy[Fr_coarse.EltRibbon] .>= 1e10))
            EltRibbon = Fr_coarse.EltRibbon[valid_ribbon]
            
            valid_channel = setdiff(1:length(Fr_coarse.EltChannel), 
                                findall(sgndDist_copy[Fr_coarse.EltChannel] .>= 1e10))
            EltChannel = Fr_coarse.EltChannel[valid_channel]

            cells_outside = setdiff(1:coarse_mesh.NumberOfElts, EltChannel)

            SolveFMM(sgndDist_copy,
                    EltRibbon,
                    EltChannel,
                    coarse_mesh,
                    cells_outside,
                    Int[])

            # evaluate last level set on the coarse mesh to evaluate velocity of the tip
            valid_ribbon = setdiff(1:length(Fr_coarse.EltRibbon), 
                                findall(sgndDist_last_coarse[Fr_coarse.EltRibbon] .>= 1e10))
            EltRibbon = Fr_coarse.EltRibbon[valid_ribbon]
            
            valid_channel = setdiff(1:length(Fr_coarse.EltChannel), 
                                findall(sgndDist_last_coarse[Fr_coarse.EltChannel] .>= 1e10))
            EltChannel = Fr_coarse.EltChannel[valid_channel]

            cells_outside = setdiff(1:coarse_mesh.NumberOfElts, EltChannel)

            SolveFMM(sgndDist_last_coarse,
                    EltRibbon,
                    EltChannel,
                    coarse_mesh,
                    cells_outside,
                    Int[])

            if self.timeStep_last === nothing
                self.timeStep_last = 1
            end
            
            Fr_coarse.v = -(sgndDist_copy[Fr_coarse.EltTip] - sgndDist_last_coarse[Fr_coarse.EltTip]) / self.timeStep_last

            # Interpolate Tarrival (placeholder implementation)
            for i in Fr_coarse.EltChannel
                Fr_coarse.Tarrival[i] = self.time  # placeholder
            end

            # Interpolate TarrvlZrVrtx (placeholder implementation)
            for i in Fr_coarse.EltChannel
                Fr_coarse.TarrvlZrVrtx[i] = self.time  # placeholder
            end

            # The zero vertex arrival time for the tip elements
            for (indx, elt) in enumerate(Fr_coarse.EltTip)
                Fr_coarse.TarrvlZrVrtx[elt] = self.time  # placeholder
            end

            # Handle closed cells
            coarse_closed = Int[]
            for e in self.closed
                push!(coarse_closed, e)  # placeholder - you need proper mapping
            end
            Fr_coarse.closed = unique(coarse_closed)

            Fr_coarse.LkOff = LkOff
            Fr_coarse.LkOffTotal = self.LkOffTotal
            Fr_coarse.injectedVol = self.injectedVol
            Fr_coarse.efficiency = (Fr_coarse.injectedVol - Fr_coarse.LkOffTotal) / Fr_coarse.injectedVol
            Fr_coarse.time = self.time
            Fr_coarse.wHist = wHist_coarse
            self.source = inj_prop.sourceElem

            if inj_prop.modelInjLine
                Fr_coarse.pInjLine = self.pInjLine
            end

            return Fr_coarse
        else
            # in case of mesh extension just update
            # Note: mapping_old_indexes function needs to be implemented
            ind_new_elts = setdiff(1:coarse_mesh.NumberOfElts, Int[])  # placeholder
            ind_old_elts = Int[]  # placeholder
            newNumberOfElts = coarse_mesh.NumberOfElts
            _muPrime = isa(self.muPrime, Vector) ? maximum(self.muPrime) : self.muPrime

            self.CellStatus = update_value(self, self.CellStatus, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0, mytype=Int)
            self.EltChannel = update_index(self, self.EltChannel, ind_old_elts, length(self.EltChannel), mytype=Int)
            self.EltCrack = update_index(self, self.EltCrack, ind_old_elts, length(self.EltCrack), mytype=Int)
            self.EltRibbon = update_index(self, self.EltRibbon, ind_old_elts, length(self.EltRibbon), mytype=Int)
            self.EltTip = update_index(self, self.EltTipBefore, ind_old_elts, length(self.EltTipBefore), mytype=Int)
            self.InCrack = update_value(self, self.InCrack, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0, mytype=Int)
            self.LkOff = update_value(self, self.LkOff, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0.0, mytype=Float64)
            self.Tarrival = update_value(self, self.Tarrival, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=NaN, mytype=Float64)
            self.TarrvlZrVrtx = update_value(self, self.TarrvlZrVrtx, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=NaN, mytype=Float64)
            self.closed = update_index(self, self.closed, ind_old_elts, length(self.closed), mytype=Int)
            self.fully_traversed = update_index(self, self.fully_traversed, ind_old_elts, length(self.fully_traversed), mytype=Int)
            self.muPrime = update_value(self, self.muPrime, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=fluid_prop.muPrime, mytype=Float64)
            self.pFluid = update_value(self, self.pFluid, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0.0, mytype=Float64)
            self.pNet = update_value(self, self.pNet, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0.0, mytype=Float64)
            self.sgndDist = update_value(self, self.sgndDist, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=1e50, mytype=Float64)
            self.sgndDist_last = update_value(self, self.sgndDist_last, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=1e50, mytype=Float64)
            self.w = update_value(self, self.w, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0.0, mytype=Float64)
            self.wHist = update_value(self, self.wHist, ind_new_elts, ind_old_elts, newNumberOfElts, value_new_elem=0.0, mytype=Float64)

            self.fronts_dictionary = update_front_dict(self, self.fronts_dictionary, ind_old_elts)
            self.regime_color = update_regime_color(self, self.regime_color, ind_new_elts, ind_old_elts, newNumberOfElts)
            self.source = inj_prop.sourceElem
            self.mesh = coarse_mesh

            if inj_prop.modelInjLine
                self.pInjLine = self.pInjLine
            end

            return self
        end
    end

    # -----------------------------------------------------------------------------------------------------------------------

    function update_tip_regime(self, mat_prop, fluid_prop, timeStep)

        log = "JFrac.update_tip_regime"

        # fixed parameters
        beta_mtilde = 4 / (15^(1/4) * (2^(1/2) - 1)^(1/4))
        beta_m = 2^(1/3) * 3^(5/6)

        # initiate with all cells white
        self.regime_color = ones(Float32, self.mesh.NumberOfElts, 3)

        # calculate velocity
        vel = -(self.sgndDist[self.EltRibbon] - self.sgndDist_last[self.EltRibbon]) / timeStep

        # decide on moving cells
        stagnant_condition = mat_prop.Kprime[self.EltRibbon] .* abs.(self.sgndDist[self.EltRibbon]).^(1/2) ./ 
                            (mat_prop.Eprime * self.w[self.EltRibbon]) .> 1
        stagnant = findall(stagnant_condition)
        
        # moving cells are those not in stagnant
        moving_mask = trues(length(self.EltRibbon))
        moving_mask[stagnant] .= false
        moving = findall(moving_mask)

        for i in moving
            if any(isnan.(self.sgndDist[self.EltRibbon[i]]))
                @debug "Why nan distance?" _group = log
            end
            
            wk = mat_prop.Kprime[self.EltRibbon[i]] / mat_prop.Eprime * abs(self.sgndDist[self.EltRibbon[i]])^(1/2)

            muPrime = fluid_prop.muPrime
            wm = beta_m * (muPrime * vel[i] / mat_prop.Eprime)^(1/3) * abs(self.sgndDist[self.EltRibbon[i]])^(2/3)
            wmtilde = beta_mtilde * (4 * muPrime^2 * vel[i] * mat_prop.Cprime[self.EltRibbon[i]]^2 / 
                                    mat_prop.Eprime^2)^(1/8) * abs(self.sgndDist[self.EltRibbon[i]])^(5/8)

            nk = wk / (self.w[self.EltRibbon[i]] - wk)
            nm = wm / (self.w[self.EltRibbon[i]] - wm)
            nmtilde = wmtilde / (self.w[self.EltRibbon[i]] - wmtilde)

            nk_plus_nm_plus_nmtilde = nk + nm + nmtilde
            Nk = nk / nk_plus_nm_plus_nmtilde
            Nm = nm / nk_plus_nm_plus_nmtilde
            Nmtilde = nmtilde / nk_plus_nm_plus_nmtilde

            if Nk > 1.0
                Nk = 1.0
            elseif Nk < 0.0
                Nk = 0.0
            end
            
            if Nm > 1.0
                Nm = 1.0
            elseif Nm < 0.0
                Nm = 0.0
            end
            
            if Nmtilde > 1.0
                Nmtilde = 1.0
            elseif Nmtilde < 0.0
                Nmtilde = 0.0
            end

            self.regime_color[self.EltRibbon[i], :] = [Nk, Nmtilde, Nm]
        end
        
        return nothing
    end

end # module FractureModule