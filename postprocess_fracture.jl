# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 12.06.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""

module PostprocessFracture

    using JLD2
    using Logging
    using Statistics
    using Interpolations
    using LinearAlgebra
    using CSV
    using DataFrames
    using JSON

    include("utility.jl")
    include("HF_reference_solutions.jl")
    include("labels.jl")

    using .Utility: ReadFracture
    using .HFReferenceSolutions: HF_analytical_sol, get_fracture_dimensions_analytical
    using .Labels:

    export load_fractures, rename_simulation, get_fracture_variable, get_fracture_variable_at_point, get_fracture_variable_slice_interpolated,
        get_fracture_variable_slice_cell_center, get_HF_analytical_solution, get_HF_analytical_solution_at_point, get_fracture_dimensions_analytical_with_properties,
        write_fracture_variable_csv_file, read_fracture_variable_csv_file, write_fracture_mesh_csv_file, append_to_json_file, get_extremities_cells,
        get_front_intercepts, write_properties_csv_file, get_fracture_geometric_parameters, get_Ffront_as_vector, get_fracture_fp, 
        get_velocity_as_vector, get_velocity_slice, 

    slash = Sys.iswindows() ? '\\' : '/'

    #-----------------------------------------------------------------------------------------------------------------------

    """
        load_fractures(address=nothing, sim_name="simulation", time_period=0.0, time_srs=nothing, step_size=1, load_all=false)

        This function returns a list of the fractures. If address and simulation name are not provided, results from the
        default address and having the default name will be loaded.

        # Arguments
        - `address::Union{String, Nothing}`: the folder address containing the saved files. If it is not provided,
                                        simulation from the default folder (_simulation_data_PyFrac) will be loaded.
        - `sim_name::String`: the simulation name from which the fractures are to be loaded. If not
                                        provided, simulation with the default name (Simulation) will be loaded.
        - `time_period::Float64`: time period between two successive fractures to be loaded. if not provided,
                                        all fractures will be loaded.
        - `time_srs::Union{Vector{Float64}, Float64, Int, Nothing}`: if provided, the fracture stored at the closest time after the given times
                                        will be loaded.
        - `step_size::Int`: the number of time steps to skip before loading the next fracture. If not
                                        provided, all of the fractures will be loaded.
        - `load_all::Bool`: avoid jumping time steps too close to each other

        # Returns
        - `Tuple{Vector, Any}`: a list of fractures and properties
    """
    function load_fractures(address=nothing, sim_name='simulation', time_period=0.0, time_srs=nothing, step_size=1, load_all=false):

        logger = Logging.current_logger()
        @info "Returning fractures..."

        if address === nothing:
            address = "." * slash * "_simulation_data_PyFrac"
        end

        if address[end] != slash
            address = address * slash
        end

        if time_srs !== nothing
            if isa(time_srs, Number)
                time_srs = [Float64(time_srs)]
            elseif isa(time_srs, Vector)
                time_srs = Float64.(time_srs)
            end
        end

        sim_full_name = sim_name
        sim_full_path = address * sim_full_name
        properties_file = sim_full_path * slash * "properties.jld2"

        if isfile(properties_file)
            properties = load(properties_file)
        else
            error("Data not found. The address might be incorrect")
        end

        fileNo = 0
        next_t = 0.0
        t_srs_indx = 1
        fracture_list = []

        if time_srs !== nothing
            if length(time_srs) == 0
                return fracture_list
            end
            next_t = time_srs[t_srs_indx]
        end

        # time at wich the first fracture file was modified
        while fileNo < 500000
            # trying to load next file. exit loop if not found
            filename = sim_full_path * slash * sim_full_name * "_file_" * string(fileNo) * ".jld2"
            
            if isfile(filename)
                try
                    ff = load(filename)
                    fileNo += step_size
                    
                    if load_all
                        @info "Returning fracture at " * string(ff.time) * " s"
                        push!(fracture_list, ff)
                    else
                        if 1.0 - next_t / ff.time >= -1e-8
                            # if the current fracture time has advanced the output time period
                            @info "Returning fracture at " * string(ff.time) * " s"
                            push!(fracture_list, ff)

                            if t_srs_given
                                if t_srs_indx < length(time_srs)
                                    t_srs_indx += 1
                                    next_t = time_srs[t_srs_indx]
                                end
                                if ff.time > maximum(time_srs)
                                    break
                                end
                            else
                                next_t = ff.time + time_period
                            end
                        end
                    end
                catch e
                    @error "Error loading file $filename: $e"
                    break
                end
            else
                break
            end
        end

        if fileNo >= 500000
            error("too many files.")
        end

        if length(fracture_list) == 0
            error("Fracture list is empty")
        end

        return fracture_list, properties
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        rename_simulation(address=nothing, sim_name="simulation", sim_name_new=nothing)

        This function renames a given simulation. The time stamp of the simulation is copied from the old name.

        # Arguments
        - `address::Union{String, Nothing}`: the folder address containing the saved files. If it is not provided,
                                        simulation from the default folder (_simulation_data_PyFrac) will be loaded.
        - `sim_name::String`: the simulation name which is to be renamed.
        - `sim_name_new::Union{String, Nothing}`: the name to be given to the simulation.
    """
    function rename_simulation(address=nothing, sim_name="simulation", sim_name_new=nothing)
        @info "Renaming simulation..."

        if sim_name_new === nothing
            sim_name_new = sim_name * "new"
        end
        
        slash = Sys.iswindows() ? '\\' : '/'

        if address === nothing
            address = "." * slash * "_simulation_data_PyFrac"
        end

        if address[end] != slash
            address = address * slash
        end

        sim_full_name = sim_name
        sim_full_name_new = sim_name_new
        sim_full_path = address * sim_full_name
        sim_full_path_new = address * sim_full_name_new
        properties_file = sim_full_path * slash * "properties.jld2"

        fileNo = 0

        while fileNo < 5000
            # trying to load next file. exit loop if not found
            old_filename = sim_full_path * slash * sim_full_name * "_file_" * string(fileNo) * ".jld2"
            new_filename = sim_full_path * slash * sim_full_name_new * "_file_" * string(fileNo) * ".jld2"
            
            if isfile(old_filename)
                try
                    mv(old_filename, new_filename)
                catch e
                    @error "Error renaming file $old_filename: $e"
                    break
                end
            else
                break
            end

            fileNo += 1
        end

        if fileNo >= 5000
            error("too many files.")
        end

        if isdir(sim_full_path)
            mv(sim_full_path, sim_full_path_new)
        end
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_variable(fracture_list, variable, edge=4, return_time=false)

        This function returns the required variable from a fracture list.

        # Arguments
        - `fracture_list::Vector`: the fracture list from which the variable is to be extracted.
        - `variable::String`: the variable to be extracted. See labels.supported_variables of the
                            labels module for a list of supported variables.
        - `edge::Int`: the edge of the cell that will be plotted. This is for variables that
                    are evaluated on the cell edges instead of cell center. It can have a
                    value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
        - `return_time::Bool`: if true, the times at which the fractures are stored will also be returned.

        # Returns
        - `variable_list::Vector`: a list containing the extracted variable from each of the fracture. The 
                                dimension and type of each member of the list depends upon the variable type.
        - `time_srs::Vector`: a list of times at which the fractures are stored.
    """
    function get_fracture_variable(fracture_list, variable, edge=4, return_time=false)
        variable_list = []
        time_srs = []
        legend_coord = []

        if variable == "time" || variable == "t"
            for i in fracture_list
                push!(variable_list, i.time)
                push!(time_srs, i.time)
            end

        elseif variable == "width" || variable == "w" || variable == "surface"
            for i in fracture_list
                push!(variable_list, i.w)
                push!(time_srs, i.time)
            end

        elseif variable == "fluid pressure" || variable == "pf"
            for i in fracture_list
                push!(variable_list, i.pFluid)
                push!(time_srs, i.time)
            end

        elseif variable == "net pressure" || variable == "pn"
            for i in fracture_list
                push!(variable_list, i.pNet)
                push!(time_srs, i.time)
            end

        elseif variable == "front velocity" || variable == "v"
            for i in fracture_list
                vel = fill(NaN, i.mesh.NumberOfElts)
                vel[i.EltTip] .= i.v
                push!(variable_list, vel)
                push!(time_srs, i.time)
            end

        elseif variable == "Reynolds number" || variable == "Re"
            if fracture_list[end].ReynoldsNumber === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.ReynoldsNumber[edge+1])
                    push!(time_srs, i.time)
                elseif i.ReynoldsNumber !== nothing
                    push!(variable_list, mean(i.ReynoldsNumber, dims=1)[:])
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end

        elseif variable == "fluid flux" || variable == "ff"
            if fracture_list[end].fluidFlux === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.fluidFlux[edge+1])
                    push!(time_srs, i.time)
                elseif i.fluidFlux !== nothing
                    push!(variable_list, mean(i.fluidFlux, dims=1)[:])
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end

        elseif variable == "fluid velocity" || variable == "fv"
            if fracture_list[end].fluidVelocity === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.fluidVelocity[edge+1])
                    push!(time_srs, i.time)
                elseif i.fluidVelocity !== nothing
                    push!(variable_list, mean(i.fluidVelocity, dims=1)[:])
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end

        elseif variable == "pressure gradient x" || variable == "dpdx"
            for i in fracture_list
                dpdxLft = (i.pNet[i.EltCrack] - i.pNet[i.mesh.NeiElements[i.EltCrack, 1]]) * 
                        i.InCrack[i.mesh.NeiElements[i.EltCrack, 1]]
                dpdxRgt = (i.pNet[i.mesh.NeiElements[i.EltCrack, 2]] - i.pNet[i.EltCrack]) * 
                        i.InCrack[i.mesh.NeiElements[i.EltCrack, 2]]
                dpdx = zeros(i.mesh.NumberOfElts)
                dpdx[i.EltCrack] .= mean([dpdxLft, dpdxRgt], dims=1)[:]
                push!(variable_list, dpdx)
                push!(time_srs, i.time)
            end

        elseif variable == "pressure gradient y" || variable == "dpdy"
            for i in fracture_list
                dpdyBtm = (i.pNet[i.EltCrack] - i.pNet[i.mesh.NeiElements[i.EltCrack, 3]]) * 
                        i.InCrack[i.mesh.NeiElements[i.EltCrack, 3]]
                dpdxtop = (i.pNet[i.mesh.NeiElements[i.EltCrack, 4]] - i.pNet[i.EltCrack]) * 
                        i.InCrack[i.mesh.NeiElements[i.EltCrack, 4]]
                dpdy = zeros(i.mesh.NumberOfElts)
                dpdy[i.EltCrack] .= mean([dpdyBtm, dpdxtop], dims=1)[:]
                push!(variable_list, dpdy)
                push!(time_srs, i.time)
            end

        elseif variable == "fluid flux as vector field" || variable == "ffvf"
            if fracture_list[end].fluidFlux_components === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.fluidFlux_components[edge+1])
                    push!(time_srs, i.time)
                elseif i.fluidFlux_components !== nothing
                    push!(variable_list, i.fluidFlux_components)
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end

        elseif variable == "fluid velocity as vector field" || variable == "fvvf"
            if fracture_list[end].fluidVelocity_components === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.fluidVelocity_components[edge+1])
                    push!(time_srs, i.time)
                elseif i.fluidFlux_components !== nothing
                    push!(variable_list, i.fluidVelocity_components)
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end

        elseif variable == "effective viscosity" || variable == "ev"
            if fracture_list[end].effVisc === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.effVisc[edge+1])
                    push!(time_srs, i.time)
                elseif i.effVisc !== nothing
                    push!(variable_list, mean(i.effVisc, dims=1)[:])
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end
        
        elseif variable == "prefactor G" || variable == "G"
            if fracture_list[end].G === nothing
                error("Variable not saved")
            end
            for i in fracture_list
                if edge < 0 || edge > 4
                    error("Edge can be an integer between and including 0 and 4.")
                end
                if edge < 4
                    push!(variable_list, i.G[edge+1])
                    push!(time_srs, i.time)
                elseif i.G !== nothing
                    push!(variable_list, mean(i.G, dims=1)[:])
                    push!(time_srs, i.time)
                else
                    push!(variable_list, fill(NaN, i.mesh.NumberOfElts))
                end
            end

        elseif variable in ("front_dist_min", "d_min", "front_dist_max", "d_max", "front_dist_mean", "d_mean")
            for i in fracture_list
                if length(i.source) != 0
                    source_loc = i.mesh.CenterCoor[i.source[1], :]
                end
                # coordinate of the zero vertex in the tip cells
                front_intersect_dist = sqrt.((i.Ffront[:, [1, 3]][:] .- source_loc[1]) .^ 2 .+
                                            (i.Ffront[:, [2, 4]][:] .- source_loc[2]) .^ 2)
                if variable == "front_dist_mean" || variable == "d_mean"
                    push!(variable_list, mean(front_intersect_dist))
                elseif variable == "front_dist_max" || variable == "d_max"
                    push!(variable_list, maximum(front_intersect_dist))
                elseif variable == "front_dist_min" || variable == "d_min"
                    push!(variable_list, minimum(front_intersect_dist))
                end
                push!(time_srs, i.time)
            end
            
        elseif variable == "mesh"
            for i in fracture_list
                push!(variable_list, i.mesh)
                push!(time_srs, i.time)
            end

        elseif variable == "efficiency" || variable == "ef"
            for i in fracture_list
                push!(variable_list, i.efficiency)
                push!(time_srs, i.time)
            end
                
        elseif variable == "volume" || variable == "V"
            for i in fracture_list
                push!(variable_list, i.FractureVolume)
                push!(time_srs, i.time)
            end
                
        elseif variable == "leak off" || variable == "lk"
            for i in fracture_list
                push!(variable_list, i.LkOff)
                push!(time_srs, i.time)
            end
                
        elseif variable == "leaked off volume" || variable == "lkv"
            for i in fracture_list
                push!(variable_list, sum(i.LkOffTotal[i.EltCrack]))
                push!(time_srs, i.time)
            end
                
        elseif variable == "aspect ratio" || variable == "ar"
            for fr in fracture_list
                x_coords = vcat(fr.Ffront[:, 1], fr.Ffront[:, 3])
                x_len = maximum(x_coords) - minimum(x_coords)
                y_coords = vcat(fr.Ffront[:, 2], fr.Ffront[:, 4])
                y_len = maximum(y_coords) - minimum(y_coords)
                push!(variable_list, x_len / y_len)
                push!(time_srs, fr.time)
            end

        elseif variable == "chi"
            for i in fracture_list
                vel = fill(NaN, i.mesh.NumberOfElts)
                vel[i.EltTip] .= i.v
                push!(variable_list, vel)
                push!(time_srs, i.time)
            end

        elseif variable == "regime"
            if hasproperty(fracture_list[1], :regime_color)
                for i in fracture_list
                    push!(variable_list, i.regime_color)
                    push!(time_srs, i.time)
                end
            else
                error("The regime cannot be found. Saving of regime is most likely not enabled.\n" *
                    " See the saveRegime flag of SimulationProperties class.")
            end

        elseif variable == "source elements" || variable == "se"
            for fr in fracture_list
                push!(variable_list, fr.source)
                push!(time_srs, fr.time)
            end

        elseif variable == "injection line pressure" || variable == "ilp"
            for fr in fracture_list
                if fr.pInjLine === nothing
                    error("It seems that injection line is not solved. Injection line pressure is not available")
                else
                    push!(variable_list, fr.pInjLine)
                end
                push!(time_srs, fr.time)
            end

        elseif variable == "injection rate" || variable == "ir"
            for fr in fracture_list
                if fr.injectionRate === nothing
                    error("It seems that injection line is not solved. Injection rate is not available")
                else
                    push!(variable_list, fr.injectionRate)
                end
                push!(time_srs, fr.time)
            end

        elseif variable == "total injection rate" || variable == "tir"
            for fr in fracture_list
                if fr.injectionRate === nothing
                    error("It seems that injection line is not solved. Injection rate is not available")
                else
                    push!(variable_list, sum(fr.injectionRate))
                end
                push!(time_srs, fr.time)
            end

        else
            error("The variable type is not correct.")
        end

        if !return_time
            return variable_list
        elseif variable == "regime"
            return variable_list, legend_coord, time_srs
        else
            return variable_list, time_srs
        end
    end#-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_variable_at_point(fracture_list, variable, point, edge=4, return_time=true)

        This function returns the required variable from a fracture list at the given point.

        # Arguments
        - `fracture_list::Vector`: the fracture list from which the variable is to be extracted.
        - `variable::String`: the variable to be extracted. See supported_variables of the
                            Labels module for a list of supported variables.
        - `point::Vector`: the point at which the given variable is plotted against time [x, y].
        - `edge::Int`: the edge of the cell that will be plotted. This is for variables that
                    are evaluated on the cell edges instead of cell center. It can have a
                    value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
        - `return_time::Bool`: if true, the times at which the fractures are stored will also be returned.

        # Returns
        - `variable_list::Vector`: a list containing the extracted variable from each of the fracture. The 
                                dimension and type of each member of the list depends upon the variable type.
        - `time_srs::Vector`: a list of times at which the fractures are stored.
    """
    function get_fracture_variable_at_point(fracture_list, variable, point, edge=4, return_time=true)
        logger = Logging.current_logger()
        
        if variable ∉ supported_variables
            error("Variable not supported")
        end

        return_list = []

        if variable in ["front intercepts", "fi"]
            return_list = get_front_intercepts(fracture_list, point)
            if return_time
                return return_list, get_fracture_variable(fracture_list, "t")
            else
                return return_list
            end
        else
            var_values, time_list = get_fracture_variable(fracture_list, variable, edge=edge, return_time=true)
        end

        if variable in unidimensional_variables
            return_list = var_values
        else
            for i in 1:length(fracture_list)
                if variable in bidimensional_variables

                    points = fracture_list[i].mesh.CenterCoor
                    values = var_values[i]
                    
                    try
                        interpolator = LinearInterpolation(points, values, extrapolation_bc=NaN)
                        value_point = interpolator(point[1], point[2])
                        
                        if isnan(value_point)
                            @warn "Point outside fracture."
                        end
                        
                        push!(return_list, value_point)
                    catch e
                        @warn "Interpolation failed: $e"
                        push!(return_list, NaN)
                    end
                end
            end
        end

        if return_time
            return return_list, time_list
        else
            return return_list
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_variable_slice_interpolated(var_value, mesh, point1=nothing, point2=nothing)

        This function returns the given fracture variable on a given slice of the domain. Two points are to be given that
        will be joined to form the slice. The values on the slice are interpolated from the values available on the cell
        centers.

        # Arguments
        - `var_value::Vector{Float64}`: the value of the variable on each cell of the domain.
        - `mesh::CartesianMesh`: the CartesianMesh object describing the mesh.
        - `point1::Union{Vector{Float64}, Nothing}`: the left point from which the slice should pass [x, y].
        - `point2::Union{Vector{Float64}, Nothing}`: the right point from which the slice should pass [x, y].

        # Returns
        - `value_samp_points::Vector{Float64}`: the values of the variable at the sampling points.
        - `sampling_line::Vector{Float64}`: the distance of the point where the value is provided from the center of the slice.
    """
    function get_fracture_variable_slice_interpolated(var_value, mesh, point1=nothing, point2=nothing)
        if !(typeof(var_value) <: AbstractVector{<:Number})
            error("Variable value should be provided in the form of array with the size equal to the number of elements in the mesh!")
        elseif length(var_value) != mesh.NumberOfElts
            error("Given array is not equal to the number of elements in mesh!")
        end

        if point1 === nothing
            point1 = [-mesh.Lx, 0.0]
        end
        if point2 === nothing
            point2 = [mesh.Lx, 0.0]
        end

        # Convert to mutable arrays if needed
        point1 = Float64.(point1)
        point2 = Float64.(point2)

        # the code below find the extreme points of the line joining the two given points with the current mesh
        if point2[1] == point1[1]
            point1[2] = -mesh.Ly
            point2[2] = mesh.Ly
        elseif point2[2] == point1[2]
            point1[1] = -mesh.Lx
            point2[1] = mesh.Lx
        else
            slope = (point2[2] - point1[2]) / (point2[1] - point1[1])
            y_intrcpt_lft = slope * (-mesh.Lx - point1[1]) + point1[2]
            y_intrcpt_rgt = slope * (mesh.Lx - point1[1]) + point1[2]
            x_intrcpt_btm = (-mesh.Ly - point1[2]) / slope + point1[1]
            x_intrcpt_top = (mesh.Ly - point1[2]) / slope + point1[1]

            if abs(y_intrcpt_lft) < mesh.Ly
                point1[1] = -mesh.Lx
                point1[2] = y_intrcpt_lft
            end
            if y_intrcpt_lft > mesh.Ly
                point1[1] = x_intrcpt_top
                point1[2] = mesh.Ly
            end
            if y_intrcpt_lft < -mesh.Ly
                point1[1] = x_intrcpt_btm
                point1[2] = -mesh.Ly
            end

            if abs(y_intrcpt_rgt) < mesh.Ly
                point2[1] = mesh.Lx
                point2[2] = y_intrcpt_rgt
            end
            if y_intrcpt_rgt > mesh.Ly
                point2[1] = x_intrcpt_top
                point2[2] = mesh.Ly
            end
            if y_intrcpt_rgt < -mesh.Ly
                point2[1] = x_intrcpt_btm
                point2[2] = -mesh.Ly
            end
        end

        # Create sampling points
        x_samples = range(point1[1], point2[1], length=105)
        y_samples = range(point1[2], point2[2], length=105)
        sampling_points = hcat(x_samples, y_samples)

        # Interpolate values at sampling points
        try
            interpolator = LinearInterpolation(mesh.CenterCoor, var_value, 
                                            extrapolation_bc=NaN)
            value_samp_points = [interpolator(sampling_points[i, 1], sampling_points[i, 2]) 
                            for i in 1:size(sampling_points, 1)]
        catch e
            value_samp_points = fill(NaN, 105)
        end

        # Calculate sampling line distances
        sampling_line_lft = sqrt.((sampling_points[1:52, 1] .- sampling_points[53, 1]) .^ 2 .+
                                (sampling_points[1:52, 2] .- sampling_points[53, 2]) .^ 2)
        sampling_line_rgt = sqrt.((sampling_points[53:end, 1] .- sampling_points[53, 1]) .^ 2 .+
                                (sampling_points[53:end, 2] .- sampling_points[53, 2]) .^ 2)
        sampling_line = vcat(-reverse(sampling_line_lft), sampling_line_rgt)

        return value_samp_points, sampling_line
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_variable_slice_cell_center(var_value, mesh, point=nothing, orientation="horizontal")

        This function returns the given fracture variable on a given slice of the domain. Two slice is constructed from the
        given point and the orientation. The values on the slice are taken from the cell centers.

        # Arguments
        - `var_value::Vector{Float64}`: the value of the variable on each cell of the domain.
        - `mesh::CartesianMesh`: the CartesianMesh object describing the mesh.
        - `point::Union{Vector{Float64}, Nothing}`: the point from which the slice should pass [x, y]. If it does not lie on a cell
                                        center, the closest cell center will be taken. By default, [0., 0.] will be
                                        taken.
        - `orientation::String`: the orientation according to which the slice is made in the case the
                                        plotted values are not interpolated and are taken at the cell centers.
                                        Any of the four ("vertical", "horizontal", "ascending" and "descending")
                                        orientation can be used.

        # Returns
        - `var_value::Vector{Float64}`: the values of the variable at the sampling points.
        - `sampling_line::Vector{Float64}`: the distance of the point where the value is provided from the center of the slice.
        - `sampling_cells::Vector{Int}`: the cells on the mesh along with the slice is made.
    """
    function get_fracture_variable_slice_cell_center(var_value, mesh, point=nothing, orientation="horizontal")
        if !(typeof(var_value) <: AbstractVector{<:Number})
            error("Variable value should be provided in the form of array with the size equal to the number of elements in the mesh!")
        elseif length(var_value) != mesh.NumberOfElts
            error("Given array is not equal to the number of elements in mesh!")
        end

        if point === nothing
            point = [0.0, 0.0]
        end
        if !(orientation in ("horizontal", "vertical", "increasing", "decreasing"))
            error("Given orientation is not supported. Possible options:\n 'horizontal', 'vertical', 'increasing', 'decreasing'")
        end

        zero_cell = mesh.locate_element(point[1], point[2])
        if isnan(zero_cell)
            error("The given point does not lie in the grid!")
        end

        zero_cell = Int(zero_cell)
        sampling_cells = Int[]
        
        if orientation == "vertical"
            bottom_cells = Int[]
            current_cell = zero_cell
            while current_cell > 0
                push!(bottom_cells, current_cell)
                current_cell -= mesh.nx
            end
            sampling_cells = vcat(reverse(bottom_cells[2:end]), [zero_cell])
            
            current_cell = zero_cell + mesh.nx
            while current_cell <= mesh.NumberOfElts
                push!(sampling_cells, current_cell)
                current_cell += mesh.nx
            end
            
        elseif orientation == "horizontal"
            row_start = ((zero_cell - 1) ÷ mesh.nx) * mesh.nx + 1
            row_end = row_start + mesh.nx - 1
            sampling_cells = collect(row_start:row_end)
            
        elseif orientation == "increasing"
            bottom_half = Int[]
            current_cell = zero_cell
            while current_cell > 0 && current_cell <= mesh.NumberOfElts
                push!(bottom_half, current_cell)
                next_cell = current_cell - mesh.nx - 1
                if next_cell <= 0 || next_cell > mesh.NumberOfElts
                    break
                end
                current_cell = next_cell
            end
            
            bottom_half_filtered = Int[]
            for cell in bottom_half
                if mesh.CenterCoor[cell, 1] <= mesh.CenterCoor[zero_cell, 1]
                    push!(bottom_half_filtered, cell)
                end
            end
            
            top_half = Int[]
            current_cell = zero_cell
            while current_cell > 0 && current_cell <= mesh.NumberOfElts
                next_cell = current_cell + mesh.nx + 1
                if next_cell <= 0 || next_cell > mesh.NumberOfElts
                    break
                end
                current_cell = next_cell
                push!(top_half, current_cell)
            end
            
            top_half_filtered = Int[]
            for cell in top_half
                if cell <= mesh.NumberOfElts && mesh.CenterCoor[cell, 1] >= mesh.CenterCoor[zero_cell, 1]
                    push!(top_half_filtered, cell)
                end
            end
            
            sampling_cells = vcat(reverse(bottom_half_filtered), top_half_filtered)
            
        elseif orientation == "decreasing"
            bottom_half = Int[]
            current_cell = zero_cell
            while current_cell > 0 && current_cell <= mesh.NumberOfElts
                push!(bottom_half, current_cell)
                next_cell = current_cell - mesh.nx + 1
                if next_cell <= 0 || next_cell > mesh.NumberOfElts
                    break
                end
                current_cell = next_cell
            end
            
            bottom_half_filtered = Int[]
            for cell in bottom_half
                if mesh.CenterCoor[cell, 1] >= mesh.CenterCoor[zero_cell, 1]
                    push!(bottom_half_filtered, cell)
                end
            end
            
            top_half = Int[]
            current_cell = zero_cell
            while current_cell > 0 && current_cell <= mesh.NumberOfElts
                next_cell = current_cell + mesh.nx - 1
                if next_cell <= 0 || next_cell > mesh.NumberOfElts
                    break
                end
                current_cell = next_cell
                push!(top_half, current_cell)
            end
            
            top_half_filtered = Int[]
            for cell in top_half
                if cell <= mesh.NumberOfElts && mesh.CenterCoor[cell, 1] <= mesh.CenterCoor[zero_cell, 1]
                    push!(top_half_filtered, cell)
                end
            end
            
            sampling_cells = vcat(reverse(bottom_half_filtered), top_half_filtered)
        end

        sampling_cells = filter(x -> x > 0 && x <= mesh.NumberOfElts, sampling_cells)

        if length(sampling_cells) == 0
            sampling_line = Float64[]
            return Float64[], sampling_line, Int[]
        end

        sampling_len = sqrt((mesh.CenterCoor[sampling_cells[1], 1] - mesh.CenterCoor[sampling_cells[end], 1])^2 +
                        (mesh.CenterCoor[sampling_cells[1], 2] - mesh.CenterCoor[sampling_cells[end], 2])^2)

        # making x-axis centered at zero for the 1D slice. Necessary to have same reference with different meshes and
        # analytical solution plots.
        sampling_line = range(0, sampling_len, length=length(sampling_cells)) .- sampling_len / 2

        return var_value[sampling_cells], collect(sampling_line), sampling_cells
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_HF_analytical_solution(regime, variable, mat_prop, inj_prop, mesh=nothing, fluid_prop=nothing,
                                time_srs=nothing, length_srs=nothing, h=nothing, samp_cell=nothing, gamma=nothing)

        Get analytical solution for hydraulic fracturing problem.

        # Arguments
        - `regime::String`: the regime for which the analytical solution is to be evaluated.
        - `variable::String`: the variable to be extracted.
        - `mat_prop`: material properties.
        - `inj_prop`: injection properties.
        - `mesh`: the mesh object (optional).
        - `fluid_prop`: fluid properties (optional).
        - `time_srs`: time series (optional).
        - `length_srs`: length series (optional).
        - `h`: parameter h (optional).
        - `samp_cell`: sample cell (optional).
        - `gamma`: parameter gamma (optional).

        # Returns
        - `return_list::Vector`: list of computed values.
        - `mesh_list::Vector`: list of mesh objects.
    """
    function get_HF_analytical_solution(regime, variable, mat_prop, inj_prop, mesh=nothing, fluid_prop=nothing,
                                    time_srs=nothing, length_srs=nothing, h=nothing, samp_cell=nothing, gamma=nothing)
        
        if time_srs === nothing && length_srs === nothing
            error("Either time series or lengths series is to be provided.")
        end

        # Initialize parameters
        Kc_1 = regime == "E_K" ? mat_prop.Kc1 : nothing
        Cij = regime == "E_E" ? mat_prop.Cij : nothing
        density = regime == "MDR" && fluid_prop !== nothing ? fluid_prop.density : nothing

        # Calculate V0
        V0 = nothing
        if size(inj_prop.injectionRate, 1) > 2
            V0 = inj_prop.injectionRate[1, 2] * inj_prop.injectionRate[2, 1]
        end

        # Check for fluid properties requirement
        muPrime = nothing
        if regime in ["M", "MDR", "Mt", "PKN", "Mp"]
            if fluid_prop === nothing
                error("Fluid properties required for " * regime * " type analytical solution")
            end
            muPrime = fluid_prop.muPrime
        end

        # Set sample cell
        if samp_cell === nothing
            samp_cell = Int(length(mat_prop.Kprime) / 2)
        end

        # Determine series length
        srs_length = time_srs !== nothing ? length(time_srs) : length(length_srs)

        mesh_list = []
        return_list = []

        for i in 1:srs_length
            # Get length and time
            length_val = length_srs !== nothing ? length_srs[i] : nothing
            time_val = time_srs !== nothing ? time_srs[i] : nothing

            if variable in ["time", "t", "width", "w", "net pressure", "pn", "front velocity", "v"]
                # Handle mesh creation
                mesh_i = mesh
                if mesh === nothing && variable in ["width", "w", "net pressure", "pn"]
                    x_len, y_len = get_fracture_dimensions_analytical_with_properties(regime,
                                                                                    time_srs[i],
                                                                                    mat_prop,
                                                                                    inj_prop,
                                                                                    fluid_prop=fluid_prop,
                                                                                    h=h,
                                                                                    samp_cell=samp_cell,
                                                                                    gamma=gamma)
                    # Assuming CartesianMesh is available
                    mesh_i = CartesianMesh(x_len, y_len, 151, 151)
                end

                
                # Get injection rate (assuming it's a 2D array)
                injection_rate = size(inj_prop.injectionRate, 1) >= 1 ? inj_prop.injectionRate[1, 1] : inj_prop.injectionRate[1]
                
                # Call analytical solution (assuming HF_analytical_sol is defined)
                # Note: required_string should be defined somewhere
                t, r, p, w, v, actvElts = HF_analytical_sol(regime,
                                                        mesh_i,
                                                        mat_prop.Eprime,
                                                        injection_rate,
                                                        inj_point=inj_prop.sourceCoordinates,
                                                        muPrime = muPrime,
                                                        Kprime=mat_prop.Kprime[samp_cell],
                                                        Cprime=mat_prop.Cprime[samp_cell],
                                                        length=length_val,
                                                        t=time_val,
                                                        Kc_1=Kc_1,
                                                        h=h,
                                                        density=density,
                                                        Cij=Cij,
                                                        gamma=gamma,
                                                        required=get_required_string(variable), # Assuming this function exists
                                                        Vinj=V0)
                
                push!(mesh_list, mesh_i)

                # Add results based on variable
                if variable in ["time", "t"]
                    push!(return_list, t)
                elseif variable in ["width", "w"]
                    push!(return_list, w)
                elseif variable in ["net pressure", "pn"]
                    push!(return_list, p)
                elseif variable in ["front velocity", "v"]
                    push!(return_list, v)
                end

            elseif variable in ["front_dist_min", "d_min", "front_dist_max", "d_max", "front_dist_mean", "d_mean", "radius", "r"]
                x_len, y_len = get_fracture_dimensions_analytical_with_properties(regime,
                                                                                time_val,
                                                                                mat_prop,
                                                                                inj_prop,
                                                                                fluid_prop=fluid_prop,
                                                                                h=h,
                                                                                samp_cell=samp_cell,
                                                                                gamma=gamma)
                
                if variable in ["radius", "r"]
                    push!(return_list, x_len)
                elseif variable in ["front_dist_min", "d_min"]
                    push!(return_list, y_len)
                elseif variable in ["front_dist_max", "d_max"]
                    push!(return_list, x_len)
                elseif variable in ["front_dist_mean", "d_mean"]
                    if regime in ("E_K", "E_E")
                        error("Mean distance not available.")
                    else
                        push!(return_list, x_len)
                    end
                end
            else
                error("The variable type is not correct or the analytical solution not available. Select one of the following:\n" *
                    "-- 'r' or 'radius'\n" * 
                    "-- 'w' or 'width'\n" * 
                    "-- 'pn' or 'net pressure'\n" * 
                    "-- 'v' or 'front velocity'\n" *
                    "-- 'd_min' or 'front_dist_min'\n" *
                    "-- 'd_max' or 'front_dist_max'\n" *
                    "-- 'd_mean' or 'front_dist_mean'\n")
            end
        end

        return return_list, mesh_list
    end

    # Helper function to get required string (needs to be implemented based on your requirements)
    function get_required_string(variable)
        # This should map variable names to required strings for HF_analytical_sol
        required_map = Dict(
            "time" => "t",
            "t" => "t",
            "width" => "w", 
            "w" => "w",
            "net pressure" => "pn",
            "pn" => "pn",
            "front velocity" => "v",
            "v" => "v"
        )
        return get(required_map, variable, "")
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_HF_analytical_solution_at_point(regime, variable, point, mat_prop, inj_prop, fluid_prop=nothing, time_srs=nothing,
                                            length_srs=nothing, h=nothing, samp_cell=nothing, gamma=nothing)

        Get analytical solution for hydraulic fracturing problem at a specific point.

        # Arguments
        - `regime::String`: the regime for which the analytical solution is to be evaluated.
        - `variable::String`: the variable to be extracted.
        - `point::Vector{Float64}`: the point [x, y] at which the solution is evaluated.
        - `mat_prop`: material properties.
        - `inj_prop`: injection properties.
        - `fluid_prop`: fluid properties (optional).
        - `time_srs`: time series (optional).
        - `length_srs`: length series (optional).
        - `h`: parameter h (optional).
        - `samp_cell`: sample cell (optional).
        - `gamma`: parameter gamma (optional).

        # Returns
        - `values_point::Vector`: values at the specified point.
    """
    function get_HF_analytical_solution_at_point(regime, variable, point, mat_prop, inj_prop, fluid_prop=nothing, time_srs=nothing,
                                                length_srs=nothing, h=nothing, samp_cell=nothing, gamma=nothing)
        
        values_point = Float64[]

        # Determine series length
        srs_length = time_srs !== nothing ? length(time_srs) : length(length_srs)

        # Create mesh
        mesh_Lx = point[1] == 0.0 ? 1.0 : 2 * abs(point[1])
        mesh_Ly = point[2] == 0.0 ? 1.0 : 2 * abs(point[2])
        mesh = CartesianMesh(mesh_Lx, mesh_Ly, 5, 5)

        for i in 1:srs_length
            # Prepare time and length arrays
            time = time_srs !== nothing ? [time_srs[i]] : nothing
            length_val = length_srs !== nothing ? [length_srs[i]] : nothing

            # Get analytical solution
            value_mesh, mesh_list = get_HF_analytical_solution(regime,
                                                            variable,
                                                            mat_prop,
                                                            inj_prop,
                                                            mesh=mesh,
                                                            fluid_prop=fluid_prop,
                                                            time_srs=time,
                                                            length_srs=length_val,
                                                            h=h,
                                                            samp_cell=samp_cell,
                                                            gamma=gamma)
            
            if variable in ["front_dist_min", "d_min", "front_dist_max", "d_max", "front_dist_mean", "d_mean", "radius", "r", "t", "time"]
                push!(values_point, value_mesh[1])
            elseif point == [0.0, 0.0]
                # Assuming mesh_list[1].CenterElts is available and properly indexed
                center_elt_index = mesh_list[1].CenterElts
                push!(values_point, value_mesh[1][center_elt_index])
            else
                value_point = value_mesh[1][19]
                push!(values_point, value_point)
            end
        end

        return values_point
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_dimensions_analytical_with_properties(regime, time_srs, mat_prop, inj_prop, fluid_prop=nothing,
                                                        h=nothing, samp_cell=nothing, gamma=nothing)

        Get fracture dimensions from analytical solution with given properties.

        # Arguments
        - `regime::String`: the regime for which the analytical solution is to be evaluated.
        - `time_srs`: time series.
        - `mat_prop`: material properties.
        - `inj_prop`: injection properties.
        - `fluid_prop`: fluid properties (optional).
        - `h`: parameter h (optional).
        - `samp_cell`: sample cell (optional).
        - `gamma`: parameter gamma (optional).

        # Returns
        - `Tuple{Float64, Float64}`: x_length and y_length of the fracture.
    """
    function get_fracture_dimensions_analytical_with_properties(regime, time_srs, mat_prop, inj_prop, fluid_prop=nothing,
                                                            h=nothing, samp_cell=nothing, gamma=nothing)
        
        # Initialize parameters
        Kc_1 = regime == "E_K" ? mat_prop.Kc1 : nothing
        density = regime == "MDR" && fluid_prop !== nothing ? fluid_prop.density : nothing

        # Check for fluid properties requirement
        muPrime = nothing
        if regime in ("M", "Mt", "PKN", "MDR", "Mp", "La")
            if fluid_prop === nothing
                error("Fluid properties required to evaluate analytical solution")
            end
            muPrime = fluid_prop.muPrime
        end

        # Set sample cell
        if samp_cell === nothing
            samp_cell = Int(length(mat_prop.Kprime) / 2)
        end

        # Calculate V0
        V0 = nothing
        if size(inj_prop.injectionRate, 1) > 2
            V0 = inj_prop.injectionRate[1, 2] * inj_prop.injectionRate[2, 1]  # Julia indexing
        end

        # Get injection rate
        Q0 = size(inj_prop.injectionRate, 1) >= 1 ? inj_prop.injectionRate[1, 1] : inj_prop.injectionRate[1]
        
        # Get fracture dimensions (assuming get_fracture_dimensions_analytical is implemented)
        x_len, y_len = get_fracture_dimensions_analytical(regime,
                                                        maximum(time_srs),
                                                        mat_prop.Eprime,
                                                        Q0,
                                                        muPrime,
                                                        Kprime=mat_prop.Kprime[samp_cell],
                                                        Cprime=mat_prop.Cprime[samp_cell],
                                                        Kc_1=Kc_1,
                                                        h=h,
                                                        density=density,
                                                        gamma=gamma,
                                                        Vinj=V0)

        return x_len, y_len
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        write_fracture_variable_csv_file(file_name, fracture_list, variable, point=nothing, edge=4)

        This function writes fracture variable from each fracture in the list as a csv file. The variable from each of
        the fracture in the list will saved in a row of the csv file. If a variable is bi-dimensional, a point can be
        given at which the variable is to be saved.

        # Arguments
        - `file_name::String`: the name of the file to be written.
        - `fracture_list::Vector`: the fracture list from which the variable is to be extracted.
        - `variable::String`: the variable to be saved. See supported_variables of the
                            Labels module for a list of supported variables.
        - `point::Union{Vector{Float64}, Nothing}`: the point in the mesh at which the given variable is saved [x, y]. If the
                                                point is not given, the variable will be saved on the whole mesh.
        - `edge::Int`: the edge of the cell that will be saved. This is for variables that
                    are evaluated on the cell edges instead of cell center. It can have a
                    value from 0 to 4 (0->left, 1->right, 2->bottom, 3->top, 4->average).
    """
    function write_fracture_variable_csv_file(file_name, fracture_list, variable, point=nothing, edge=4)
        logger = Logging.current_logger()
        
        if variable ∉ supported_variables
            error(err_msg_variable)
        end

        var_values, time_list = get_fracture_variable(fracture_list, variable, edge=edge, return_time=true)

        return_list = []
        header = String[]

        if point === nothing
            if !(typeof(var_values[1]) <: AbstractArray)
                var_values = [Float64[value] for value in var_values]
                header = ["time, s", "value"]
            else
                header = vcat(["time, s"], ["$i cell" for i in 1:length(var_values[1])])
            end
            
            for (time, values) in zip(time_list, var_values)
                push!(return_list, vcat([time], values))
            end
        else
            for i in 1:length(fracture_list)
                try
                    points = fracture_list[i].mesh.CenterCoor
                    values = var_values[i]
                    
                    interpolator = LinearInterpolation(points, values, extrapolation_bc=NaN)
                    value_point = interpolator(point[1], point[2])
                    
                    if isnan(value_point)
                        @warn "Point outside fracture."
                    end
                    push!(return_list, value_point)
                catch e
                    @warn "Interpolation failed: $e"
                    push!(return_list, NaN)
                end
            end
        end

        @assert typeof(var_values[1]) <: AbstractArray

        if file_name[end-3:end] != ".csv"
            file_name = file_name * ".csv"
        end

        if point === nothing
            df_data = hcat(time_list, hcat([v for v in var_values]...))
            column_names = header
            df = DataFrame(df_data, column_names)
        else
            df = DataFrame(time=time_list, value=return_list)
        end

        CSV.write(file_name, df)
    end


    #-----------------------------------------------------------------------------------------------------------------------
    """
        read_fracture_variable_csv_file(file_name)

        This function returns the required variable from the csv file.

        # Arguments
        - `file_name::String`: the name of the file to be read.

        # Returns
        - `variable_list::Matrix{Float64}`: a matrix containing the extracted variable from each of the fracture. The 
                                        dimension and type depends upon the variable type.
    """
    function read_fracture_variable_csv_file(file_name)
        if file_name[end-3:end] != ".csv"
            file_name = file_name * ".csv"
        end

        variable_list = CSV.read(file_name, Matrix, header=false)
        
        return variable_list
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        write_fracture_mesh_csv_file(file_name, mesh_list, time_steps=nothing)

        This function writes important data of a mesh as a csv file. The csv contains (in a row vector) the number of
        elements, hx, hy, nx, ny, the flattened connectivity matrix and the flattened node coordinates. Each row of the csv
        corresponds to an entry in the mesh_list.

        # Arguments
        - `file_name::String`: the name of the file to be written.
        - `mesh_list::Vector`: the mesh list from which the data is to be extracted.
        - `time_steps::Union{Vector{Float64}, Nothing}`: time steps corresponding to meshes (optional).
    """
    function write_fracture_mesh_csv_file(file_name, mesh_list, time_steps=nothing)
        return_list = []

        if time_steps !== nothing
            for (ind, i) in enumerate(mesh_list)
                export_mesh = [time_steps[ind], Float64(i.NumberOfElts)]
                export_mesh = vcat(export_mesh, [i.hx, i.hy, Float64(i.nx), Float64(i.ny)])
                export_mesh = vcat(export_mesh, vec(i.CenterCoor))  # flattening CenterCoor
                push!(return_list, export_mesh)
            end
            header = ["Time, s", "Number of cells", "dx", "dy", "nx", "ny", "cell centers..."]
        else
            for i in mesh_list
                export_mesh = [Float64(i.NumberOfElts)]
                export_mesh = vcat(export_mesh, [i.hx, i.hy, Float64(i.nx), Float64(i.ny)])
                export_mesh = vcat(export_mesh, vec(i.CenterCoor))  # flattening CenterCoor
                push!(return_list, export_mesh)
            end
            header = ["Number of cells", "dx", "dy", "nx", "ny", "cell centers..."]
        end

        if file_name[end-3:end] != ".csv"
            file_name = file_name * ".csv"
        end

        open(file_name, "w") do io
            writedlm(io, header, ',')
            writedlm(io, return_list, ',')
        end
    end
    #-----------------------------------------------------------------------------------------------------------------------


    """
        append_to_json_file(file_name, content, action, key=nothing, delete_existing_filename=false)

        This function appends data of a mesh as a json file.

        # Arguments
        - `file_name::String`: the name of the file to be written.
        - `content`: the content to be written (list, dictionary, etc.)
        - `action::String`: action to take. Current options are:
                        'append2keyASnewlist' - append content as a new list in a list of lists
                        'append2keyAND2list' - append content to existing list
                        'dump_this_dictionary' - dump only the content of the dictionary
                        'extend_dictionary' - extend existing dictionary with new content
        - `key::Union{String, Nothing}`: a string that describes the information you are passing.
        - `delete_existing_filename::Bool`: whether to delete existing file.
    """
    function append_to_json_file(file_name, content, action, key=nothing, delete_existing_filename=false)
        logger = Logging.current_logger()


        # 1) Check if the file_name is a Json file
        if file_name[end-4:end] != ".json"
            file_name = file_name * ".json"
        end

        # 3) Check if the file already exists
        if isfile(file_name) && delete_existing_filename
            rm(file_name)
            @warn "File " * file_name * " existed and it will be Removed!"
        end

        # 4) Check if the file already exists
        if isfile(file_name)
            # The file exists
            try
                # Read existing data
                data = open(file_name, "r") do f
                    JSON.parse(f)
                end

                if action in ["append2keyASnewlist", "append2keyAND2list"] && key !== nothing
                    if haskey(data, key)  # the key exists and we need just to add the value
                        if isa(data[key], Vector)  # the data that is already there is a list and a key is provided
                            push!(data[key], content)
                        elseif action == "append2keyAND2list"
                            data[key] = [data[key], content]
                        elseif action == "append2keyASnewlist"
                            data[key] = [[data[key]], [content]]
                        end
                    else
                        if action == "append2keyAND2list"
                            to_write = Dict(key => content)
                        elseif action == "append2keyASnewlist"
                            to_write = Dict(key => [content])
                        end
                        merge!(data, to_write)
                        open(file_name, "w") do f
                            JSON.print(f, data)
                        end
                        return
                    end
                elseif action == "dump_this_dictionary"
                    open(file_name, "w") do f
                        JSON.print(f, content)
                    end
                    return
                elseif action == "extend_dictionary"
                    if isa(content, Dict)
                        merge!(data, content)
                        open(file_name, "w") do f
                            JSON.print(f, data)
                        end
                        return
                    else
                        error("DUMP TO JSON ERROR: You should provide a dictionary")
                    end
                else
                    error("DUMP TO JSON ERROR: action not supported OR key not provided")
                end

                # Write updated data
                open(file_name, "w") do f
                    JSON.print(f, data)
                end
            catch e
                @error "Error reading or writing JSON file: $e"
                rethrow(e)
            end
        else
            # The file does not exist, create a new one
            if action == "append2keyAND2list" || action == "append2keyASnewlist"
                to_write = Dict(key => content)
                open(file_name, "w") do f
                    JSON.print(f, to_write)
                end
                return
            elseif action == "dump_this_dictionary"
                open(file_name, "w") do f
                    JSON.print(f, content)
                end
                return
            else
                error("DUMP TO JSON ERROR: action not supported")
            end
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_extremities_cells(Fr_list)

        This function returns the extreme points for each of the fracture in the list.

        # Arguments
        - `Fr_list::Vector`: the fracture list

        # Returns
        - `extremities::Matrix{Int}`: the [left, right, bottom, top] extremities of each of the fracture in the list.
    """
    function get_extremities_cells(Fr_list)
        extremities = zeros(Int, length(Fr_list), 4)

        for (indx, fracture) in enumerate(Fr_list)
            # Find rightmost point
            max_intrsct1_x = argmax(fracture.Ffront[:, 1])
            max_intrsct2_x = argmax(fracture.Ffront[:, 3])
            if fracture.Ffront[max_intrsct1_x, 1] > fracture.Ffront[max_intrsct2_x, 3]
                extremities[indx, 2] = fracture.EltTip[max_intrsct1_x]
            else
                extremities[indx, 2] = fracture.EltTip[max_intrsct2_x]
            end

            # Find leftmost point
            min_intrsct1_x = argmin(fracture.Ffront[:, 1])
            min_intrsct2_x = argmin(fracture.Ffront[:, 3])
            if fracture.Ffront[min_intrsct1_x, 1] < fracture.Ffront[min_intrsct2_x, 3]
                extremities[indx, 1] = fracture.EltTip[min_intrsct1_x]
            else
                extremities[indx, 1] = fracture.EltTip[min_intrsct2_x]
            end

            # Find topmost point
            max_intrsct1_y = argmax(fracture.Ffront[:, 2])
            max_intrsct2_y = argmax(fracture.Ffront[:, 4])
            if fracture.Ffront[max_intrsct1_y, 2] > fracture.Ffront[max_intrsct2_y, 4]
                extremities[indx, 4] = fracture.EltTip[max_intrsct1_y]
            else
                extremities[indx, 4] = fracture.EltTip[max_intrsct2_y]
            end

            # Find bottommost point
            min_intrsct1_y = argmin(fracture.Ffront[:, 2])
            min_intrsct2_y = argmin(fracture.Ffront[:, 4])
            if fracture.Ffront[min_intrsct1_y, 2] < fracture.Ffront[min_intrsct2_y, 4]
                extremities[indx, 3] = fracture.EltTip[min_intrsct1_y]
            else
                extremities[indx, 3] = fracture.EltTip[min_intrsct2_y]
            end
        end

        return extremities
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_front_intercepts(fr_list, point)

        This function returns the top, bottom, left and right intercepts on the front of the horizontal and vertical lines
        drawn from the given point.

        # Arguments
        - `fr_list::Vector`: the given fracture list.
        - `point::Vector{Float64}`: the point from the horizontal and vertical lines are drawn.

        # Returns
        - `intercepts::Vector{Vector{Float64}}`: list of top, bottom, left and right intercepts for each fracture in the list
    """
    function get_front_intercepts(fr_list, point)
        logger = Logging.current_logger()
        intercepts = Vector{Vector{Float64}}()

        for fr in fr_list
            # Initialize intercepts to NaN if not available
            intrcp_top = intrcp_btm = intrcp_lft = intrcp_rgt = [NaN]  
            
            # The cell in which the given point lie
            pnt_cell = fr.mesh.locate_element(point[1], point[2])
            
            if !(pnt_cell in fr.EltChannel)
                @warn "Point is not inside fracture!"
            else
                # The y coordinate of the cell
                pnt_cell_y = fr.mesh.CenterCoor[pnt_cell, 2]
                
                # All the cells with the same y coord
                cells_x_axis = findall(x -> x == pnt_cell_y, fr.mesh.CenterCoor[:, 2])
                
                # The tip cells with the same y coord
                tipCells_x_axis = intersect(fr.EltTip, cells_x_axis)

                # The code below removes the tip cells which are directly at right and left of the cell containing the point
                # but have the front line partially passing through them. For them, the horizontal line drawn from the given
                # point will pass through the cell but not from the front line.
                if length(tipCells_x_axis) > 2
                    invalid_cell = trues(length(tipCells_x_axis))
                    for (indx, cell) in enumerate(tipCells_x_axis)
                        in_tip_cells = findall(x -> x == cell, fr.EltTip)
                        if length(in_tip_cells) > 0
                            tip_idx = in_tip_cells[1]
                            if (point[2] > fr.Ffront[tip_idx, 2] && point[2] <= fr.Ffront[tip_idx, 4]) || 
                            (point[2] < fr.Ffront[tip_idx, 2] && point[2] >= fr.Ffront[tip_idx, 4])
                                invalid_cell[indx] = false
                            end
                        end
                    end
                    # Remove invalid cells
                    valid_indices = findall(!, invalid_cell)
                    tipCells_x_axis = tipCells_x_axis[valid_indices]
                end

                # Find out the left and right cells
                lft_cell = NaN
                rgt_cell = NaN
                if length(tipCells_x_axis) == 2
                    if fr.mesh.CenterCoor[tipCells_x_axis[1], 1] < point[1]
                        lft_cell = tipCells_x_axis[1]
                        rgt_cell = tipCells_x_axis[2]
                    else
                        lft_cell = tipCells_x_axis[2]
                        rgt_cell = tipCells_x_axis[1]
                    end
                end

                pnt_cell_x = fr.mesh.CenterCoor[pnt_cell, 1]
                cells_y_axis = findall(x -> x == pnt_cell_x, fr.mesh.CenterCoor[:, 1])
                tipCells_y_axis = intersect(fr.EltTip, cells_y_axis)

                # The code below removes the tip cells which are directly at top and bottom of the cell containing the point
                # but have the front line partially passing through them. For them, the vertical line drawn from the given
                # point will pass through the cell but not from the front line.
                if length(tipCells_y_axis) > 2
                    invalid_cell = trues(length(tipCells_y_axis))
                    for (indx, cell) in enumerate(tipCells_y_axis)
                        in_tip_cells = findall(x -> x == cell, fr.EltTip)
                        if length(in_tip_cells) > 0
                            tip_idx = in_tip_cells[1]
                            if (point[1] > fr.Ffront[tip_idx, 1] && point[1] <= fr.Ffront[tip_idx, 3]) || 
                            (point[1] < fr.Ffront[tip_idx, 1] && point[1] >= fr.Ffront[tip_idx, 3])
                                invalid_cell[indx] = false
                            end
                        end
                    end
                    # Remove invalid cells
                    valid_indices = findall(!, invalid_cell)
                    tipCells_y_axis = tipCells_y_axis[valid_indices]
                end

                btm_cell = NaN
                top_cell = NaN
                if length(tipCells_y_axis) == 2
                    if fr.mesh.CenterCoor[tipCells_y_axis[1], 2] < point[2]
                        btm_cell = tipCells_y_axis[1]
                        top_cell = tipCells_y_axis[2]
                    else
                        btm_cell = tipCells_y_axis[2]
                        top_cell = tipCells_y_axis[1]
                    end
                end

                # Find indices in EltTip
                top_in_tip = findall(x -> x == top_cell, fr.EltTip)
                btm_in_tip = findall(x -> x == btm_cell, fr.EltTip)
                lft_in_tip = findall(x -> x == lft_cell, fr.EltTip)
                rgt_in_tip = findall(x -> x == rgt_cell, fr.EltTip)

                # Find the intersection using the equations of the front lines in the tip cells
                if length(top_in_tip) > 0
                    tip_idx = top_in_tip[1]
                    intrcp_top = [fr.Ffront[tip_idx, 4] + 
                            (fr.Ffront[tip_idx, 4] - fr.Ffront[tip_idx, 2]) / 
                            (fr.Ffront[tip_idx, 3] - fr.Ffront[tip_idx, 1]) * 
                            (point[1] - fr.Ffront[tip_idx, 3])]
                end

                if length(btm_in_tip) > 0
                    tip_idx = btm_in_tip[1]
                    intrcp_btm = [fr.Ffront[tip_idx, 4] + 
                            (fr.Ffront[tip_idx, 4] - fr.Ffront[tip_idx, 2]) / 
                            (fr.Ffront[tip_idx, 3] - fr.Ffront[tip_idx, 1]) * 
                            (point[1] - fr.Ffront[tip_idx, 3])]
                end

                if length(lft_in_tip) > 0
                    tip_idx = lft_in_tip[1]
                    intrcp_lft = [(point[2] - fr.Ffront[tip_idx, 4]) / 
                            (fr.Ffront[tip_idx, 4] - fr.Ffront[tip_idx, 2]) * 
                            (fr.Ffront[tip_idx, 3] - fr.Ffront[tip_idx, 1]) + 
                            fr.Ffront[tip_idx, 3]]
                end

                if length(rgt_in_tip) > 0
                    tip_idx = rgt_in_tip[1]
                    intrcp_rgt = [(point[2] - fr.Ffront[tip_idx, 4]) / 
                            (fr.Ffront[tip_idx, 4] - fr.Ffront[tip_idx, 2]) * 
                            (fr.Ffront[tip_idx, 3] - fr.Ffront[tip_idx, 1]) + 
                            fr.Ffront[tip_idx, 3]]
                end
            end

            push!(intercepts, [intrcp_top[1], intrcp_btm[1], intrcp_lft[1], intrcp_rgt[1]])
        end

        return intercepts
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        write_properties_csv_file(file_name, properties)

        This function writes the properties of a simulation as a csv file. The csv contains (in a row vector) Eprime, K1c
        , Cl, mu, rho_f, Q and t_inj

        # Arguments
        - `file_name::String`: the name of the file to be written.
        - `properties::Tuple`: the properties of the fracture loaded
    """
    function write_properties_csv_file(file_name, properties)
        # Determine the length of output list based on injectionRate structure
        if size(properties[3].injectionRate, 2) > 1
            output_list = Vector{Union{Float64, Nothing}}(nothing, 7)
        else
            output_list = Vector{Union{Float64, Nothing}}(nothing, 6)
        end

        # Fill the output list with properties
        output_list[1] = properties[1].Eprime
        output_list[2] = properties[1].K1c[1]
        output_list[3] = properties[1].Cl
        output_list[4] = properties[2].viscosity
        output_list[5] = properties[2].density

        if size(properties[3].injectionRate, 2) > 1
            output_list[6] = properties[3].injectionRate[2, 1]
            output_list[7] = properties[3].injectionRate[1, 2]
        else
            output_list[6] = properties[3].injectionRate[2, 1]
        end

        # Add .csv extension if needed
        if file_name[end-3:end] != ".csv"
            file_name = file_name * ".csv"
        end

        # Create DataFrame and write to CSV
        df = DataFrame(property_values = output_list)
        CSV.write(file_name, df)
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_geometric_parameters(fr_list)

        This function computes geometric parameters of fractures including height, maximum breadth, average breadth, and variance of breadth.

        # Arguments
        - `fr_list::Vector`: the fracture list

        # Returns
        - `Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`: 
        (height, max_breadth, avg_breadth, var_breadth)
    """
    function get_fracture_geometric_parameters(fr_list)
        max_breadth = fill(NaN, length(fr_list))
        avg_breadth = fill(NaN, length(fr_list))
        var_breadth = fill(NaN, length(fr_list))
        height = fill(NaN, length(fr_list))
        iter = 1

        for jk in fr_list
            # Get left and right front vectors
            if length(jk.source) != 0
                _, left, right = get_Ffront_as_vector(jk, jk.mesh.CenterCoor[jk.source[1], :])
            else
                _, left, right = get_Ffront_as_vector(jk, [0.0, 0.0])
            end

            # Compute breadth
            if size(left, 1) == size(right, 1)
                breadth_values = abs.(left[:, 1] - right[:, 1])
                breadth = vcat(breadth_values, left[:, 2])
            else
                min_size = min(size(left, 1), size(right, 1))
                breadth_diff = abs.(left[1:min_size, 1] - right[1:min_size, 1])
                y_coords = vcat(left[1:min_size, 2], right[1:min_size, 2])
                breadth = vcat(breadth_diff, y_coords)
            end

            # Compute statistics
            max_breadth[iter] = maximum(breadth[1:length(breadth)÷2])  # First half contains breadth values
            avg_breadth[iter] = mean(breadth[1:length(breadth)÷2])
            var_breadth[iter] = var(breadth[1:length(breadth)÷2])

            # Compute height
            all_y_coords = vcat(jk.Ffront[:, 2], jk.Ffront[:, 4])
            height[iter] = abs(maximum(all_y_coords) - minimum(all_y_coords))

            iter = iter + 1
        end

        return height, max_breadth, avg_breadth, var_breadth
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_Ffront_as_vector(frac, inj_p)

        This function processes fracture front data and returns front vectors.

        # Arguments
        - `frac`: fracture object
        - `inj_p::Vector{Float64}`: injection point coordinates [x, y]

        # Returns
        - `Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}`: (Ffront, left, right)
    """
    function get_Ffront_as_vector(frac, inj_p)
        # Create masks for different quadrants
        # Lower left quadrant
        mask13 = (frac.Ffront[:, 1] .<= inj_p[1]) .& (frac.Ffront[:, 2] .<= inj_p[2])
        mask24 = (frac.Ffront[:, 3] .<= inj_p[1]) .& (frac.Ffront[:, 4] .<= inj_p[2])
        lowLef_points1 = frac.Ffront[mask13, 1:2]
        lowLef_points2 = frac.Ffront[mask24, 3:4]
        lowLef = vcat(lowLef_points1, lowLef_points2)
        
        if size(lowLef, 1) > 0
            # Sort by y-coordinate in descending order (flip sorting)
            sort_indices = sortperm(lowLef[:, 2], rev=true)
            lowLef = lowLef[sort_indices, :]
        end

        # Lower right quadrant
        mask13 = (frac.Ffront[:, 1] .>= inj_p[1]) .& (frac.Ffront[:, 2] .<= inj_p[2])
        mask24 = (frac.Ffront[:, 3] .>= inj_p[1]) .& (frac.Ffront[:, 4] .<= inj_p[2])
        lowRig_points1 = frac.Ffront[mask13, 1:2]
        lowRig_points2 = frac.Ffront[mask24, 3:4]
        lowRig = vcat(lowRig_points1, lowRig_points2)
        
        if size(lowRig, 1) > 0
            # Sort by y-coordinate in ascending order
            sort_indices = sortperm(lowRig[:, 2])
            lowRig = lowRig[sort_indices, :]
        end

        # Upper left quadrant
        mask13 = (frac.Ffront[:, 1] .<= inj_p[1]) .& (frac.Ffront[:, 2] .>= inj_p[2])
        mask24 = (frac.Ffront[:, 3] .<= inj_p[1]) .& (frac.Ffront[:, 4] .>= inj_p[2])
        upLef_points1 = frac.Ffront[mask13, 1:2]
        upLef_points2 = frac.Ffront[mask24, 3:4]
        upLef = vcat(upLef_points1, upLef_points2)
        
        if size(upLef, 1) > 0
            # Sort by y-coordinate in descending order (flip sorting)
            sort_indices = sortperm(upLef[:, 2], rev=true)
            upLef = upLef[sort_indices, :]
        end

        # Upper right quadrant
        mask13 = (frac.Ffront[:, 1] .>= inj_p[1]) .& (frac.Ffront[:, 2] .>= inj_p[2])
        mask24 = (frac.Ffront[:, 3] .>= inj_p[1]) .& (frac.Ffront[:, 4] .>= inj_p[2])
        upRig_points1 = frac.Ffront[mask13, 1:2]
        upRig_points2 = frac.Ffront[mask24, 3:4]
        upRig = vcat(upRig_points1, upRig_points2)
        
        if size(upRig, 1) > 0
            # Sort by y-coordinate in ascending order
            sort_indices = sortperm(upRig[:, 2])
            upRig = upRig[sort_indices, :]
        end

        # Construct Ffront
        if size(lowLef, 1) > 0 && size(lowLef, 2) >= 2
            if size(lowLef, 1) > 0 && size(lowRig, 1) > 0 && size(upRig, 1) > 0 && size(upLef, 1) > 0
                Ffront = vcat(lowLef, lowRig, upRig, upLef, lowLef[1:1, :])  # Add first point to close the loop
            else
                Ffront = upRig
                if size(upLef, 1) > 0
                    Ffront = vcat(Ffront, upLef)
                end
                if size(Ffront, 1) > 0
                    Ffront = vcat(Ffront, Ffront[1:1, :])  # Add first point to close the loop
                end
            end
        else
            if size(upRig, 1) > 0 && size(upLef, 1) > 0
                Ffront = vcat(upRig, upLef, upRig[1:1, :])  # Add first point to close the loop
            else
                Ffront = Array{Float64}(undef, 0, 2)  # Empty matrix
            end
        end

        # Construct left and right boundaries
        left = vcat(lowLef, upLef)
        if size(left, 1) > 0
            # Remove duplicates and sort by y-coordinate
            left_unique = unique(left, dims=1)
            if size(left_unique, 1) > 0
                sort_indices = sortperm(left_unique[:, 2])
                left = left_unique[sort_indices, :]
            end
        else
            left = Array{Float64}(undef, 0, 2)  # Empty matrix
        end

        right = vcat(lowRig, upRig)
        if size(right, 1) > 0
            # Remove duplicates and sort by y-coordinate
            right_unique = unique(right, dims=1)
            if size(right_unique, 1) > 0
                sort_indices = sortperm(right_unique[:, 2])
                right = right_unique[sort_indices, :]
            end
        else
            right = Array{Float64}(undef, 0, 2)  # Empty matrix
        end

        return Ffront, left, right
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_fracture_fp(fr_list)

        This function returns the fracture front points for each fracture in the list.

        # Arguments
        - `fr_list::Vector`: the fracture list

        # Returns
        - `fp_list::Vector{Matrix{Float64}}`: list of fracture front points matrices
    """
    function get_fracture_fp(fr_list)
        fp_list = Matrix{Float64}[]
        iter = 1

        for jk in fr_list
            if length(jk.source) != 0
                # Get the first element of the tuple (Ffront) and append to list
                fp_list.append!(get_Ffront_as_vector(jk, jk.mesh.CenterCoor[jk.source[1], :])[1])
            else
                # Get the first element of the tuple (Ffront) and append to list
                fp_list.append!(get_Ffront_as_vector(jk, [0.0, 0.0])[1])
            end
            iter = iter + 1
        end

        return fp_list
    end

    #-----------------------------------------------------------------------------------------------------------------------
    include("elastohydrodynamic_solver.jl")
    using ElastohydrodynamicSolver: calculate_fluid_flow_characteristics_laminar

    """
        get_velocity_as_vector(Solid, Fluid, Fr_list)

        This function gets the velocity components of the fluid flux for a given list of fractures

        # Arguments
        - `Solid`: Instance of the class MaterialProperties - see related documentation
        - `Fluid`: Instance of the class FluidProperties - see related documentation
        - `Fr_list`: List of Instances of the class Fracture - see related documentation

        # Returns
        - `Tuple{Vector{Matrix{Float64}}, Vector{Float64}}`: 
        List containing a matrix with the information about the fluid velocity for each of the edges of any mesh element,
        List of time stations
    """
    function get_velocity_as_vector(Solid, Fluid, Fr_list)
        fluid_vel_list = Matrix{Float64}[]
        time_srs = Float64[]

        for i in Fr_list
            fluid_flux, fluid_vel, Rey_num, fluid_flux_components, fluid_vel_components = 
                calculate_fluid_flow_characteristics_laminar(i.w,
                                                        i.pFluid,
                                                        Solid.SigmaO,
                                                        i.mesh,
                                                        i.EltCrack,
                                                        i.InCrack,
                                                        Fluid.muPrime,
                                                        Fluid.density)
            
            # fluid_vel_components_for_one_elem = [fx left edge, fy left edge, fx right edge, fy right edge, 
            #                                    fx bottom edge, fy bottom edge, fx top edge, fy top edge]
            #
            #                 6  7
            #               (ux,uy)
            #           o---top edge---o
            #     0  1  |              |    2  3
            #   (ux,uy)left          right(ux,uy)
            #           |              |
            #           o-bottom edge--o
            #               (ux,uy)
            #                 4  5
            #
            push!(fluid_vel_list, fluid_vel_components)
            push!(time_srs, i.time)
        end

        return fluid_vel_list, time_srs
    end

    #-----------------------------------------------------------------------------------------------------------------------
    """
        get_velocity_slice(Solid, Fluid, Fr_list, initial_point, vel_direction="ux", orientation="horizontal")

        This function returns, at each time station, the velocity component in x or y direction along a horizontal or vertical section passing
        through a given point.

        WARNING: ASSUMING NO MESH COARSENING OR REMESHING WITH DOMAIN COMPRESSION

        # Arguments
        - `Solid`: Instance of the class MaterialProperties - see related documentation
        - `Fluid`: Instance of the class FluidProperties - see related documentation
        - `Fr_list`: List of Instances of the class Fracture - see related documentation
        - `initial_point`: coordinates of the point where to draw the slice
        - `vel_direction`: component of the velocity vector, it can be "ux" or "uy"
        - `orientation`: it can be "horizontal" or "vertical"

        # Returns
        - `Tuple{Vector{Vector{Float64}}, Vector{Float64}, Vector{Vector{Float64}}}`: 
        set of velocities, set of times, set of points along the slice, where the velocity is given
    """
    function get_velocity_slice(Solid, Fluid, Fr_list, initial_point, vel_direction="ux", orientation="horizontal")
        # initial_point - of the slice
        fluid_vel_list, time_srs = get_velocity_as_vector(Solid, Fluid, Fr_list)
        nOFtimes = length(time_srs)
        
        list_of_sampling_lines = Vector{Float64}[]
        list_of_fluid_vel_lists = Vector{Float64}[]

        for i in 1:nOFtimes
            # 1) get the coordinates of the points in the slices
            vector_to_be_lost = zeros(Int, Fr_list[i].mesh.NumberOfElts)
            NotUsd_var_values, sampling_line_center, sampling_cells = get_fracture_variable_slice_cell_center(vector_to_be_lost,
                                                                                                            Fr_list[i].mesh,
                                                                                                            point = initial_point,
                                                                                                            orientation = orientation)
            hx = Fr_list[i].mesh.hx  # element horizontal size
            hy = Fr_list[i].mesh.hy  # element vertical size
            
            # get the coordinates along the slice where you are getting the values
            indx1, indx2, offset = 0, 0, 0.0
            if vel_direction == "ux" && orientation == "horizontal"
                indx1, indx2, offset = 1, 3, hx * 0.5
            elseif vel_direction == "ux" && orientation == "vertical"
                indx1, indx2, offset = 5, 7, hy * 0.5
            elseif vel_direction == "uy" && orientation == "horizontal"
                indx1, indx2, offset = 2, 4, hx * 0.5
            elseif vel_direction == "uy" && orientation == "vertical"
                indx1, indx2, offset = 6, 8, hy * 0.5
            end

            # combining the two list of locations where I get the velocity
            sampling_line_center1 = sampling_line_center .- offset
            sampling_line_center2 = sampling_line_center .+ offset
            sampling_line = vcat(sampling_line_center1, sampling_line_center2)
            push!(list_of_sampling_lines, sampling_line)

            # 2) get the velocity values
            EltCrack_i = Fr_list[i].EltCrack
            fluid_vel_list_i = fluid_vel_list[i]

            vector_to_be_lost1 = zeros(Fr_list[i].mesh.NumberOfElts)
            vector_to_be_lost1[EltCrack_i] = fluid_vel_list_i[indx1, :]
            vector_to_be_lost2 = zeros(Fr_list[i].mesh.NumberOfElts)
            vector_to_be_lost2[EltCrack_i] = fluid_vel_list_i[indx2, :]

            # Create interleaved velocity list using reshape and permutedims
            vel1_sampled = vector_to_be_lost1[sampling_cells]
            vel2_sampled = vector_to_be_lost2[sampling_cells]
            
            # Interleave the two arrays
            n = length(vel1_sampled)
            fluid_vel_list_final_i = Vector{Float64}(undef, 2n)
            fluid_vel_list_final_i[1:2:end] = vel1_sampled
            fluid_vel_list_final_i[2:2:end] = vel2_sampled
            
            push!(list_of_fluid_vel_lists, fluid_vel_list_final_i)
        end

        return list_of_fluid_vel_lists, time_srs, list_of_sampling_lines
    end
end # module PostprocessFracture