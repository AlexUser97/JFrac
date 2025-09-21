# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac elasticity module on Julia language.
"""

module Visualization

    include("postprocess_fracture.jl")
    include("properties.jl")
    include("labels.jl")
    using .PostprocessFracture: get_fracture_variable, get_fracture_variable_at_point, get_HF_analytical_solution, get_HF_analytical_solution_at_point 
    using .Properties: PlotProperties, LabelProperties, MaterialProperties, FluidProperties
    using .Labels: supported_variables, supported_projections, unidimensional_variables, suitable_elements

    using PyPlot
    using Logging
    using OpenCV

    export plot_fracture_list, plot_fracture_list_slice, plot_fracture_list_at_point, plot_fracture_variable_as_vector,
    plot_variable_vs_time, plot_fracture_variable_as_image, plot_fracture_variable_as_surface, plot_fracture_surface,
    plot_fracture_variable_as_contours, plot_fracture_slice_interpolated, plot_fracture_slice_cell_center, plot_analytical_solution_slice,
    plot_analytical_solution_at_point, plot_scale_3D, plot_slice_3D, plot_footprint_analytical, plot_analytical_solution, get_HF_analytical_solution_footprint,
    plot_injection_source, animate_simulation_results, text3d, zoom_factory, to_precision, save_images_to_video, remove_zeros, get_elements, plot_regime,
    mkmtTriangle, fill_mkmtTriangle, plot_points_to_mkmtTriangle


    """
        plot_fracture_list(fracture_list, variable="footprint", projection=nothing, elements=nothing, plot_prop=nothing,
                        fig=nothing, edge=4, contours_at=nothing, labels=nothing, mat_properties=nothing,
                        backGround_param=nothing, plot_non_zero=true, source_loc=[0.0, 0.0])

        This function plots the fracture evolution with time. The state of the fracture at different times is provided in
        the form of a list of Fracture objects.

        # Arguments
        - `fracture_list::Vector{Fracture}`: the list of Fracture objects giving the evolution of fracture with time.
        - `variable::String="footprint"`: the variable to be plotted.
        - `projection::Union{String, Nothing}=nothing`: a string specifying the projection.
        - `elements::Union{Vector{Int}, Nothing}=nothing`: the elements to be plotted.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `edge::Int=4`: the edge of the cell that will be plotted.
        - `contours_at::Union{Vector, Nothing}=nothing`: the values at which the contours are to be plotted.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `mat_properties::Union{MaterialProperties, Nothing}=nothing`: the material properties.
        - `backGround_param::Union{String, Nothing}=nothing`: the parameter according to which the mesh will be colormapped.
        - `plot_non_zero::Bool=true`: if true, only non-zero values will be plotted.
        - `source_loc::Vector{Float64}=[0.0, 0.0]`: source location.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_list(fracture_list::Vector, variable::String="footprint", projection::Union{String, Nothing}=nothing, 
                            elements::Union{Vector{Int}, Nothing}=nothing, plot_prop::Union{PlotProperties, Nothing}=nothing,
                            fig::Union{PyPlot.Figure, Nothing}=nothing, edge::Int=4, contours_at::Union{Vector, Nothing}=nothing,
                            labels::Union{LabelProperties, Nothing}=nothing, mat_properties::Union{MaterialProperties, Nothing}=nothing,
                            backGround_param::Union{String, Nothing}=nothing, plot_non_zero::Bool=true, 
                            source_loc::Vector{Float64}=[0.0, 0.0])

        @info "Plotting $variable..."
        
        if !isa(fracture_list, Vector)
            throw(ArgumentError("The provided fracture_list is not Vector type object!"))
        end
        
        if length(fracture_list) == 0
            throw(ArgumentError("Provided fracture list is empty!"))
        end
        
        if !(variable in supported_variables)
            throw(ArgumentError("Variable $variable is not supported"))
        end
        
        if projection === nothing
            projection = supported_projections[variable][1]
        elseif !(projection in supported_projections[variable])
            throw(ArgumentError("The given projection is not supported for '$variable'. Select one of the following: $(supported_projections[variable])"))
        end
        
        if plot_prop === nothing
            plot_prop = PlotProperties()
        end
        
        if labels === nothing
            labels = LabelProperties(variable, "whole mesh", projection)
        end
        
        max_Lx = 0.0
        max_Ly = 0.0
        largest_mesh = nothing
        
        for fracture in fracture_list
            if fracture.mesh.Lx > max_Lx
                largest_mesh = fracture.mesh
                max_Lx = fracture.mesh.Lx
            end
            if fracture.mesh.Ly > max_Ly
                largest_mesh = fracture.mesh
                max_Ly = fracture.mesh.Ly
            end
        end
        
        # Initialize figure if not provided
        if fig === nothing
            fig = PyPlot.figure()
        end
        
        if variable == "mesh"
            if backGround_param !== nothing && mat_properties === nothing
                throw(ArgumentError("Material properties are required to color code background"))
            end
            if projection == "2D"
                fig = largest_mesh.plot(fig=fig, material_prop=mat_properties, 
                                    backGround_param=backGround_param, plot_prop=plot_prop)
            else
                fig = largest_mesh.plot_3D(fig=fig, material_prop=mat_properties,
                                        backGround_param=backGround_param, plot_prop=plot_prop)
            end
            
        elseif variable == "footprint"
            if projection == "2D"
                for fracture in fracture_list
                    fig = fracture.plot_front(fig=fig, plot_prop=plot_prop)
                end
            else
                for fracture in fracture_list
                    fig = fracture.plot_front_3D(fig=fig, plot_prop=plot_prop)
                end
            end
            
        elseif variable in ["source elements", "se"]
            for fracture in fracture_list
                fig = plot_injection_source(fracture, fig=fig, plot_prop=plot_prop)
            end
            
        else
            var_val_list = nothing
            time_list = nothing
            var_val_copy = nothing
            
            if variable == "chi"
                vel_list, time_list = get_fracture_variable(fracture_list, "v", edge=edge, return_time=true)
                var_val_list = []
                for i in vel_list
                    actual_ki = 2 * mat_properties.Cprime * mat_properties.Eprime ./ 
                                (sqrt.(i) * mat_properties.Kprime)
                    push!(var_val_list, actual_ki)
                end
                
            elseif variable == "regime"
                var_val_list, legend_coord, time_list = get_fracture_variable(fracture_list, variable, 
                                                                            edge=edge, return_time=true)
                
            else
                var_val_list, time_list = get_fracture_variable(fracture_list, variable, 
                                                            edge=edge, return_time=true)
            end
            
            if var_val_list !== nothing
                var_val_copy = deepcopy(var_val_list)
                for i in 1:length(var_val_copy)
                    var_val_copy[i] = var_val_copy[i] / labels.unitConversion
                end
            end
            
            vmin, vmax = Inf, -Inf
            if projection != "2D_vectorfield" && var_val_copy !== nothing
                var_value_tmp = deepcopy(var_val_copy)
                if elements !== nothing
                    var_value_tmp = [v[elements] for v in var_value_tmp]
                end
                if plot_non_zero
                    # Flatten and filter non-zero values for vmin/vmax calculation
                    all_values = Float64[]
                    for arr in var_value_tmp
                        if isa(arr, Vector)
                            filtered = arr[abs.(arr) .> 1e-16]
                            append!(all_values, filtered[isfinite.(filtered)])
                        end
                    end
                    if length(all_values) > 0
                        if variable in ("p", "pressure")
                            non_zero = abs.(all_values) .> 0
                            if any(non_zero)
                                med_val = median(all_values[non_zero])
                                vmin, vmax = -0.2 * med_val, 1.5 * med_val
                            end
                        else
                            vmin, vmax = minimum(all_values), maximum(all_values)
                        end
                    end
                end
            end
            
            if variable == "regime" && var_val_list !== nothing
                for i in 1:length(var_val_list)
                    fig = plot_regime(var_val_copy[i], fracture_list[i].mesh, 
                                    elements=fracture_list[i].EltRibbon, fig=fig)
                end
                
            elseif variable in unidimensional_variables && var_val_list !== nothing
                fig = plot_variable_vs_time(time_list, var_val_list, fig=fig, 
                                        plot_prop=plot_prop, label=labels.legend)
                
            elseif variable in supported_variables && variable ∉ ["mesh", "footprint", "se", "source elements"] && var_val_list !== nothing
                if projection != "2D_vectorfield"
                    if plot_non_zero
                        for indx in 1:length(var_val_copy)
                            # Remove zeros by setting them to NaN
                            vals = var_val_copy[indx]
                            if isa(vals, Vector)
                                zero_indices = abs.(vals) .< 1e-16
                                vals[zero_indices] .= NaN
                                var_val_copy[indx] = vals
                            end
                        end
                    end
                end
                
                if variable == "surface"
                    plot_prop.colorMap = "cool"
                    for i in 1:length(var_val_list)
                        fig = plot_fracture_surface(var_val_copy[i], fracture_list[i].mesh, fig=fig,
                                                plot_prop=plot_prop, plot_colorbar=false, elements=elements,
                                                vmin=vmin, vmax=vmax)
                    end
                    
                elseif projection == "2D_clrmap"
                    for i in 1:length(var_val_list)
                        fig = plot_fracture_variable_as_image(var_val_copy[i], fracture_list[i].mesh, fig=fig,
                                                            plot_prop=plot_prop, elements=elements,
                                                            plt_colorbar=false, vmin=vmin, vmax=vmax)
                    end
                    
                elseif projection == "2D_contours"
                    for i in 1:length(var_val_list)
                        labels.legend = "t= " * to_precision(time_list[i], plot_prop.dispPrecision)
                        plot_prop.lineColor = plot_prop.colorsList[(i-1) % length(plot_prop.colorsList) + 1]
                        fig = plot_fracture_variable_as_contours(var_val_copy[i], fracture_list[i].mesh, fig=fig,
                                                            plot_prop=plot_prop, contours_at=contours_at,
                                                            plt_colorbar=false, vmin=vmin, vmax=vmax)
                    end
                elseif projection == "3D"
                    for i in 1:length(var_val_list)
                        fig = plot_fracture_variable_as_surface(var_val_copy[i], fracture_list[i].mesh, fig=fig,
                                                            plot_prop=plot_prop, plot_colorbar=false,
                                                            elements=elements, vmin=vmin, vmax=vmax)
                    end
                elseif projection == "2D_vectorfield"
                    for i in 1:length(var_val_list)
                        if !any(isnan.(var_val_copy[i]))
                            # fracture_list[i].EltCrack => ribbon+tip+other in crack
                            # fracture_list[i].EltChannel => ribbon+other in crack
                            elements_where_to_plot = fracture_list[i].EltChannel
                            # Alternative options:
                            # elements_where_to_plot = setdiff(fracture_list[i].EltChannel, fracture_list[i].EltRibbon)
                            # elements_where_to_plot = setdiff(elements_where_to_plot, unique(vcat(fracture_list[i].mesh.NeiElements[fracture_list[i].EltRibbon]...)))
                            fig = plot_fracture_variable_as_vector(var_val_copy[i], fracture_list[i].mesh,
                                                                elements_where_to_plot, fig=fig)
                        end
                    end
                end
            end
        end

        # Set labels
        ax = fig.get_axes()[1]
        ax.set_xlabel(labels.xLabel)
        ax.set_ylabel(labels.yLabel)
        ax.set_title(labels.figLabel)

    if projection == "3D" && variable ∉ ["mesh", "footprint", "se", "source elements"]
            ax.set_zlabel(labels.zLabel)
            sm = PyPlot.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                        norm=PyPlot.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cb = PyPlot.colorbar(sm, alpha=plot_prop.alpha)
            cb.set_label(labels.colorbarLabel)
            
        elseif projection in ("2D_clrmap", "2D_contours") && variable != "regime"
            im = ax.images
            if length(im) > 0
                cb = PyPlot.colorbar(im[end], ax=ax, pad=0.05)
                cb.set_label(labels.colorbarLabel)
            end
        end
        
        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_list_slice(fracture_list, variable="width", point1=nothing, point2=nothing, projection="2D", plot_prop=nothing,
                                fig=nothing, edge=4, labels=nothing, plot_cell_center=false, orientation="horizontal",
                                extreme_points=nothing, export2Json=false, export2Json_assuming_no_remeshing=true)

        This function plots the fracture evolution on a given slice of the domain. Two points are to be given that will be
        joined to form the slice. The values on the slice are either interpolated from the values available on the cell
        centers. Exact values on the cell centers can also be plotted.

        # Arguments
        - `fracture_list::Vector{Fracture}`: the list of Fracture objects giving the evolution of fracture with time.
        - `variable::String="width"`: the variable to be plotted.
        - `point1::Union{Vector{Float64}, Nothing}=nothing`: the left point from which the slice should pass [x, y].
        - `point2::Union{Vector{Float64}, Nothing}=nothing`: the right point from which the slice should pass [x, y].
        - `projection::String="2D"`: a string specifying the projection. It can either '3D' or '2D'.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `edge::Int=4`: the edge of the cell that will be plotted.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `plot_cell_center::Bool=false`: if True, the discrete values at the cell centers will be plotted.
        - `orientation::String="horizontal"`: the orientation according to which the slice is made.
        - `extreme_points::Union{Matrix{Float64}, Nothing}=nothing`: An empty array of shape (2, 2).
        - `export2Json::Bool=false`: If you set it to True the function will return a dictionary with the data.
        - `export2Json_assuming_no_remeshing::Bool=true`: Assuming no remeshing for export.

        # Returns
        - `Union{PyPlot.Figure, Dict}`: A Figure object or dictionary with plot data.
    """

    function plot_fracture_list_slice(fracture_list::Vector, variable::String="width", point1::Union{Vector{Float64}, Nothing}=nothing, 
                                    point2::Union{Vector{Float64}, Nothing}=nothing, projection::String="2D", 
                                    plot_prop::Union{PlotProperties, Nothing}=nothing,
                                    fig::Union{PyPlot.Figure, Nothing}=nothing, edge::Int=4, 
                                    labels::Union{LabelProperties, Nothing}=nothing, plot_cell_center::Bool=false, 
                                    orientation::String="horizontal", extreme_points::Union{Matrix{Float64}, Nothing}=nothing, 
                                    export2Json::Bool=false, export2Json_assuming_no_remeshing::Bool=true)
        
        if !(variable in supported_variables)
            throw(ArgumentError("Variable $variable is not supported"))
        end

        if variable in unidimensional_variables
            throw(ArgumentError("The given variable does not vary spatially."))
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
            if plot_cell_center
                plot_prop.lineStyle = "."
            end
        end

        if labels === nothing
            labels = LabelProperties(variable, "slice", projection)
        end

        mesh_list = get_fracture_variable(fracture_list, "mesh", edge=edge, return_time=false)
        var_val_list, time_list = get_fracture_variable(fracture_list, variable, edge=edge, return_time=true)

        var_val_copy = deepcopy(var_val_list)
        for i in 1:length(var_val_copy)
            var_val_copy[i] = var_val_copy[i] / labels.unitConversion
        end

        # find maximum and minimum to set the viewing limits on axis
        vmin, vmax = Inf, -Inf
        for i in var_val_copy
            # Remove inf and nan values
            finite_vals = i[isfinite.(i)]
            if length(finite_vals) > 0
                if variable in ("p", "pressure")
                    non_zero = abs.(finite_vals) .> 0
                    if any(non_zero)
                        med_val = median(finite_vals[non_zero])
                        i_min, i_max = -0.2 * med_val, 1.5 * med_val
                    else
                        i_min, i_max = Inf, -Inf
                    end
                else
                    i_min, i_max = minimum(finite_vals), maximum(finite_vals)
                end
                vmin, vmax = min(vmin, i_min), max(vmax, i_max)
            end
        end

        label = labels.legend

        if export2Json
            to_write = Dict{String, Any}(
                "size_of_data" => length(time_list),
                "time_list" => time_list
            )
        end

        for i in 1:length(var_val_list)
            labels.legend = label * " t= " * to_precision(time_list[i], plot_prop.dispPrecision)
            plot_prop.lineColor = plot_prop.colorsList[((i-1) % length(plot_prop.colorsList)) + 1]
            
            if occursin("2D", projection)
                if plot_cell_center
                    result = plot_fracture_slice_cell_center(var_val_copy[i], mesh_list[i], point=point1,
                                                        orientation=orientation, fig=fig, plot_prop=plot_prop,
                                                        vmin=vmin, vmax=vmax, plot_colorbar=false, labels=labels,
                                                        extreme_points=extreme_points, export2Json=export2Json)
                    
                    if length(result) == 4
                        fig, sampling_line_out, var_value_selected, sampling_cells = result
                    else
                        fig = result[1]
                        sampling_line_out = result[2]
                        var_value_selected = result[3]
                        sampling_cells = result[4]
                    end
                    
                    if i == 1 && export2Json && export2Json_assuming_no_remeshing
                        to_write[variable * "_sampling_coords_"] = collect(sampling_line_out)
                        to_write[variable * "_sampling_cells"] = collect(sampling_cells)
                    end
                    if export2Json && !export2Json_assuming_no_remeshing
                        to_write[variable * "_sampling_coords_" * string(i-1)] = collect(sampling_line_out)
                        to_write[variable * "_sampling_cells_" * string(i-1)] = collect(sampling_cells)
                        to_write[variable * "_" * string(i-1)] = collect(var_value_selected)
                    end
                    if export2Json && export2Json_assuming_no_remeshing
                        to_write[string(i-1)] = collect(var_value_selected)
                    end
                else
                    fig = plot_fracture_slice_interpolated(var_val_copy[i], mesh_list[i], point1=point1, point2=point2,
                                                        fig=fig, plot_prop=plot_prop, vmin=vmin, vmax=vmax,
                                                        plot_colorbar=false, labels=labels, export2Json=export2Json)
                end
                
                if !export2Json
                    ax_tv = fig.get_axes()[1]
                    ax_tv.set_xlabel("meter")
                    ax_tv.set_ylabel("meter")
                    ax_tv.set_title("Top View")

                    # making colorbar
                    im = ax_tv.images
                    if length(im) > 0
                        cb = PyPlot.colorbar(im[end], ax=ax_tv, pad=0.05)
                        cb.set_label(labels.colorbarLabel)
                    end

                    if length(fig.get_axes()) > 1
                        ax_slice = fig.get_axes()[2]
                        ax_slice.set_ylabel(labels.colorbarLabel)
                        ax_slice.set_xlabel("(x,y) " * labels.xLabel)
                    end
                end

            elseif projection == "3D" && !export2Json
                fig = plot_slice_3D(var_val_copy[i], mesh_list[i], point1=point1, point2=point2, fig=fig,
                                plot_prop=plot_prop, vmin=vmin, vmax=vmax, label=labels.legend)
                ax_slice = fig.get_axes()[1]
                ax_slice.set_xlabel("meter")
                ax_slice.set_ylabel("meter")
                ax_slice.set_zlabel(labels.zLabel)
                ax_slice.title(labels.figLabel)
            else
                throw(ArgumentError("Given Projection is not correct!"))
            end
        end

        if plot_prop.plotLegend && !export2Json && length(fig.get_axes()) > 0
            ax_slice = fig.get_axes()[end]
            ax_slice.legend()
        end

        if export2Json
            return to_write
        else
            return fig
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_list_at_point(fracture_list, variable="width", point=nothing, plot_prop=nothing, fig=nothing,
                                    edge=4, labels=nothing)

        This function plots the fracture evolution on a given point.

        # Arguments
        - `fracture_list::Vector{Fracture}`: the list of Fracture objects giving the evolution of fracture with time.
        - `variable::String="width"`: the variable to be plotted.
        - `point::Union{Vector{Float64}, Nothing}=nothing`: the point at which the given variable is plotted against time [x, y].
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `edge::Int=4`: the edge of the cell that will be plotted.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_list_at_point(fracture_list::Vector, variable::String="width", point::Union{Vector{Float64}, Nothing}=nothing, 
                                        plot_prop::Union{PlotProperties, Nothing}=nothing, fig::Union{PyPlot.Figure, Nothing}=nothing,
                                        edge::Int=4, labels::Union{LabelProperties, Nothing}=nothing)
        
        if !(variable in supported_variables)
            throw(ArgumentError("Variable $variable is not supported"))
        end

        if variable in unidimensional_variables
            throw(ArgumentError("The given variable does not vary spatially."))
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if labels === nothing
            labels = LabelProperties(variable, "point", "2D")
        end

        if point === nothing
            point = [0.0, 0.0]
        end

        point_values, time_list = get_fracture_variable_at_point(fracture_list, variable, point=point, edge=edge)

        point_values = point_values / labels.unitConversion

        fig = plot_variable_vs_time(time_list, point_values, fig=fig, plot_prop=plot_prop, label=labels.legend)

        ax = fig.get_axes()[1]
        ax.set_xlabel("time (\$s\$)")
        ax.set_ylabel(labels.colorbarLabel)
        ax.set_title(labels.figLabel)
        if plot_prop.plotLegend
            ax.legend()
        end

        plot_prop_fp = PlotProperties(line_color="k")
        labels_fp = LabelProperties("footprint", "whole mesh", "2D")
        labels_fp.figLabel = ""
        fig_image = plot_fracture_list([fracture_list[end]], variable="footprint", projection="2D",
                                    plot_prop=plot_prop_fp, labels=labels_fp)

        labels_2D = LabelProperties(variable, "whole mesh", "2D_clrmap")
        labels_2D.figLabel = "Sampling Point"
        fig_image = plot_fracture_list([fracture_list[end]], variable=variable, projection="2D_clrmap",
                                    fig=fig_image, plot_prop=plot_prop, edge=edge, labels=labels_2D)

        ax_image = fig_image.get_axes()[1]
        ax_image.plot([point[1]], [point[2]], "ko")

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_variable_as_vector(var_value, mesh, Elements_to_plot, fig=nothing)

        This function plots a given 2D vector field.

        # Arguments
        - `var_value`: an array with each column having the following information:
                    [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge,
                        fy bottom edge, fx top edge, fy top edge]
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `Elements_to_plot`: list of cell names on whose edges plot the vectors.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_variable_as_vector(var_value, mesh, Elements_to_plot, fig::Union{PyPlot.Figure, Nothing}=nothing)
        
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(111)
        else
            ax = fig.get_axes()[1]
        end

        # Create U vector (x-components)
        U = vcat(var_value[1, Elements_to_plot], var_value[3, Elements_to_plot])
        U = vcat(U, var_value[5, Elements_to_plot])
        U = vcat(U, var_value[7, Elements_to_plot])
        U = vec(U)  # Equivalent to np.ndarray.flatten

        # Create V vector (y-components)
        V = vcat(var_value[2, Elements_to_plot], var_value[4, Elements_to_plot])
        V = vcat(V, var_value[6, Elements_to_plot])
        V = vcat(V, var_value[8, Elements_to_plot])
        V = vec(V)  # Equivalent to np.ndarray.flatten

        # Create X coordinates
        X = vcat(mesh.CenterCoor[Elements_to_plot, 1] .- mesh.hx * 0.5, 
                mesh.CenterCoor[Elements_to_plot, 1] .+ mesh.hx * 0.5)
        X = vcat(X, mesh.CenterCoor[Elements_to_plot, 1])
        X = vcat(X, mesh.CenterCoor[Elements_to_plot, 1])
        X = vec(X)

        # Create Y coordinates
        Y = vcat(mesh.CenterCoor[Elements_to_plot, 2], 
                mesh.CenterCoor[Elements_to_plot, 2])
        Y = vcat(Y, mesh.CenterCoor[Elements_to_plot, 2] .- mesh.hy * 0.5)
        Y = vcat(Y, mesh.CenterCoor[Elements_to_plot, 2] .+ mesh.hy * 0.5)
        Y = vec(Y)

        # Calculate magnitude
        M = hypot.(U, V)  # Element-wise hypotenuse

        ax.quiver(X, Y, U, V, M, pivot="mid")

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_variable_vs_time(time_list, value_list, fig=nothing, plot_prop=nothing, label=nothing)

        This function plots a given list of values against time.

        # Arguments
        - `time_list::Union{Vector, Array}`: the list of times.
        - `value_list::Union{Vector, Array}`: the list of values.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `label::Union{String, Nothing}=nothing`: the label given to the plot line.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_variable_vs_time(time_list, value_list, fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                plot_prop::Union{PlotProperties, Nothing}=nothing, label::Union{String, Nothing}=nothing)
        
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(111)
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if plot_prop.plotLegend && label !== nothing
            label_copy = label
        else
            label_copy = nothing
        end

        if plot_prop.graphScaling == "linear"
            ax.plot(time_list, value_list, plot_prop.lineStyle, color=plot_prop.lineColor, label=label_copy)

        elseif plot_prop.graphScaling == "loglog"
            ax.loglog(time_list, value_list, plot_prop.lineStyle, color=plot_prop.lineColor, label=label_copy)

        elseif plot_prop.graphScaling == "semilogx"
            ax.semilogx(time_list, value_list, plot_prop.lineStyle, color=plot_prop.lineColor, label=label_copy)

        elseif plot_prop.graphScaling == "semilogy"
            ax.semilogy(time_list, value_list, plot_prop.lineStyle, color=plot_prop.lineColor, label=label_copy)
            
        else
            throw(ArgumentError("Graph scaling type $(plot_prop.graphScaling) not supported"))
        end

        if plot_prop.plotLegend
            ax.legend()
        end

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_variable_as_image(var_value, mesh, fig=nothing, plot_prop=nothing, elements=nothing, vmin=nothing,
                                        vmax=nothing, plt_colorbar=true)

        This function plots the 2D fracture variable in the form of a colormap.

        # Arguments
        - `var_value::Array`: a ndarray of the length of the number of cells in the mesh.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `elements::Union{Vector{Int}, Nothing}=nothing`: the elements to be plotted.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.
        - `plt_colorbar::Bool=true`: if True, colorbar will be plotted.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_variable_as_image(var_value, mesh, fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                            plot_prop::Union{PlotProperties, Nothing}=nothing, 
                                            elements::Union{Vector{Int}, Nothing}=nothing, 
                                            vmin::Union{Float64, Nothing}=nothing,
                                            vmax::Union{Float64, Nothing}=nothing, 
                                            plt_colorbar::Bool=true)
        
        if elements !== nothing
            var_value_fullMesh = fill(NaN, mesh.NumberOfElts)
            var_value_fullMesh[elements] .= var_value[elements]
            var_value = var_value_fullMesh
        end

        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(111)
        else
            ax = fig.get_axes()[1]
        end

        x = reshape(mesh.CenterCoor[:, 1], (mesh.ny, mesh.nx))
        y = reshape(mesh.CenterCoor[:, 2], (mesh.ny, mesh.nx))

        var_value_2D = reshape(var_value, (mesh.ny, mesh.nx))

        dx = (x[1, 2] - x[1, 1]) / 2.0
        dy = (y[2, 1] - y[1, 1]) / 2.0
        extent = [x[1, 1] - dx, x[end, end] + dx, y[1, 1] - dy, y[end, end] + dy]

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if vmin === nothing && vmax === nothing
            # Remove inf and nan values
            finite_vals = var_value[isfinite.(var_value)]
            if length(finite_vals) > 0
                vmin, vmax = minimum(finite_vals), maximum(finite_vals)
            else
                vmin, vmax = 0.0, 1.0  # Default values if no finite values
            end
        end

        cax = ax.imshow(var_value_2D, cmap=plot_prop.colorMap, interpolation=plot_prop.interpolation,
                        extent=extent, alpha=0.8, vmin=vmin, vmax=vmax, origin="lower")

        if plt_colorbar
            PyPlot.colorbar(cax)
        end

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_variable_as_surface(var_value, mesh, fig=nothing, plot_prop=nothing, plot_colorbar=true, elements=nothing,
                                        vmin=nothing, vmax=nothing)

        This function plots the 2D fracture variable in the form of a surface.

        # Arguments
        - `var_value::Array`: a ndarray of the length of the number of cells in the mesh.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `plot_colorbar::Bool=true`: if True, colorbar will be plotted.
        - `elements::Union{Vector{Int}, Nothing}=nothing`: the elements to be plotted.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_variable_as_surface(var_value, mesh, fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                            plot_prop::Union{PlotProperties, Nothing}=nothing, 
                                            plot_colorbar::Bool=true, elements::Union{Vector{Int}, Nothing}=nothing,
                                            vmin::Union{Float64, Nothing}=nothing, vmax::Union{Float64, Nothing}=nothing)
        
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.gca(projection="3d")
            # Note: zoom_factory is not implemented here - you may need to add it separately
            # scale = 1.1
            # zoom_factory(ax, base_scale=scale)
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if elements === nothing
            elements = collect(1:mesh.NumberOfElts)
        end

        if vmin === nothing && vmax === nothing
            # Remove inf and nan values
            finite_vals = var_value[isfinite.(var_value)]
            if length(finite_vals) > 0
                vmin, vmax = minimum(finite_vals), maximum(finite_vals)
            else
                vmin, vmax = 0.0, 1.0  # Default values if no finite values
            end
        end

        ax.plot_trisurf(mesh.CenterCoor[elements, 1],  # x coordinates (1-indexed)
                        mesh.CenterCoor[elements, 2],  # y coordinates (1-indexed)
                        var_value[elements],           # z values
                        cmap=plot_prop.colorMap,
                        linewidth=plot_prop.lineWidth,
                        alpha=plot_prop.alpha,
                        vmin=vmin,
                        vmax=vmax)

        if plot_colorbar
            sm = PyPlot.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                        norm=PyPlot.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            PyPlot.colorbar(sm, alpha=plot_prop.alpha)
        end

        ax.set_zlim(vmin, vmax)

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_surface(width, mesh, fig=nothing, plot_prop=nothing, plot_colorbar=true, elements=nothing,
                            vmin=nothing, vmax=nothing)

        This function plots the 2D fracture variable in the form of a surface.

        # Arguments
        - `width::Array`: the fracture width.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `plot_colorbar::Bool=true`: if True, colorbar will be plotted.
        - `elements::Union{Vector{Int}, Nothing}=nothing`: the elements to be plotted.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_surface(width, mesh, fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                plot_prop::Union{PlotProperties, Nothing}=nothing, 
                                plot_colorbar::Bool=true, elements::Union{Vector{Int}, Nothing}=nothing,
                                vmin::Union{Float64, Nothing}=nothing, vmax::Union{Float64, Nothing}=nothing)
        
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.gca(projection="3d")
            # Note: zoom_factory is not implemented here - you may need to add it separately
            # scale = 1.1
            # zoom_factory(ax, base_scale=scale)
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if elements === nothing
            elements = collect(1:mesh.NumberOfElts)
        end

        if vmin === nothing && vmax === nothing
            # Remove inf and nan values
            finite_vals = width[isfinite.(width)]
            if length(finite_vals) > 0
                vmin, vmax = minimum(finite_vals), maximum(finite_vals)
            else
                vmin, vmax = 0.0, 1.0  # Default values if no finite values
            end
        end

        # Plot upper surface (width/2)
        ax.plot_trisurf(mesh.CenterCoor[elements, 1],     # x coordinates (1-indexed)
                        mesh.CenterCoor[elements, 2],     # y coordinates (1-indexed)
                        width[elements] / 2,              # upper surface
                        cmap=plot_prop.colorMap,
                        linewidth=plot_prop.lineWidth,
                        alpha=plot_prop.alpha,
                        vmin=vmin,
                        vmax=vmax)

        # Plot lower surface (-width/2)
        ax.plot_trisurf(mesh.CenterCoor[elements, 1],     # x coordinates (1-indexed)
                        mesh.CenterCoor[elements, 2],     # y coordinates (1-indexed)
                        -width[elements] / 2,             # lower surface
                        cmap=plot_prop.colorMap,
                        linewidth=plot_prop.lineWidth,
                        alpha=plot_prop.alpha,
                        vmin=vmin,
                        vmax=vmax)

        if plot_colorbar
            sm = PyPlot.cm.ScalarMappable(cmap=plot_prop.colorMap,
                                        norm=PyPlot.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            PyPlot.colorbar(sm, alpha=plot_prop.alpha)
        end

        ax.set_zlim(vmin, vmax)

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_variable_as_contours(var_value, mesh, fig=nothing, plot_prop=nothing, plt_backGround=true,
                                        plt_colorbar=true, contours_at=nothing, vmin=nothing, vmax=nothing)

        This function plots the contours of the 2D fracture variable.

        # Arguments
        - `var_value::Array`: a ndarray of the length of the number of cells in the mesh.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `plt_backGround::Bool=true`: if True, the colormap of the variable will also be plotted.
        - `plt_colorbar::Bool=true`: if True, colorbar will be plotted.
        - `contours_at::Union{Vector, Nothing}=nothing`: the values at which the contours are to be plotted.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_variable_as_contours(var_value, mesh, fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                            plot_prop::Union{PlotProperties, Nothing}=nothing, 
                                            plt_backGround::Bool=true, plt_colorbar::Bool=true, 
                                            contours_at::Union{Vector, Nothing}=nothing,
                                            vmin::Union{Float64, Nothing}=nothing, 
                                            vmax::Union{Float64, Nothing}=nothing)
        
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(111)
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        x = reshape(mesh.CenterCoor[:, 1], (mesh.ny, mesh.nx))
        y = reshape(mesh.CenterCoor[:, 2], (mesh.ny, mesh.nx))

        var_value_2D = reshape(var_value, (mesh.ny, mesh.nx))

        dx = (x[1, 2] - x[1, 1]) / 2.0
        dy = (y[2, 1] - y[1, 1]) / 2.0
        extent = [x[1, 1] - dx, x[end, end] + dx, y[1, 1] - dy, y[end, end] + dy]

        if vmin === nothing && vmax === nothing
            # Remove inf and nan values
            finite_vals = var_value[isfinite.(var_value)]
            if length(finite_vals) > 0
                vmin, vmax = minimum(finite_vals), maximum(finite_vals)
            else
                vmin, vmax = 0.0, 1.0  # Default values if no finite values
            end
        end

        if plt_backGround
            cax = ax.imshow(var_value_2D, cmap=plot_prop.colorMap, interpolation=plot_prop.interpolation,
                            extent=extent, vmin=vmin, vmax=vmax, origin="lower")

            if plt_colorbar
                fig.colorbar(cax)
            end
        end

        if contours_at === nothing
            contours_at = vmin .+ (vmax - vmin) * [0.01, 0.3, 0.5, 0.7, 0.9]
        end

        CS = ax.contour(x, y, var_value_2D, contours_at, colors=plot_prop.lineColor)

        PyPlot.clabel(CS)

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_slice_interpolated(var_value, mesh, point1=nothing, point2=nothing, fig=nothing, plot_prop=nothing, vmin=nothing,
                                        vmax=nothing, plot_colorbar=true, labels=nothing, plt_2D_image=true, export2Json=false)

        This function plots the fracture on a given slice of the domain. Two points are to be given that will be
        joined to form the slice. The values on the slice are interpolated from the values available on the cell
        centers.

        # Arguments
        - `var_value::Array`: a ndarray with the length of the number of cells in the mesh.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `point1::Union{Vector{Float64}, Nothing}=nothing`: the left point from which the slice should pass [x, y].
        - `point2::Union{Vector{Float64}, Nothing}=nothing`: the right point from which the slice should pass [x, y].
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.
        - `plot_colorbar::Bool=true`: if True, colorbar will be plotted.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `plt_2D_image::Bool=true`: if True, a subplot showing the colormap and the slice will also be plotted.
        - `export2Json::Bool=false`: if True, suppress logging.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_fracture_slice_interpolated(var_value, mesh, point1::Union{Vector{Float64}, Nothing}=nothing, 
                                            point2::Union{Vector{Float64}, Nothing}=nothing, 
                                            fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                            plot_prop::Union{PlotProperties, Nothing}=nothing,
                                            vmin::Union{Float64, Nothing}=nothing, 
                                            vmax::Union{Float64, Nothing}=nothing, 
                                            plot_colorbar::Bool=true, 
                                            labels::Union{LabelProperties, Nothing}=nothing, 
                                            plt_2D_image::Bool=true, 
                                            export2Json::Bool=false)
        
        if !export2Json
            @info "Plotting slice..."
        end
        
        if plt_2D_image
            if fig === nothing
                fig = PyPlot.figure()
                ax_2D = fig.add_subplot(211)
                ax_slice = fig.add_subplot(212)
            else
                ax_2D = fig.get_axes()[1]
                ax_slice = fig.get_axes()[2]
            end
        else
            if fig === nothing
                fig = PyPlot.figure()
                ax_slice = fig.add_subplot(111)
            else
                if length(fig.get_axes()) > 1
                    ax_slice = fig.get_axes()[2]
                else
                    ax_slice = fig.get_axes()[1]
                end
            end
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if plt_2D_image
            x = reshape(mesh.CenterCoor[:, 1], (mesh.ny, mesh.nx))
            y = reshape(mesh.CenterCoor[:, 2], (mesh.ny, mesh.nx))

            var_value_2D = reshape(var_value, (mesh.ny, mesh.nx))

            dx = (x[1, 2] - x[1, 1]) / 2.0
            dy = (y[2, 1] - y[1, 1]) / 2.0
            extent = [x[1, 1] - dx, x[end, end] + dx, y[1, 1] - dy, y[end, end] + dy]

            im_2D = ax_2D.imshow(var_value_2D, cmap=plot_prop.colorMap, interpolation=plot_prop.interpolation,
                                extent=extent, vmin=vmin, vmax=vmax, origin="lower")

            if plot_colorbar
                # Simple colorbar approach
                fig.colorbar(im_2D, ax=ax_2D, pad=0.05)
            end
        end

        if point1 === nothing
            point1 = [-mesh.Lx, 0.0]
        end
        if point2 === nothing
            point2 = [mesh.Lx, 0.0]
        end

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

        if plt_2D_image
            ax_2D.plot([point1[1], point2[1]], [point1[2], point2[2]], 
                    plot_prop.lineStyle, color=plot_prop.lineColor)
        end

        # Create sampling points
        x_samples = range(point1[1], stop=point2[1], length=105)
        y_samples = range(point1[2], stop=point2[2], length=105)
        sampling_points = hcat(x_samples, y_samples)

        # Note: griddata needs to be implemented or imported
        # For now, assuming there's a griddata function available
        try
            value_samp_points = griddata(mesh.CenterCoor, var_value, sampling_points, 
                                    method="linear", fill_value=NaN)
        catch e
            # Fallback if griddata is not available
            value_samp_points = fill(NaN, size(sampling_points, 1))
        end

        # Calculate sampling line distances
        sampling_line_lft = sqrt.((sampling_points[1:52, 1] .- sampling_points[53, 1]).^2 .+ 
                                (sampling_points[1:52, 2] .- sampling_points[53, 2]).^2)
        sampling_line_rgt = sqrt.((sampling_points[53:end, 1] .- sampling_points[53, 1]).^2 .+ 
                                (sampling_points[53:end, 2] .- sampling_points[53, 2]).^2)
        sampling_line = vcat(-reverse(sampling_line_lft), sampling_line_rgt)

        legend = nothing
        if labels !== nothing
            legend = labels.legend
        end

        ax_slice.plot(sampling_line, value_samp_points, plot_prop.lineStyle, 
                    color=plot_prop.lineColor, label=legend)

        # Set x ticks and labels
        tick_indices = [1, 21, 42, 53, 63, 84, 105]
        ax_slice.set_xticks(sampling_line[tick_indices])

        xtick_labels = String[]
        for i in tick_indices
            if i <= size(sampling_points, 1)
                label = "(" * to_precision(sampling_points[i, 1], plot_prop.dispPrecision) * ", " * 
                    to_precision(sampling_points[i, 2], plot_prop.dispPrecision) * ")"
                push!(xtick_labels, label)
            end
        end
        ax_slice.set_xticklabels(xtick_labels)

        if vmin !== nothing && vmax !== nothing
            ax_slice.set_ylim((vmin - 0.1*vmin, vmax + 0.1*vmax))
        end

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_fracture_slice_cell_center(var_value, mesh, point=nothing, orientation="horizontal", fig=nothing, plot_prop=nothing,
                                    vmin=nothing, vmax=nothing, plot_colorbar=true, labels=nothing, plt_2D_image=true,
                                    extreme_points=nothing, export2Json=false)

        This function plots the fracture on a given slice of the domain. A points along with the direction of the slice is
        given to form the slice. The slice is made from the center of the cell containing the given point along the given
        orientation.

        # Arguments
        - `var_value::Array`: a ndarray with the length of the number of cells in the mesh.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `point::Union{Vector{Float64}, Nothing}=nothing`: the point from which the slice should pass [x, y].
        - `orientation::String="horizontal"`: the orientation according to which the slice is made.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.
        - `plot_colorbar::Bool=true`: if True, colorbar will be plotted.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `plt_2D_image::Bool=true`: if True, a subplot showing the colormap and the slice will also be plotted.
        - `extreme_points::Union{Matrix{Float64}, Nothing}=nothing`: An empty array of shape (2, 2).
        - `export2Json::Bool=false`: if True, suppress some plotting operations.

        # Returns
        - `Tuple`: (Figure, x_plot_coord, var_value[sampling_cells], sampling_cells)
    """

    function plot_fracture_slice_cell_center(var_value, mesh, point::Union{Vector{Float64}, Nothing}=nothing, 
                                            orientation::String="horizontal", 
                                            fig::Union{PyPlot.Figure, Nothing}=nothing, 
                                            plot_prop::Union{PlotProperties, Nothing}=nothing,
                                            vmin::Union{Float64, Nothing}=nothing, 
                                            vmax::Union{Float64, Nothing}=nothing, 
                                            plot_colorbar::Bool=true, 
                                            labels::Union{LabelProperties, Nothing}=nothing, 
                                            plt_2D_image::Bool=true,
                                            extreme_points::Union{Matrix{Float64}, Nothing}=nothing, 
                                            export2Json::Bool=false)
        
        if !export2Json
            @info "Plotting slice..."
            if plt_2D_image
                if fig === nothing
                    fig = PyPlot.figure()
                    ax_2D = fig.add_subplot(211)
                    ax_slice = fig.add_subplot(212)
                else
                    ax_2D = fig.get_axes()[1]
                    ax_slice = fig.get_axes()[2]
                end
            else
                if fig === nothing
                    fig = PyPlot.figure()
                    ax_slice = fig.add_subplot(111)
                else
                    ax_slice = fig.get_axes()[1]
                end
            end
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
            plot_prop.lineStyle = "."
        end

        if plt_2D_image && !export2Json
            x = reshape(mesh.CenterCoor[:, 1], (mesh.ny, mesh.nx))
            y = reshape(mesh.CenterCoor[:, 2], (mesh.ny, mesh.nx))

            var_value_2D = reshape(var_value, (mesh.ny, mesh.nx))

            dx = (x[1, 2] - x[1, 1]) / 2.0
            dy = (y[2, 1] - y[1, 1]) / 2.0
            extent = [x[1, 1] - dx, x[end, end] + dx, y[1, 1] - dy, y[end, end] + dy]

            im_2D = ax_2D.imshow(var_value_2D, cmap=plot_prop.colorMap, interpolation=plot_prop.interpolation,
                                extent=extent, vmin=vmin, vmax=vmax, origin="lower")

            if plt_2D_image && plot_colorbar
                # Simple approach without make_axes_locatable
                fig.colorbar(im_2D, ax=ax_2D, pad=0.05)
            end
        end

        if point === nothing
            point = [0.0, 0.0]
        end
        
        if !(orientation in ("horizontal", "vertical", "increasing", "decreasing"))
            throw(ArgumentError("Given orientation is not supported. Possible options: 'horizontal', 'vertical', 'increasing', 'decreasing'"))
        end

        zero_cell_result = mesh.locate_element(point[1], point[2])
        zero_cell = zero_cell_result[1]
        
        if any(isnan.(zero_cell))
            throw(ArgumentError("The given point does not lie in the grid!"))
        end

        # Convert to integer index
        zero_cell = Int(round(zero_cell))

        sampling_cells = Int[]
        x_plot_coord = Float64[]

        if orientation == "vertical"
            # Collect cells going vertically
            sampling_cells_top = Int[]
            cell = zero_cell
            while cell >= 1
                pushfirst!(sampling_cells_top, cell)
                cell -= mesh.nx
            end
            
            sampling_cells_bottom = Int[]
            cell = zero_cell + mesh.nx
            while cell <= mesh.NumberOfElts
                push!(sampling_cells_bottom, cell)
                cell += mesh.nx
            end
            
            sampling_cells = vcat(sampling_cells_top, sampling_cells_bottom)
            x_plot_coord = mesh.CenterCoor[sampling_cells, 2]  # y-coordinates for vertical slice

        elseif orientation == "horizontal"
            row = (zero_cell - 1) ÷ mesh.nx + 1
            start_cell = (row - 1) * mesh.nx + 1
            end_cell = row * mesh.nx
            sampling_cells = collect(start_cell:end_cell)
            x_plot_coord = mesh.CenterCoor[sampling_cells, 1]  # x-coordinates for horizontal slice

        elseif orientation == "increasing"  # diagonal ascending
            # Bottom half (diagonal down-left)
            bottom_half = Int[]
            cell = zero_cell
            while cell >= 1 && cell <= mesh.NumberOfElts
                if mesh.CenterCoor[cell, 1] <= mesh.CenterCoor[zero_cell, 1]
                    pushfirst!(bottom_half, cell)
                end
                # Move diagonally down-left (1-indexed)
                next_row = ((cell - 1) ÷ mesh.nx) + 1 - 1
                next_col = ((cell - 1) % mesh.nx) + 1 - 1
                if next_row >= 1 && next_col >= 1
                    cell = (next_row - 1) * mesh.nx + next_col
                else
                    break
                end
            end
            
            # Top half (diagonal up-right)
            top_half = Int[]
            cell = zero_cell
            while cell >= 1 && cell <= mesh.NumberOfElts
                if mesh.CenterCoor[cell, 1] >= mesh.CenterCoor[zero_cell, 1]
                    push!(top_half, cell)
                end
                # Move diagonally up-right
                next_row = ((cell - 1) ÷ mesh.nx) + 1 + 1
                next_col = ((cell - 1) % mesh.nx) + 1 + 1
                if next_row <= mesh.ny && next_col <= mesh.nx
                    cell = (next_row - 1) * mesh.nx + next_col
                else
                    break
                end
            end
            
            # Remove duplicates and sort
            sampling_cells = vcat(reverse(bottom_half), top_half[2:end])  # Remove duplicate center cell
            x_plot_coord = vcat(-sqrt.(sum((mesh.CenterCoor[bottom_half, :] .- mesh.CenterCoor[zero_cell, :]').^2, dims=2)[:]),
                            sqrt.(sum((mesh.CenterCoor[top_half, :] .- mesh.CenterCoor[zero_cell, :]').^2, dims=2)[:]))

        elseif orientation == "decreasing"  # diagonal descending
            # Bottom half (diagonal down-right)
            bottom_half = Int[]
            cell = zero_cell
            while cell >= 1 && cell <= mesh.NumberOfElts
                if mesh.CenterCoor[cell, 1] >= mesh.CenterCoor[zero_cell, 1]
                    pushfirst!(bottom_half, cell)
                end
                # Move diagonally down-right (1-indexed)
                next_row = ((cell - 1) ÷ mesh.nx) + 1 - 1
                next_col = ((cell - 1) % mesh.nx) + 1 + 1
                if next_row >= 1 && next_col <= mesh.nx
                    cell = (next_row - 1) * mesh.nx + next_col
                else
                    break
                end
            end
            
            # Top half (diagonal up-left)
            top_half = Int[]
            cell = zero_cell
            while cell >= 1 && cell <= mesh.NumberOfElts
                if mesh.CenterCoor[cell, 1] <= mesh.CenterCoor[zero_cell, 1]
                    push!(top_half, cell)
                end
                # Move diagonally up-left
                next_row = ((cell - 1) ÷ mesh.nx) + 1 + 1
                next_col = ((cell - 1) % mesh.nx) + 1 - 1
                if next_row <= mesh.ny && next_col >= 1
                    cell = (next_row - 1) * mesh.nx + next_col
                else
                    break
                end
            end
            
            # Remove duplicates and sort
            sampling_cells = vcat(reverse(bottom_half), top_half[2:end])  # Remove duplicate center cell
            x_plot_coord = vcat(-sqrt.(sum((mesh.CenterCoor[bottom_half, :] .- mesh.CenterCoor[zero_cell, :]').^2, dims=2)[:]),
                            sqrt.(sum((mesh.CenterCoor[top_half, :] .- mesh.CenterCoor[zero_cell, :]').^2, dims=2)[:]))
        end

        if plt_2D_image && !export2Json
            ax_2D.plot(mesh.CenterCoor[sampling_cells, 1], mesh.CenterCoor[sampling_cells, 2], 
                    "k.", linewidth=plot_prop.lineWidth, alpha=plot_prop.alpha, markersize=1)
        end

        if !export2Json
            ax_slice.plot(x_plot_coord, var_value[sampling_cells], plot_prop.lineStyle, 
                        color=plot_prop.lineColor, label=(labels !== nothing ? labels.legend : nothing))
        end

        # Set up ticks
        if length(sampling_cells) > 7
            mid = length(sampling_cells) ÷ 2
            half_1st = 1:(mid ÷ 3):min(3, mid)
            half_2nd = (mid + (mid ÷ 3)):((mid ÷ 3)):length(sampling_cells)
            if length(half_2nd) < 3 && length(sampling_cells) > 0
                half_2nd = vcat(half_2nd, length(sampling_cells))
            end
            x_ticks_indices = vcat(half_1st[1:min(3, end)], [mid + 1], half_2nd[min(end, 3):end])
            x_ticks_indices = unique(sort(x_ticks_indices))
            x_ticks_indices = x_ticks_indices[x_ticks_indices .<= length(x_plot_coord)]
        else
            x_ticks_indices = 1:min(length(sampling_cells), 7)
        end

        if !export2Json && length(x_ticks_indices) > 0 && length(x_plot_coord) > 0
            ax_slice.set_xticks(x_plot_coord[x_ticks_indices])
            
            xtick_labels = String[]
            for i in x_ticks_indices
                if i <= length(sampling_cells)
                    label = "(" * to_precision(round(mesh.CenterCoor[sampling_cells[i], 1], digits=3), plot_prop.dispPrecision) * 
                        ", " * to_precision(round(mesh.CenterCoor[sampling_cells[i], 2], digits=3), plot_prop.dispPrecision) * ")"
                    push!(xtick_labels, label)
                end
            end
            ax_slice.set_xticklabels(xtick_labels)
            
            if vmin !== nothing && vmax !== nothing
                ax_slice.set_ylim((vmin - 0.1*vmin, vmax + 0.1*vmax))
            end
        end

        if extreme_points !== nothing && length(sampling_cells) > 0
            extreme_points[1, :] = mesh.CenterCoor[sampling_cells[1], :]
            extreme_points[2, :] = mesh.CenterCoor[sampling_cells[end], :]
        end

        if export2Json
            fig = nothing
        end

        return fig, x_plot_coord, var_value[sampling_cells], sampling_cells
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_analytical_solution_slice(regime, variable, mat_prop, inj_prop, mesh=nothing, fluid_prop=nothing, fig=nothing,
                                point1=nothing, point2=nothing, time_srs=nothing, length_srs=nothing, h=nothing, samp_cell=nothing,
                                plot_prop=nothing, labels=nothing, gamma=nothing, plt_top_view=false)

        This function plots slice of the given analytical solution. It can be used to compare simulation results by
        superimposing on the figure obtained from the slice plot function.

        # Arguments
        - `regime::String`: the string specifying the limiting case solution to be plotted.
        - `variable::String`: the variable to be plotted. Possible options are 'w', 'width' or 'p', 'pressure'.
        - `mat_prop::MaterialProperties`: the MaterialProperties object giving the material properties.
        - `inj_prop::InjectionProperties`: the InjectionProperties object giving the injection properties.
        - `mesh::Union{CartesianMesh, Nothing}=nothing`: a CartesianMesh class object describing the grid.
        - `fluid_prop::Union{FluidProperties, Nothing}=nothing`: the FluidProperties object giving the fluid properties.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: figure object to superimpose the image.
        - `point1::Union{Vector{Float64}, Nothing}=nothing`: the left point from which the slice should pass [x, y].
        - `point2::Union{Vector{Float64}, Nothing}=nothing`: the right point from which the slice should pass [x, y].
        - `time_srs::Union{Vector, Nothing}=nothing`: the times at which the analytical solution is to be plotted.
        - `length_srs::Union{Vector, Nothing}=nothing`: the length at which the analytical solution is to be plotted.
        - `h::Union{Float64, Nothing}=nothing`: the height of fracture in case of height contained hydraulic fractures.
        - `samp_cell::Union{Int, Nothing}=nothing`: the cell from where the values of the parameter to be taken.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `gamma::Union{Float64, Nothing}=nothing`: the aspect ratio, used in the case of elliptical fracture.
        - `plt_top_view::Bool=false`: if True, top view will be plotted also.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_analytical_solution_slice(regime::String, variable::String, mat_prop, inj_prop, 
                                        mesh=nothing, 
                                        fluid_prop::Union{FluidProperties, Nothing}=nothing, 
                                        fig::Union{PyPlot.Figure, Nothing}=nothing,
                                        point1::Union{Vector{Float64}, Nothing}=nothing, 
                                        point2::Union{Vector{Float64}, Nothing}=nothing, 
                                        time_srs::Union{Vector, Nothing}=nothing, 
                                        length_srs::Union{Vector, Nothing}=nothing, 
                                        h::Union{Float64, Nothing}=nothing, 
                                        samp_cell::Union{Int, Nothing}=nothing,
                                        plot_prop::Union{PlotProperties, Nothing}=nothing, 
                                        labels::Union{LabelProperties, Nothing}=nothing, 
                                        gamma::Union{Float64, Nothing}=nothing, 
                                        plt_top_view::Bool=false)
        
        if !(variable in supported_variables)
            throw(ArgumentError("Variable $variable is not supported"))
        end

        if variable in ("time", "t", "front_dist_min", "d_min", "front_dist_max", "d_max",
                        "front_dist_mean", "d_mean")
            throw(ArgumentError("The given variable does not vary spatially."))
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end
        plot_prop_cp = deepcopy(plot_prop)  # Assuming copy.copy equivalent

        if labels === nothing
            labels = LabelProperties(variable, "slice", "2D")
        end

        analytical_list, mesh_list = get_HF_analytical_solution(regime, variable, mat_prop, inj_prop,
                                                            mesh=mesh, fluid_prop=fluid_prop, time_srs=time_srs,
                                                            length_srs=length_srs, h=h, samp_cell=samp_cell, gamma=gamma)
        
        for i in 1:length(analytical_list)
            analytical_list[i] = analytical_list[i] / labels.unitConversion
            if variable in ("pn", "pressure")
                # Set negative pressures to zero
                analytical_list[i][analytical_list[i] .< 0] .= 0.0
            end
        end

        # finding maximum and minimum values in complete list
        analytical_value = deepcopy(analytical_list)
        vmin, vmax = Inf, -Inf
        for i in analytical_value
            # Remove inf, -inf and nan values
            finite_vals = i[isfinite.(i)]
            if length(finite_vals) > 0
                if variable in ("p", "pressure")
                    non_zero = abs.(finite_vals) .> 0
                    if any(non_zero)
                        med_val = median(finite_vals[non_zero])
                        i_min, i_max = -0.2 * med_val, 1.5 * med_val
                    else
                        i_min, i_max = Inf, -Inf
                    end
                else
                    i_min, i_max = minimum(finite_vals), maximum(finite_vals)
                end
                vmin, vmax = min(vmin, i_min), max(vmax, i_max)
            end
        end

        plot_prop_cp.colorMap = plot_prop.colorMaps[2]
        plot_prop_cp.lineStyle = plot_prop.lineStyleAnal
        plot_prop_cp.lineWidth = plot_prop.lineWidthAnal
        
        for i in 1:length(analytical_list)
            labels.legend = "analytical (" * regime * ") t= " * to_precision(time_srs[i], plot_prop.dispPrecision)
            plot_prop_cp.lineColor = plot_prop_cp.colorsList[((i-1) % length(plot_prop_cp.colorsList)) + 1]

            fig = plot_fracture_slice_interpolated(analytical_list[i], mesh_list[i], point1=point1, point2=point2,
                                                fig=fig, plot_prop=plot_prop_cp, vmin=vmin, vmax=vmax,
                                                plot_colorbar=false, labels=labels, plt_2D_image=plt_top_view)
        end
        
        if plt_top_view
            ax_tv = fig.get_axes()[1]
            ax_tv.set_xlabel("meter")
            ax_tv.set_ylabel("meter")
            ax_tv.set_title("Top View")

            # making colorbar
            im = ax_tv.images
            if length(im) > 0
                # Simple approach without make_axes_locatable
                cb = fig.colorbar(im[end], ax=ax_tv, pad=0.05)
                cb.set_label(labels.colorbarLabel)
            end

            ax_slice = fig.get_axes()[2]
        else
            ax_slice = fig.get_axes()[1]
        end
        
        ax_slice.set_ylabel(labels.colorbarLabel)
        ax_slice.set_xlabel("(x,y) " * labels.xLabel)

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_analytical_solution_at_point(regime, variable, mat_prop, inj_prop, fluid_prop=nothing, fig=nothing, point=nothing,
                                        time_srs=nothing, length_srs=nothing, h=nothing, samp_cell=nothing, plot_prop=nothing,
                                        labels=nothing, gamma=nothing)

        This function plots the given analytical solution at a given point. It can be used to compare simulation results by
        superimposing on the figure obtained from the plot at point function.

        # Arguments
        - `regime::String`: the string specifying the limiting case solution to be plotted.
        - `variable::String`: the variable to be plotted. Possible options are 'w', 'width' or 'p', 'pressure'.
        - `mat_prop::MaterialProperties`: the MaterialProperties object giving the material properties.
        - `inj_prop::InjectionProperties`: the InjectionProperties object giving the injection properties.
        - `fluid_prop::Union{FluidProperties, Nothing}=nothing`: the FluidProperties object giving the fluid properties.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: figure object to superimpose the image.
        - `point::Union{Vector{Float64}, Nothing}=nothing`: the point at which the solution to be plotted [x, y].
        - `time_srs::Union{Vector, Nothing}=nothing`: the times at which the analytical solution is to be plotted.
        - `length_srs::Union{Vector, Nothing}=nothing`: the length at which the analytical solution is to be plotted.
        - `h::Union{Float64, Nothing}=nothing`: the height of fracture in case of height contained hydraulic fractures.
        - `samp_cell::Union{Int, Nothing}=nothing`: the cell from where the values of the parameter to be taken.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `gamma::Union{Float64, Nothing}=nothing`: the aspect ratio, used in the case of elliptical fracture.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_analytical_solution_at_point(regime::String, variable::String, mat_prop, inj_prop, 
                                            fluid_prop::Union{FluidProperties, Nothing}=nothing, 
                                            fig::Union{PyPlot.Figure, Nothing}=nothing,
                                            point::Union{Vector{Float64}, Nothing}=nothing,
                                            time_srs::Union{Vector, Nothing}=nothing, 
                                            length_srs::Union{Vector, Nothing}=nothing, 
                                            h::Union{Float64, Nothing}=nothing, 
                                            samp_cell::Union{Int, Nothing}=nothing, 
                                            plot_prop::Union{PlotProperties, Nothing}=nothing,
                                            labels::Union{LabelProperties, Nothing}=nothing, 
                                            gamma::Union{Float64, Nothing}=nothing)
        
        @info "PyFrac.plot_analytical_solution_at_point"
        
        if !(variable in supported_variables)
            throw(ArgumentError("Variable $variable is not supported"))
        end

        if time_srs === nothing && length_srs === nothing
            throw(ArgumentError("Either time series or length series is to be provided!"))
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end
        plot_prop_cp = deepcopy(plot_prop)  # Assuming copy.copy equivalent

        if labels === nothing
            labels_given = false
            labels = LabelProperties(variable, "point", "2D")
        else
            labels_given = true
        end

        if point === nothing
            point = [0.0, 0.0]
        end
        
        analytical_list = get_HF_analytical_solution_at_point(regime, variable, point, mat_prop, inj_prop,
                                                        fluid_prop=fluid_prop, length_srs=length_srs,
                                                        time_srs=time_srs, h=h, samp_cell=samp_cell, gamma=gamma)
        
        if time_srs === nothing
            time_srs = get_HF_analytical_solution_at_point(regime, "t", point, mat_prop, inj_prop,
                                                        fluid_prop=fluid_prop, length_srs=length_srs,
                                                        time_srs=time_srs, h=h, samp_cell=samp_cell, gamma=gamma)
        end

        for i in 1:length(analytical_list)
            analytical_list[i] = analytical_list[i] / labels.unitConversion
        end

        if variable in ["time", "t", "front_dist_min", "d_min", "front_dist_max", "d_max",
                        "front_dist_mean", "d_mean"]
            @warn "The given variable does not vary spatially."
        end

        plot_prop_cp.lineColor = plot_prop.lineColorAnal
        plot_prop_cp.lineStyle = plot_prop.lineStyleAnal
        plot_prop_cp.lineWidth = plot_prop.lineWidthAnal
        
        if !labels_given
            labels.legend = labels.legend * " analytical"
        end
        labels.xLabel = "time (\$s\$)"

        fig = plot_variable_vs_time(time_srs, analytical_list, fig=fig, plot_prop=plot_prop_cp, label=labels.legend)

        ax = fig.get_axes()[1]
        ax.set_xlabel(labels.xLabel)
        ax.set_ylabel(labels.colorbarLabel)
        ax.set_title(labels.figLabel)
        
        if plot_prop.plotLegend
            ax.legend()
        end

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_scale_3D(fracture, fig=nothing, plot_prop=nothing)

        This function plots lines with dimensions on the 3D fracture plot.

        # Arguments
        - `fracture::Fracture`: the fracture object
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: figure object to superimpose the image
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots
    """

    function plot_scale_3D(fracture, fig::Union{PyPlot.Figure, Nothing}=nothing, plot_prop::Union{PlotProperties, Nothing}=nothing)
        """ This function plots lines with dimensions on the 3D fracture plot."""
        
        @info "PyFrac.plot_scale_3D"
        @info "Plotting scale..."
        
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        I = fracture.Ffront[:, 1:2]  # 1-indexed columns

        max_x = maximum(I[:, 1])
        max_y = maximum(I[:, 2])
        min_x = minimum(I[:, 1])
        min_y = minimum(I[:, 2])
        
        # Create path data for dimension lines
        path_data = [
            ("MOVETO", [min_x, min_y - 2 * fracture.mesh.hy]),
            ("LINETO", [max_x, min_y - 2 * fracture.mesh.hy]),
            ("MOVETO", [min_x, min_y - 2.5 * fracture.mesh.hy]),
            ("LINETO", [min_x, min_y - 1.5 * fracture.mesh.hy]),
            ("MOVETO", [max_x, min_y - 2.5 * fracture.mesh.hy]),
            ("LINETO", [max_x, min_y - 1.5 * fracture.mesh.hy]),
            ("MOVETO", [min_x - 2.5 * fracture.mesh.hx, min_y - fracture.mesh.hy]),
            ("LINETO", [min_x - 2.5 * fracture.mesh.hx, max_y]),
            ("MOVETO", [min_x - 3.0 * fracture.mesh.hx, min_y - fracture.mesh.hy]),
            ("LINETO", [min_x - 2.0 * fracture.mesh.hx, min_y - fracture.mesh.hy]),
            ("MOVETO", [min_x - 3.0 * fracture.mesh.hx, max_y]),
            ("LINETO", [min_x - 2.0 * fracture.mesh.hx, max_y])
        ]

        # Extract codes and vertices
        codes = [item[1] for item in path_data]
        verts = [item[2] for item in path_data]
        
        # Note: Path and PathPatch functionality may need to be implemented or imported
        # This is a simplified approach using direct line plotting
        
        # Plot horizontal dimension line
        ax.plot([min_x, max_x], [min_y - 2 * fracture.mesh.hy, min_y - 2 * fracture.mesh.hy], [0, 0], "k-", linewidth=1)
        
        # Plot horizontal dimension markers
        ax.plot([min_x, min_x], [min_y - 2.5 * fracture.mesh.hy, min_y - 1.5 * fracture.mesh.hy], [0, 0], "k-", linewidth=1)
        ax.plot([max_x, max_x], [min_y - 2.5 * fracture.mesh.hy, min_y - 1.5 * fracture.mesh.hy], [0, 0], "k-", linewidth=1)
        
        # Plot vertical dimension line
        ax.plot([min_x - 2.5 * fracture.mesh.hx, min_x - 2.5 * fracture.mesh.hx], [min_y - fracture.mesh.hy, max_y], [0, 0], "k-", linewidth=1)
        
        # Plot vertical dimension markers
        ax.plot([min_x - 3.0 * fracture.mesh.hx, min_x - 2.0 * fracture.mesh.hx], [min_y - fracture.mesh.hy, min_y - fracture.mesh.hy], [0, 0], "k-", linewidth=1)
        ax.plot([min_x - 3.0 * fracture.mesh.hx, min_x - 2.0 * fracture.mesh.hx], [max_y, max_y], [0, 0], "k-", linewidth=1)

        if plot_prop.textSize === nothing
            plot_prop.textSize = maximum([fracture.mesh.hx, fracture.mesh.hy])  # Fixed: was hx, hx
        end

        y_len = to_precision(max_y - min_y + fracture.mesh.hy, plot_prop.dispPrecision)
        # Note: text3d may need to be implemented or imported
        # For now, using regular text with 3D coordinates
        try
            ax.text(min_x - 2.5 * fracture.mesh.hx - 5 * plot_prop.textSize, (max_y + min_y) / 2, 0,
                    y_len * "m", fontsize=plot_prop.textSize, color="k")
        catch e
            # Fallback if 3D text is not available
            ax.text2D(0.05, 0.95, y_len * "m", transform=ax.transAxes, fontsize=plot_prop.textSize, color="k")
        end
        
        x_len = to_precision(max_x - min_x + fracture.mesh.hy, plot_prop.dispPrecision)  # Note: using hy as in original
        try
            ax.text((max_x + min_x) / 2, min_y - 2 * fracture.mesh.hy - 2 * plot_prop.textSize, 0,
                    x_len * "m", fontsize=plot_prop.textSize, color="k")
        catch e
            # Fallback if 3D text is not available
            ax.text2D(0.05, 0.90, x_len * "m", transform=ax.transAxes, fontsize=plot_prop.textSize, color="k")
        end

        ax.grid(false)
        ax.set_frame_on(false)
        ax.set_axis_off()

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_slice_3D(var_value, mesh, point1=nothing, point2=nothing, fig=nothing, plot_prop=nothing, vmin=nothing, vmax=nothing,
                    label=nothing)

        This function plots the fracture on a given slice of the domain in 3D. Two points are to be given that will be
        joined to form the slice. The values on the slice are interpolated from the values available on the cell
        centers.

        # Arguments
        - `var_value::Array`: a ndarray with the length of the number of cells in the mesh.
        - `mesh::CartesianMesh`: a CartesianMesh object giving the discretization of the domain.
        - `point1::Union{Vector{Float64}, Nothing}=nothing`: the left point from which the slice should pass [x, y].
        - `point2::Union{Vector{Float64}, Nothing}=nothing`: the right point from which the slice should pass [x, y].
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: the figure to superimpose on. New figure will be made if not provided.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `vmin::Union{Float64, Nothing}=nothing`: the minimum value to be used to colormap and make the colorbar.
        - `vmax::Union{Float64, Nothing}=nothing`: the maximum value to be used to colormap and make the colorbar.
        - `label::Union{String, Nothing}=nothing`: the label of plotted line to be used for legend.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_slice_3D(var_value, mesh, point1::Union{Vector{Float64}, Nothing}=nothing, 
                        point2::Union{Vector{Float64}, Nothing}=nothing, 
                        fig::Union{PyPlot.Figure, Nothing}=nothing, 
                        plot_prop::Union{PlotProperties, Nothing}=nothing, 
                        vmin::Union{Float64, Nothing}=nothing, 
                        vmax::Union{Float64, Nothing}=nothing,
                        label::Union{String, Nothing}=nothing)
        
        @info "PyFrac.plot_slice_3D"
        @info "Plotting slice in 3D..."

        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
            plot_prop.lineStyle = "k--"
        end

        if point1 === nothing
            point1 = [-mesh.Lx, 0.0]
        end
        if point2 === nothing
            point2 = [mesh.Lx, 0.0]
        end
        
        # Create sampling points
        x_samples = range(point1[1], stop=point2[1], length=100)
        y_samples = range(point1[2], stop=point2[2], length=100)
        sampling_points = hcat(x_samples, y_samples)

        # Note: griddata needs to be implemented or imported
        # For now, assuming there's a griddata function available
        try
            value_samp_points = griddata(mesh.CenterCoor, var_value, sampling_points, 
                                    method="linear", fill_value=NaN)
        catch e
            # Fallback if griddata is not available
            value_samp_points = fill(NaN, size(sampling_points, 1))
        end

        ax.plot(sampling_points[:, 1], sampling_points[:, 2], value_samp_points,
                plot_prop.lineStyle, color=plot_prop.lineColor, label=label)
                
        if vmin === nothing && vmax === nothing
            # Remove inf and nan values for vmin/vmax calculation
            finite_vals = var_value[isfinite.(var_value)]
            if length(finite_vals) > 0
                vmin, vmax = minimum(finite_vals), maximum(finite_vals)
            else
                vmin, vmax = 0.0, 1.0  # Default values
            end
        end
        
        ax.set_zlim(vmin, vmax)

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_footprint_analytical(regime, mat_prop, inj_prop, fluid_prop=nothing, time_srs=nothing, h=nothing, samp_cell=nothing,
                                fig=nothing, plot_prop=nothing, gamma=nothing, inj_point=nothing)

        This function plots footprint of the analytical solution fracture. It can be used to compare simulation results by
        superimposing on the figure obtained from the footprint plot.

        # Arguments
        - `regime::String`: the string specifying the limiting case solution to be plotted.
                ========    ============================
                option      limiting solution
                ========    ============================
                'M'         viscosity storage
                'Mt'        viscosity leak-off
                'K'         toughness storage
                'Kt'        toughness leak-off
                'PKN'       PKN
                'KGD_K'     KGD toughness
                'MDR'       MDR turbulent viscosity
                'E_K'       anisotropic toughness
                'E_E'       anisotropic elasticity
                ========    ============================
        - `mat_prop::MaterialProperties`: the MaterialProperties object giving the material properties.
        - `inj_prop::InjectionProperties`: the InjectionProperties object giving the injection properties.
        - `fluid_prop::Union{FluidProperties, Nothing}=nothing`: the FluidProperties object giving the fluid properties.
        - `time_srs::Union{Vector, Nothing}=nothing`: the times at which the analytical solution is to be plotted.
        - `h::Union{Float64, Nothing}=nothing`: the height of fracture in case of height contained hydraulic fractures.
        - `samp_cell::Union{Int, Nothing}=nothing`: the cell from where the values of the parameter to be taken.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: figure object to superimpose the image.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `gamma::Union{Float64, Nothing}=nothing`: the aspect ratio, used in the case of elliptical fracture.
        - `inj_point::Union{Vector, Nothing}=nothing`: a list of size 2, giving the x and y coordinate of the injection point.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_footprint_analytical(regime::String, mat_prop, inj_prop;
                                    fluid_prop=nothing, time_srs=nothing,
                                    h=nothing, samp_cell=nothing,
                                    fig=nothing, plot_prop=nothing,
                                    gamma=nothing, inj_point=nothing)
        @info "Plotting analytical footprint..."

        # Create or use existing figure
        if fig === nothing
            fig = PyPlot.figure()
        end

        # Get current axes (create if doesn't exist)
        ax = PyPlot.gca()

        # Use default plot properties if not provided
        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        # Get analytical solution footprint patches
        footprint_patches = get_HF_analytical_solution_footprint(
            regime, mat_prop, inj_prop, plot_prop,
            fluid_prop=fluid_prop, time_srs=time_srs,
            h=h, samp_cell=samp_cell,
            gamma=gamma, inj_point=inj_point
        )

        # Add each patch to the plot
        for patch in footprint_patches
            ax.add_patch(patch)

            # Handle 3D case if needed (PyPlot doesn't have direct art3d equivalent)
            if hasproperty(ax, :get_zlim)
                # For 3D plots, you might need to use different approach
                # This is a placeholder for 3D conversion logic
                # In PyPlot, 3D patches are typically handled with mplot3D toolkit
            end
        end

        # Set equal aspect ratio for proper scaling
        ax.set_aspect("equal")

        # Update figure canvas
        fig.canvas.draw_idle()

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_analytical_solution(regime, variable, mat_prop, inj_prop, mesh=nothing, fluid_prop=nothing, fig=nothing,
                                projection="2D", time_srs=nothing, length_srs=nothing, h=nothing, samp_cell=nothing, plot_prop=nothing,
                                labels=nothing, contours_at=nothing, gamma=nothing)

        This function plots the analytical solution according to the given regime. It can be used to compare simulation
        results by superimposing on the figure obtained from the plot function.

        # Arguments
        - `regime::String`: the string specifying the limiting case solution to be plotted.
                ========    ============================
                option      limiting solution
                ========    ============================
                'M'         viscosity storage
                'Mt'        viscosity leak-off
                'K'         toughness storage
                'Kt'        toughness leak-off
                'PKN'       PKN
                'KGD_K'     KGD toughness
                'MDR'       MDR turbulent viscosity
                'E_K'       anisotropic toughness
                'E_E'       anisotropic elasticity
                ========    ============================
        - `variable::String`: the variable to be plotted. Possible options are 'w', 'width' or 'p', 'pressure'.
        - `mat_prop::MaterialProperties`: the MaterialProperties object giving the material properties.
        - `inj_prop::InjectionProperties`: the InjectionProperties object giving the injection properties.
        - `mesh::Union{CartesianMesh, Nothing}=nothing`: a CartesianMesh class object describing the grid.
        - `fluid_prop::Union{FluidProperties, Nothing}=nothing`: the FluidProperties object giving the fluid properties.
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: figure object to superimpose the image.
        - `projection::String="2D"`: a string specifying the projection.
        - `time_srs::Union{Vector, Nothing}=nothing`: the times at which the analytical solution is to be plotted.
        - `length_srs::Union{Vector, Nothing}=nothing`: the length at which the analytical solution is to be plotted.
        - `h::Union{Float64, Nothing}=nothing`: the height of fracture in case of height contained hydraulic fractures.
        - `samp_cell::Union{Int, Nothing}=nothing`: the cell from where the values of the parameter to be taken.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `contours_at::Union{Vector, Nothing}=nothing`: the values at which the contours are to be plotted.
        - `gamma::Union{Float64, Nothing}=nothing`: the aspect ratio, used in the case of elliptical fracture.

        # Returns
        - `PyPlot.Figure`: A Figure object that can be used superimpose further plots.
    """

    function plot_analytical_solution(regime::String, variable::String, mat_prop, inj_prop, 
                                    mesh=nothing, 
                                    fluid_prop::Union{FluidProperties, Nothing}=nothing, 
                                    fig::Union{PyPlot.Figure, Nothing}=nothing,
                                    projection::String="2D", 
                                    time_srs::Union{Vector, Nothing}=nothing, 
                                    length_srs::Union{Vector, Nothing}=nothing, 
                                    h::Union{Float64, Nothing}=nothing, 
                                    samp_cell::Union{Int, Nothing}=nothing, 
                                    plot_prop::Union{PlotProperties, Nothing}=nothing,
                                    labels::Union{LabelProperties, Nothing}=nothing, 
                                    contours_at::Union{Vector, Nothing}=nothing, 
                                    gamma::Union{Float64, Nothing}=nothing)
        
        @info "PyFrac.plot_analytical_solution"
        @info "Plotting analytical $variable $regime solution..."
        
        if !(variable in supported_variables)
            throw(ArgumentError("Variable $variable is not supported"))
        end

        if labels === nothing
            labels_given = false
            labels = LabelProperties(variable, "whole mesh", projection)
        else
            labels_given = true
        end

        if variable == "footprint"
            fig = plot_footprint_analytical(regime, mat_prop, inj_prop, fluid_prop=fluid_prop,
                                        time_srs=time_srs, h=h, samp_cell=samp_cell, fig=fig,
                                        plot_prop=plot_prop, gamma=gamma,
                                        inj_point=inj_prop.sourceCoordinates)
        else
            if plot_prop === nothing
                plot_prop = PlotProperties()
            end
            plot_prop_cp = deepcopy(plot_prop)  # Assuming copy.copy equivalent

            analytical_list, mesh_list = get_HF_analytical_solution(regime, variable, mat_prop, inj_prop,
                                                                mesh=mesh, fluid_prop=fluid_prop, time_srs=time_srs,
                                                                length_srs=length_srs, h=h, samp_cell=samp_cell,
                                                                gamma=gamma)

            for i in 1:length(analytical_list)
                analytical_list[i] = analytical_list[i] / labels.unitConversion
            end

            analytical_value = deepcopy(analytical_list)
            vmin, vmax = Inf, -Inf
            for i in analytical_value
                # Remove inf and nan values
                finite_vals = i[isfinite.(i)]
                if length(finite_vals) > 0
                    i_min, i_max = minimum(finite_vals), maximum(finite_vals)
                    vmin, vmax = min(vmin, i_min), max(vmax, i_max)
                end
            end

            if variable in unidimensional_variables
                plot_prop_cp.lineStyle = plot_prop.lineStyleAnal
                plot_prop_cp.lineColor = plot_prop.lineColorAnal
                plot_prop_cp.lineWidth = plot_prop.lineWidthAnal
                if !labels_given
                    labels.legend = labels.legend * " analytical"
                end
                labels.xLabel = "time (\$s\$)"
                fig = plot_variable_vs_time(time_srs, analytical_list, fig=fig,
                                        plot_prop=plot_prop_cp, label=labels.legend)
                projection = "2D"
            else
                if projection == "2D_clrmap"
                    for i in 1:length(analytical_list)
                        fig = plot_fracture_variable_as_image(analytical_list[i], mesh_list[i], fig=fig,
                                                            plot_prop=plot_prop_cp, vmin=vmin, vmax=vmax)
                    end
                elseif projection == "2D_contours"
                    for i in 1:length(analytical_list)
                        fig = plot_fracture_variable_as_contours(analytical_list[i], mesh_list[i], fig=fig,
                                                            plot_prop=plot_prop_cp, contours_at=contours_at,
                                                            vmin=vmin, vmax=vmax)
                    end
                elseif projection == "3D"
                    for i in 1:length(analytical_list)
                        fig = plot_fracture_variable_as_surface(analytical_list[i], mesh_list[i], fig=fig,
                                                            plot_prop=plot_prop_cp, plot_colorbar=false,
                                                            vmin=vmin, vmax=vmax)
                    end
                end
            end
        end

        ax = fig.get_axes()[1]
        ax.set_xlabel(labels.xLabel)
        ax.set_ylabel(labels.yLabel)
        ax.set_title(labels.figLabel)
        
        if variable != "footprint"
            if projection == "3D"
                ax.set_zlabel(labels.zLabel)
                sm = PyPlot.cm.ScalarMappable(cmap=plot_prop_cp.colorMap,
                                            norm=PyPlot.Normalize(vmin=vmin, vmax=vmax))
                sm._A = []
                cb = PyPlot.colorbar(sm, alpha=plot_prop_cp.alpha)
                cb.set_label(labels.colorbarLabel * " analytical")
            elseif projection in ("2D_clrmap", "2D_contours")
                im = ax.images
                if length(im) > 0
                    cb = im[end].colorbar
                    cb.set_label(labels.colorbarLabel * " analytical")
                end
            elseif projection == "2D"
                ax.set_title(labels.figLabel)
                if plot_prop_cp.plotLegend
                    ax.legend()
                end
            end
        end

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_HF_analytical_solution_footprint(regime, mat_prop, inj_prop, plot_prop, fluid_prop=nothing, time_srs=nothing,
                                            h=nothing, samp_cell=nothing, gamma=nothing, inj_point=nothing)

        This function returns footprint of the analytical solution in the form of patches

        # Arguments
        - `regime::String`: the regime string
        - `mat_prop::MaterialProperties`: material properties
        - `inj_prop::InjectionProperties`: injection properties
        - `plot_prop::PlotProperties`: plot properties
        - `fluid_prop::Union{FluidProperties, Nothing}=nothing`: fluid properties
        - `time_srs::Union{Vector, Nothing}=nothing`: time series
        - `h::Union{Float64, Nothing}=nothing`: fracture height
        - `samp_cell::Union{Int, Nothing}=nothing`: sample cell
        - `gamma::Union{Float64, Nothing}=nothing`: aspect ratio
        - `inj_point::Union{Vector, Nothing}=nothing`: injection point

        # Returns
        - `Vector`: list of patches
    """

    function get_HF_analytical_solution_footprint(regime::String, mat_prop, inj_prop, plot_prop, 
                                                fluid_prop::Union{FluidProperties, Nothing}=nothing, 
                                                time_srs::Union{Vector, Nothing}=nothing,
                                                h::Union{Float64, Nothing}=nothing, 
                                                samp_cell::Union{Int, Nothing}=nothing, 
                                                gamma::Union{Float64, Nothing}=nothing, 
                                                inj_point::Union{Vector, Nothing}=nothing)
        """ This function returns footprint of the analytical solution in the form of patches"""

        if time_srs === nothing
            throw(ArgumentError("Time series is to be provided."))
        end

        if regime == "E_K"
            Kc_1 = mat_prop.Kc1
        else
            Kc_1 = nothing
        end

        if regime == "MDR"
            density = fluid_prop.density
        else
            density = nothing
        end

        if samp_cell === nothing
            samp_cell = Int(length(mat_prop.Kprime) / 2)
        end

        if regime in ["K", "M"]
            Cprime = nothing
        else
            Cprime = mat_prop.Cprime[samp_cell]
        end

        if regime == "K"
            muPrime = nothing
        else
            muPrime = fluid_prop.muPrime
        end

        if regime == "M"
            Kprime = nothing
        else
            Kprime = mat_prop.Kprime[samp_cell]
        end

        if regime == "PKN" && h === nothing
            throw(ArgumentError("Fracture height is required to plot PKN fracture!"))
        end

        if size(inj_prop.injectionRate, 2) > 1
            V0 = inj_prop.injectionRate[1, 2] * inj_prop.injectionRate[2, 1]
        else
            V0 = nothing
        end

        # Import matplotlib patches
        mpatches = pyimport("matplotlib.patches")
        
        return_patches = []
        for i in time_srs
            if size(inj_prop.injectionRate, 2) > 1
                if i > inj_prop.injectionRate[1, 2]
                    Q0 = 0.0
                else
                    Q0 = inj_prop.injectionRate[2, 1]
                end
            else
                Q0 = inj_prop.injectionRate[2, 1]
            end

            # Note: assuming max() function is available for arrays
            _muPrime = isa(muPrime, Array) ? maximum(muPrime) : muPrime
            x_len, y_len = get_fracture_dimensions_analytical(regime, i, mat_prop.Eprime, Q0,
                                                            muPrime=_muPrime, Kprime=Kprime, Cprime=Cprime,
                                                            Kc_1=Kc_1, h=h, density=density, gamma=gamma, Vinj=V0)

            if inj_point === nothing
                inj_point = [0.0, 0.0]
            end

            if regime in ("M", "Mt", "K", "Kt", "E", "MDR")
                patch = mpatches.Circle((inj_point[1], inj_point[2]), x_len,
                                    edgecolor=plot_prop.lineColorAnal, facecolor="none")
                push!(return_patches, patch)
            elseif regime in ("PKN", "KGD_K")
                patch = mpatches.Rectangle(xy=(-x_len + inj_point[1], -y_len + inj_point[2]), 
                                        width=2 * x_len, height=2 * y_len,
                                        edgecolor=plot_prop.lineColorAnal, facecolor="none")
                push!(return_patches, patch)
            elseif regime in ("E_K", "E_E")
                patch = mpatches.Ellipse(xy=(inj_point[1], inj_point[2]), width=2 * x_len, 
                                    height=2 * y_len, edgecolor=plot_prop.lineColorAnal, facecolor="none")
                push!(return_patches, patch)
            else
                throw(ArgumentError("Regime $regime not supported."))
            end
        end

        return return_patches
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_injection_source(frac, fig=nothing, plot_prop=nothing)

        This function plots the location of the source.

        # Arguments
        - `frac::Fracture`: fracture object
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: figure object
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: plot properties

        # Returns
        - `PyPlot.Figure`: figure object
    """

    function plot_injection_source(frac, fig::Union{PyPlot.Figure, Nothing}=nothing, plot_prop::Union{PlotProperties, Nothing}=nothing)
        """
        This function plots the location of the source.
        """

        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(111)
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        ax.plot(frac.mesh.CenterCoor[frac.source, 1], frac.mesh.CenterCoor[frac.source, 2], 
                ".", color=plot_prop.lineColor)

        ax.plot(frac.mesh.CenterCoor[frac.sink, 1], frac.mesh.CenterCoor[frac.sink, 2], 
                ".", color="w")

        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        animate_simulation_results(fracture_list, variable="footprint", projection=nothing, elements=nothing,
                                    plot_prop=nothing, edge=4, contours_at=nothing, labels=nothing, mat_properties=nothing,
                                    backGround_param=nothing, block_figure=false, plot_non_zero=true, pause_time=0.2,
                                    save_images=false, images_address=".")

        This function plots the fracture evolution with time. The state of the fracture at different times is provided in
        the form of a list of Fracture objects.

        # Arguments
        - `fracture_list::Vector{Fracture}`: the list of Fracture objects giving the evolution of fracture with time.
        - `variable::Union{String, Vector{String}}="footprint"`: the variable to be plotted.
        - `projection::Union{String, Nothing}=nothing`: a string specifying the projection.
        - `elements::Union{Vector{Int}, Nothing}=nothing`: the elements to be plotted.
        - `plot_prop::Union{PlotProperties, Nothing}=nothing`: the properties to be used for the plot.
        - `edge::Int=4`: the edge of the cell that will be plotted.
        - `contours_at::Union{Vector, Nothing}=nothing`: the values at which the contours are to be plotted.
        - `labels::Union{LabelProperties, Nothing}=nothing`: the labels to be used for the plot.
        - `mat_properties::Union{MaterialProperties, Nothing}=nothing`: the material properties.
        - `backGround_param::Union{String, Nothing}=nothing`: the parameter according to which the mesh will be colormapped.
        - `block_figure::Bool=false`: if True, a key would be needed to be pressed to proceed to the next frame.
        - `plot_non_zero::Bool=true`: if true, only non-zero values will be plotted.
        - `pause_time::Float64=0.2`: time (in seconds) between two successive updates of frames.
        - `save_images::Bool=false`: if true, images will be saved.
        - `images_address::String="."`: directory to save images.
    """
    function animate_simulation_results(fracture_list::Vector, variable::Union{String, Vector{String}}="footprint", 
                                    projection::Union{String, Nothing}=nothing, elements::Union{Vector{Int}, Nothing}=nothing,
                                    plot_prop::Union{PlotProperties, Nothing}=nothing, edge::Int=4, 
                                    contours_at::Union{Vector, Nothing}=nothing, labels::Union{LabelProperties, Nothing}=nothing, 
                                    mat_properties::Union{MaterialProperties, Nothing}=nothing,
                                    backGround_param::Union{String, Nothing}=nothing, block_figure::Bool=false, 
                                    plot_non_zero::Bool=true, pause_time::Float64=0.2,
                                    save_images::Bool=false, images_address::String=".")
        
        @info "PyFrac.animate_simulation_results"
        
        if isa(variable, String)
            variable = [variable]
        end
        figures = Vector{Union{PyPlot.Figure, Nothing}}(nothing, length(variable))

        setFigPos = true
        for (Fr_i, fracture) in enumerate(fracture_list)
            for (indx, plt_var) in enumerate(variable)
                @info "Plotting solution at $(fracture.time)..."
                if plot_prop === nothing
                    plot_prop = PlotProperties()
                end

                if figures[indx] !== nothing
                    ax = figures[indx].get_axes()[1]  # save axes from last figure
                    PyPlot.figure(figures[indx].number)
                    PyPlot.clf()  # clear figure
                    figures[indx].add_axes(ax)  # add axis to the figure
                end

                if plt_var == "footprint"
                    figures[indx] = fracture.plot_fracture(variable="mesh", mat_properties=mat_properties,
                                                        projection=projection, backGround_param=backGround_param,
                                                        fig=figures[indx], plot_prop=plot_prop)

                    plot_prop.lineColor = "k"
                    figures[indx] = fracture.plot_fracture(variable="footprint", projection=projection,
                                                        fig=figures[indx], plot_prop=plot_prop, labels=labels)

                else
                    fp_projection = "2D"
                    if projection !== nothing
                        if occursin("2D", projection)
                            fp_projection = "2D"
                        else
                            fp_projection = "3D"
                        end
                    end
                    fig_labels = LabelProperties(plt_var, "whole mesh", fp_projection)
                    fig_labels.figLabel = ""

                    figures[indx] = fracture.plot_fracture(variable="footprint", projection=fp_projection,
                                                        fig=figures[indx], labels=fig_labels)
                    
                    elems = elements
                    if elements === nothing
                        elems = get_elements(suitable_elements[plt_var], fracture)
                    end
                    figures[indx] = fracture.plot_fracture(variable=plt_var, projection=projection,
                                                        elements=elems, mat_properties=mat_properties,
                                                        fig=figures[indx], plot_prop=plot_prop,
                                                        edge=edge, contours_at=contours_at,
                                                        labels=labels, plot_non_zero=plot_non_zero)
                end

                # plotting source elements
                plot_injection_source(fracture, fig=figures[indx])

                # plotting closed cells
                if length(fracture.closed) > 0
                    plot_prop.lineColor = "orangered"
                    figures[indx] = fracture.mesh.identify_elements(fracture.closed, fig=figures[indx],
                                                                plot_prop=plot_prop, plot_mesh=false,
                                                                print_number=false)
                end
                
                # plot the figure
                PyPlot.ion()
                PyPlot.pause(pause_time)

                if save_images
                    image_name = plt_var * string(Fr_i)
                    PyPlot.savefig(images_address * "/" * image_name * ".png")
                end
            end
            
            # set figure position
            if setFigPos
                for i in 1:length(variable)
                    PyPlot.figure(i)
                    # Note: figure positioning may not work the same way in Julia/PyPlot
                    # This is matplotlib-specific functionality
                    try
                        # This part is matplotlib-specific and may need adjustment
                        mngr = PyPlot.get_current_fig_manager()
                        x_offset = 650 * (i - 1)
                        y_offset = 50
                        if i >= 3
                            x_offset = (i - 3) * 650
                            y_offset = 500
                        end
                        mngr.window.setGeometry(x_offset, y_offset, 640, 545)
                    catch e
                        # Silently continue if figure positioning fails
                    end
                end
                setFigPos = false
            end

            if block_figure
                PyPlot.pause(0.5)
                PyPlot.ion()
                PyPlot.show()
                PyPlot.waitforbuttonpress()
            end
        end
        PyPlot.show(block=true)
    end
    #-----------------------------------------------------------------------------------------------------------------------


    """
        text3d(ax, xyz, s, zdir="z", size=nothing, angle=0, usetex=false, kwargs...)

        Plots the string 's' on the axes 'ax', with position 'xyz', size 'size',
        and rotation angle 'angle'.  'zdir' gives the axis which is to be treated
        as the third dimension.  usetex is a boolean indicating whether the string
        should be interpreted as latex or not.  Any additional keyword arguments
        are passed on to transform_path.

        Note: zdir affects the interpretation of xyz.
    """
    function text3d(ax, xyz, s, zdir="z", size=nothing, angle=0, usetex=false; kwargs...)

        x, y, z = xyz
        if zdir == "y"
            xy1, z1 = (x, z), y
        elseif zdir == "x"  # Fixed: was duplicate "y" condition
            xy1, z1 = (y, z), x
        else
            xy1, z1 = (x, y), z
        end

        # Import required matplotlib modules
        matplotlib_text = pyimport("matplotlib.text")
        matplotlib_patches = pyimport("matplotlib.patches")
        matplotlib_transforms = pyimport("matplotlib.transforms")
        art3d = pyimport("mpl_toolkits.mplot3d.art3d")

        # Create text path
        text_path = matplotlib_text.TextPath((0, 0), s, size=size, usetex=usetex)
        
        # Create transformation
        trans = matplotlib_transforms.Affine2D().rotate(angle).translate(xy1[1], xy1[2])

        # Create patch and add to axes
        p1 = matplotlib_patches.PathPatch(trans.transform_path(text_path); kwargs...)
        ax.add_patch(p1)
        art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        zoom_factory(ax, base_scale=2.0)

        Creates a zoom functionality for the given axes.
    """
    function zoom_factory(ax, base_scale=2.0)
        function zoom_fun(event)
            # get the current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cur_xrange = (cur_xlim[2] - cur_xlim[1]) * 0.5
            cur_yrange = (cur_ylim[2] - cur_ylim[1]) * 0.5
            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location
            
            if event.button == "up"
                # deal with zoom in
                scale_factor = 1.0 / base_scale
            elseif event.button == "down"
                # deal with zoom out
                scale_factor = base_scale
            else
                # deal with something that should never happen
                scale_factor = 1.0
            end
            
            # set new limits
            ax.set_xlim([xdata - cur_xrange * scale_factor,
                        xdata + cur_xrange * scale_factor])
            ax.set_ylim([ydata - cur_yrange * scale_factor,
                        ydata + cur_yrange * scale_factor])
            PyPlot.draw() # force re-draw
        end

        fig = ax.get_figure() # get the figure of interest
        # attach the call back
        fig.canvas.mpl_connect("scroll_event", zoom_fun)

        # return the function
        return zoom_fun
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        to_precision(x, p)

        returns a string representation of x formatted with a precision of p

        Based on the webkit javascript implementation taken from here:
        https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp  
    """
    function to_precision(x, p)

        x = float(x)

        if x == 0.0
            return "0." * "0"^(p-1)
        end

        out = String[]

        if x < 0
            push!(out, "-")
            x = -x
        end

        e = Int(floor(log10(x)))
        tens = 10.0^(e - p + 1)
        n = floor(x/tens)

        if n < 10.0^(p - 1)
            e = e - 1
            tens = 10.0^(e - p + 1)
            n = floor(x / tens)
        end

        if abs((n + 1.0) * tens - x) <= abs(n * tens - x)
            n = n + 1
        end

        if n >= 10^p
            n = n / 10.0
            e = e + 1
        end

        m = string(round(n, sigdigits=p))

        # Remove trailing zeros and decimal point if not needed
        if occursin(".", m)
            m = rstrip(m, '0')
            m = rstrip(m, '.')
        end

        if e < -2 || e >= p
            push!(out, string(m[1]))
            if p > 1
                push!(out, ".")
                if length(m) > 1
                    append!(out, [string(c) for c in m[2:min(end, p)]])
                else
                    append!(out, ["0" for _ in 2:p])
                end
            end
            push!(out, "e")
            if e > 0
                push!(out, "+")
            end
            push!(out, string(e))
        elseif e == (p - 1)
            push!(out, m)
        elseif e >= 0
            if length(m) >= e + 1
                push!(out, m[1:e+1])
                if e + 1 < length(m)
                    push!(out, ".")
                    push!(out, m[e+2:end])
                end
            else
                push!(out, m)
                push!(out, "0"^(e + 1 - length(m)))
            end
        else
            push!(out, "0.")
            append!(out, ["0" for _ in 1:-(e+1)])
            push!(out, m)
        end

        return join(out)
    end

    #-----------------------------------------------------------------------------------------------------------------------


    """
        save_images_to_video(image_folder, video_name="movie")

        This function makes a video from the images in the given folder.
    """
    function save_images_to_video(image_folder, video_name="movie")
        """ This function makes a video from the images in the given folder."""
        
        @info "PyFrac.save_images_to_video"
        
        # Note: OpenCV.jl or similar package would be needed for this functionality
        # This is a conceptual translation - you'll need to install appropriate packages
        
        if !occursin(".avi", video_name)
            video_name = video_name * ".avi"
        end

        # Get list of images
        images = filter(x -> endswith(x, ".png"), readdir(image_folder))
        sort!(images)  # Sort to ensure proper order
        
        if length(images) == 0
            @warn "No PNG images found in folder $image_folder"
            return
        end

        # Read first image to get dimensions
        first_image_path = joinpath(image_folder, images[1])
        frame = OpenCV.imread(first_image_path)
        height, width = size(frame, 1), size(frame, 2)

        # Create video writer (conceptual - OpenCV.jl API may differ)
        # video = OpenCV.VideoWriter(video_name, OpenCV.CAP_ANY, 1, (width, height))
        
        img_no = 0
        for image in images
            @info "adding image no $img_no"
            image_path = joinpath(image_folder, image)
            frame = OpenCV.imread(image_path)
            # video.write(frame)  # Conceptual
            # OpenCV.waitKey(1)
            img_no += 1
        end

        # OpenCV.destroyAllWindows()
        # video.release()
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        remove_zeros(var_value, mesh, plot_boundary=false)

        Remove zeros from the variable values.
    """
    function remove_zeros(var_value, mesh, plot_boundary=false)
        if plot_boundary
            zero = falses(mesh.NumberOfElts)
            zero[abs.(var_value) .< 3 * eps()] .= true
            for i in 1:mesh.NumberOfElts-1
                not_left = zero[i] && (i+1 <= length(zero) ? zero[i + 1] : false)
                not_right = zero[i] && (i-1 >= 1 ? zero[i - 1] : false)
                not_bottom = zero[i] && (i - mesh.nx >= 1 ? zero[i - mesh.nx] : false)
                not_top = zero[i] && ((i + mesh.nx) % mesh.NumberOfElts + 1 <= length(zero) ? 
                                    zero[(i + mesh.nx) % mesh.NumberOfElts + 1] : false)
                if not_left && not_right && not_bottom && not_top
                    var_value[i] = NaN
                end
            end
            if mesh.NumberOfElts >= 1
                var_value[mesh.NumberOfElts] = NaN
            end
        else
            var_value[abs.(var_value) .< 3 * eps()] .= NaN
        end
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        get_elements(specifier, fr)

        Get elements based on specifier.
    """
    function get_elements(specifier, fr)
        if specifier == "crack"
            return fr.EltCrack
        elseif specifier == "channel"
            return fr.EltChannel
        elseif specifier == "tip"
            return fr.EltTip
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------


    """
        plot_regime(var_value, mesh, fig=nothing, elements=nothing)

        This function plots the fracture regime with the color code defined by Dontsov. Plotting is done at the ribbon
        cells. The colorbar is replaced by the colorcoded triangle.

        # Arguments
        - `var_value::Array`: List containing the color code at the tip.
        - `mesh::CartesianMesh`: mesh of the current timestep
        - `fig::Union{PyPlot.Figure, Nothing}=nothing`: Figure of the current footprint
        - `elements::Union{Vector{Int}, Nothing}=nothing`: the elements to be plotted.

        # Returns
        - `PyPlot.Figure`: Adapted figure
    """
    function plot_regime(var_value, mesh, fig::Union{PyPlot.Figure, Nothing}=nothing, elements::Union{Vector{Int}, Nothing}=nothing)
        """
        This function plots the fracture regime with the color code defined by Dontsov. Plotting is done at the ribbon
        cells. The colorbar is replaced by the colorcoded triangle.

        Args:
            var_value (list):                   -- List containing the color code at the tip.
            mesh (object):                      -- mesh of the current timestep
            fig (figure):                       -- Figure of the current footprint
            elements (ndarray):                 -- the elements to be plotted.

            Return:
            fig (figure):                       -- Adapted figure

        """

        # getting the extent of the figure
        x = reshape(mesh.CenterCoor[:, 1], (mesh.ny, mesh.nx))  # 1-indexed
        y = reshape(mesh.CenterCoor[:, 2], (mesh.ny, mesh.nx))  # 1-indexed

        dx = (x[1, 2] - x[1, 1]) / 2.0
        dy = (y[2, 1] - y[1, 1]) / 2.0

        extent = [x[1, 1] - dx, x[end, end] + dx, y[1, 1] - dy, y[end, end] + dy]

        # selecting only the relevant elements
        if elements !== nothing
            var_value_fullMesh = ones(mesh.NumberOfElts, 3)
            var_value_fullMesh[elements, :] = var_value[elements, :]
            var_value = var_value_fullMesh
        end

        # re-arrange the solution for plotting
        var_value_2D = reshape(var_value, (mesh.ny, mesh.nx, 3))

        # decide where we are not stagnant
        if elements !== nothing
            prod_vals = prod(var_value[elements, :] .== [1.0 1.0 1.0], dims=2)
            non_stagnant = findall(vec(prod_vals) .!= 1.0)
        else
            non_stagnant = Int[]
        end

        # use footprint if provided
        if fig === nothing
            fig = PyPlot.figure()
            ax = fig.add_subplot(111)
        else
            ax = fig.get_axes()[1]
            # Note: getting and restoring lines is matplotlib-specific
            # l = ax.get_lines()  # This would need PyCall access
            
            fig.clf()
            fig.add_subplot(121)
            # Restoring lines would need PyCall implementation
            # for line in l
            #     ax.plot(line.get_data()[1], line.get_data()[2], "k")
            # end
        end

        # plotting the colored cells
        ax = fig.get_axes()[1]
        ax.imshow(var_value_2D, extent=extent, origin="lower")

        # plotting the triangle with the location of the tip cells
        leg = fig.add_subplot(122)
        leg = mkmtTriangle(leg)
        leg = fill_mkmtTriangle(leg)
        
        if elements !== nothing && length(non_stagnant) > 0
            plot_points_to_mkmtTriangle(leg, var_value[elements[non_stagnant], :])
        end

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        mkmtTriangle(fig)

        This function draws the Maxwell triangle used to highlight the regime dominant in the ribbon cell.

        # Arguments
        - `fig`: The figure to place the Maxwell triangle in.

        # Returns
        - Modified figure
    """
    function mkmtTriangle(fig)
        """
        This function draws the Maxwell triangle used to highlight the regime dominant in the ribbon cell.

        Args:
            fig (figure):               -- The figure to place the Maxwell triangle in.

        """

        # Plot the triangle
        a = 1.0 / sqrt(3)
        fig.plot([0.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.5/a, 0.0], "k-")
        fig.axis([-0.25, 1.2, -0.2, 1.05])
        # Remove axes
        fig.axis("off")
        #Label the corners of the triangle
        fig.text(1.0, 0, raw"$k$", fontsize=18, verticalalignment="top")
        fig.text(-0.1, 0, raw"$m$", fontsize=18, verticalalignment="top")
        fig.text(0.45, 0.575/a, raw"$\tilde{m}$", fontsize=18, verticalalignment="top")

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------


    """
        fill_mkmtTriangle(fig)

        This function colors the Maxwell triangle used to highlight the regime dominant in the ribbon cell.

        # Arguments
        - `fig`: The figure with the Maxwell triangle to color.

        # Returns
        - Modified figure
    """
    function fill_mkmtTriangle(fig)
        """
        This function colors the Maxwell triangle used to highlight the regime dominant in the ribbon cell.

        Args:
            fig (figure):               -- The figure with the Maxwell triangle to color.

        """

        # Generate an image with 300x300 pixels
        Nlignes = 300
        Ncol = 300
        img = zeros(Nlignes, Ncol, 4)
        dx = 2.0 / (Ncol - 1)
        dy = 1.0 / (Nlignes - 1)

        # choose color of pixels.
        for i in 1:(Ncol-1)
            for j in 1:(Nlignes-1)
                x = -1.0 + (i-1) * dx  # 0-indexed calculation
                y = (j-1) * dy         # 0-indexed calculation
                v = y
                r = (x + 1 - v) / 2.0
                b = 1.0 - v - r
                if (r >= 0) && (r <= 1.0) && (v >= 0) && (v <= 1.0) && (b >= 0) && (b <= 1.0)
                    img[j, i, :] = [r, v, b, 1.0]
                else
                    img[j, i, :] = [1.0, 1.0, 1.0, 0.0]
                end
            end
        end
        a = 1.0 / sqrt(3)
        fig.imshow(img, origin="lower", extent=[0.0, 1, 0.0, 0.5 / a])

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_points_to_mkmtTriangle(fig, rgbpoints)

        This function plots a set of points in the m-k-mtilde triangle

        # Arguments
        - `fig`: The figure with the Maxwell triangle to place the points.
        - `rgbpoints`: Color code in RGB of the points to plot.

        # Returns
        - Modified figure
    """
    function plot_points_to_mkmtTriangle(fig, rgbpoints)
        """
        This function plots a set of points in the m-k-mtilde triangle

        Args:
            fig (figure):               -- The figure with the Maxwell triangle to place the points.
            rgbpoints (ndarraz):        -- Color code in RGB of the points to plot.

        """

        nOFpoits = size(rgbpoints, 1)
        x = zeros(nOFpoits)
        y = zeros(nOFpoits)
        
        # Transform color into coordinates
        a = 1.0 / sqrt(3)
        for k in 1:nOFpoits
            rgb = rgbpoints[k, :]
            somme = sum(rgb)
            if somme != 0  # Avoid division by zero
                x[k] = ((rgb[1] - rgb[3]) / sqrt(3) / somme) / (2*a) + 0.5
                y[k] = 0.5/a * rgb[2] / somme
            else
                x[k] = 0.5
                y[k] = 0.5/a * 0.333  # Default position if sum is zero
            end
        end
        
        # Plot the points
        fig.plot(x, y, "k.", markersize=9)
        
        return fig
    end

end # module Visualization