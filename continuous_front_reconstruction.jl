# -*- coding: utf-8 -*-
"""
This file is a part of JFrac.
Realization of Pyfrac on Julia language.

"""
module ContinuousFrontReconstruction

include("properties.jl")
include("level_set.jl")
using .Properties: PlotProperties
using .LevelSet: 
using PyPlot
import PyPlot: plotgrid
using LinearAlgebra
using Statistics
using DataStructures

export itertools_chain_from_iterable, append_to_typelists, distance, copute_area_of_a_polygon, pointtolinedistance, is_inside_the_triangle, recompute_LS_at_tip_cells, findangle, elements, findcommon, filltable, ISinsideFracture, get_fictitius_cell_type, get_fictitius_cell_specific_names, get_fictitius_cell_names, get_fictitius_cell_all_names, get_LS_on_i_fictitius_cell, find_fictitius_cells, split_central_from_noncentral_intersections, define_orientation_type1, define_orientation_type2, define_orientation_type3OR4, move_intersections_to_the_center_when_inRibbon_type3, move_intersections_to_the_center_when_inRibbon_type1, move_intersections_to_the_center_when_inRibbon_type4, split_type4SubType4_from_rest, get_mesh_info_for_computing_intersections, find_x_OR_y_intersections, find_edge_ID, find_xy_intersections_type3_case_2_intersections, check_if_point_inside_cell, reorder_intersections, find_xy_intersections_type3_case_0_1_2_intersections, find_xy_intersections_type1, find_xy_intersections_with_cell_center, process_fictitius_cells_3, process_fictitius_cells_1, process_fictitius_cells_2, process_fictitius_cells_4, get_next_cell_name, get_next_cell_name_from_first, UpdateListsFromContinuousFrontRec, you_advance_more_than_2_cells, plotgrid, plot_final_reconstruction, plot_xy_points, plot_two_fronts, plot_cells


"""
    plotgrid(mesh, ax)

Plots the 2D mesh grid

# Arguments
- `mesh::CartesianMesh`: mesh object
- `ax::PyPlot.PyObject`: an axes object from a PyPlot figure

# Returns
- nothing - it only plots the mesh grid
"""
function plotgrid(mesh::CartesianMesh, ax::PyPlot.PyObject)
    # set the four corners of the rectangular mesh
    ax.set_xlim([-mesh.Lx - mesh.hx / 2, mesh.Lx + mesh.hx / 2])
    ax.set_ylim([-mesh.Ly - mesh.hy / 2, mesh.Ly + mesh.hy / 2])

    # Add rectangle for each cell
    patches = PyPlot.matplotlib.patches.Patch[]

    for i in 1:mesh.NumberOfElts
        vertex_indices = mesh.Connectivity[i, :]
        vertices = mesh.VertexCoor[vertex_indices, :]

        polygon = PyPlot.matplotlib.patches.Polygon(vertices, true)
        push!(patches, polygon)
    end

    plot_prop = PlotProperties()

    # Создаем коллекцию патчей

    p = PyPlot.matplotlib.collections.PatchCollection(
        patches,
        alpha=plot_prop.alpha,
        edgecolor=plot_prop.lineColor,
        linewidth=plot_prop.lineWidth
    )

    ax.add_collection(p)
    ax.axis("equal")

    return nothing
end

"""
    plot_cell_lists(mesh, list; fig=nothing, mycolor="g", mymarker="_", shiftx=0.01, shifty=0.01, annotate_cellName=false, grid=true)

Plot an identifier at the position of each cell in a given list.

Use this function to plot even more than one list and see the difference between them.
You can customize the color, specify the shift of the identifier with respect to the cell center.
You can customize the marker used for the identifier.

# Arguments
- `mesh::CartesianMesh`: mesh object
- `list::Vector{Int}`: a list of int representing the cell names you want to mark
- `fig::Union{PyPlot.Figure, Nothing}=nothing`: a PyPlot figure object. If `nothing`, a new figure is created.
- `mycolor::String="g"`: a valid string that specify the color
- `mymarker::String="_"`: a valid marker string to control the marker
- `shiftx::Float64=0.01`: float representing the amount of shift of the identifier from the cell center. The number represents the percentage of the cell size in x direction.
- `shifty::Float64=0.01`: float representing the amount of shift of the identifier from the cell center. The number represents the percentage of the cell size in y direction.
- `annotate_cellName::Bool=false`: True or False to decide if you want to plot the cell name
- `grid::Bool=true`: True or False to decide if you want to plot the grid

# Returns
- `PyPlot.Figure`: the figure object
"""
function plot_cell_lists(mesh::CartesianMesh, list::Vector{Int}; 
                         fig::Union{PyPlot.Figure, Nothing}=nothing, 
                         mycolor::String="g", mymarker::String="_", 
                         shiftx::Float64=0.01, shifty::Float64=0.01,
                         annotate_cellName::Bool=false, grid::Bool=true)

    local ax
    if fig === nothing
        fig = PyPlot.figure()
        ax = fig.add_subplot(111)
    else
        ax = fig.get_axes()[1]
    end

    x_positions = mesh.CenterCoor[list, 1] .+ mesh.hx .* shiftx
    y_positions = mesh.CenterCoor[list, 2] .+ mesh.hy .* shifty
    
    ax.plot(x_positions, y_positions, marker=mymarker, color=mycolor, linestyle="none")

    if annotate_cellName
        x_center = mesh.CenterCoor[list, 1]
        y_center = mesh.CenterCoor[list, 2]
        for (i, txt) in enumerate(list)
            ax.annotate(string(txt), (x_center[i], y_center[i]))
        end
    end

    if grid
        plotgrid(mesh, ax)
    end
    PyPlot.show()
    return fig
end

"""
    plot_ray_tracing_numpy_results(mesh, x, y, poly, inside)

Plot the results from the function "ray_tracing_numpy"

# Arguments
- `mesh::CartesianMesh`: mesh object
- `x::Vector{Float64}`: an array containing the x coordinates of the points tested
- `y::Vector{Float64}`: an array containing the y coordinates of the points tested
- `poly::Matrix{Float64}`: a matrix containing the x and y coordinates of the polygon
- `inside::Vector{Int}`: inside is a binary vector containing 1 (true) and 0 (false) that results from the function ray_tracing_numpy

# Returns
- nothing - it only plots the mesh grid
"""
function plot_ray_tracing_numpy_results(mesh::CartesianMesh, 
                                      x::Vector{Float64}, 
                                      y::Vector{Float64}, 
                                      poly::Matrix{Float64}, 
                                      inside::Vector{Int})
    fig = PyPlot.figure()
    inside_indices = findall(!iszero, inside)
    if !isempty(inside_indices)
        plt.plot(x[inside_indices], y[inside_indices], marker=".", color="Green", linestyle="none")
    end
    
    outside_indices = findall(iszero, inside)
    
    if !isempty(outside_indices)
        plt.plot(x[outside_indices], y[outside_indices], marker="x", color="Red", linestyle="none")
    end
    
    if size(poly, 1) > 0
        poly_x = [poly[:, 1]; poly[1, 1]]
        poly_y = [poly[:, 2]; poly[1, 2]]
        plt.plot(poly_x, poly_y, marker=".", color="Blue", linestyle="-")
    end
    
    ax = fig.get_axes()[1]
    plotgrid(mesh, ax)
    return nothing
end

"""
    ray_tracing_numpy(x, y, poly)

given a polygon this function tests if each of the points in a given set is inside or outside

The answer is obtained by drawing an horizontal line on the right side of the point

# Arguments
- `x::Vector{Float64}`: an array containing the x coordinates of the points to be tested e.g.: np.asarray([[0.,5.]])
- `y::Vector{Float64}`: an array containing the y coordinates of the points to be tested e.g.: np.asarray([[0.,5.]])
- `poly::Matrix{Float64}`: a matrix containing the x and y coordinated of the poligons e.g.: np.asarray([[-1,-1],[1,-1],[0,1]])

# Returns
- `::Vector{Bool}`: an array of booleans ordered as x
"""
function ray_tracing_numpy(x::Vector{Float64}, y::Vector{Float64}, poly::Matrix{Float64})::Vector{Bool}
    # make the array with the answer at the points (np.bool_ : Boolean(True or False) stored as a byte)
    # assume that the points are all outside (0 is false)
    # Handle scalar inputs
    if isa(x, Number) && isa(y, Number)
        x = [x]
        y = [y]
    end
    
    # make the array with the answer at the points
    # assume that the points are all outside (false)
    inside = falses(length(x))

    # initializing the parameters
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[1, 1], poly[1, 2]
    n = size(poly, 1)
    
    for i in 1:(n+1) # i in [0,1,...,n]
        # modulo returns the first index when the element n is called (starting from 0)
        idx_mod = (i % n == 0) ? n : (i % n)
        p2x, p2y = poly[idx_mod, 1], poly[idx_mod, 2]
        
        # compute the indexes of the points from which an horizontal line might intersect with the segment
        # In Julia: findall для получения индексов, логические операторы &, |, !
        condition1 = (y .> min(p1y, p2y)) .& (y .<= max(p1y, p2y)) .& (x .<= max(p1x, p2x))
        idx_points = findall(condition1) # 1 is True in Julia (1-based indexing)

        # if the front segment is not horizontal compute the x coord of the intersection of an horizontal line and the
        # front only in the case the front is on the right side of the point (this latter requirement is expressed by
        # the fact that we are taking y[idx]
        if p1y != p2y
            y_subset = y[idx_points]
            x_subset = x[idx_points]
            xints_vals = (y_subset .- p1y) .* (p2x - p1x) / (p2y - p1y) .+ p1x #-->this might give problems of tolerance
        end

        # if the segment is vertical switch directly from true to false and vice versa
        if p1x == p2x
            inside[idx_points] .= .!inside[idx_points]
        # if the segment is NOT vertical you have to decide:
        #      p1
        #       \          outside
        #         \
        #           \
        #          *--\----------->
        #               \  *------>
        #     inside      \
        #                   p2
        #
        # if x < xints the point is inside, otherwise it is outside
        elseif !isempty(idx_points)
            if p1y != p2y # Только если предыдущее условие выполнялось
                condition2 = x_subset .<= xints_vals
                idx_final = idx_points[condition2]
            else
                idx_final = Int[]
            end
            inside[idx_final] .= .!inside[idx_final]
        end

        p1x, p1y = p2x, p2y
    end

    return inside
end

"""
    find_indexes_repeatd_elements(list)

This function returns all the indexes of the repeated elements

Example: giving the following list:
      0  1  2  3  4  5  6  7
list=[10,15,33,33,18,22,16,22]
it returns all indexes of repeated elements [2,3,5,7]

# Arguments
- `list::Vector`: list e.g.: [10,15,33,33,18,22,16,22]

# Returns
- `Vector{Int}`: list with all the indexes of the repeated elements
"""
function find_indexes_repeatd_elements(list::Vector)::Vector{Int}
    sort_indexes = sortperm(list)
    sorted_list = list[sort_indexes]
    vals, first_indexes = unique(sorted_list, dims=1, keepdims=true)
    
    unique_vals = unique(list)
    counts = [count(==(val), list) for val in unique_vals]
    first_occurrence_indices = [findfirst(==(val), list) for val in unique_vals]
    
    value_to_indices = Dict{eltype(list), Vector{Int}}()
    
    for (i, val) in enumerate(list)
        if !haskey(value_to_indices, val)
            value_to_indices[val] = Int[]
        end
        push!(value_to_indices[val], i)
    end
    repeated_indices = Int[]
    for (val, indices) in value_to_indices
        if length(indices) > 1
            append!(repeated_indices, indices)
        end
    end
    sort!(repeated_indices)
    return repeated_indices
end

"""
    Point
This class represents the concept of a point
"""
mutable struct Point
    name::String
    x::Float64
    y::Float64
    """
        Point(name, x, y)
    Constructor method
    # Arguments
    - `name::String`: point name
    - `x::Float64`: x coordinate of a point
    - `y::Float64`: y coordinate of a point
    """
    function Point(name::String, x::Float64, y::Float64)
        new(name, x, y)
    end
end

"""
    distance(p1, p2)
Compute the euclidean distance between the two points
# Arguments
- `p1::Point`: object of type Point - first point
- `p2::Point`: object of type Point - second point
# Returns
- `Float64`: euclidean distance between the points
"""
function distance(p1::Point, p2::Point)::Float64
    return hypot(p2.x - p1.x, p2.y - p1.y)
end

"""
    copute_area_of_a_polygon(x, y)

Use the Shoelace formula (Gauss area formula or surveyor's formula) to compute the area of a polygon.

# Arguments
- `x::Vector{Float64}`: x coordinates of the points defining the polygon (closed front) e.g.: [0.0,1.0,0.0] for a triangle
- `y::Vector{Float64}`: y coordinates of the points defining the polygon (closed front) e.g.: [0.0,0.0,1.0] for a triangle

# Returns
- `Float64`: float representing the area of the polygon
"""
function copute_area_of_a_polygon(x::Vector{Float64}, y::Vector{Float64})::Float64

    if length(x) != length(y)
        error("FRONT RECONSTRUCTION ERROR: bad coordinate size.")
    end
    n::Int64 = length(x)

    area::Float64 = abs(dot(x[1:n-1], y[2:n]) + x[n] * y[1] - dot(x[2:n], y[1:n-1]) - x[1] * y[n]) / 2.0
    return area
end

"""
    pointtolinedistance(x0, x1, x2, y0, y1, y2)

Compute the minimum euclidean distance from a point of coordinates (x0,y0) to a the line passing through 2 points.
The function works only for planar problems.

# Arguments
- `x0::Float64`: float representing the x coordinate of the point
- `x1::Float64`: float representing the x coordinate of the first point contained by the line
- `x2::Float64`: float representing the x coordinate of the second point contained by the line
- `y0::Float64`: float representing the y coordinate of the point
- `y1::Float64`: float representing the y coordinate of the first point contained by the line
- `y2::Float64`: float representing the y coordinate of the second point contained by the line

# Returns
- `Float64`: float representing the shortest euclidean distance
"""
function pointtolinedistance(x0::Float64, x1::Float64, x2::Float64, y0::Float64, y1::Float64, y2::Float64)::Float64
    if x1 == x2 && y1 == y2
        error("FRONT RECONSTRUCTION ERROR: line defined by two coincident points")
    else
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / sqrt((-x1 + x2)^2 + (-y1 + y2)^2)
    end
end

"""
    elements(typeindex, nodeindex, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID)

This function handles two cases: a and b. It returns respectively:
 - in case a), the names of the 2 elements bounded by the EDGE where the node lies
 - in case b), the names of the 4 elements having the node as a VERTEX

# Arguments
- `typeindex::Vector{Int64}`: set of booleans that tells if 0 that the corresponding node is on the edge of a cell otherwise it is coincident to a vertex
- `nodeindex::Int64`: the index, in edgeORvertexID, of the node that we are selecting
- `connectivityedgeselem::Vector{Vector{Int64}}`: see mesh.Connectivityedgeselem
- `Connectivitynodeselem::Vector{Vector{Int64}}`: see mesh.Connectivitynodeselem
- `edgeORvertexID::Vector{Int64}`: set of IDs with the meaning of vertex ID or edge ID

# Returns
- `Vector{Int64}`: set of cell names
"""
function elements(typeindex::Vector{Int64}, nodeindex::Int64, connectivityedgeselem::Vector{Vector{Int64}}, 
                 Connectivitynodeselem::Vector{Vector{Int64}}, edgeORvertexID::Vector{Int64})::Vector{Int64}
    
    # CASE a)
    if typeindex[nodeindex] == 0  # the node is one the edge of a cell
        cellOfNodei = connectivityedgeselem[edgeORvertexID[nodeindex]]
    # CASE b)
    else # the node is one vertex of a cell
        cellOfNodei = Connectivitynodeselem[edgeORvertexID[nodeindex]]
    end
    return cellOfNodei
end

"""
    findcommon(nodeindex0, nodeindex1, typeindex, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID)

Given two points we return the cells that are in common between them

# Arguments
- `nodeindex0::Int64`: position of the node 0 inside the list of the found intersections that defines the front
- `nodeindex1::Int64`: position of the node 1 inside the list of the found intersections that defines the front
- `typeindex::Vector{Int64}`: array that specify if a node at the front is an existing vertex or an intersection with the edge
- `connectivityedgeselem::Vector{Vector{Int64}}`: given an edge number, it will return all the elements that have the given edge
- `Connectivitynodeselem::Vector{Vector{Int64}}`: given a node number, it will return all the elements that have the given node
- `edgeORvertexID::Vector{Int64}`: list that contains for each node at the front the number of the vertex or of the edge where it lies

# Returns
- `Vector{Int64}`: list of elements
"""
function findcommon(nodeindex0::Int64, nodeindex1::Int64, typeindex::Vector{Int64}, 
                   connectivityedgeselem::Vector{Vector{Int64}}, Connectivitynodeselem::Vector{Vector{Int64}}, 
                   edgeORvertexID::Vector{Int64})::Vector{Int64}
    
    cellOfNodei = elements(typeindex, nodeindex0, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID)
    cellOfNodeip1 = elements(typeindex, nodeindex1, connectivityedgeselem, Connectivitynodeselem, edgeORvertexID)
    diff = setdiff(cellOfNodei, cellOfNodeip1)  # Return the unique values in cellOfNodei that are not in cellOfNodeip1.
    common = setdiff(cellOfNodei, diff)  # Return the unique values in cellOfNodei that are not in diff.
    return common
end

"""
    filltable(nodeVScommonelementtable, nodeindex, common, sgndDist_k, column)

we define a node as the intersection between the zero of the level set and the grid made of elements
given two elements, this function returns the common element/elements between two nodes that are supposed to
be one after the other along the 

# Arguments
- `nodeVScommonelementtable::Matrix{Int64}`: 
- `nodeindex::Int64`: 
- `common::Vector{Int64}`: 
- `sgndDist_k::Vector{Float64}`: 
- `column::Int64`: 

# Returns
- `Tuple{Matrix{Int64}, Bool}`: 
"""
function filltable(nodeVScommonelementtable::Matrix{Int64}, nodeindex::Int64, common::Vector{Int64}, sgndDist_k::Vector{Float64}, column::Int64)::Tuple{Matrix{Int64}, Bool}
    # we define a node as the intersection between the zero of the level set and the grid made of elements
    # given two elements, this function returns the common element/elements between two nodes that are supposed to
    # be one after the other along the 

    if length(common) == 1
        nodeVScommonelementtable[nodeindex, column] = common[1]
        exitstatus = true
    elseif length(common) > 1
        """
        situations with two common elements:
           |      |                  |      |           |      |
        ___|______|____           ___*======*====    ___|_*____*____
           |      |                 ||      |           |/     |\        
        ___*______*____           ___*______|___     ___/______|_\__
           |      |                 ||      |          /|      |  \  
        ___|______|____           __||______|____    _/_|______|___\___
           |      |                 ||      |           |      |
        In this situation take the i with LS<0 as tip
        (...if you choose LS>0 as tip you will not find zero vertexes then...)
        """
        nodeVScommonelementtable[nodeindex, column] = common[argmax(sgndDist_k[common])]
        #nodeVScommonelementtable[nodeindex,column] = common[argmin(sgndDist_k[common])]
        exitstatus = true
    elseif length(common) == 0
        #raise SystemExit('FRONT RECONSTRUCTION ERROR: two consecutive nodes does not belongs to a common cell')
        exitstatus = false
    end
    return nodeVScommonelementtable, exitstatus
end

"""
    ISinsideFracture(i, mesh, sgndDist_k)

    you are in cell i
    you want to know if points 0,1,2,3 are inside or outside of the fracture
    -extrapolate the level set at those points by taking the level set (LS) at the center of the neighbors cells
    -if at the point the LS is < 0 then the point is inside
      _   _   _   _   _   _
    | _ | _ | _ | _ | _ | _ |
    | _ | _ | _ | _ | _ | _ |
    | _ | e | a | f | _ | _ |
    | _ | _ 3 _ 2 _ | _ | _ |
    | _ | d | i | b | _ | _ |
    | _ | _ 0 _ 1 _ | _ | _ |
    | _ | h | c | g | _ | _ |
    | _ | _ | _ | _ | _ | _ |
"""
function ISinsideFracture(i::Int64, mesh, sgndDist_k::Vector{Float64})::Vector{Bool}
    #                         0     1      2      3
    #       NeiElements[i]->[left, right, bottom, up]
    [left_elem, right_elem, bottom_elem, top_elem] = [1, 2, 3, 4]

    a = mesh.NeiElements[i, top_elem]
    b = mesh.NeiElements[i, right_elem]
    c = mesh.NeiElements[i, bottom_elem]
    d = mesh.NeiElements[i, left_elem]
    e = mesh.NeiElements[d, top_elem]
    f = mesh.NeiElements[b, top_elem]
    g = mesh.NeiElements[b, bottom_elem]
    h = mesh.NeiElements[d, bottom_elem]

    hcid_mean = mean(sgndDist_k[[h, c, i, d]])
    cgbi_mean = mean(sgndDist_k[[c, g, b, i]])
    ibfa_mean = mean(sgndDist_k[[i, b, f, a]])
    diae_mean = mean(sgndDist_k[[d, i, a, e]])
    answer_on_vertexes = [hcid_mean<0, cgbi_mean<0, ibfa_mean<0, diae_mean<0]
    return answer_on_vertexes
end

"""
    findangle(x1, y1, x2, y2, x0, y0, mac_precision)

Compute the angle with respect to the horizontal direction between the segment from a point of coordinates (x0,y0)
and orthogonal to a the line passing through 2 points. The function works only for planar problems.

Args:
:param x0: coordinate x point
:param y0: coordinate y first
:param x1: coordinate x first point that defines the line
:param y1: coordinate y first point that defines the line
:param x2: coordinate x second point that defines the line
:param y2: coordinate y second point that defines the line

Returns:
:return: angle, xintersections, yintersections

"""
function findangle(x1::Float64, y1::Float64, x2::Float64, y2::Float64, x0::Float64, y0::Float64, mac_precision::Float64)::Tuple{Float64, Float64, Float64}
    # ------ only for plotting purposes -----
    dist_p1p2 = distance(Point(0,x1,y1),Point(0,x2,y2))
    if abs(x2 - x1)/dist_p1p2 < mac_precision/1000  # the front is a vertical line
        x = x2
        y = y0
        angle = 0.0
    elseif abs(y2 - y1)/dist_p1p2 < mac_precision/1000  # the front is an horizontal line
        angle = π/2
        x = x0
        y = y2
    else
        # m and q1 are the coefficients of the line defined by (x1,y1) and (x2,y2): y = m * x + q1
        # q2 is the coefficients of the line defined by (x,y) and (x0,y0): y = -1/m * x + q2
        m = (y2 - y1) / (x2 - x1)
        q1 = y2 - m * x2
        q2 = y0 + x0 / m
        x = (q2 - q1) * m / (m * m + 1)
        y = m * x + q1
        # angle = atan(abs((y-y0))/abs((x-x0))) naive way of computing the angle
    # ---------------------------------------------------

    # here we use directly points 1 and 2 to find the angle instead of finding the intersection between the normal from a point to the
    # front and then computing the angle

    if y2!=y1
        dx_over_dy =  abs((x2-x1))/abs((y2-y1))
        return atan(dx_over_dy), x, y
    else
        return π/2, x0, y2
    end
end

function plot_final_reconstruction(
    mesh::Any,
    list_of_xintersection::Vector{Vector{Float64}},
    list_of_yintersection::Vector{Vector{Float64}},
    anularegion::Vector{Int64},
    sgndDist_k::Vector{Float64},
    newRibbon::Vector{Int64},
    listofTIPcells::Vector{Int64},
    list_of_xintersectionsfromzerovertex::Vector{Vector{Float64}},
    list_of_yintersectionsfromzerovertex::Vector{Vector{Float64}},
    list_of_vertexID::Vector{Vector{Int64}},
    oldRibbon::Vector{Int64}
)
    # fig = None
    # if fig is None:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     A = np.full(mesh.NumberOfElts, np.nan)
    #     A[anularegion] = sgndDist_k[anularegion]
    #     from visualization import plot_fracture_variable_as_image
    #     fig = plot_fracture_variable_as_image(A, mesh, fig=fig)
    # else:
    #     ax = fig.get_axes()[0]
    # find positive non ribbon
    # nonRibbon = np.setdiff1d(anularegion, Ribbon)
    # Positive_nonRibbon = nonRibbon[np.where(sgndDist_k[nonRibbon] > 0)[0]]
    # plt.plot(mesh.CenterCoor[Ribbon, 0], mesh.CenterCoor[Ribbon, 1], ".", marker="_", color='g')
    # plt.plot(mesh.CenterCoor[Positive_nonRibbon, 0], mesh.CenterCoor[Positive_nonRibbon, 1], ".", marker="+",
    #          color='r')
    # Negative_nonRibbon = np.setdiff1d(nonRibbon, Positive_nonRibbon)
    # if Negative_nonRibbon.size > 0:  # plot them
    #     plt.plot(mesh.CenterCoor[Negative_nonRibbon, 0], mesh.CenterCoor[Negative_nonRibbon, 1], ".",
    #              marker="_", color='b')
    
    A::Vector{Float64} = fill(NaN, mesh.NumberOfElts)
    A[anularegion] = sgndDist_k[anularegion]
    # from visualization import plot_fracture_variable_as_image
    # figure = plot_fracture_variable_as_image(A, mesh, fig=fig)
    # ax = figure.get_axes()[0]
    for iiii in 1:length(list_of_vertexID)
        vertexID::Vector{Int64} = list_of_vertexID[iiii]
        xintersectionsfromzerovertex::Vector{Float64} = list_of_xintersectionsfromzerovertex[iiii]
        yintersectionsfromzerovertex::Vector{Float64} = list_of_yintersectionsfromzerovertex[iiii]
        xintersection::Vector{Float64} = list_of_xintersection[iiii]
        yintersection::Vector{Float64} = list_of_yintersection[iiii]
        xtemp::Vector{Float64} = copy(xintersection)
        ytemp::Vector{Float64} = copy(yintersection)
        push!(xtemp, xtemp[1]) # close the front
        push!(ytemp, ytemp[1]) # close the front
        # plt.plot(mesh.CenterCoor[listofTIPcells, 0], mesh.VertexCoor[mesh.Connectivity[Ribbon,0],1], '.',color='violet')
        PyPlot.plot(xtemp, ytemp, "-o")
        n::Int64 = length(xintersectionsfromzerovertex)
        for i in 1:n
            # very nice peace of code below: I am randoimly specific positions from the matrix
            next_i_idx::Int64 = mod1(i + 1, n)
            PyPlot.plot([mesh.VertexCoor[vertexID[next_i_idx], 1], xintersectionsfromzerovertex[i]], 
                       [mesh.VertexCoor[vertexID[next_i_idx], 2], yintersectionsfromzerovertex[i]], "-r")
        end
        # plt.plot(mesh.VertexCoor[vertexID, 0], mesh.VertexCoor[vertexID, 1], '.', color='red')
        PyPlot.plot(mesh.VertexCoor[vertexID, 1], mesh.VertexCoor[vertexID, 2], ".", color="red")
        # plt.plot(xintersectionsfromzerovertex, yintersectionsfromzerovertex, '.', color='red')
        PyPlot.plot(xintersectionsfromzerovertex, yintersectionsfromzerovertex, ".", color="red")
    end
    # plt.plot(mesh.CenterCoor[newRibbon,0], mesh.CenterCoor[newRibbon,1], '.',color='orange')
    PyPlot.plot(mesh.CenterCoor[newRibbon, 1], mesh.CenterCoor[newRibbon, 2], ".", color="orange")
    # plt.plot(mesh.CenterCoor[oldRibbon,0]*1.05, mesh.CenterCoor[oldRibbon,1], '.',color='green')
    PyPlot.plot(mesh.CenterCoor[oldRibbon, 1] * 1.05, mesh.CenterCoor[oldRibbon, 2], ".", color="green")
    # plt.plot(mesh.CenterCoor[listofTIPcells, 0] + mesh.hx / 10, mesh.CenterCoor[listofTIPcells, 1] + mesh.hy / 10, '.', color='blue')
    PyPlot.plot(mesh.CenterCoor[listofTIPcells, 1] + mesh.hx / 10, mesh.CenterCoor[listofTIPcells, 2] + mesh.hy / 10, ".", color="blue")
    # plt.show()
    # return fig
    return nothing
end

function plot_xy_points(anularegion, mesh, sgndDist_k, Ribbon, x, y, fig=nothing, annotate_cellName=false, annotate_edgeName=false, annotatePoints=true, grid=true, oldfront=nothing)
        #fig = nothing
        if fig === nothing
            fig = plt.figure()
            ax = fig.add_subplot(111)
            A = fill(NaN, mesh.NumberOfElts)
            A[anularegion] = sgndDist_k[anularegion]
            # from visualization import plot_fracture_variable_as_image
            # fig = plot_fracture_variable_as_image(A, mesh, fig=fig)
        else
            ax = fig.get_axes()[1]
        end
        # find positive non ribbon
        nonRibbon = setdiff(anularegion, Ribbon)
        Positive_nonRibbon = nonRibbon[findall(sgndDist_k[nonRibbon] .> 0)]
        plt.plot(mesh.CenterCoor[Ribbon, 1], mesh.CenterCoor[Ribbon, 2], ".", marker="_", color="green")
        plt.plot(mesh.CenterCoor[Positive_nonRibbon, 1], mesh.CenterCoor[Positive_nonRibbon, 2], ".", marker="+",
                 color="red")
        Negative_nonRibbon = setdiff(nonRibbon, Positive_nonRibbon)
        if length(Negative_nonRibbon) > 0  # plot them
            plt.plot(mesh.CenterCoor[Negative_nonRibbon, 1], mesh.CenterCoor[Negative_nonRibbon, 2], ".",
                     marker="_", color="blue")
        end
        plt.plot(convert(Array, x), convert(Array, y), ".-", color="red")

        if annotate_cellName
            x_center = mesh.CenterCoor[anularegion, 1]
            y_center = mesh.CenterCoor[anularegion, 2]
            for i in 1:length(anularegion)
                txt = anularegion[i]
                ax.annotate(txt, (x_center[i], y_center[i]))
            end
        end

        if annotatePoints
            points = 1:length(x)
            offset = sqrt(mesh.hx^2 + mesh.hy^2) / 15
            for i in 1:length(points)
                txt = points[i]
                ax.annotate(txt,
                            xy=(x[i], y[i]),
                            xycoords="data",
                            xytext=(x[i] + offset, y[i] + offset),
                            textcoords="data",
                            arrowprops=Dict("arrowstyle" => "->",
                            "connectionstyle" => "arc3"))
            end
        end

        if annotate_edgeName
            edges_in_anularegion = unique(vcat(mesh.Connectivityelemedges[anularegion]...))
            x_center = Float64[]
            y_center = Float64[]
            for i in edges_in_anularegion
                node_0 = mesh.Connectivityedgesnodes[i][1]
                node_1 = mesh.Connectivityedgesnodes[i][2]
                push!(x_center, mean([mesh.VertexCoor[node_0, 1], mesh.VertexCoor[node_1, 1]]))
                push!(y_center, mean([mesh.VertexCoor[node_0, 2], mesh.VertexCoor[node_1, 2]]))
            end
            for i in 1:length(edges_in_anularegion)
                txt = edges_in_anularegion[i]
                ax.annotate(txt, (x_center[i], y_center[i]))
            end
        end

        if grid
            plotgrid(mesh, ax)
        end

        if oldfront !== nothing
            n = size(oldfront, 1)
            for i in 1:n
                plt.plot([oldfront[i, 1], oldfront[i, 3]],
                         [oldfront[i, 2], oldfront[i, 4]], "-g")
            end
        end
        plt.show()
        return fig
end


function plot_two_fronts(mesh, newfront=nothing, oldfront=nothing, fig=nothing, grid=true, cells=nothing)
    # fig = nothing
    if fig === nothing
        fig = plt.figure()
        ax = fig.add_subplot(111)
        A = fill(0.0, mesh.NumberOfElts)
        # from visualization import plot_fracture_variable_as_image
        # fig = plot_fracture_variable_as_image(A, mesh, fig=fig)
    else
        ax = fig.get_axes()[1]
    end

    if grid
        plotgrid(mesh, ax)
    end

    if oldfront !== nothing
        n = size(oldfront, 1)
        for i in 1:n
            plt.plot([oldfront[i, 1], oldfront[i, 3]],
                     [oldfront[i, 2], oldfront[i, 4]], "-g")
        end
    end

    if newfront !== nothing
        n = size(newfront, 1)
        for i in 1:n
            plt.plot([newfront[i, 1], newfront[i, 3]],
                     [newfront[i, 2], newfront[i, 4]], "-b")
        end
    end

    if cells !== nothing
        plt.plot(mesh.CenterCoor[cells, 1], mesh.CenterCoor[cells, 2], ".", marker="_", color="green")
    end

    plt.show()
    return fig
end

function plot_cells(anularegion, mesh, sgndDist_k, Ribbon, list, fig=nothing, annotate_cellName=false, grid=true)
    if fig === nothing
        fig = plt.figure()
        ax = fig.add_subplot(111)
        A = fill(NaN, mesh.NumberOfElts)
        A[anularegion] = sgndDist_k[anularegion]
        # from visualization import plot_fracture_variable_as_image

        # fig = plot_fracture_variable_as_image(A, mesh, fig=fig)
    else
        ax = fig.get_axes()[1]
    end


    # find positive non ribbon
    nonRibbon = setdiff(anularegion, Ribbon)
    Positive_nonRibbon = nonRibbon[findall(sgndDist_k[nonRibbon] .> 0)]
    plt.plot(mesh.CenterCoor[Ribbon, 1], mesh.CenterCoor[Ribbon, 2], ".", marker="_", color="green")
    plt.plot(mesh.CenterCoor[Positive_nonRibbon, 1], mesh.CenterCoor[Positive_nonRibbon, 2], ".", marker="+", color="red")
    Negative_nonRibbon = setdiff(nonRibbon, Positive_nonRibbon)
    if length(Negative_nonRibbon) > 0  # plot them
        plt.plot(mesh.CenterCoor[Negative_nonRibbon, 1], mesh.CenterCoor[Negative_nonRibbon, 2], ".",
                 marker="_", color="blue")
    end

    plt.plot(mesh.CenterCoor[list, 1] + mesh.hx * 0.1,
             mesh.CenterCoor[list, 2] + mesh.hy * 0.1, ".", color="yellow")
    if annotate_cellName
        x_center = mesh.CenterCoor[anularegion, 1]
        y_center = mesh.CenterCoor[anularegion, 2]
        for i in 1:length(anularegion)
            txt = anularegion[i]
            ax.annotate(txt, (x_center[i], y_center[i]))
        end
    end
    if grid
        plotgrid(mesh, ax)
    end
    plt.show()
    return fig
end

function get_fictitius_cell_type(LS)
    number_of_negative_cells = sum(Int.(LS .< 0))
    if number_of_negative_cells == 1
        return 3
    elseif number_of_negative_cells == 3
        return 4
    else
        if (LS[1] > 0 && LS[3] > 0) || (LS[1] < 0 && LS[3] < 0)
            return 2
        else
            return 1
        end
    end
end

function get_fictitius_cell_specific_names(index_to_output, fictitius_cells, NeiElements)
    """
     _ _ _ _ _ _
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|
    |_|_|a|b|_|_|
    |_|_|i|c|_|_|
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|

    for understanding the first operation think that you are in the cell i --> take the cell a,b,c
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                              0     1      2      3
    """
    [left_elem, right_elem, bottom_elem, top_elem] = [1, 2, 3, 4]

    a = NeiElements[fictitius_cells, top_elem]
    c = NeiElements[fictitius_cells, right_elem]
    b = NeiElements[fictitius_cells + 1, top_elem]

    if index_to_output == "left right"
        m0 = hcat(a, b)
        m1 = hcat(fictitius_cells, c)

    elseif index_to_output == "bottom top"
        m0 = hcat(c, b)
        m1 = hcat(fictitius_cells, a)
    end
    return m0, m1
end

function get_fictitius_cell_names(index_to_output, fictitius_cells, NeiElements)
    """
     _ _ _ _ _ _
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|
    |_|_|a|b|_|_|
    |_|_|i|c|_|_|
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|

    for understanding the first operation think that you are in the cell i --> take the cell a,b,c
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                              0     1      2      3
    """
    [left_elem, right_elem, bottom_elem, top_elem] = [1, 2, 3, 4]


    a = NeiElements[fictitius_cells, top_elem]
    c = NeiElements[fictitius_cells, right_elem]
    b = NeiElements[fictitius_cells + 1, top_elem]
    full_matrix = hcat(fictitius_cells, c, b, a)
    to_return = Int64[]
    for i in 1:length(index_to_output)
        push!(to_return, full_matrix[i, index_to_output[i]])
    end
    return convert(Array, [to_return])
end

function get_fictitius_cell_all_names(fictitius_cells, NeiElements)
    """
     _ _ _ _ _ _
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|
    |_|_|a|b|_|_|
    |_|_|i|c|_|_|
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|

    for understanding the first operation think that you are in the cell i --> take the cell a,b,c
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                              0     1      2      3
    """
    [left_elem, right_elem, bottom_elem, top_elem] = [1, 2, 3, 4]


    a = NeiElements[fictitius_cells, top_elem]
    c = NeiElements[fictitius_cells, right_elem]
    b = NeiElements[c, top_elem]

    return hcat(fictitius_cells, c, b, a)
end

function get_LS_on_i_fictitius_cell(columns_to_output, fictitius_cells, NeiElements, sgndDist_k)
    """
     _ _ _ _ _ _
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|
    |_|_|a|b|_|_|
    |_|_|i|c|_|_|
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|

    for understanding the first operation think that you are in the cell i --> take the cell a,b,c
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                              0     1      2      3
    """
    [left_elem, right_elem, bottom_elem, top_elem] = [1, 2, 3, 4]

    if columns_to_output == "icba" || columns_to_output == "iabc" || columns_to_output == "ibca"
        a = NeiElements[fictitius_cells, top_elem]
        c = NeiElements[fictitius_cells, right_elem]
        # when close to the boundary c might be the fictitious cell itself
        b = NeiElements[c, top_elem]

        #creating a matrix with 4 columns: i_cells, c, b, a substituting the value of the signed distance
        return sgndDist_k[hcat(fictitius_cells, c, b, a)]

    elseif columns_to_output == "ab"
        a = NeiElements[fictitius_cells, top_elem]
        c = NeiElements[fictitius_cells, right_elem]
        # when close to the boundary c might be the fictitious cell itself
        b = NeiElements[c, top_elem]

        #creating a matrix with 2 columns: b, a substituting the value of the signed distance
        return sgndDist_k[hcat(a, b)]

    elseif columns_to_output == "i"

        # creating a matrix with 1 columns: i_cells, substituting the value of the signed distance
        return sgndDist_k[fictitius_cells]
    end
end

function find_fictitius_cells(anularegion, NeiElements, sgndDist_k)
    """
    This function has vectorized operations.
    This function returns a list of "valid" "fictitius cells".
    A fictitius cell is made of 4 cells of the mesh e.g. cells i,a,b,c in the mesh below.
    A fictitius cell is represented by the name of the element in position i in the mesh below
    A valid fictitius cell is a cell where at least one vertex has Level Set<0 and at least one has LS>0
    A valid fictitius cell is important because we know the front is passing through it.
    The front will always enter and exit the fictitius cell from two different edges.
    The front can't exit the fictitious cell from a vertex because we set LS -(machine precision) where it was 0.

     _ _ _ _ _ _
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|
    |_|_|a|b|_|_|
    |_|_|i|c|_|_|
    |_|_|_|_|_|_|
    |_|_|_|_|_|_|

    for understanding the first operation think that you are in the cell i --> take the cell a,b,c
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                              0     1      2      3
    """
    # log = logging.getLogger('PyFrac.continuous_front_reconstruction')
    LS = get_LS_on_i_fictitius_cell("icba", anularegion, NeiElements, sgndDist_k)

    """
    Explanation
    
    the following line:
    i_indexes_of_fictitius_cells=np.where(np.column_stack((np.all(LS > 0.,axis=1), np.all(LS < 0.,axis=1))).sum(axis=1) == 0)[0]
    
    is equivalent to:
    1-    condition_1=np.all(LS > 0.,axis=1)  
    2-    condition_2=np.all(LS < 0.,axis=1)
    3-    conditions_1_and_2=np.column_stack((condition_1, condition_2))
    4-    false_for_fictitius_cells=conditions_1_and_2.sum(axis=1)
    5-    i_indexes_of_fictitius_cells=np.where(false_for_fictitius_cells == 0)[0]
    
    that means
    1-   create a vector with True if in a row of LS alla the values are > 0  
    2-   create a vector with True if in a row of LS alla the values are < 0
    3-   create a matrix with the columns defined by the 2 vectors computed above
    4-   for each row sum the True/False of the first column with True/False of the second.
         the possibilities are:
         False + False = False (0)
         False + True  = True  (1)
         True + False  = True  (1)
    5-   Find the indexes of False values, i.e. of the i cells in each valid fictitius cell
    
    NOTE: I am expectin non empty list of fictitius cells        
    """
    all_positive = all.(eachrow(LS .> 0.0))
    all_negative = all.(eachrow(LS .< 0.0))
    conditions_matrix = hcat(all_positive, all_negative)
    false_for_fictitius_cells = sum(conditions_matrix, dims=2)
    i_indexes_of_fictitius_cells = findall(vec(false_for_fictitius_cells) .== 0)

    try
        if length(i_indexes_of_fictitius_cells) < 1
            error("FRONT RECONSTRUCTION ERROR: The front does not exist")
        end
    catch e
        # log.error("The front does not exist")
        println("Error: The front does not exist")
    end

    if any(vec(LS[i_indexes_of_fictitius_cells, :]) .> 10.0^40)
        exitstatus = true
        return exitstatus, nothing, nothing, nothing, nothing
    else
        exitstatus = false

        """
            Whe define the fictitius cell types:
        
            type 1        |   type 2        |    type 3       |    type 4
            2(+) & 2(-)   |   2(+) & 2(-)   |    3(+) & 1(-)  |    3(-) & 1(+)  
            + ------ -    |   + ------ -    |    + ------ -   |    - ------ +         
            |        |    |   |        |    |    |        |   |    |        |
            |        |    |   |        |    |    |        |   |    |        |
            + ------ -    |   - ------ +    |    + ------ +   |    - ------ -
            
            With the following lines we want to find the cells of type number 2
         
                        LS_i=LS[i_indexes_of_fictitius_cells,0]
                        LS_c=LS[i_indexes_of_fictitius_cells,1]
                        LS_b=LS[i_indexes_of_fictitius_cells,2]
                        LS_a=LS[i_indexes_of_fictitius_cells,3]
                        
                        LS_i_times_LS_c=np.prod([LS_i, LS_c], axis=0)
                        LS_b_times_LS_a=np.prod([LS_b, LS_a], axis=0)
                        LS_a_times_LS_c=np.prod([LS_a, LS_c], axis=0)
                        LS_b_times_LS_i=np.prod([LS_b,LS_i], axis=0)
                    
                        i_indexes_of_TYPE_2_cells = i_indexes_of_fictitius_cells[
                            np.where((LS_i_times_LS_c < 0. +
                                      LS_b_times_LS_a < 0. +
                                      LS_a_times_LS_c > 0. +
                                      LS_b_times_LS_i > 0.) == 4)[0]]
                      
            conceptually we are looking for the cases (if they exist) where two front of the same fracture or two different 
            fractures are crossing the same fictitius cell. We could call these fictitius cells as "double_front_fictitius_cells"
            but we will always coalesce the fractures in these situation, even when they are not.
            We can identify these situations considering the sign of the level set at the vertexes of the fictitius cell
            
            i  c    b  a
            +  -    +  -    is desidered 
            -  +    -  +    is desidered 
            -  +    +  -    is not desidered
            +  -    -  +    is not desidered
            +  +    -  -    is not desidered
            -  -    +  +    is not desidered
            -  -    -  -    is not possible
            +  +    +  +    is not possible
            all the rest    is not desidered
            
            the product between columns should give the following signs:
            
            i*c  b*a  a*c  b*i
             -    -    +    +    is desidered 
             
            some tests are performed on the results: 
            
            i*c  b*a  a*c  b*i
            <0?  <0?  >0?  >0?
             1    1    1    1    True=1, is desidered 
             
            we can summ all the "Trues" and check if the results is a sharp 4.
                 
            """

        LS_i = LS[i_indexes_of_fictitius_cells, 1]
        LS_c = LS[i_indexes_of_fictitius_cells, 2]
        LS_b = LS[i_indexes_of_fictitius_cells, 3]
        LS_a = LS[i_indexes_of_fictitius_cells, 4]

        LS_i_times_LS_c = LS_i .* LS_c
        LS_b_times_LS_a = LS_b .* LS_a
        LS_a_times_LS_c = LS_a .* LS_c
        LS_b_times_LS_i = LS_b .* LS_i

        condition1 = Int.(LS_i_times_LS_c .< 0.0)
        condition2 = Int.(LS_b_times_LS_a .< 0.0)
        condition3 = Int.(LS_a_times_LS_c .> 0.0)
        condition4 = Int.(LS_b_times_LS_i .> 0.0)
        
        sums = condition1 + condition2 + condition3 + condition4
        i_indexes_of_TYPE_2_cells = i_indexes_of_fictitius_cells[findall(sums .== 4)]

        """
        Whe define the fictitius cell types:
        
        type 1        |   type 2        |    type 3       |    type 4
        2(+) & 2(-)   |   2(+) & 2(-)   |    3(+) & 1(-)  |    3(-) & 1(+)  
        + ------ -    |   + ------ -    |    + ------ -   |    - ------ +         
        |        |    |   |        |    |    |        |   |    |        |
        |        |    |   |        |    |    |        |   |    |        |
        + ------ -    |   - ------ +    |    + ------ +   |    - ------ -
        
        cell type 2 has been recognized just above.
        Now we want to find types 1 and 3
        Remember that:
        type 1 OR 4: 2(+) & 2(-)               ----> the product of the LS on all the vertexes will be +
        type 3: 3(+) & 1(-)  OR  4(-) & 1(+)   ----> the product of the LS on all the vertexes will be -
        
        Use this last result to distinguish between i_indexes_of_TYPES_3_and_4_cells and i_indexes_of_TYPES_1_and_2_cells
         
        """
        product_check = LS_i_times_LS_c .* LS_b_times_LS_a
        i_indexes_of_TYPES_3_and_4_cells_IN_i_indexes_of_fictitius_cells = findall(product_check .< 0.0)
        i_indexes_of_TYPES_3_and_4_cells = i_indexes_of_fictitius_cells[i_indexes_of_TYPES_3_and_4_cells_IN_i_indexes_of_fictitius_cells]
        i_indexes_of_TYPES_1_and_2_cells = setdiff(i_indexes_of_fictitius_cells, i_indexes_of_TYPES_3_and_4_cells)
        i_indexes_of_TYPE_1_cells = setdiff(i_indexes_of_TYPES_1_and_2_cells, i_indexes_of_TYPE_2_cells)

        condition_type3 = Int.(LS_i[i_indexes_of_TYPES_3_and_4_cells_IN_i_indexes_of_fictitius_cells] .> 0.0) +
                         Int.(LS_c[i_indexes_of_TYPES_3_and_4_cells_IN_i_indexes_of_fictitius_cells] .> 0.0) +
                         Int.(LS_b[i_indexes_of_TYPES_3_and_4_cells_IN_i_indexes_of_fictitius_cells] .> 0.0)
        
        i_indexes_of_TYPE_3_cells_temp = findall(condition_type3 .> 1)
        i_indexes_of_TYPE_3_cells = i_indexes_of_TYPES_3_and_4_cells[i_indexes_of_TYPE_3_cells_temp]
        i_indexes_of_TYPE_4_cells = setdiff(i_indexes_of_TYPES_3_and_4_cells, i_indexes_of_TYPE_3_cells)

        i_1_2_3_4_FC_names = vcat(anularegion[i_indexes_of_TYPE_1_cells], 
                                  anularegion[i_indexes_of_TYPE_2_cells],
                                  anularegion[i_indexes_of_TYPE_3_cells], 
                                  anularegion[i_indexes_of_TYPE_4_cells])

        # the following is a test that can be removed for speed
        # try:
        #     if np.setdiff1d(anularegion[i_indexes_of_fictitius_cells],np.concatenate((anularegion[i_indexes_of_TYPE_1_cells], anularegion[i_indexes_of_TYPE_2_cells], anularegion[i_indexes_of_TYPE_3_cells], anularegion[i_indexes_of_TYPE_4_cells]))).size > 0:
        #         raise Exception('FRONT RECONSTRUCTION ERROR: this function has an error')
        # except RuntimeError:
        #     log.error("FRONT RECONSTRUCTION ERROR: this function has an error")

        # make dictionaries:
        i_1_2_3_4_FC_type = Dict(string.(anularegion[i_indexes_of_TYPE_1_cells]) .=> ones(Int, length(i_indexes_of_TYPE_1_cells)))
        merge!(i_1_2_3_4_FC_type, Dict(string.(anularegion[i_indexes_of_TYPE_2_cells]) .=> fill(2, length(i_indexes_of_TYPE_2_cells))))
        merge!(i_1_2_3_4_FC_type, Dict(string.(anularegion[i_indexes_of_TYPE_3_cells]) .=> fill(3, length(i_indexes_of_TYPE_3_cells))))
        merge!(i_1_2_3_4_FC_type, Dict(string.(anularegion[i_indexes_of_TYPE_4_cells]) .=> fill(4, length(i_indexes_of_TYPE_4_cells))))

        dict_FC_names = Dict(string.(i_1_2_3_4_FC_names) .=> i_1_2_3_4_FC_names)

        return exitstatus, i_1_2_3_4_FC_names, length(i_indexes_of_TYPE_2_cells), i_1_2_3_4_FC_type, dict_FC_names
    end
end

function split_central_from_noncentral_intersections(indexesFC_TYPE_, Fracturelist, mesh, sgndDist_k)
    # if the sum of LS=0 then the front is passing in the middle of the cell
    LS_TYPE_ = get_LS_on_i_fictitius_cell("icba", Fracturelist[indexesFC_TYPE_], mesh.NeiElements, sgndDist_k)
    # todo: one could think to speed up the code by setting the condition below to >=0 but I am not sure it should be
    # for alle the cells
    central_intersections = findall(sum(LS_TYPE_, dims=2) .== 0)

    if length(central_intersections) > 0
        other_intersections = setdiff(1:length(indexesFC_TYPE_), central_intersections)
        return indexesFC_TYPE_[central_intersections], indexesFC_TYPE_[other_intersections]
    else
        return Int64[], indexesFC_TYPE_
    end
end

function define_orientation_type1(T1_other_intersections, mesh, sgndDist_k)
    """
            type 1        |                 |                 |
            2(+) & 2(-)   |                 |                 |
                 2        |        3        |        1        |        0
            - ------ -    |   - ------ +    |    + ------ -   |    + ------ +
            |        |    |   |        |    |    |        |   |    |        |
            |        |    |   |        |    |    |        |   |    |        |
            + ------ +    |   - ------ +    |    + ------ -   |    - ------ -
    """
    LS_TYPE_1 = get_LS_on_i_fictitius_cell("ab", T1_other_intersections, mesh.NeiElements, sgndDist_k)
    testvector = [2, 1]
    orientation = (LS_TYPE_1 .> 0) * testvector
    orientation = orientation'
    """
    now the orientation is like this:
            type 1        |                 |                 |
            2(+) & 2(-)   |                 |                 |
                 0        |        1        |        2        |        3
            - ------ -    |   - ------ +    |    + ------ -   |    + ------ +
            |        |    |   |        |    |    |        |   |    |        |
            |        |    |   |        |    |    |        |   |    |        |
            + ------ +    |   - ------ +    |    + ------ -   |    - ------ -
            
    We want to make it as above by applying a function
    f : a + bx + cx^2 + dx^3
    f(0)->2, f(1)->3, f(2)->1, f(3)->0
     a=2, b=23/6, c=-7/2, d=2/3
    """
    f(x) = 2 + 23*x/6 - 7*x*x/2 + 2*x*x*x/3
    return Int.(f.(orientation))
end

function define_orientation_type2(T2_other_intersections, mesh, sgndDist_k)
    # todo: check the boolean
    """
            type 2        |
            2(+) & 2(-)   |
                 0        |        1
            + ------ -    |   - ------ +
            |        |    |   |        |
            |        |    |   |        |
            - ------ +    |   + ------ -
    """
    LS_TYPE_2 = get_LS_on_i_fictitius_cell("i", T2_other_intersections, mesh.NeiElements, sgndDist_k)
    return (LS_TYPE_2 .> 0.)'
end

function define_orientation_type3OR4(type, Tx_other_intersections, mesh, sgndDist_k)
    #log = logging.getLogger('PyFrac.continuous_front_reconstruction')
    """
                        type 3        |                 |                 |                 |
                        3(+) & 1(-)   |                 |                 |                 |
    RETURNED                 1        |        2        |        3        |        0        |
    ORIENTATION:        + ------ +    |   + ------ -    |    - ------ +   |    + ------ +   |    a ------ b
                        |        |    |   |        |    |    |        |   |    |        |   |    |        |
                        |        |    |   |        |    |    |        |   |    |        |   |    |        |
                        + ------ -    |   + ------ +    |    + ------ +   |    - ------ +   |    i ------ c

                        type 4        |                 |                 |                 |
    RETURNED            3(-) & 1(+)   |                 |                 |                 |
    ORIENTATION:             1        |        2        |        3        |        0        |
                        - ------ -    |   - ------ +    |    + ------ -   |    - ------ -   |    a ------ b
                        |        |    |   |        |    |    |        |   |    |        |   |    |        |
                        |        |    |   |        |    |    |        |   |    |        |   |    |        |
                        - ------ +    |   - ------ -    |    - ------ -   |    + ------ -   |    i ------ c
    """
    LS_TYPE_3or4 = get_LS_on_i_fictitius_cell("icba", Tx_other_intersections, mesh.NeiElements, sgndDist_k)
    testvector = [1, 2, 3, 4]
    if type == "3"
        LS_TYPE_3or4 = Int.(LS_TYPE_3or4 .< 0.0) # True (1) when is Negative otherwise False (0)
    elseif type == "4"
        LS_TYPE_3or4 = Int.(LS_TYPE_3or4 .> 0.0) # True (1) when is Positive otherwise False (0)
    end
    for i in 1:size(LS_TYPE_3or4, 1)
        LS_TYPE_3or4[i, :] = LS_TYPE_3or4[i, :] .* testvector
    end

    # if (np.sum(LS_TYPE_3or4, axis=1)-1)[0] == 4 :
    #  log.debug("stop")
    return sum(LS_TYPE_3or4, dims=2) .- 1
end

function move_intersections_to_the_center_when_inRibbon_type3(indexesFC_T3_central_inters,
                                                            indexesFC_T3_other_inters,
                                                            Fracturelist,
                                                            mesh,
                                                            sgndDist_k,
                                                            Ribbon)

    # define the orientation of all the cells in indexesFC_T3_other_inters
    T3_orientations = define_orientation_type3OR4("3", Fracturelist[indexesFC_T3_other_inters], mesh, sgndDist_k)

    # get the names of the negative cells in the FC in indexesFC_T3_other_inters
    T3_other_intersections_name_of_negatives = get_fictitius_cell_names(T3_orientations, Fracturelist[indexesFC_T3_other_inters], mesh.NeiElements)

    # check if the negative cells are Ribbon cells
    T3_other_intersections_test_if_Ribbon = Int.(vec(T3_other_intersections_name_of_negatives) .∈ Ref(Ribbon))

    # let to be examined by the next if condition only the cells that are within ribbon
    indexes_temp = findall(T3_other_intersections_test_if_Ribbon .== 1)
    indexesFC_T3_2_intersections = indexesFC_T3_other_inters[indexes_temp]
    indexesFC_T3_0_1_2_intersections = setdiff(indexesFC_T3_other_inters, indexesFC_T3_2_intersections)

    # compute the level set at the point at the center of the fictitius cell
    # you can compute it by simply taking the average of the LS value that is known at the vertexes of
    # the fictitius cell
    if length(indexesFC_T3_2_intersections) > 0
        LS_TYPE_3 = get_LS_on_i_fictitius_cell("iabc", Fracturelist[indexesFC_T3_2_intersections], mesh.NeiElements, sgndDist_k)

        # if the level set value at that point is > 0 then the front is intersecting the
        # ribbon cell, so what you have to do is move the name of the cells where to take the centre as
        # intersecting point
        to_move_to_the_center = findall(vec(sum(LS_TYPE_3, dims=2) / 4.0) .> 0.0)
        if length(to_move_to_the_center) > 0
            indexesFC_T3_central_inters = vcat(indexesFC_T3_central_inters, indexesFC_T3_2_intersections[to_move_to_the_center])
            indexesFC_T3_2_intersections = setdiff(indexesFC_T3_2_intersections, indexesFC_T3_2_intersections[to_move_to_the_center])
        end
    end
    return indexesFC_T3_central_inters, indexesFC_T3_2_intersections, indexesFC_T3_0_1_2_intersections
end

function move_intersections_to_the_center_when_inRibbon_type1(
    T1_central_intersections, T1_other_intersections, Fracturelist, mesh, sgndDist_k, Ribbon)

    # define the orientation of all the cells in T1_other_intersections
    T1_1st_negative_cell_local_index = define_orientation_type1(Fracturelist[T1_other_intersections], mesh, sgndDist_k)
    T1_2nd_negative_cell_local_index = mod.(T1_1st_negative_cell_local_index .+ 1, 4)

    # get the names of the negative cells
    T1_1st_negative_cell_name = get_fictitius_cell_names(T1_1st_negative_cell_local_index', Fracturelist[T1_other_intersections], mesh.NeiElements)
    T1_2nd_negative_cell_name = get_fictitius_cell_names(T1_2nd_negative_cell_local_index', Fracturelist[T1_other_intersections], mesh.NeiElements)


    # check if the negative cells are Ribbon cells
    T1_1st_negative_cell_checkif_Ribbon = Int.(vec(T1_1st_negative_cell_name) .∈ Ref(Ribbon))
    T1_2nd_negative_cell_checkif_Ribbon = Int.(vec(T1_2nd_negative_cell_name) .∈ Ref(Ribbon))


    # return True if any of them is in ribbon
    T1_negative_cell_checkif_Ribbon = T1_1st_negative_cell_checkif_Ribbon + T1_2nd_negative_cell_checkif_Ribbon


    # take to the next check only the cells that are within ribbon
    indexes_temp = findall(T1_negative_cell_checkif_Ribbon .== 1)
    T1_close_to_ribbon = T1_other_intersections[indexes_temp]
    T1_far_from_ribbon = setdiff(T1_other_intersections, indexes_temp)

    # compute the level set at the point at the center of the fictitius cell
    # you can compute it by simply taking the average of the LS value that is known at the vertexes of
    # the fictitius cell
    if length(T1_close_to_ribbon) > 0
        LS_TYPE_1 = get_LS_on_i_fictitius_cell("iabc", Fracturelist[T1_close_to_ribbon], mesh.NeiElements, sgndDist_k)

        # if the level set value at that point is > 0 then the front is intersecting the
        # ribbon cell, so what you have to do is move the name of the cells where to take the centre as
        # intersecting point
        to_move_to_the_center = findall(vec(sum(LS_TYPE_1, dims=2) / 4.0) .> 0.0)
        if length(to_move_to_the_center) > 0
            T1_central_intersections = vcat(T1_central_intersections, T1_close_to_ribbon[to_move_to_the_center])
            T1_close_to_ribbon = setdiff(T1_close_to_ribbon, T1_close_to_ribbon[to_move_to_the_center])
        end
    end
    return T1_central_intersections, vcat(T1_close_to_ribbon, T1_far_from_ribbon)
end

function get_mesh_info_for_computing_intersections(i, mesh, sgndDist_k)

    # get the level set at the vertex of all these fictitius cells
    LS_other_intersections = get_LS_on_i_fictitius_cell("icba", i, mesh.NeiElements, sgndDist_k)

    # get the name of the node at the center of the fictitius cell: 1
    #
    #    o---------o---------o
    #    |         |         |
    #    |    a ------- b    |
    #    |    |    |    |    |
    #    o----|----1----|----o
    #    |    |    |    |    |
    #    |    i ------- c    |
    #    |         |         |
    #    o---------o---------o
    centernode = mesh.Connectivity[i, 3]

    # get the coordinates of the vertical and horizontal line passing for the center node
    xgrid = mesh.VertexCoor[centernode, 1]
    ygrid = mesh.VertexCoor[centernode, 2]

    # get the names of the elements c b a
    # c = right
    # b = rightup
    # a = up
    #
    #    o---------o---------o
    #    |         |         |
    #    |    a ------- b    |
    #    |    |    |    |    |
    #    o----|----o----|----o
    #    |    |    |    |    |
    #    |    i ------- c    |
    #    |         |         |
    #    o---------o---------o

    #    0     1      2      3
    # NeiElements[i]->[left, right, bottom, up]
    [left_elem, right_elem, bottom_elem, top_elem] = [1, 2, 3, 4]
    #
    up = mesh.NeiElements[i, top_elem]
    right = mesh.NeiElements[i, right_elem]
    rightUp = mesh.NeiElements[i + 1, top_elem]

    # get the coordinates of the centers of the cells iabc
    allx = [mesh.CenterCoor[right, 1], mesh.CenterCoor[i, 1], mesh.CenterCoor[up, 1], mesh.CenterCoor[rightUp, 1]]
    ally = [mesh.CenterCoor[up, 2], mesh.CenterCoor[rightUp, 2], mesh.CenterCoor[right, 2], mesh.CenterCoor[i, 2]]

    return xgrid, ygrid, allx, ally, LS_other_intersections
end

function find_x_OR_y_intersections(intersection_with, XorYgrid, numberofinters, allx, ally, float_precision, LS)

    # get the coefficients for
    alphaXorY = Array{float_precision}(undef, 4, numberofinters)

    # compute the alphaX and alphaY vectors that will be used to compute the intersections
    if intersection_with == "x"  # intersection between the front and the Horizontal line
        allXorY = allx
        alphaXorY[[1, 3], :] = (XorYgrid - ally[[1, 3], :])
        alphaXorY[[2, 4], :] = -(XorYgrid - ally[[2, 4], :])

    elseif intersection_with == "y"     # intersection between the front and the Vertical line
        allXorY = ally
        alphaXorY[[1, 3], :] = (XorYgrid - allx[[1, 3], :])
        alphaXorY[[2, 4], :] = -(XorYgrid - allx[[2, 4], :])
    end

    # compute some products fo the computations of the intersections
    doubledotProdwithAlphaXorY = alphaXorY' .* LS
    innerProdwithAlphaXorY = sum(doubledotProdwithAlphaXorY, dims=2)

    # intersection between the front and the Vertical or Horizontal line
    YorX = sum(doubledotProdwithAlphaXorY .* allXorY', dims=2) ./ vec(innerProdwithAlphaXorY)

    return YorX
end

function find_edge_ID(xCandidate, xgrid, yCandidate, ygrid, mesh, i)

    # define if the xintersection is between the right cells or on the left ones
    x_position = Int.(xCandidate .- xgrid .> 0)

    # define the edge ID [ leftIDs, rightIDs ]
    #                         0       1
    IDs0, IDs1 = get_fictitius_cell_specific_names("left right", i, mesh.NeiElements)
    row_idx = 0:size(x_position, 1)-1
    # very nice peace of code below: I am randoimly specific positions from the matrix
    list_temp_0 = Int64[]
    list_temp_1 = Int64[]
    for j in 1:size(row_idx, 1)
        push!(list_temp_0, IDs0[j, x_position[j]])
        push!(list_temp_1, IDs1[j, x_position[j]])
    end
    IDs0 = reshape(list_temp_0, :, 1)
    IDs1 = reshape(list_temp_1, :, 1)
    edge_x = Int64[]
    for j in 1:size(IDs0, 1)
        push!(edge_x, intersect(mesh.Connectivityelemedges[IDs0[j]], mesh.Connectivityelemedges[IDs1[j]])[1])
    end

    # define if the yintersection is between the top cells or between the bottom ones
    y_position = Int.(yCandidate .- ygrid .> 0)
    # define the edge ID [ bottomID, topID ]
    #                         0       1
    IDs0, IDs1 = get_fictitius_cell_specific_names("bottom top", i, mesh.NeiElements)
    row_idx = 0:size(y_position, 1)-1
    # very nice peace of code below: I am randoimly specific positions from the matrix
    list_temp_0 = Int64[]
    list_temp_1 = Int64[]
    for j in 1:size(row_idx, 1)
        push!(list_temp_0, IDs0[j, y_position[j]])
        push!(list_temp_1, IDs1[j, y_position[j]])
    end
    IDs0 = reshape(list_temp_0, :, 1)
    IDs1 = reshape(list_temp_1, :, 1)
    edge_y = Int64[]
    for j in 1:size(IDs0, 1)
        push!(edge_y, intersect(mesh.Connectivityelemedges[IDs0[j]], mesh.Connectivityelemedges[IDs1[j]])[1])
    end
    return edge_x, edge_y
end

function find_xy_intersections_type3_case_2_intersections(return_info, indexesFC_T3_2_intersections,
                                                      Fracturelist, mesh, sgndDist_k, float_precision)
    i = Fracturelist[indexesFC_T3_2_intersections]
    #
    xgrid, ygrid, allx, ally, LS_other_intersections = get_mesh_info_for_computing_intersections(i, mesh, sgndDist_k)

    # intersection between the front and the Horizontal line
    xCandidate = find_x_OR_y_intersections("x", ygrid, length(i), allx, ally, float_precision, LS_other_intersections)
    # and ygrid

    # intersection between the front and the Vertical line
    yCandidate = find_x_OR_y_intersections("y", xgrid, length(i), allx, ally, float_precision, LS_other_intersections)
    # and xgrid

    if length(xCandidate) > 0
        edge_x, edge_y = find_edge_ID(xCandidate, xgrid, yCandidate, ygrid, mesh, i)
    end

    if return_info == "return xy"
        edgeORvertexID = Int64[]
        if length(xCandidate) > 0
            x, y, edge_2_inter = reorder_intersections(Fracturelist,
                                                       xCandidate,
                                                       yCandidate,
                                                       xgrid,
                                                       ygrid,
                                                       edge_x,
                                                       edge_y,
                                                       mesh,
                                                       indexesFC_T3_2_intersections)
            edgeORvertexID = vcat(edgeORvertexID, edge_2_inter)
        else
            x = y = Float64[]
        end
        return x, y, edgeORvertexID

    elseif return_info == "return all xy"
        return xCandidate, xgrid, ygrid, yCandidate, convert(Array, edge_x), convert(Array, edge_y)
    end
end

function check_if_point_inside_cell(xORy_grid, xORy_Candidate, hx_OR_hy, mac_precision)
    xORy_max = xORy_grid + hx_OR_hy * 0.5
    xORy_min = xORy_grid - hx_OR_hy * 0.5
    return ((xORy_Candidate .- xORy_max) / hx_OR_hy .<= mac_precision/10000) .* ((xORy_Candidate .- xORy_min) / hx_OR_hy .>= -mac_precision/10000)
end

function reorder_intersections(Fracturelist,
                          xCandidate_2_inter,
                          yCandidate_2_inter,
                          xgrid_2_inter,
                          ygrid_2_inter,
                          edge_x_2_inter,
                          edge_y_2_inter,
                          mesh,
                          indexesFC_Tx_2_intersections_local)
    """
    We need to order the points: does it comes irst x-intersection or y-intersection?
    We need to understand the location of the points in the fictitius cell (A1, A2, B1, B2) and WHERE the front
    is coming (L R T B) i.e. the previous fictitius cell.

            |     ____|____     |
                 |T        |
     _ _ _ _|_ _ |_ _ |_ _ |_ _ | _ _ _ _ _ _
                 |         |
         ___|____|____|____|____|___
        |        |    B2   |        |
     _ _| _ | _ _|_A1_|_A2_| _ _|_ _|_ _ _
        |L       |    B1   |       R|
        |___|____|____|____|____|___|
                 |         |
    _ _ _ _ | _ _|_ _ |_ _ |_ _ | _ _ _ _
                 |B        |
            |    |____|____|    |

            |         |         |

    The following table is providing us a way of deciding if we need to get fist the intersection with the horizontal
    axis or the vertical one. If we consider the former case then we have 0 otherwise 1.

          (A1-B1)  (A2-B2)  (A2-B1)  (A1-B2)            (A1-B1)  (A2-B2)  (A2-B1)  (A1-B2)             0 1 2 3        This vector form is more easy to access
      L     A1       B2       B1       A1             L    x        y        y        x             0  0 1 1 0        i values 0       - 1       - 2         - 3
      R     B1       A2       A2       B2      <=>    R    y        x        x        y       <=>   1  1 0 0 1   <=>  j values 0 1 2 3 - 0 1 2 3 - 0 1 2  3  - 0  1  2  3
      B     B1       A2       B1       A1             B    y        x        y        x             2  1 0 1 0        indexes  0 1 2 3   4 5 6 7   8 9 10 11 - 12 13 14 15
      T     A1       B2       A2       B2             T    x        y        x        y             3  0 1 0 1                 0 1 1 0 - 1 0 0 1 - 1 0  1  0 -  0  1  0  1
    """

    # take the names of the fictitius cells
    i = Fracturelist[indexesFC_Tx_2_intersections_local]
    # take all 4 neighbours elements
    #                         0     1      2      3
    #       NeiElements[i]->[left, right, bottom, up]
    matrix_with_rows_where_to_search = mesh.NeiElements[i]
    # thake the index of the previous fictitius cell (where the front is coming)
    array_with_the_number_to_be_searched_in_each_row = Fracturelist[indexesFC_Tx_2_intersections_local .- 1]
    # OLD matrix of indexes:
    # m=np.asarray([[0,1,1,0],
    #               [1,0,0,1],
    #               [0,1,0,1],
    #               [1,0,1,0]],dtype=int)
    # I have written the matrix as an array beaqcuse is easier to access
    m = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]

    # array of 1D coordinates
    ij = fill(3, length(indexesFC_Tx_2_intersections_local))

    # find the i values
    # todo: may be a more efficient way exist
    for jj in 1:length(indexesFC_Tx_2_intersections_local)
        ij[jj] = findall(matrix_with_rows_where_to_search[jj, :] .== array_with_the_number_to_be_searched_in_each_row[jj])[1]
    end
    # multiply each i value by 4 to compute the index for the array
    ij = ij * 4
    """
    the following lines are providing the values on the table below depending on the test conditions: 

      x>xgrid ?
    False   True
      0       2  False 
                       y>ygrid ?
      3       1  True

    the values will fill the second column of the matrix ij
    """
    a = 2 * Int.(xCandidate_2_inter .- xgrid_2_inter .> 0)
    b = Int.(yCandidate_2_inter .- ygrid_2_inter .> 0)
    ij = ij + a .* (-2 * b + 1) + 3 * b

    # now we need to get the values from the matrix m by knowing the indexes
    # that have been stored in the matrix ij
    m = m[ij]
    # if m[i] is zero take as first the intersection with x axes, otherwise with y axes
    # todo: may be a more efficient way exist
    edge_2_inter = Vector{Vector{Int64}}()
    xtemp = Vector{Vector{Float64}}()
    ytemp = Vector{Vector{Float64}}()
    for jj in 1:length(m)
        if m[jj] == 0
            push!(xtemp, [xCandidate_2_inter[jj], xgrid_2_inter[jj]])
            push!(ytemp, [ygrid_2_inter[jj], yCandidate_2_inter[jj]])

            push!(edge_2_inter, [edge_x_2_inter[jj], edge_y_2_inter[jj]])
        else
            push!(xtemp, [xgrid_2_inter[jj], xCandidate_2_inter[jj]])
            push!(ytemp, [yCandidate_2_inter[jj], ygrid_2_inter[jj]])

            push!(edge_2_inter, [edge_y_2_inter[jj], edge_x_2_inter[jj]])
        end
    end

    return xtemp, ytemp, edge_2_inter
end

function find_xy_intersections_type3_case_0_1_2_intersections(indexesFC_T3_0_1_2_intersections,
        Fracturelist, mesh, sgndDist_k, float_precision, mac_precision)
    # one assumption behind this function is that the front is curved within cells type 3
    # so there will be always an intersection with both axes whereas within cells type 1 you can have straight front
    # and thus more checks are needed

    xCandidate, xgrid, ygrid, yCandidate, edge_x, edge_y = find_xy_intersections_type3_case_2_intersections("return all xy",
                                                                                indexesFC_T3_0_1_2_intersections,
                                                                                Fracturelist,
                                                                                mesh,
                                                                                sgndDist_k,
                                                                                float_precision)
    # we can have 0, 1, 2 intersections.
    # In the latter case they need to be ordered properly.
    # Now we will count the intersections per FC
    is_X_inside_the_cell_Answer = Int.(check_if_point_inside_cell(xgrid, xCandidate, mesh.hx, mac_precision))
    is_Y_inside_the_cell_Answer = Int.(check_if_point_inside_cell(ygrid, yCandidate, mesh.hy, mac_precision))
    number_of_intersections = is_X_inside_the_cell_Answer + is_Y_inside_the_cell_Answer

    # find where I have two intersections
    temp_indexes = findall(number_of_intersections .== 2)

    # separate the case with two intersection from the case with 0 or 1
    indexesFC_T3_2_intersections_local = indexesFC_T3_0_1_2_intersections[temp_indexes]
    xCandidate_2_inter = xCandidate[temp_indexes]
    ygrid_2_inter = ygrid[temp_indexes]
    yCandidate_2_inter = yCandidate[temp_indexes]
    xgrid_2_inter = xgrid[temp_indexes]
    edge_x_2_inter = edge_x[temp_indexes]
    edge_y_2_inter = edge_y[temp_indexes]

    indexesFC_T3_0_1_intersection_local = setdiff(indexesFC_T3_0_1_2_intersections, temp_indexes)
    xCandidate = xCandidate[setdiff(1:length(xCandidate), temp_indexes)]
    ygrid = ygrid[setdiff(1:length(ygrid), temp_indexes)]
    yCandidate = yCandidate[setdiff(1:length(yCandidate), temp_indexes)]
    xgrid = xgrid[setdiff(1:length(xgrid), temp_indexes)]
    edge_x = edge_x[setdiff(1:length(edge_x), temp_indexes)]
    edge_y = edge_y[setdiff(1:length(edge_y), temp_indexes)]
    is_X_inside_the_cell_Answer = is_X_inside_the_cell_Answer[setdiff(1:length(is_X_inside_the_cell_Answer), temp_indexes)]
    is_Y_inside_the_cell_Answer = is_Y_inside_the_cell_Answer[setdiff(1:length(is_Y_inside_the_cell_Answer), temp_indexes)]
    number_of_intersections = number_of_intersections[setdiff(1:length(number_of_intersections), temp_indexes)]

    # find where I have no intersections
    temp_indexes = findall(number_of_intersections .== 0)

    # separate the case with 0 intersection from the case with 1
    indexesFC_T3_0_intersection_local = indexesFC_T3_0_1_intersection_local[temp_indexes]
    indexesFC_T3_1_intersection_local = setdiff(indexesFC_T3_0_1_intersection_local, temp_indexes)
    xCandidate = xCandidate[setdiff(1:length(xCandidate), temp_indexes)]
    ygrid = ygrid[setdiff(1:length(ygrid), temp_indexes)]
    yCandidate = yCandidate[setdiff(1:length(yCandidate), temp_indexes)]
    xgrid = xgrid[setdiff(1:length(xgrid), temp_indexes)]
    edge_x = edge_x[setdiff(1:length(edge_x), temp_indexes)]
    edge_y = edge_y[setdiff(1:length(edge_y), temp_indexes)]
    is_X_inside_the_cell_Answer = is_X_inside_the_cell_Answer[setdiff(1:length(is_X_inside_the_cell_Answer), temp_indexes)]
    is_Y_inside_the_cell_Answer = is_Y_inside_the_cell_Answer[setdiff(1:length(is_Y_inside_the_cell_Answer), temp_indexes)]

    # check if the steps before have been done correctly
    if length(is_X_inside_the_cell_Answer) > 0
        if maximum(is_X_inside_the_cell_Answer) > 1
            error("FRONT RECONSTRUCTION ERROR: the processing of the cells is not correct")
        end
    end

    # processing the case of single intersection
    # setting xCandidate and xgrid within an unique array: xCandidate
    edge_1_inters = copy(edge_x)
    temp_indexes = findall(is_Y_inside_the_cell_Answer .== 1)
    edge_1_inters[temp_indexes] = edge_y[temp_indexes]
    temp_indexes = findall(is_Y_inside_the_cell_Answer .== 1)
    xCandidate[temp_indexes] = xgrid[temp_indexes]
    temp_indexes = findall(is_X_inside_the_cell_Answer .== 1)
    yCandidate[temp_indexes] = ygrid[temp_indexes]

    # we need to order the points
    if length(indexesFC_T3_2_intersections_local) > 0
        xCandidate_2_inter, yCandidate_2_inter, edge_2_inter = reorder_intersections(Fracturelist,
                                                                                     xCandidate_2_inter,
                                                                                     yCandidate_2_inter,
                                                                                     xgrid_2_inter,
                                                                                     ygrid_2_inter,
                                                                                     edge_x_2_inter,
                                                                                     edge_y_2_inter,
                                                                                     mesh,
                                                                                     indexesFC_T3_2_intersections_local)
    else 
        edge_2_inter = Vector{Vector{Int64}}()
    end

    return indexesFC_T3_0_intersection_local,
           indexesFC_T3_1_intersection_local,
           indexesFC_T3_2_intersections_local,
           xCandidate, yCandidate, edge_1_inters,
           xCandidate_2_inter, yCandidate_2_inter, edge_2_inter
end

function find_xy_intersections_type1(indexesFC_T1_1_2_intersections,
                                 Fracturelist, mesh, sgndDist_k, float_precision, mac_precision)

    # 1 or 2 intersections per fictitius cell are allowed

    # we expect to have some NaN or some points outside of the cells
    xCandidate,
     xgrid,
     ygrid,
     yCandidate,
     edge_x, edge_y = find_xy_intersections_type3_case_2_intersections("return all xy",
                                                                        indexesFC_T1_1_2_intersections,
                                                                        Fracturelist,
                                                                        mesh,
                                                                        sgndDist_k,
                                                                        float_precision)


    # check the exstistance of nan
    # if any of xCandidate or any of yCandidate is nan, substitute it with the coordinate of a point out of the mesh
    if any(isnan.(xCandidate)) || any(isnan.(yCandidate))
        nan_values = findall(isnan.(xCandidate))
        xCandidate[nan_values] .= mesh.CenterCoor[1, 1] - mesh.hx
        nan_values = findall(isnan.(yCandidate))
        xCandidate[nan_values] .= mesh.CenterCoor[1, 2] - mesh.hy
    end


    # we can have 1, 2 intersections.
    # In the latter case they need to be ordered properly.
    # Now we will count the intersections per FC
    is_X_inside_the_cell_Answer = Int.(check_if_point_inside_cell(xgrid, xCandidate, mesh.hx, mac_precision))
    is_Y_inside_the_cell_Answer = Int.(check_if_point_inside_cell(ygrid, yCandidate, mesh.hy, mac_precision))
    number_of_intersections = is_X_inside_the_cell_Answer + is_Y_inside_the_cell_Answer

    # find where I have two intersections
    temp_indexes = findall(number_of_intersections .== 2)

    # separate the case with two intersection from the case with 1
    indexesFC_T1_2_intersections_local = indexesFC_T1_1_2_intersections[temp_indexes]
    xCandidate_2_inter = xCandidate[temp_indexes]
    ygrid_2_inter = ygrid[temp_indexes]
    yCandidate_2_inter = yCandidate[temp_indexes]
    xgrid_2_inter = xgrid[temp_indexes]
    edge_x_2_inter = edge_x[temp_indexes]
    edge_y_2_inter = edge_y[temp_indexes]

    indexesFC_T1_1_intersection_local = setdiff(indexesFC_T1_1_2_intersections, temp_indexes)
    xCandidate = xCandidate[setdiff(1:length(xCandidate), temp_indexes)]
    ygrid = ygrid[setdiff(1:length(ygrid), temp_indexes)]
    yCandidate = yCandidate[setdiff(1:length(yCandidate), temp_indexes)]
    xgrid = xgrid[setdiff(1:length(xgrid), temp_indexes)]
    edge_x = edge_x[setdiff(1:length(edge_x), temp_indexes)]
    edge_y = edge_y[setdiff(1:length(edge_y), temp_indexes)]
    is_X_inside_the_cell_Answer = is_X_inside_the_cell_Answer[setdiff(1:length(is_X_inside_the_cell_Answer), temp_indexes)]
    is_Y_inside_the_cell_Answer = is_Y_inside_the_cell_Answer[setdiff(1:length(is_Y_inside_the_cell_Answer), temp_indexes)]


    # check if the steps before have been done correctly
    if length(is_X_inside_the_cell_Answer) > 0
        if maximum(is_X_inside_the_cell_Answer) > 1
            error("FRONT RECONSTRUCTION ERROR: the processing of the cells is not correct")
        end
    end


    # processing the case of single intersection
    # setting xCandidate and xgrid within an unique array: xCandidate
    edge_1_inters = copy(edge_x)
    temp_indexes = findall(is_Y_inside_the_cell_Answer .== 1)
    edge_1_inters[temp_indexes] = edge_y[temp_indexes]
    temp_indexes = findall(is_Y_inside_the_cell_Answer .== 1)
    xCandidate[temp_indexes] = xgrid[temp_indexes]
    temp_indexes = findall(is_X_inside_the_cell_Answer .== 1)
    yCandidate[temp_indexes] = ygrid[temp_indexes]

    if length(indexesFC_T1_2_intersections_local) > 0
        xCandidate_2_inter, yCandidate_2_inter, edge_2_inter = reorder_intersections(Fracturelist,
                                                                                     xCandidate_2_inter,
                                                                                     yCandidate_2_inter,
                                                                                     xgrid_2_inter,
                                                                                     ygrid_2_inter,
                                                                                     edge_x_2_inter,
                                                                                     edge_y_2_inter,
                                                                                     mesh,
                                                                                     indexesFC_T1_2_intersections_local)
    else 
        edge_2_inter = Vector{Vector{Int64}}()
    end

    return indexesFC_T1_1_intersection_local,
           indexesFC_T1_2_intersections_local,
           xCandidate, yCandidate, edge_1_inters,
           xCandidate_2_inter, yCandidate_2_inter, edge_2_inter
end

function find_xy_intersections_with_cell_center(indexesFC_Tx_central_inters, Fracturelist, mesh)

    # define a more practical name
    i = Fracturelist[indexesFC_Tx_central_inters]

    # take the coordinates
    centernode = mesh.Connectivity[i, 3]
    x = mesh.VertexCoor[centernode, 1]
    y = mesh.VertexCoor[centernode, 2]

    edgeORvertexID = Int64[]

    # append infos
    if length(centernode) > 0
        edgeORvertexID = vcat(edgeORvertexID, centernode)  # intersecting with a vertex
    end
    return x, y, edgeORvertexID
end

function process_fictitius_cells_3(indexesFC_TYPE_3, Args, x, y, typeindex, edgeORvertexID)

    Fracturelist, Ribbon, mesh, sgndDist_k, float_precision, mac_precision = Args

    # find when you have an intersection with the cell center or when you have two intersections
    indexesFC_T3_central_inters, indexesFC_T3_other_inters = split_central_from_noncentral_intersections(indexesFC_TYPE_3, Fracturelist, mesh, sgndDist_k)

    # if the intersection will be in the ribbon cell, move these cells
    # from the other intersections to the central intersection list
    indexesFC_T3_central_inters,
     indexesFC_T3_2_intersections,
     indexesFC_T3_0_1_2_intersections = move_intersections_to_the_center_when_inRibbon_type3(indexesFC_T3_central_inters,
                                                                                    indexesFC_T3_other_inters,
                                                                                    Fracturelist,
                                                                                    mesh,
                                                                                    sgndDist_k,
                                                                                    Ribbon)


    # find the intersections with the center
    # 1 intersection
    T3_x_inters_center,
     T3_y_inters_center,
     T3_edgeORvertexID_center = find_xy_intersections_with_cell_center(indexesFC_T3_central_inters,
                                                                        Fracturelist,
                                                                        mesh)

    # set the found intersections
    for j in 1:length(indexesFC_T3_central_inters)
        temp_index = indexesFC_T3_central_inters[j]
        x[temp_index] = [T3_x_inters_center[j]]
        y[temp_index] = [T3_y_inters_center[j]]
        edgeORvertexID[temp_index] = [T3_edgeORvertexID_center[j]]
        typeindex[temp_index] = [1]
    end


    # find the intersections with the vertical and horizontal axes passing throug the cell center
    # 2 intersections
    if length(indexesFC_T3_2_intersections) > 0
        T3_x_inters,
         T3_y_inters,
         T3_edgeORvertexID = find_xy_intersections_type3_case_2_intersections("return xy",
                                                                               indexesFC_T3_2_intersections,
                                                                               Fracturelist,
                                                                               mesh,
                                                                               sgndDist_k,
                                                                               float_precision)
    end
    # set the found intersections
    for j in 1:length(indexesFC_T3_2_intersections)
        temp_index = indexesFC_T3_2_intersections[j]
        x[temp_index] = T3_x_inters[j]
        y[temp_index] = T3_y_inters[j]
        edgeORvertexID[temp_index] = T3_edgeORvertexID[j]
        typeindex[temp_index] = [0, 0]
    end

    # 0,1,2 intersections
    if length(indexesFC_T3_0_1_2_intersections) > 0
        indexesFC_T3_0_intersection_local,
         indexesFC_T3_1_intersection_local,
         indexesFC_T3_2_intersections_local,
         xCandidate, yCandidate, edge_1_inters,
         xCandidate_2_inter, yCandidate_2_inter, edge_2_inter = find_xy_intersections_type3_case_0_1_2_intersections(indexesFC_T3_0_1_2_intersections,
                                                                                                                    Fracturelist,
                                                                                                                    mesh, sgndDist_k, float_precision, mac_precision)
        # set the found intersections
        for j in 1:length(indexesFC_T3_0_intersection_local)
            temp_index = indexesFC_T3_0_intersection_local[j]
            x[temp_index] = Float64[]
            y[temp_index] = Float64[]
            edgeORvertexID[temp_index] = Int64[]
            typeindex[temp_index] = Int64[]
        end

        for j in 1:length(indexesFC_T3_1_intersection_local)
            temp_index = indexesFC_T3_1_intersection_local[j]
            x[temp_index] = [xCandidate[j]]
            y[temp_index] = [yCandidate[j]]
            edgeORvertexID[temp_index] = [edge_1_inters[j]]
            typeindex[temp_index] = [0]
        end

        for j in 1:length(indexesFC_T3_2_intersections_local)
            temp_index = indexesFC_T3_2_intersections_local[j]
            x[temp_index] = xCandidate_2_inter[j]
            y[temp_index] = yCandidate_2_inter[j]
            edgeORvertexID[temp_index] = edge_2_inter[j]
            typeindex[temp_index] = [0, 0]
        end
    end

    return x, y, typeindex, edgeORvertexID
end

function process_fictitius_cells_1(indexesFC_TYPE_1, Args, x, y, typeindex, edgeORvertexID)

    Fracturelist, Ribbon, mesh, sgndDist_k, float_precision, mac_precision = Args

    # find when you have an intersection with the cell center or when you have two intersections
    indexesFC_T1_central_inters, indexesFC_T1_other_inters = split_central_from_noncentral_intersections(indexesFC_TYPE_1, Fracturelist, mesh, sgndDist_k)

    # if the intersection will be in the ribbon cell, move these cells
    # from the other intersections to the central intersection list
    indexesFC_T1_central_inters,
     indexesFC_T1_1_2_intersections = move_intersections_to_the_center_when_inRibbon_type1(indexesFC_T1_central_inters,
                                                                                  indexesFC_T1_other_inters,
                                                                                  Fracturelist,
                                                                                  mesh,
                                                                                  sgndDist_k,
                                                                                  Ribbon)



    # find the intersections with the center
    # 1 intersection
    T1_x_inters_center,
     T1_y_inters_center,
     T1_edgeORvertexID_center = find_xy_intersections_with_cell_center(indexesFC_T1_central_inters,
                                                                        Fracturelist,
                                                                        mesh)

    # set the found intersections
    for j in 1:length(indexesFC_T1_central_inters)
        temp_index = indexesFC_T1_central_inters[j]
        x[temp_index] = [T1_x_inters_center[j]]
        y[temp_index] = [T1_y_inters_center[j]]
        edgeORvertexID[temp_index] = [T1_edgeORvertexID_center[j]]
        typeindex[temp_index] = [1]
    end

    if length(indexesFC_T1_1_2_intersections) > 0
        # find the intersections with the vertical and horizontal axes passing throug the cell center
        indexesFC_T1_1_intersection_local,
         indexesFC_T1_2_intersections_local,
         xCandidate, yCandidate, edge_1_inters,
         xCandidate_2_inter, yCandidate_2_inter, edge_2_inter = find_xy_intersections_type1(indexesFC_T1_1_2_intersections,
                                                             Fracturelist,
                                                             mesh,
                                                             sgndDist_k,
                                                             float_precision,
                                                             mac_precision)
        for j in 1:length(indexesFC_T1_1_intersection_local)
            temp_index = indexesFC_T1_1_intersection_local[j]
            x[temp_index] = [xCandidate[j]]
            y[temp_index] = [yCandidate[j]]
            edgeORvertexID[temp_index]  = [edge_1_inters[j]]
            typeindex[temp_index] = [0]
        end

        for j in 1:length(indexesFC_T1_2_intersections_local)
            temp_index = indexesFC_T1_2_intersections_local[j]
            x[temp_index] = xCandidate_2_inter[j]
            y[temp_index] = yCandidate_2_inter[j]
            edgeORvertexID[temp_index] = edge_2_inter[j]
            typeindex[temp_index] = [0, 0]
        end
    end


    return x, y, typeindex, edgeORvertexID
end

function split_type4SubType4_from_rest(indexesFC_TYPE_4, Fracturelist, mesh, sgndDist_k, Ribbon)
    # check in each FC if any of the negative cells are in ribbon
    # if none of them are in Ribbon then add the index of the FC to indexesFC_TYPE_4_ST4
    #todo: it may be done more efficiently
    indexesFC_TYPE_4_ST4 = Int64[]
    jj_list = Int64[]
    orientation4 = define_orientation_type3OR4("4", Fracturelist[indexesFC_TYPE_4], mesh, sgndDist_k)
    icbamesh = get_fictitius_cell_all_names(Fracturelist[indexesFC_TYPE_4], mesh.NeiElements)
    for jj in 1:length(indexesFC_TYPE_4)
        negative_cells = icbamesh[jj, setdiff(1:4, orientation4[jj])]
        if sum(Int.(negative_cells .∈ Ref(Ribbon))) < 1
            push!(jj_list, jj)
        end
    end
    if length(jj_list) > 0
        indexesFC_TYPE_4_ST4 = vcat(indexesFC_TYPE_4_ST4, indexesFC_TYPE_4[jj_list])
        indexesFC_TYPE_4_ST01235 = setdiff(indexesFC_TYPE_4, indexesFC_TYPE_4[jj_list])
        return indexesFC_TYPE_4_ST4, indexesFC_TYPE_4_ST01235
    else
        return Int64[], indexesFC_TYPE_4
    end
end

function move_intersections_to_the_center_when_inRibbon_type4(indexesFC_T4_central_inters,
                                                         indexesFC_T4_other_inters,
                                                         Fracturelist,
                                                         mesh,
                                                         sgndDist_k)
    # NOTE: WE DO NOT NEED THE RIBBON FOR THIS...
    # compute the level set at the point at the center of the fictitius cell
    # you can compute it by simply taking the average of the LS value that is known at the vertexes of
    # the fictitius cell
    if length(indexesFC_T4_other_inters) > 0
        LS_TYPE_4 = get_LS_on_i_fictitius_cell("iabc", Fracturelist[indexesFC_T4_other_inters], mesh.NeiElements, sgndDist_k)

        # if the level set value at that point is > 0 then the front is intersecting the
        # ribbon cell, so what you have to do is move the name of the cells where to take the centre as
        # intersecting point
        to_move_to_the_center = findall(vec(sum(LS_TYPE_4, dims=2) / 4.0) .> 0.0)
        if length(to_move_to_the_center) > 0
            indexesFC_T4_central_inters = vcat(indexesFC_T4_central_inters, indexesFC_T4_other_inters[to_move_to_the_center])
            indexesFC_T4_other_inters = setdiff(indexesFC_T4_other_inters, indexesFC_T4_other_inters[to_move_to_the_center])
        end
    end
    return indexesFC_T4_central_inters, indexesFC_T4_other_inters
end

function process_fictitius_cells_4(indexesFC_TYPE_4, Args, x, y, typeindex, edgeORvertexID)
    """
    type 4 -> 3(-) & 1(+)
    POSSIBILITIES: (R = ribbon)
                                      |                 |                 |                 |                  |
                                      |                 |                 |                 |                  |
                             0        |        1        |        2        |        3        |        4         |        5
                       R- ------ -R   |   - ------ -    |   R- ------ -   |   R- ------ -   |    - ------ -R   |    - ------ -
                        |        |    |   |        |    |    |        |   |    |        |   |    |        |    |    |        |
                        |        |    |   |        |    |    |        |   |    |        |   |    |        |    |    |        |
                       R- ------ +    |  R- ------ +    |    - ------ +   |   R- ------ +   |   R- ------ +    |    - ------ +

    IN THEORY ONE SHOULD DO:
    plan:
          - group types 0 and 4 -> they have all ribbon or two opposite ribbon & the - opposite to + is non ribbon
                  types 2 and 5 -> they have no ribbon or one that is opposite to cell with +
                  types 1 and 3 -> all the other cases

          - look for 0,1   intersections in types 0 and 4 -> if any intersection is present, it will be with the center
                     0,1,2 intersections in types 2 and 5 -> can be present up to 2 intersections. But we need to check and force intersections to the center when present a ribbon
                     0,1   intersections in types 1 and 3 -> intersections with the cell center will naturally appears

          - put toghether: 0 intersections
                           1 intersections
                           2 intersections

    definitions:
          - group types 0 and 4 -> cells_04
                  types 2 and 5 -> cells_25
                  types 1 and 3 -> cells_13

    IN PRACTISE WE CAN SAVE ALL THE SPLITTING MENTIONED BEFORE BY DOING THE NEXT
    """

    Fracturelist, Ribbon, mesh, sgndDist_k, float_precision, mac_precision = Args

    # 1) find when you have an intersection with the cell center - of any subtype
    indexesFC_T4_central_inters, indexesFC_T4_other_inters = split_central_from_noncentral_intersections(
        indexesFC_TYPE_4, Fracturelist, mesh, sgndDist_k)

    # 2) split type4 SubType4 from the rest of cells
    indexesFC_TYPE_4_ST4, indexesFC_TYPE_4_ST01235 = split_type4SubType4_from_rest(indexesFC_T4_other_inters, Fracturelist, mesh, sgndDist_k, Ribbon)


    # 3) force the cell to be at the center if the LS at the center is POSITIVE -> 1 intersection (do not include subtype4)
    indexesFC_T4_central_inters,
     indexesFC_T4_0_1_2_intersections = move_intersections_to_the_center_when_inRibbon_type4(indexesFC_T4_central_inters,
                                                                                    indexesFC_TYPE_4_ST01235,
                                                                                    Fracturelist,
                                                                                    mesh,
                                                                                    sgndDist_k)


    # 4) find intersections with the center for forced and non-forced cells
    T4_x_inters_center,
     T4_y_inters_center,
     T4_edgeORvertexID_center = find_xy_intersections_with_cell_center(indexesFC_T4_central_inters,
                                                                        Fracturelist,
                                                                        mesh)
    # set the found intersections
    for j in 1:length(indexesFC_T4_central_inters)
        temp_index = indexesFC_T4_central_inters[j]
        x[temp_index] = [T4_x_inters_center[j]]
        y[temp_index] = [T4_y_inters_center[j]]
        edgeORvertexID[temp_index] = [T4_edgeORvertexID_center[j]]
        typeindex[temp_index] = [1]
    end

    # 5) find 0,1,2 intersections for all the other cells subtypes including subtype4
    #    one assumption is that the front is curved within cells type 4
    #    so there will be always an intersection with both axes whereas within cells type 1 you can have straight front
    #    and thus more checks are needed
    # 0,1,2 intersections
    if length(indexesFC_TYPE_4_ST4) > 0
        indexesFC_T4_0_1_2_intersections = vcat(indexesFC_T4_0_1_2_intersections, indexesFC_TYPE_4_ST4)
    end
    indexesFC_T4_0_intersection_local,
     indexesFC_T4_1_intersection_local,
     indexesFC_T4_2_intersections_local,
     xCandidate, yCandidate, edge_1_inters,
     xCandidate_2_inter, yCandidate_2_inter, edge_2_inter = find_xy_intersections_type3_case_0_1_2_intersections(indexesFC_T4_0_1_2_intersections,
                                                                                                                Fracturelist,
                                                                                                                mesh, sgndDist_k, float_precision, mac_precision)
    # set the found intersections
    for j in 1:length(indexesFC_T4_0_intersection_local)
        temp_index = indexesFC_T4_0_intersection_local[j]
        x[temp_index] = Float64[]
        y[temp_index] = Float64[]
        edgeORvertexID[temp_index] = Int64[]
        typeindex[temp_index] = Int64[]
    end

    for j in 1:length(indexesFC_T4_1_intersection_local)
        temp_index = indexesFC_T4_1_intersection_local[j]
        x[temp_index] = [xCandidate[j]]
        y[temp_index] = [yCandidate[j]]
        edgeORvertexID[temp_index] = [edge_1_inters[j]]
        typeindex[temp_index] = [0]
    end

    for j in 1:length(indexesFC_T4_2_intersections_local)
        temp_index = indexesFC_T4_2_intersections_local[j]
        x[temp_index] = xCandidate_2_inter[j]
        y[temp_index] = yCandidate_2_inter[j]
        edgeORvertexID[temp_index] = edge_2_inter[j]
        typeindex[temp_index] = [0, 0]
    end

    return x, y, typeindex, edgeORvertexID
end

function process_fictitius_cells_2(indexesFC_TYPE_2, Args, x, y, typeindex, edgeORvertexID)
    """
    type 2        |
    2(+) & 2(-)   |
         0        |        1
    + ------ -    |   - ------ +
    |        |    |   |        |
    |        |    |   |        |
    - ------ +    |   + ------ -
    """

    Fracturelist, Ribbon, mesh, sgndDist_k, float_precision, mac_precision = Args

    # define the orientation of all the cells in indexesFC_TYPE_2
    T2_orientations = define_orientation_type2(Fracturelist[indexesFC_TYPE_2], mesh, sgndDist_k)

    # get the names of the negative cells in the FC in indexesFC_TYPE_2
    T2_negative_cells = get_fictitius_cell_names(T2_orientations, Fracturelist[indexesFC_TYPE_2], mesh.NeiElements)

    # check if the negative cells are Ribbon cells
    T2_negative_cells_test_if_Ribbon = Int.(vec(T2_negative_cells) .∈ Ref(Ribbon))

    # let to be examined by the next if condition only the cells that are within ribbon
    indexes_temp = findall(T2_negative_cells_test_if_Ribbon .== 1)
    indexesFC_T2_close_to_ribbon = indexesFC_TYPE_2[indexes_temp]
    indexesFC_T2_far_from_ribbon = setdiff(indexesFC_TYPE_2, indexesFC_T2_close_to_ribbon)

    # compute the level set at the point at the center of the fictitius cell
    # you can compute it by simply taking the average of the LS value that is known at the vertexes of
    # the fictitius cell
    if length(indexesFC_T2_close_to_ribbon) > 0
        LS_TYPE_2 = get_LS_on_i_fictitius_cell("iabc", Fracturelist[indexesFC_T2_close_to_ribbon], mesh.NeiElements, sgndDist_k)

        # if the level set value at that point is > 0 then the front is intersecting the
        # ribbon cell, so what you have to do is move the name of the cells where to take the centre as
        # intersecting point
        to_move_to_the_center = findall(vec(sum(LS_TYPE_2, dims=2) / 4.0) .> 0.0)
        if length(to_move_to_the_center) > 0
            indexesFC_T2_central_inters = indexesFC_T2_close_to_ribbon[to_move_to_the_center]
            indexesFC_T2_close_to_ribbon = setdiff(indexesFC_T2_close_to_ribbon, indexesFC_T2_close_to_ribbon[to_move_to_the_center])
        else
            indexesFC_T2_central_inters = Int64[]
        end
    else
        indexesFC_T2_central_inters = Int64[]
    end

    # find the intersections with the center
    # 1 intersection
    if length(indexesFC_T2_central_inters) > 0
        T2_x_inters_center,
         T2_y_inters_center,
         T2_edgeORvertexID_center = find_xy_intersections_with_cell_center(indexesFC_T2_central_inters,
                                                                            Fracturelist,
                                                                            mesh)
        # set the found intersections
        for j in 1:length(indexesFC_T2_central_inters)
            temp_index = indexesFC_T2_central_inters[j]
            x[temp_index] = [T2_x_inters_center[j]]
            y[temp_index] = [T2_y_inters_center[j]]
            edgeORvertexID[temp_index] = [T2_edgeORvertexID_center[j]]
            typeindex[temp_index] = [1]
        end
    end

    # find the intersections with the vertical and horizontal axes passing through the cell center
    # 2 intersections
    if length(indexesFC_T2_close_to_ribbon) > 0 || length(indexesFC_T2_far_from_ribbon) > 0
        indexesFC_T2_1_2_intersections = vcat(indexesFC_T2_close_to_ribbon, indexesFC_T2_far_from_ribbon)
        
        indexesFC_T2_1_intersection_local,
         indexesFC_T2_2_intersections_local,
         xCandidate, yCandidate, edge_1_inters,
         xCandidate_2_inter, yCandidate_2_inter, edge_2_inter = find_xy_intersections_type1(indexesFC_T2_1_2_intersections,
                                                             Fracturelist,
                                                             mesh,
                                                             sgndDist_k,
                                                             float_precision,
                                                             mac_precision)
        
        # set the found intersections
        for j in 1:length(indexesFC_T2_1_intersection_local)
            temp_index = indexesFC_T2_1_intersection_local[j]
            x[temp_index] = [xCandidate[j]]
            y[temp_index] = [yCandidate[j]]
            edgeORvertexID[temp_index] = [edge_1_inters[j]]
            typeindex[temp_index] = [0]
        end

        for j in 1:length(indexesFC_T2_2_intersections_local)
            temp_index = indexesFC_T2_2_intersections_local[j]
            x[temp_index] = xCandidate_2_inter[j]
            y[temp_index] = yCandidate_2_inter[j]
            edgeORvertexID[temp_index] = edge_2_inter[j]
            typeindex[temp_index] = [0, 0]
        end
    end

    return x, y, typeindex, edgeORvertexID
end

function get_next_cell_name(current_cell_name, previous_cell_name, FC_type, Args)
    # log = logging.getLogger('PyFrac.continuous_front_reconstruction')
    mesh, dict_Ribbon, sgndDist_k = Args
    dict_of_possibilities = Dict{String, Int64}()
    """
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                             0     1      2      3
    """
    if FC_type == 1
        orientation = define_orientation_type1(current_cell_name, mesh, sgndDist_k)
        if orientation == 0 || orientation == 2  # quasi-horizontal direction
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 2]
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 1]
        else  # quasi-vertical direction
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 4]
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 4])] = mesh.NeiElements[current_cell_name, 3]
        end
    elseif FC_type == 2
        orientation = define_orientation_type2(current_cell_name, mesh, sgndDist_k)
        if orientation == 0
            if string(mesh.NeiElements[current_cell_name + 1, 4]) in keys(dict_Ribbon) && string(current_cell_name) in keys(dict_Ribbon)
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 4]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 4])] = mesh.NeiElements[current_cell_name, 1]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 2]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 3]
            else
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 4]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 4]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 1]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 3]
            end
        else  # orientation == 1
            if string(mesh.NeiElements[current_cell_name, 2]) in keys(dict_Ribbon) && string(mesh.NeiElements[current_cell_name, 4]) in keys(dict_Ribbon)
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 4]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 4]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 1]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 3]
            else
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 4]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 4])] = mesh.NeiElements[current_cell_name, 1]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 2]
                dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 3]
            end
        end
    else
        if FC_type == 3
            orientation = define_orientation_type3OR4("3", current_cell_name, mesh, sgndDist_k)
        elseif FC_type == 4
            orientation = define_orientation_type3OR4("4", current_cell_name, mesh, sgndDist_k)
        else 
            error("FRONT RECONSTRUCTION ERROR: Unknown cell type")
        end

        if orientation == 0
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 1]
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 3]
        elseif orientation == 1
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 3])] = mesh.NeiElements[current_cell_name, 2]
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 3]
        elseif orientation == 2
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 2])] = mesh.NeiElements[current_cell_name, 4]
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 4])] = mesh.NeiElements[current_cell_name, 2]
        elseif orientation == 3
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 1])] = mesh.NeiElements[current_cell_name, 4]
            dict_of_possibilities[string(mesh.NeiElements[current_cell_name, 4])] = mesh.NeiElements[current_cell_name, 1]
        else 
            error("FRONT RECONSTRUCTION ERROR: Wrong orientation")
        end
    end

    try
        if string(previous_cell_name) ∉ keys(dict_of_possibilities)
            throw(ErrorException(""))
        else
            return dict_of_possibilities[string(previous_cell_name)]
        end
    catch e
        # log.debug("The previous fictitious cell is not neighbour of the current fictitious cell")
        println("The previous fictitious cell is not neighbour of the current fictitious cell")
    end
end

function get_next_cell_name_from_first(first_cell_name, FC_type, mesh, sgndDist_k)
    """
    remembrer the usage of NeiElements[i]->[left, right, bottom, up]
                                             0     1      2      3
    """
    if FC_type == 1
        orientation = define_orientation_type1(first_cell_name, mesh, sgndDist_k)
        if orientation == 0 || orientation == 2  # quasi-horizontal direction
            next_cell_name = mesh.NeiElements[first_cell_name, 1]
        else  # quasi-vertical direction
            next_cell_name = mesh.NeiElements[first_cell_name, 3]
        end
    elseif FC_type == 2
        next_cell_name = mesh.NeiElements[first_cell_name, 1]
    elseif FC_type == 3
        orientation = define_orientation_type3OR4("3", first_cell_name, mesh, sgndDist_k)
        if orientation == 0 || orientation == 1
            next_cell_name = mesh.NeiElements[first_cell_name, 3]
        else
            next_cell_name = mesh.NeiElements[first_cell_name, 4]
        end
    elseif FC_type == 4
        orientation = define_orientation_type3OR4("4", first_cell_name, mesh, sgndDist_k)
        if orientation == 0 || orientation == 1
            next_cell_name = mesh.NeiElements[first_cell_name, 3]
        else
            next_cell_name = mesh.NeiElements[first_cell_name, 4]
        end
    else 
        error("FRONT RECONSTRUCTION ERROR: Unknown cell type")
    end
    return next_cell_name
end

function itertools_chain_from_iterable(lsts)
    log = @logger "PyFrac.continuous_front_reconstruction"
    # lsts example: [[1], [], [2,3], [4], [], [5,6]]
    try
        # Check all elements are iterable (arrays)
        if any(x -> !(x isa AbstractVector), lsts)
            throw(ArgumentError("The list contains an element not between square brackets"))
        end
        return reduce(vcat, lsts)
    catch e
        @error log e.msg
        return nothing
    end
end


function append_to_typelists(cell_index, cell_type, type1, type2, type3, type4)
    if cell_type == 1
        push!(type1, cell_index)
    elseif cell_type == 2
        push!(type2, cell_index)
    elseif cell_type == 3
        push!(type3, cell_index)
    elseif cell_type == 4
        push!(type4, cell_index)
    end
    return type1, type2, type3, type4
end

function is_inside_the_triangle(p_center, p_zero_vertex, p1, p2, mac_precision, area_of_a_cell)
    #
    # This function answer to the question:
    # is the point inside a triangle?
    # (given the coordinates)
    T1x = [p_center.x, p1.x, p2.x]
    T1y = [p_center.y, p1.y, p2.y]
    T2x = [p_center.x, p1.x, p_zero_vertex.x]
    T2y = [p_center.y, p1.y, p_zero_vertex.y]
    T3x = [p_center.x, p2.x, p_zero_vertex.x]
    T3y = [p_center.y, p2.y, p_zero_vertex.y]
    Tx = [p1.x, p2.x, p_zero_vertex.x]
    Ty = [p1.y, p2.y, p_zero_vertex.y]
    
    area_sum = copute_area_of_a_polygon(T1x, T1y) +
               copute_area_of_a_polygon(T2x, T2y) +
               copute_area_of_a_polygon(T3x, T3y)
    area_diff = area_sum - copute_area_of_a_polygon(Tx, Ty)
    
    if (area_diff) / area_of_a_cell < mac_precision
        return true
    else
        return false
    end
end

function recompute_LS_at_tip_cells(sgndDist_k, p_zero_vertex, p_center, p1, p2, mac_precision, area_of_a_cell, zero_level_set_value)
    # find the distance from the cell center to the front
    p1x = p1.x
    p1y = p1.y
    p2x = p2.x
    p2y = p2.y
    p0x = p_center.x
    p0y = p_center.y
    
    distance_center_to_front = pointtolinedistance(p0x, p1x, p2x, p0y, p1y, p2y)
    
    # we do not allow to have LS == 0
    if distance_center_to_front == 0
        distance_center_to_front = -zero_level_set_value
    else
        """
        Now we want to understand if the sign of the LS at the cell center is positive or negative.
        This question is equivalent of asking if the center of the cell is inside or outside of the fracture.
        Again, the latter question is equivalent of asking if the center of the cell belongs to the triangle 
        that is made by considering the zero vertex and the two points of intersection of the front with the cell. 
        If it is true, then the cell center is inside the fracture, otherwise it is outside
        """
        # Предполагается, что is_inside_the_triangle определена выше
        if is_inside_the_triangle(p_center, p_zero_vertex, p1, p2, mac_precision, area_of_a_cell)
            sgndDist_k[p_center.name] = -distance_center_to_front
        else
            sgndDist_k[p_center.name] = +distance_center_to_front
        end
    end
    return sgndDist_k
end

"""
    TODO: CORRECT MISTAKES!!!
    reconstruct_front_continuous(sgndDist_k, anularegion, Ribbon, eltsChannel, mesh, 
                                recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge, 
                                lstTmStp_EltCrack0=nothing, oldfront=nothing)

description of the function.

Args:
    `sgndDist_k::Vector{Float64}`: vector that contains the distance from the center of the cell to the front. 
                They are negative if the point is inside the fracture otherwise they are positive
    `anularegion::Vector{Int64}`: name of the cells where we expect to be the front
    `Ribbon::Vector{Int64}`: name of the ribbon elements
    `lstTmStp_EltCrack0::Union{Vector{Int64}, Nothing}` = nothing

Returns:
    tipcells (integers): -- descriptions.
    nextribboncells (integers): -- descriptions.
    vertexes (integers): -- descriptions.
    alphas (float): -- descriptions.
    orthogonaldistances (float): -- descriptions.
    ...
"""
function reconstruct_front_continuous(sgndDist_k::Vector{Float64}, anularegion::Vector{Int64}, Ribbon::Vector{Int64}, eltsChannel::Vector{Int64}, mesh::CartesianMesh,
                                    recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge::Bool,
                                    lstTmStp_EltCrack0=nothing, oldfront=nothing)
    
    log = @logger "PyFrac.continuous_front_reconstruction"
    
    recompute_front = false
    float_precision = Float64
    mac_precision = 100 * sqrt(eps(Float64))
    zero_level_set_value = minimum([mesh.hx, mesh.hy]) / 1000.0
    area_of_a_cell = mesh.hx * mesh.hy

    """
    -1) - Prerequistite to be checked here: 
    none of the cells at the boundary should have negative Level set
    """
    if any(sgndDist_k[mesh.Frontlist] .< 0)
        @warn log "Some cells at the boundary of the mesh have negative level set"
        negativefront = findall(sgndDist_k .< 0)
        intwithFrontlist = intersect(negativefront, mesh.Frontlist)
        correct_size_of_pstv_region = [false, true, false]
        return intwithFrontlist, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, nothing, nothing, nothing, nothing
    end

    """
    0) - Set all the LS==0 to -zero_level_set_value
    In this way we avoid to deal with the situations where the front is crossing exactly a vertex of an i     
    """
    zerovertexes = findall(sgndDist_k .== 0)
    if length(zerovertexes) > 0
        sgndDist_k[zerovertexes] .= -zero_level_set_value
    end

    """
    1) - define a dictionary for the Ribbon cells
       - find a list of valid fictitius cells (inamesOfFC) that can belongs to more than one fracture
       - compute the LS at the valid fictitius cells (LSofFC)
       - compute the "i names" of the FC of different types 
           Whe define the fictitius cell types:

        type 1        |   type 2        |    type 3       |    type 4
        2(+) & 2(-)   |   2(+) & 2(-)   |    3(+) & 1(-)  |    3(-) & 1(+)  
        + ------ -    |   + ------ -    |    + ------ -   |    - ------ +         
        |        |    |   |        |    |    |        |   |    |        |
        |        |    |   |        |    |    |        |   |    |        |
        + ------ -    |   - ------ +    |    + ------ +   |    - ------ -
    """
    dict_Ribbon = Dict(string(i) => true for i in Ribbon)
    
    exitstatus, i_1_2_3_4_names_of_fictitiuscells, number_of_type_2_cells, i_1_2_3_4_FC_type, dict_FC_names = find_fictitius_cells(anularegion, mesh.NeiElements, sgndDist_k)

    # the values of the LS in the fictitius cells might be not computed at all the cells, so increase the thickness of the band
    if exitstatus
        if any(sgndDist_k[mesh.Frontlist] .< 0)
            @info log "increasing the thickness of the band will not help to reconstruct the front because it will be outside of the mesh: Failing the time-step"
            correct_size_of_pstv_region = [false, false, true]
            return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, nothing, nothing, nothing, nothing
        else
            @debug log "I am increasing the thickness of the band (directive from find fictitius cells routine)"
            # from utility import plot_as_matrix
            # K = zeros(Float64, mesh.NumberOfElts)
            # K[anularegion] = sgndDist_k[anularegion]
            # plot_as_matrix(K, mesh)
            correct_size_of_pstv_region = [false, false, false]
            return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, sgndDist_k, nothing, nothing, nothing
        end
    end

    """
    2) - define the fractures
    """
    # each cell of type 1, 3, 4 count as 1, type 2 counts as 2

    NofCells_to_explore::Int64 = length(i_1_2_3_4_names_of_fictitiuscells) + 2 * number_of_type_2_cells
    NofCells_explored::Int64 = 0

    list_of_Fracturelists::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    list_of_Cells_type_1_list::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    list_of_Cells_type_2_list::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    list_of_Cells_type_3_list::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    list_of_Cells_type_4_list::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    Args::Vector{Any} = [mesh, dict_Ribbon, sgndDist_k]

    # I require more than 3 cells to define a single fracture
    if NofCells_to_explore > 3
        # do until you have explored all the cells
        while NofCells_explored < NofCells_to_explore
            Fracturelist::Vector{Int64} = Int64[]
            Cells_type_1_list::Vector{Int64} = Int64[]
            Cells_type_2_list::Vector{Int64} = Int64[]
            Cells_type_3_list::Vector{Int64} = Int64[]
            Cells_type_4_list::Vector{Int64} = Int64[]

            first_cell_name_key::String = first(keys(dict_FC_names))
            first_cell_name::Int64 = dict_FC_names[first_cell_name_key]
            
            push!(Fracturelist, first_cell_name)
            
            first_cell_type_key::String = string(first_cell_name)
            first_cell_type::Int64 = i_1_2_3_4_FC_type[first_cell_type_key]
            
            index_for_append::Int64 = length(Fracturelist)
            Cells_type_1_list, Cells_type_2_list, Cells_type_3_list, Cells_type_4_list = 
                append_to_typelists(index_for_append, first_cell_type, Cells_type_1_list, Cells_type_2_list, Cells_type_3_list, Cells_type_4_list)
                
            delete!(dict_FC_names, first_cell_name_key)
            NofCells_explored += 1

            # todo: in case of cells of type 2 this have to be reviewed: do not delete cell of type 2
            next_cell_name::Int64 = get_next_cell_name_from_first(first_cell_name, first_cell_type, mesh, sgndDist_k)

            while next_cell_name != first_cell_name
                NofCells_explored += 1
                push!(Fracturelist, next_cell_name)
                # we need this check because it happens that some cells of type i are not in the fictitious cells
                # in that case compute the type on the fly 3536
                LSet_temp::Vector{Float64} = get_LS_on_i_fictitius_cell("iabc", next_cell_name, mesh.NeiElements, sgndDist_k)
                LSet_first_value::Float64 = LSet_temp[1]

                next_cell_name_str::String = string(next_cell_name)
                if !haskey(i_1_2_3_4_FC_type, next_cell_name_str)
                    if any(LSet_first_value > 10.0^40)
                        fict_cells_for_fracture = get_fictitius_cell_all_names(Fracturelist, mesh.NeiElements)
                        flattened_fict_cells = vcat(fict_cells_for_fracture...)
                        unique_fict_cells = unique(flattened_fict_cells)
                        
                        intwithFrontlist::Vector{Int64} = intersect(unique_fict_cells, mesh.Frontlist)
                        if length(intwithFrontlist) > 0
                            @info log "The new front reaches the boundary. Remeshing"
                            correct_size_of_pstv_region::Vector{Bool} = [false, true, false]
                            # Returning the intersection between the fictitius cells and the frontlist as tip in order to decide the direction of remeshing
                            # (in case of anisotropic remeshing)
                            return (intwithFrontlist, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, nothing, nothing, nothing, nothing)
                        else
                            @debug log "I am increasing the thickness of the band (tip i cell not found in the anularegion)"
                            correct_size_of_pstv_region::Vector{Bool} = [false, false, false]
                            return (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, sgndDist_k, nothing, nothing, nothing)
                        end
                    end
                    cell_type::Int64 = get_fictitius_cell_type(LSet_first_value)
                else
                    cell_type::Int64 = i_1_2_3_4_FC_type[next_cell_name_str]
                end

                index_for_append_current::Int64 = length(Fracturelist)
                Cells_type_1_list, Cells_type_2_list, Cells_type_3_list, Cells_type_4_list = 
                    append_to_typelists(index_for_append_current, cell_type, Cells_type_1_list, Cells_type_2_list, Cells_type_3_list, Cells_type_4_list)
                    
                if haskey(dict_FC_names, next_cell_name_str) && cell_type != 2
                    delete!(dict_FC_names, next_cell_name_str)
                end
                
                previous_cell_name::Int64 = Fracturelist[end-1] # second last cell in the list and the one where we are coming
                next_cell_name = get_next_cell_name(next_cell_name, previous_cell_name, cell_type, Args)
            end # while next_cell_name != first_cell_name

            # now we check if the fracture is good or not
            if length(Fracturelist) > 3
                push!(list_of_Fracturelists, Fracturelist)
                push!(list_of_Cells_type_1_list, Cells_type_1_list)
                push!(list_of_Cells_type_2_list, Cells_type_2_list)
                push!(list_of_Cells_type_3_list, Cells_type_3_list)
                push!(list_of_Cells_type_4_list, Cells_type_4_list)
            else
                fict_cells_for_small_fracture = get_fictitius_cell_all_names(Fracturelist, mesh.NeiElements)
                flattened_fict_cells_small = vcat(fict_cells_for_small_fracture...)
                unique_fict_cells_small = unique(flattened_fict_cells_small)
                
                intwithFrontlist::Vector{Int64} = intersect(unique_fict_cells_small, mesh.Frontlist)
                if length(intwithFrontlist) > 0
                    @info log "The new front reaches the boundary. Remeshing"
                    correct_size_of_pstv_region::Vector{Bool} = [false, false, true]
                    # Returning the intersection between the fictitius cells and the frontlist as tip in order to decide the direction of remeshing
                    # (in case of anisotropic remeshing)
                    return (intwithFrontlist, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, nothing, nothing, nothing, nothing)
                else
                    @debug log "<< I REJECT A POSSIBLE FRCTURE FRONT BECAUSE IS TOO SMALL >>"
                    @debug log "set the LS of the positive cells to be -machine precision"
                    all_cells_of_all_FC_of_this_small_fracture::Vector{Int64} = unique(get_fictitius_cell_all_names(Fracturelist, mesh.NeiElements)...) 
                    
                    sgnd_values_in_small_fracture = sgndDist_k[all_cells_of_all_FC_of_this_small_fracture]
                    index_of_positives::Vector{Int64} = findall(sgnd_values_in_small_fracture .> 0)
                    
                    cells_to_modify = all_cells_of_all_FC_of_this_small_fracture[index_of_positives]
                    sgndDist_k[cells_to_modify] .= -zero_level_set_value
                end
            end # if length(Fracturelist) > 3
        end # while NofCells_explored < NofCells_to_explore

        if length(list_of_Fracturelists) < 1
            error("FRONT RECONSTRUCTION ERROR: not valid fractures have been found! Remember that you could have several fractures of too small size to be tracked")
        end
    else
        error("FRONT RECONSTRUCTION ERROR: not enough cells to define even one fracture front!")
    end


    """
    3) - process fracture fronts

    to be output at the end:
    """

    list_of_xintersections_for_all_closed_paths::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    list_of_yintersections_for_all_closed_paths::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    list_of_typeindex_for_all_closed_paths::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    list_of_edgeORvertexID_for_all_closed_paths::Vector{Vector{Int64}} = Vector{Vector{Int64}}()

    for j in 1:length(list_of_Fracturelists)
        if recompute_front
            break
        end
        
        Fracturelist::Vector{Int64} = list_of_Fracturelists[j]
        
        typeindex::Vector{Vector{Int64}} = [Int64[] for _ in Fracturelist]
        edgeORvertexID::Vector{Vector{Int64}} = [Int64[] for _ in Fracturelist]
        x::Vector{Vector{Float64}} = [Float64[] for _ in Fracturelist]
        y::Vector{Vector{Float64}} = [Float64[] for _ in Fracturelist]

        Args::Vector{Any} = [Fracturelist, Ribbon, mesh, sgndDist_k, float_precision, mac_precision]

        indexesFC_TYPE_1::Vector{Int64} = list_of_Cells_type_1_list[j]
        indexesFC_TYPE_2::Vector{Int64} = list_of_Cells_type_2_list[j]
        indexesFC_TYPE_3::Vector{Int64} = list_of_Cells_type_3_list[j]
        indexesFC_TYPE_4::Vector{Int64} = list_of_Cells_type_4_list[j]

        if length(indexesFC_TYPE_3) > 0
            x, y, typeindex, edgeORvertexID = process_fictitius_cells_3(indexesFC_TYPE_3, Args, x, y, typeindex, edgeORvertexID)
        end

        if length(indexesFC_TYPE_1) > 0
            x, y, typeindex, edgeORvertexID = process_fictitius_cells_1(indexesFC_TYPE_1, Args, x, y, typeindex, edgeORvertexID)
        end

        if length(indexesFC_TYPE_4) > 0
            x, y, typeindex, edgeORvertexID = process_fictitius_cells_4(indexesFC_TYPE_4, Args, x, y, typeindex, edgeORvertexID)
        end
        
        if length(indexesFC_TYPE_2) > 0
            @debug log "Type 2 to be tested"
            correct_size_of_pstv_region::Vector{Bool} = [false, false, true]
            return (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, correct_size_of_pstv_region, nothing, nothing, nothing, nothing)
            # [x, y, typeindex, edgeORvertexID] = process_fictitius_cells_2(indexesFC_TYPE_4, Args, x, y, typeindex, edgeORvertexID)
            # error("FRONT RECONSTRUCTION ERROR: type 2 to be tested")
        end

        """
        vocabulary:
        xintersection:= x coordinates
        yintersection:= y coordinates
        typeindex:= 0 if node intersecting an edge, 1 if intersecting an existing vertex of the mesh
        edgeORvertexID:= index of the vertex or index of the edge 
        """
        
        xintersection::Vector{Float64} = itertools_chain_from_iterable(x)
        yintersection::Vector{Float64} = itertools_chain_from_iterable(y)
        
        typeindex_flat::Vector{Int64} = itertools_chain_from_iterable(typeindex)
        edgeORvertexID_flat::Vector{Int64} = itertools_chain_from_iterable(edgeORvertexID)
        

        """
        The closed front area is implicitly not smaller than 1/2 the area of the cell
        Now we impose a threshold: the area of a closed front should be > area of a cell
        - Compute the front area.
        - If the area > area cell => - find the names of the positive cells
                                    - set the level set of the positive cells artificially to be -mac precision       
        """
        deleteTHEfront::Bool = false
        closed_front_area::Float64 = 0.0
        
        if length(xintersection) > 0
            closed_front_area = copute_area_of_a_polygon(xintersection, yintersection)
        else
            closed_front_area = 0.0
        end
        
        if closed_front_area <= area_of_a_cell * 1.01
            @debug log "A small front of area = $(round(100 * closed_front_area / area_of_a_cell, digits=4))% of the single cell has been deleted"
            deleteTHEfront = true
        elseif (maximum(xintersection) - minimum(xintersection) < mesh.hx) || (maximum(yintersection) - minimum(yintersection) < mesh.hy)
            @debug log "A front of area = $(round(100 * closed_front_area / area_of_a_cell, digits=4))% of the single cell has been deleted because it was longh and thin within a column or raw of elements"
            deleteTHEfront = true
        end

        if deleteTHEfront
            fict_cells_for_small_fracture = get_fictitius_cell_all_names(Fracturelist, mesh.NeiElements)
            all_cells_of_all_FC_of_this_small_fracture::Vector{Int64} = unique(vcat(fict_cells_for_small_fracture...))
            
            sgnd_values_in_small_fracture = sgndDist_k[all_cells_of_all_FC_of_this_small_fracture]
            index_of_positives::Vector{Int64} = findall(sgnd_values_in_small_fracture .> 0)
            
            cells_to_modify = all_cells_of_all_FC_of_this_small_fracture[index_of_positives]
            sgndDist_k[cells_to_modify] .= -zero_level_set_value
            
            recompute_front = true
        else
            push!(list_of_xintersections_for_all_closed_paths, xintersection)
            push!(list_of_yintersections_for_all_closed_paths, yintersection)
            push!(list_of_typeindex_for_all_closed_paths, typeindex_flat)
            push!(list_of_edgeORvertexID_for_all_closed_paths, edgeORvertexID_flat)
        end
    end 

    global_list_of_TIPcells::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    global_list_of_TIPcellsONLY::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    global_list_of_distances::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    global_list_of_angles::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    global_list_of_vertexpositionwithinthecell::Vector{Vector{Int64}} = Vector{Vector{Int64}}()
    global_list_of_vertexpositionwithinthecellTIPcellsONLY::Vector{Vector{Int64}} = Vector{Vector{Int64}}()

    sgndDist_k_new::Vector{Float64} = copy(sgndDist_k)

    list_of_xintersectionsfromzerovertex::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    list_of_yintersectionsfromzerovertex::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    list_of_vertexID::Vector{Vector{Int64}} = Vector{Vector{Int64}}()

    fronts_dictionary::Dict{String, Any} = Dict{String, Any}("number_of_fronts" => length(list_of_Fracturelists))

    """
    We need to compute first all the closed contours because is of fundamental importance the notion of
    inside or outside of the fracture for what will follow next 
    """
    if !recompute_front
        for j in 1:length(list_of_Fracturelists)
            xintersection::Vector{Float64} = list_of_xintersections_for_all_closed_paths[j]
            yintersection::Vector{Float64} = list_of_yintersections_for_all_closed_paths[j]
            typeindex::Vector{Int64} = list_of_typeindex_for_all_closed_paths[j]
            edgeORvertexID::Vector{Int64} = list_of_edgeORvertexID_for_all_closed_paths[j]

            """
            4) - Cleaning up the points
            """
            """
            new cleaning:
            I found this situation that needs to be resolved before proceding with the next correction.
            I will delete just one point in this case otherwise I can not identify the tip cell.
                
                                A
                |___________|___**______|___________|_   
                |           |   ||      |           |
                |           |   ||      |           |
                |     +     |   ||-     |     -     |
                |           |   ||   C  |           | 
                |___________|___**==**__|___________|_
                |           |   B   ||  |           |
                |           |       ||  |           |
                |     +     |     + ||  |     _     |
                |           |       ||  |           |
                |           |       ||  |           |
                |___________|_______**__|___________|__
                |           |      D||  |           |
                |           |       ||  |           |
                |     +     |     + ||  |     _     |
                |           |       ||  |           |
                |           |       ||  |           |
                |___________|_______**__|___________|__
            """
            to_be_deleted::Vector{Int64} = Int64[]
            for jjj in 1:length(typeindex)
                prev_jjj::Int = mod1(jjj - 1, length(typeindex))
                if typeindex[jjj] == 0 && typeindex[prev_jjj] == 0
                    if edgeORvertexID[jjj] == edgeORvertexID[prev_jjj]
                        elements_A::Vector{Int64} = Int64[]
                        if typeindex[mod1(jjj - 2, length(typeindex))] == 0
                            elements_A = mesh.Connectivityedgeselem[edgeORvertexID[mod1(jjj - 2, length(typeindex))]]
                        else
                            elements_A = mesh.Connectivitynodeselem[edgeORvertexID[mod1(jjj - 2, length(typeindex))]]
                        end
                        
                        next_jjj_plus_1::Int = mod1(jjj + 1, length(typeindex))
                        elements_D::Vector{Int64} = Int64[]
                        if typeindex[next_jjj_plus_1] == 0
                            elements_D = mesh.Connectivityedgeselem[edgeORvertexID[next_jjj_plus_1]]
                        else
                            elements_D = mesh.Connectivitynodeselem[edgeORvertexID[next_jjj_plus_1]]
                        end
                        
                        if length(intersect(elements_A, elements_D)) < 1
                            push!(to_be_deleted, jjj)
                        end
                    end
                end
            end
            
            sort!(to_be_deleted)
            counter::Int64 = 0
            for jjj in 1:length(to_be_deleted)
                value::Int64 = to_be_deleted[jjj]
                deleteat!(xintersection, value - counter)
                deleteat!(yintersection, value - counter)
                deleteat!(typeindex, value - counter)
                deleteat!(edgeORvertexID, value - counter)
                counter += 1
            end
            if counter > 0
                recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = true
                @debug log "deleted $(counter) points on the same edge but close to each other"
            end

            """
            new cleaning: 
            clean only if the elements belong to the same edge
            A and C are the corners to be deleted
            Note that removing one of the two points MIGHT change the choice of the tip cell.
            
                    ___|__________|___
                    \\|          |//   
                    A **          ** D
                    \\        //
                    |\\      //|
                    ___|_**====**_|___
                    | B     C  |
                    |          |  
            """
            recursive_deleting::Bool = true
            recursive_counter::Int64 = 0
            while recursive_deleting
                to_be_deleted_recursive::Vector{Int64} = Int64[]
                for jjj in 1:length(typeindex)
                    prev_jjj::Int = mod1(jjj - 1, length(typeindex))
                    if typeindex[jjj] == 0 && typeindex[prev_jjj] == 0
                        if edgeORvertexID[jjj] == edgeORvertexID[prev_jjj]
                            push!(to_be_deleted_recursive, prev_jjj)
                            push!(to_be_deleted_recursive, jjj)
                        end
                    end
                end
                
                sort!(to_be_deleted_recursive)
                counter = 0
                for jjj in 1:length(to_be_deleted_recursive)
                    value = to_be_deleted_recursive[jjj]
                    deleteat!(xintersection, value - counter)
                    deleteat!(yintersection, value - counter)
                    deleteat!(typeindex, value - counter)
                    deleteat!(edgeORvertexID, value - counter)
                    counter += 1
                end
                
                if counter == 0
                    recursive_deleting = false
                else
                    recursive_counter += counter
                end
            end
            
            if recursive_counter > 0
                recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = true
                @debug log "deleted $(recursive_counter) points on the same edge that leads to sub resolution"
            end

            """
            CASE 1: 
            C is the corner to be deleted unless B and D are on edges defining the same ribbon cell. In the latter case delete B and D
            Loop over the corner nodes, check if the previous point is really on edge
                                        check if the next point is really on edge
                                        check if the next and previous points shares an edge of the edges exiting front C
                                        check that the edges of the previous point and the next one belongs to the same element
                                        in order to avoid the case where -A-B-C-G-E- will see C removed
                    ___|___________|___________|_   
                    |           |           |
                    | A         |           |
                    ==**          |           |
                    |\\         | C    G    |
                    ___|_**=======**______**___F_
                    |  B       ||           |
                    |          ||           |
                    |          **  D        |
                    |           \\          |
                    |           |\\  E      |
                    ___|___________|_**________|_
                    |           |           |
            """
            """
            CASE 2: 
            B is the corner to be deleted.
            Note that removing one of the two points will not change the choice of the tip cell.
            This case DO NOT CONSIDER WHEN D IS NOT IN THE SAME EDGE OF C AND IN THE SAME CELL OF B AND C
            Loop over the corner nodes, check if the previous point is really on edge and next really on vertex
                                        or vice versa...
                                        check if the two points on the vertex shares the same edge
                                        check if the the previous and the next point are on the same element

                    ___|___________|___   
                    |           |
                    | A         |
                    ==**          |
                    |\\         | C
                    ___|_**=======**___
                    | B        ||
                    |          || 
                    |          ||
                    |          ||         
                    |          || D
                    ___|__________**____
                    |          ||
            """
            """
            CASE 3: 
            B is the corner to be deleted.
            Note that removing one of the two points will not change the choice of the tip cell.
            This case DO NOT CONSIDER WHEN G IS NOT IN ONE EDGE OF [CFED]

                    ___L___________K____________J   
                    |           |            |
                    | A         |            |
                    ==**          |            |
                    |\\         | C          |
                    ___F_**=======**____________I
                    | B         \\           |
                    |           |\\          |
                    |           | \\         |
                    |           |  \\        |
                    |           |   \\ G     |
                    ___E___________D____**______H (we only ask that G does not belong to [CFED]
                    |           |
                    
            forbidden nodes are meant to be (D,E,F,C,K,L)
            forbidden edges are meant to be all the edges of [CFED] and [FCKL]
            
                    ___|___________|____________|___   
                    |           |            |
                    |           |            |
                    ==**          |           **===
                    |\\         |          //|
                    ___|_**=======**=========**_|____
                    |           |            |
                    |           |            |
                    |           |            |
                    |           |            |
                    |           |            |
                    ___|___________|____________|__ 
                    |           |            |
            """
            vertex_indexes::Vector{Int64} = findall(x -> x == 1, typeindex)
            counter = 0
            add_at_the_end::Int64 = 0
            
            for jjj in 1:length(vertex_indexes)
                cases_1and2_passed::Bool = false
                vijjj::Int64 = vertex_indexes[jjj] - counter
                vertex_name::Int64 = edgeORvertexID[vijjj]
                edge_name_previous_point::Int64 = edgeORvertexID[mod1(vijjj - 1, length(edgeORvertexID))]
                edge_name_next_point::Int64 = edgeORvertexID[mod1(vijjj + 1, length(edgeORvertexID))]
                
                # CASE 1 - removing the corner point or the two edge points (see above)
                next_vijjj_plus_1::Int = mod1(vijjj + 1, length(typeindex))
                if typeindex[mod1(vijjj - 1, length(typeindex))] == 0 && typeindex[next_vijjj_plus_1] == 0
                    edges_of_the_corner_vertex::Vector{Int64} = mesh.Connectivitynodesedges[vertex_name] # 4 edges
                    n1::Vector{Int64} = intersect(edges_of_the_corner_vertex, edge_name_previous_point)
                    n2::Vector{Int64} = intersect(edges_of_the_corner_vertex, edge_name_next_point)
                    common_elem_name::Vector{Int64} = intersect(mesh.Connectivityedgeselem[n1], mesh.Connectivityedgeselem[n2])
                    
                    if length(common_elem_name) > 0
                        # now check if the element is in ribbon:
                        if any(in(Ribbon), common_elem_name[1])
                            index_to_delete::Int64 = next_vijjj_plus_1
                            deleteat!(xintersection, index_to_delete)
                            deleteat!(yintersection, index_to_delete)
                            deleteat!(typeindex, index_to_delete)
                            deleteat!(edgeORvertexID, index_to_delete)
                            if index_to_delete != 1
                                counter += 1
                            else
                                add_at_the_end += 1
                            end

                            if index_to_delete == 1
                                index_to_delete = mod1(vijjj - 2, length(typeindex))
                            else
                                index_to_delete = mod1(vijjj - 1, length(typeindex))
                            end
                            deleteat!(xintersection, index_to_delete)
                            deleteat!(yintersection, index_to_delete)
                            deleteat!(typeindex, index_to_delete)
                            deleteat!(edgeORvertexID, index_to_delete)
                            if index_to_delete != 1
                                counter += 1
                            else
                                add_at_the_end += 1
                            end
                        else # in case the element is not in ribbon
                            index_to_delete = vijjj
                            deleteat!(xintersection, index_to_delete)
                            deleteat!(yintersection, index_to_delete)
                            deleteat!(typeindex, index_to_delete)
                            deleteat!(edgeORvertexID, index_to_delete)
                            if index_to_delete != 1
                                counter += 1
                            else
                                add_at_the_end += 1
                            end
                        end
                    else
                        cases_1and2_passed = true
                    end
                # CASE 2 - removing the edge point
                # previous vertex is on cell node & next vertex is on cell edge
                elseif (typeindex[mod1(vijjj - 1, length(typeindex))] == 1 && typeindex[next_vijjj_plus_1] == 0)
                    edges_of_the_corner_vertex = mesh.Connectivitynodesedges[vertex_name]
                    sharing_the_element::Bool = length(intersect(mesh.Connectivitynodeselem[edge_name_previous_point], mesh.Connectivityedgeselem[edge_name_next_point])) > 0
                    edge_name_previous_point_nodes::Vector{Int64} = mesh.Connectivitynodesedges[edge_name_previous_point]
                    n1_size::Int64 = length(intersect(edges_of_the_corner_vertex, edge_name_previous_point_nodes))
                    n2_size::Int64 = length(intersect(edges_of_the_corner_vertex, edge_name_next_point))

                    if n1_size + n2_size == 2 && sharing_the_element
                        index_to_delete = next_vijjj_plus_1
                        deleteat!(xintersection, index_to_delete)
                        deleteat!(yintersection, index_to_delete)
                        deleteat!(typeindex, index_to_delete)
                        deleteat!(edgeORvertexID, index_to_delete)
                        if index_to_delete != 1
                            counter += 1
                        else
                            add_at_the_end += 1
                        end
                    else
                        cases_1and2_passed = true
                    end
                # previous vertex is on cell edge & next vertex is on cell node
                elseif (typeindex[mod1(vijjj - 1, length(typeindex))] == 0 && typeindex[next_vijjj_plus_1] == 1)
                    edges_of_the_corner_vertex = mesh.Connectivitynodesedges[vertex_name]
                    sharing_the_element = length(intersect(mesh.Connectivitynodeselem[edge_name_next_point], mesh.Connectivityedgeselem[edge_name_previous_point])) > 0
                    edge_name_next_point_nodes::Vector{Int64} = mesh.Connectivitynodesedges[edge_name_next_point]
                    n1_size = length(intersect(edges_of_the_corner_vertex, edge_name_previous_point))
                    n2_size = length(intersect(edges_of_the_corner_vertex, edge_name_next_point_nodes))
                    if n1_size + n2_size == 2 && sharing_the_element
                        index_to_delete = mod1(vijjj - 1, length(typeindex))
                        deleteat!(xintersection, index_to_delete)
                        deleteat!(yintersection, index_to_delete)
                        deleteat!(typeindex, index_to_delete)
                        deleteat!(edgeORvertexID, index_to_delete)
                        if index_to_delete != 1
                            counter += 1
                        else
                            add_at_the_end += 1
                        end
                    else
                        cases_1and2_passed = true
                    end
                end
                
                if cases_1and2_passed
                    # CASE 3 - removing the edge point
                    # PREVIOUS vertex is on cell edge or node that does not belong to any of the edges of the cell where
                    # the current vertex lies & NEXT vertex is on the edge bounding the cell where the current vertex is

                    edges_of_the_corner_vertex = mesh.Connectivitynodesedges[vertex_name]
                    # check that the next vertex is on edge
                    # check that the next vertex has an edge in common with the current vertex
                    next_vijjj_mod::Int = mod1(vijjj + 1, length(typeindex))
                    if typeindex[next_vijjj_mod] == 0 && length(intersect(edges_of_the_corner_vertex, edge_name_next_point)) == 1
                        common_cells::Vector{Int64} = intersect(mesh.Connectivitynodeselem[vertex_name, :], mesh.Connectivityedgeselem[edge_name_next_point])
                        forbidden_edges::Vector{Int64} = vcat([mesh.Connectivityelemedges[cell] for cell in common_cells]...)
                        forbidden_nodes::Vector{Int64} = unique(vcat([mesh.Connectivity[cell] for cell in common_cells]...))
                        delete_flag::Bool = false
                        
                        prev_vijjj_mod::Int = mod1(vijjj - 1, length(typeindex))
                        if typeindex[prev_vijjj_mod] == 0 # it means that the previous vertex is on edge
                            if length(intersect(forbidden_edges, edge_name_previous_point)) <= 0
                                delete_flag = true
                            end
                        else # it means that the previous vertex is on a vertex
                            if length(intersect(forbidden_nodes, edge_name_previous_point)) <= 0
                                delete_flag = true
                            end
                        end
                        
                        if delete_flag
                            index_to_delete = next_vijjj_mod
                            deleteat!(xintersection, index_to_delete)
                            deleteat!(yintersection, index_to_delete)
                            deleteat!(typeindex, index_to_delete)
                            deleteat!(edgeORvertexID, index_to_delete)
                            if index_to_delete != 1
                                counter += 1
                            else
                                add_at_the_end += 1
                            end
                        end
                        
                        # NEXT vertex is on cell edge or node that does not belong to any of the edges of the cell where
                        # the current vertex lies & PREVIOUS vertex is on the edge bounding the cell where the current vertex is

                        # check that the previous vertex is on edge
                        # check that the previous vertex has an edge in common with the current vertex
                        if typeindex[prev_vijjj_mod] == 0 && length(intersect(edges_of_the_corner_vertex, edge_name_previous_point)) == 1
                            common_cells = intersect(mesh.Connectivitynodeselem[vertex_name, :], mesh.Connectivityedgeselem[edge_name_previous_point])
                            forbidden_edges = vcat([mesh.Connectivityelemedges[cell] for cell in common_cells]...)
                            forbidden_nodes = unique(vcat([mesh.Connectivity[cell] for cell in common_cells]...))
                            delete_flag = false
                            
                            if typeindex[next_vijjj_mod] == 0 # it means that the next vertex is on edge
                                if length(intersect(forbidden_edges, edge_name_next_point)) <= 0
                                    delete_flag = true
                                end
                            else # it means that the previous vertex is on a vertex
                                if length(intersect(forbidden_nodes, edge_name_next_point)) <= 0
                                    delete_flag = true
                                end
                            end
                            
                            if delete_flag
                                index_to_delete = prev_vijjj_mod
                                deleteat!(xintersection, index_to_delete)
                                deleteat!(yintersection, index_to_delete)
                                deleteat!(typeindex, index_to_delete)
                                deleteat!(edgeORvertexID, index_to_delete)
                                if index_to_delete != 1
                                    counter += 1
                                else
                                    add_at_the_end += 1
                                end
                            end
                        end
                    end
                end
            end
            
            if counter > 0
                @debug log "deleted $(counter + add_at_the_end) edge and corner points"
            end
            
            if length(vertex_indexes) > 0
            end

            list_of_xintersections_for_all_closed_paths[j] = xintersection
            list_of_yintersections_for_all_closed_paths[j] = yintersection
            

            """
            new cleaning: 
            Before going further we need to collapse to the closest mesh node all the edges of the front that are very small
            """
            # compute all the distances between the vertexes at the front and check if some of them are smaller than the tollerance
            # x_temp = convert(Vector{Float64}, xintersection)
            # y_temp = convert(Vector{Float64}, yintersection)
            # shifted_range = vcat(2:length(xintersection), 1) # 1-based indexing
            # dxdx = (x_temp[1:length(xintersection)] - x_temp[shifted_range]) .^ 2
            # dydy = (y_temp[1:length(yintersection)] - y_temp[shifted_range]) .^ 2
            # Lcheck = sqrt.(dxdx + dydy) / sqrt(mesh.hx^2 + mesh.hy^2) .< min_ratio_front_and_edge_size
            # indexes_of_points_to_be_collapsed = findall(Lcheck)
            # x_points_to_be_collapsed = x_temp[indexes_of_points_to_be_collapsed]
            # y_points_to_be_collapsed = y_temp[indexes_of_points_to_be_collapsed]

            """
            Make a 2D table where to store info for each node found at the front. 
            The 1st column contains the TIPcell's name common with the 
            previous node in the list of nodes at the front while the second column the cell's name common with the next node.
            The nodes that have to be deleted will have same value in both columns
            """
            nodeVScommonelementtable = zeros(Int64, length(xintersection), 3)
            for nodeindex in 1:length(xintersection)
                # commonbackward contains the unique values in cellOfNodei that are in cellOfNodeim1.
                # element -1 of a list is the last element
                prev_nodeindex = mod1(nodeindex - 1, length(xintersection))
                commonbackward = findcommon(nodeindex, prev_nodeindex, typeindex, mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)
                
                # commonforward contains the unique values in cellOfNodei that are in cellOfNodeip1.
                # when nodeindex == len(xintersection)-1 then (nodeindex + 1)%len(xintersection)==0
                next_nodeindex = mod1(nodeindex + 1, length(xintersection))
                commonforward = findcommon(nodeindex, next_nodeindex, typeindex, mesh.Connectivityedgeselem, mesh.Connectivitynodeselem, edgeORvertexID)
                
                column = 1
                nodeVScommonelementtable, exitstatus = filltable(nodeVScommonelementtable, nodeindex, commonbackward, sgndDist_k, column)
                if !exitstatus
                    error("FRONT RECONSTRUCTION ERROR: two consecutive nodes does not belongs to a common cell")
                end
                
                column = 2
                nodeVScommonelementtable, exitstatus = filltable(nodeVScommonelementtable, nodeindex, commonforward, sgndDist_k, column)
                if !exitstatus
                    error("FRONT RECONSTRUCTION ERROR: two consecutive nodes does not belongs to a common cell")
                end
            end

            listofTIPcells = Int64[]
            # remove the nodes in the cells with more than 2 nodes and keep the first and the last node
            # counter = 0
            # n = length(xintersection)
            # jump = false
            for nodeindex in 1:length(xintersection)
                push!(listofTIPcells, nodeVScommonelementtable[nodeindex, 1])
            end
            # del n, jump, nodeindex, counter # В Julia сборка мусора автоматическая

            # after removing the points on the same edge, update the global list
            list_of_xintersections_for_all_closed_paths[j] = xintersection
            list_of_yintersections_for_all_closed_paths[j] = yintersection

            # In principle the following check should be activated only if the front is
            # approaching the same tip cell. The strategy is to set these shared tip cells to be negative and re-launch the code
            # At this stage if we find duplicated cells in the tip cells, than that means we should
            # impose the level set value to -machine precision in those cells and recompute the whole fractures
            # It means that we have some coalescence.

            # if j==1: fig=nothing # j=1 в Julia соответствует j=0 в Python
            # fig = plot_xy_points(anularegion, mesh, sgndDist_k, Ribbon, xintersection, yintersection, fig)
            u, c = unique(listofTIPcells, dims=1)
            # counts = countmap(listofTIPcells)
            # dup = [k for (k,v) in counts if v > 1]
            counts_dict = Dict{Int64, Int64}()
            for cell in listofTIPcells
                counts_dict[cell] = get(counts_dict, cell, 0) + 1
            end
            u_vals = collect(keys(counts_dict))
            c_vals = collect(values(counts_dict))
            dup = u_vals[c_vals .> 1]

            if length(dup) > 1
                recompute_front = true
                # set the repeated cells artificially inside the fracture
                @debug log "Recomputing the fracture front because one or more coalescing point have been found"
                @debug log "set the repeated cells artificially inside the fracture: volume error equal to $(length(dup)) cells"
                sgndDist_k[dup] .= -zero_level_set_value
                break # break here

                """
                5) - find zero vertexes, find alphas & distances 
                    Define the correct node from where compute the distance to the front
                    that node has the largest distance from the front and is inside the fracture but belongs to the tip cell  
                """
                vertexpositionwithinthecell = zeros(Int64, length(listofTIPcells))
                vertexID = zeros(Int64, length(listofTIPcells)) #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                distances = zeros(Float64, length(listofTIPcells))
                angles = zeros(Float64, length(listofTIPcells))
                xintersectionsfromzerovertex = Float64[]  #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                yintersectionsfromzerovertex = Float64[]  #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING

                # loop over the all segments at the fracture front
                # the number of segments is equal to the number of tipcells because the front is closed
                for nodeindex in 1:length(xintersection)
                    nodeindexp1 = mod1(nodeindex + 1, length(xintersection)) # take the near vertex to define an edge
                    localvertexID = Int64[]
                    localdistances = Float64[]
                    localvertexpositionwithinthecell = Int64[]
                    i = listofTIPcells[nodeindexp1]
                    # check the vertexes if they are inside or outside of the fracture
                    answer_on_vertexes = ISinsideFracture(i, mesh, sgndDist_k)
                    for jj in 1:4
                        if answer_on_vertexes[jj] # if the vertex is inside the fracture
                            p0name = mesh.Connectivity[i, jj]
                            p0x = mesh.VertexCoor[p0name, 1]
                            p0y = mesh.VertexCoor[p0name, 2]
                            push!(localvertexID, p0name)
                            push!(localvertexpositionwithinthecell, jj)
                            p1x = xintersection[nodeindex]
                            p1y = yintersection[nodeindex]
                            p2x = xintersection[nodeindexp1]
                            p2y = yintersection[nodeindexp1]
                            push!(localdistances, pointtolinedistance(p0x, p1x, p2x, p0y, p1y, p2y)) #compute the distance from the vertex to the front
                        end
                    end

                    # take the largest distance from the front
                    if length(localdistances) != 0
                        index = argmax(localdistances) 
                        
                        # take:
                        #       - name of the vertex
                        #       - local position within the cell
                        #       - distance to the front
                        vertexID[nodeindexp1] = localvertexID[index] #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                        vertexpositionwithinthecell[nodeindexp1] = localvertexpositionwithinthecell[index]
                        distances[nodeindexp1] = localdistances[index]

                        # compute the angle
                        x = mesh.VertexCoor[localvertexID[index], 1] # x coordinate of the zero vertex
                        y = mesh.VertexCoor[localvertexID[index], 2] # y coordinate of the zero vertex
                        angle_val, xint, yint = findangle(xintersection[nodeindex], yintersection[nodeindex], xintersection[nodeindexp1], yintersection[nodeindexp1], x, y, mac_precision)
                        angles[nodeindexp1] = angle_val
                        #[angle, xint, yint] = findangle(xintersection[nodeindex], yintersection[nodeindex],
                        #                                xintersection[nodeindexp1], yintersection[nodeindexp1], mesh.CenterCoor[i,1], mesh.CenterCoor[i,2])  #<--------- IT CAN BE REMOVED, IT IS ONLY FOR ONE POSSIBLE LOCAL DEBUGGING
                        if recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge
                            p_zero_vertex = Point(0, x, y)
                            p_center = Point(i, mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2])
                            p1 = Point(2, xintersection[nodeindex], yintersection[nodeindex])
                            p2 = Point(3, xintersection[nodeindexp1], yintersection[nodeindexp1])
                            sgndDist_k_new = recompute_LS_at_tip_cells(sgndDist_k_new, p_zero_vertex, p_center, p1, p2, mac_precision, area_of_a_cell, zero_level_set_value)
                        end
                        push!(xintersectionsfromzerovertex, xint) #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                        push!(yintersectionsfromzerovertex, yint) #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                    else
                        if typeindex[nodeindex] == 1 && typeindex[nodeindexp1] == 1
                            edges_node1 = mesh.Connectivitynodesedges[edgeORvertexID[nodeindex]]
                            edges_node2 = mesh.Connectivitynodesedges[edgeORvertexID[nodeindexp1]]
                            commonedge = intersect(edges_node1, edges_node2)
                            if length(commonedge) > 0
                                #            1
                                #            |
                                #         2__o__4    o is the node and the order in  connNodesEdges is [vertical_top, horizotal_left, vertical_bottom, horizotal_right]
                                #            |
                                #            3
                                # the points are on the same edge an the front it is exactly there
                                position_in_connectivity = findfirst(==(commonedge[1]), mesh.Connectivitynodesedges[edgeORvertexID[nodeindex]])
                                if position_in_connectivity !== nothing && (position_in_connectivity in [2, 4]) # horizotal_left or horizotal_right
                                    angles[nodeindexp1] = π/2.0
                                else
                                    angles[nodeindexp1] = 0.0
                                end
                                vertexID[nodeindexp1] = edgeORvertexID[nodeindexp1]
                                vertex_pos_index = findfirst(==(edgeORvertexID[nodeindexp1]), mesh.Connectivity[i, :])
                                if vertex_pos_index !== nothing
                                    vertexpositionwithinthecell[nodeindexp1] = vertex_pos_index
                                else
                                    vertexpositionwithinthecell[nodeindexp1] = 0
                                end
                                distances[nodeindexp1] = 0.0
                                x = mesh.VertexCoor[edgeORvertexID[nodeindexp1], 1]  # x coordinate of the zero vertex
                                y = mesh.VertexCoor[edgeORvertexID[nodeindexp1], 2]  # y coordinate of the zero vertex
                                xint = x
                                yint = y
                                push!(xintersectionsfromzerovertex, xint)  # <--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                                push!(yintersectionsfromzerovertex, yint)  # <--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                            end
                        else
                            error("FRONT RECONSTRUCTION ERROR: there are no nodes in the given tip cell that are inside the fracture")
                        end
                    end
                end # for nodeindex in 1:length(xintersection)

                if recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge
                end
                
                listofTIPcellsONLY = convert(Vector{Int64}, listofTIPcells) # It contains only the tip cells, not the one fully traversed
                vertexpositionwithinthecellTIPcellsONLY = convert(Vector{Int64}, vertexpositionwithinthecell)
                # distancesTIPcellsONLY=copy(distances) #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                # anglesTIPcellsONLY=copy(angles)       #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
                # vertexIDTIPcellsONLY=copy(vertexID)   #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING

                if length(xintersection) == 0
                    error("FRONT RECONSTRUCTION ERROR: front not reconstructed")
                end

                # from utility import plot_as_matrix
                # K = zeros(Float64, mesh.NumberOfElts)
                # K[listofTIPcells] = angles
                # plot_as_matrix(K, mesh)

                # from utility import plot_as_matrix
                # K = zeros(Float64, mesh.NumberOfElts)
                # K[listofTIPcells] = distances
                # plot_as_matrix(K, mesh)

                # from utility import plot_as_matrix
                # K = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
                # K[Fr_kplus1.EltTip] = Fr_kplus1.alpha
                # plot_as_matrix(K, Fr_kplus1.mesh)

                # from utility import plot_as_matrix
                # K = zeros(Float64, Fr_kplus1.mesh.NumberOfElts)
                # K[Fr_kplus1.EltTip] = Fr_kplus1.ZeroVertex
                # plot_as_matrix(K, Fr_kplus1.mesh)

                # from utility import plot_as_matrix
                # K = zeros(Float64, Fr_lstTmStp.mesh.NumberOfElts)
                # K[EltTip_k] = zrVertx_k
                # plot_as_matrix(K, Fr_lstTmStp.mesh)

                # mesh.identify_elements(listofTIPcellsONLY)
                # test=listofTIPcellsONLY
                # test1=listofTIPcellsONLY
                # for j in 2:length(listofTIPcellsONLY) # 2-based indexing in Julia
                #     element=listofTIPcellsONLY[j]
                #     test1[j]=mesh.Connectivity[element][vertexpositionwithinthecellTIPcellsONLY[j]]
                #     test[j]=vertexIDTIPcellsONLY[j]-mesh.Connectivity[element][vertexpositionwithinthecellTIPcellsONLY[j]]
                # from utility import plot_as_matrix
                # K = zeros(Float64, mesh.NumberOfElts)
                # K[listofTIPcellsONLY] = test1
                # plot_as_matrix(K, mesh)

                append!(global_list_of_TIPcells, listofTIPcells)
                append!(global_list_of_TIPcellsONLY, listofTIPcellsONLY) # listofTIPcellsONLY уже Vector{Int64}
                append!(global_list_of_distances, distances)
                append!(global_list_of_angles, angles)
                append!(global_list_of_vertexpositionwithinthecell, vertexpositionwithinthecell)
                append!(global_list_of_vertexpositionwithinthecellTIPcellsONLY, vertexpositionwithinthecellTIPcellsONLY) # vertexpositionwithinthecellTIPcellsONLY уже Vector{Int64}
                push!(list_of_xintersectionsfromzerovertex, xintersectionsfromzerovertex)
                push!(list_of_yintersectionsfromzerovertex, yintersectionsfromzerovertex)
                push!(list_of_vertexID, vertexID)
                
                fronts_dictionary["TIPcellsONLY_$j"] = listofTIPcellsONLY
                fronts_dictionary["TIPcellsANDfullytrav_$j"] = listofTIPcells #HERE THE LIST OF TIP CELLS DOES NOT CONTAIN ALL THE FULLY TRAVERSED CELLS
                fronts_dictionary["xint_$(j-1)"] = xintersection
                fronts_dictionary["yint_$(j-1)"] = yintersection
            end # if length(dup) > 1



    if !recompute_front
        """
        6) - find fully traversed elements and their alphas & distances 
        """
        # find the cells that have been passed completely by the front [CCPbF]
        # you can find them by the following reasoning:
        #
        # [CCPbF] = [cells where LS<0] - [cells at the previous channel (meaning ribbon+fracture)] - [tip cells]
        #
        # "-" means: "take away the names of"
        #
        # this is not enough, we need to account for positive cells that have been excluded from drowing the front because
        # it was having to high curvature within it. In order to find the cells I am speaking about we can use the folowing reasoning.
        #
        # [CCPbF] = [CCPbF] + neighbours of [CCPbF] - [cells at the previous channell (meaning ribbon+fracture)]  - [tip cells]
        #

        # update the levelset with the distance at the tip cells according to the distance to the reconstructed front
        # this is important in order to proper estimate the distance to the front of the fully traversed cells
        # this should not be done if we discover that we have coalescence and we would need to recompute the front location
        # according with a LS thats why we make a copy of the original sgndDist_k and we will restore it in case we see
        # that we have coalescence
        original_sgndDist_k = copy(sgndDist_k)
        sgndDist_k = sgndDist_k_new

        temp = sgndDist_k[1:length(sgndDist_k)] # 1-based indexing
        temp[temp .> 0] .= 0
        negative_cells = findall(!iszero, temp) # findall for nonzero, 1-based
        fullyfractured = setdiff(negative_cells, eltsChannel)
        fullyfractured = setdiff(fullyfractured, global_list_of_TIPcells)
        positivefullyfractured = setdiff(unique(mesh.NeiElements[fullyfractured, :][:]), global_list_of_TIPcells) # Flatten NeiElements
        positivefullyfractured = setdiff(positivefullyfractured, fullyfractured)
        positivefullyfractured = setdiff(positivefullyfractured, eltsChannel)
        if length(positivefullyfractured) > 0
            fullyfractured = vcat(fullyfractured, positivefullyfractured)
            negative_cells = vcat(negative_cells, positivefullyfractured)
        end

        left_elem, right_elem, bottom_elem, top_elem = 1, 2, 3, 4 # 1-based indexing

        if length(fullyfractured) > 0
            if recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge
                """
                If you jump here it means that 
                previously the Level set at the tip cells have been redefined
                now we have to recompute the level set in the negative cells
                because when coalescence is enforced then level set at some cells was set to -machine precision
                then the front is reconstructed and finally we are here, about to compute the alphas and distances to the front in the 
                fully traversed cells. We need to really define the level set in the cells where we set -machine precision
                """
                # level set known and unknown, cell names where the LS is known, cracked elements (including tip), mesh , empty, Specific cells that I need inwards

                # fig1 = plot_cells(anularegion, mesh, sgndDist_k, Ribbon, negative_cells, nothing, true)

                sgndDist_k = 1e50 * ones(Float64, mesh.NumberOfElts)
                sgndDist_k[global_list_of_TIPcellsONLY] = original_sgndDist_k[global_list_of_TIPcellsONLY]
                SolveFMM(sgndDist_k, global_list_of_TIPcellsONLY, negative_cells, mesh, setdiff(anularegion, negative_cells), negative_cells)
                # delta_sgndDist_k = sgndDist_k - original_sgndDist_k

                # Usage of SolveFMM:
                # 1st arg: vector with the LS value everywhere
                # 2nd arg: list of positions (or cell names) where the LS is KNOWN => it should be a set of closed fronts!
                # 3rd arg: unknown need ??
                # 4th arg: mesh obj
                # 5th arg: name of the cells where to compute the LS => we expect positive LS values here!
                # 6th arg: name of the cells where to compute the LS => we expect negative LS values here!
            end

            fullyfractured_angle = Float64[]
            fullyfractured_distance = Float64[]
            fullyfractured_vertexID = Int64[]
            fullyfractured_vertexpositionwithinthecell = Int64[]
            # loop over the fullyfractured cells
            for fullyfracturedcell in 1:length(fullyfractured) # 1-based loop
                i = fullyfractured[fullyfracturedcell]
                """
                you are in cell i
                take the level set at the center of the neighbors cells 
                    _   _   _   _   _   _
                | _ | _ | _ | _ | _ | _ |
                | _ | _ | _ | _ | _ | _ |
                | _ | e | a | f | _ | _ |
                | _ | _ 3 _ 2 _ | _ | _ |              
                | _ | d | i | b | _ | _ |
                | _ | _ 0 _ 1 _ | _ | _ |
                | _ | h | c | g | _ | _ |
                | _ | _ | _ | _ | _ | _ |

                                        1     2      3      4
                NeiElements[i]->[left, right, bottom, up] (1-based)
                """

                a = mesh.NeiElements[i, top_elem]
                b = mesh.NeiElements[i, right_elem]
                c = mesh.NeiElements[i, bottom_elem]
                d = mesh.NeiElements[i, left_elem]
                e = mesh.NeiElements[d, top_elem]
                f = mesh.NeiElements[b, top_elem]
                g = mesh.NeiElements[b, bottom_elem]
                h = mesh.NeiElements[d, bottom_elem]

                hcid = sgndDist_k[[h, c, i, d]]
                cgbi = sgndDist_k[[c, g, b, i]]
                ibfa = sgndDist_k[[i, b, f, a]]
                diae = sgndDist_k[[d, i, a, e]]
                LS = [hcid, cgbi, ibfa, diae]
                hcid_mean = mean(sgndDist_k[[h, c, i, d]])
                cgbi_mean = mean(sgndDist_k[[c, g, b, i]])
                ibfa_mean = mean(sgndDist_k[[i, b, f, a]])
                diae_mean = mean(sgndDist_k[[d, i, a, e]])
                LS_means = [hcid_mean, cgbi_mean, ibfa_mean, diae_mean]
                localvertexpositionwithinthecell = argmin(LS_means) # 1-based index
                push!(fullyfractured_vertexpositionwithinthecell, localvertexpositionwithinthecell)
                push!(fullyfractured_distance, abs(LS_means[localvertexpositionwithinthecell]))
                push!(fullyfractured_vertexID, mesh.Connectivity[i, localvertexpositionwithinthecell])
                chosenLS = LS[localvertexpositionwithinthecell]
                # compute the angle
                dLSdy = 0.5 * mesh.hy * (chosenLS[4] + chosenLS[3] - chosenLS[2] - chosenLS[1]) # Adjusted for 1-based indexing
                dLSdx = 0.5 * mesh.hx * (chosenLS[3] + chosenLS[2] - chosenLS[4] - chosenLS[1]) # Adjusted for 1-based indexing
                if dLSdy == 0.0 && dLSdx != 0.0
                    push!(fullyfractured_angle, 0.0)
                elseif dLSdy != 0.0 && dLSdx == 0.0
                    push!(fullyfractured_angle, π)
                elseif dLSdy != 0.0 && dLSdx != 0.0
                    push!(fullyfractured_angle, atan(abs(dLSdy) / abs(dLSdx)))
                else
                    error("FRONT RECONSTRUCTION ERROR: minimum of the function has been found, not expected")
                end
            end

            # finally append these informations to what computed before
            # extend in Python is vcat or append! for individual elements in Julia
            global_list_of_TIPcells = vcat(global_list_of_TIPcells, fullyfractured)
            global_list_of_distances = vcat(global_list_of_distances, fullyfractured_distance)
            global_list_of_angles = vcat(global_list_of_angles, fullyfractured_angle)
            global_list_of_vertexpositionwithinthecell = vcat(global_list_of_vertexpositionwithinthecell, fullyfractured_vertexpositionwithinthecell)

            # vertexID = vertexID + fullyfractured_vertexID #<--------- IT CAN BE REMOVED, IT IS ONLY FOR LOCAL DEBUGGING
        end

        # Cells status list store the status of all the cells in the domain
        # >>>>  update ONLY the position of the tip cells <<<<<
        # todo: the updating of the cell status seems to be duplicated in the UpdateListsFromContinuousFrontRec(..)
        CellStatusNew = zeros(Int, mesh.NumberOfElts)
        CellStatusNew[eltsChannel] .= 1
        for cell in global_list_of_TIPcells # Using 'cell' instead of 'list' to avoid confusion
            CellStatusNew[cell] = 2
        end
        CellStatusNew[Ribbon] .= 3

        # In principle the following check should be activated only if the front is
        # approaching the same tip cell
        # the strategy is to set these shared tip cells to be negative and re-launch the code
        u, c = unique_counts(global_list_of_TIPcells) # Assuming a helper function or using StatsBase.countmap
        # Simple implementation of finding duplicates:
        # u = unique(global_list_of_TIPcells)
        # c = [count(==(x), global_list_of_TIPcells) for x in u]
        dup = u[findall(x -> x > 1, c)]
        
        if length(dup) > 1
            recompute_front = true
            sgndDist_k = copy(original_sgndDist_k)
            # We compute the front and we come here. This means that the front is entering twice the same cell.
            # The strategy was to set to -machine precision the level set in those cells
            # These cells then may become fully traversed and thus the proper calculation of the distance to the front is
            # needed. In the next front reconstruction we will propagate inward the level set from the tip cells
            recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = true
            # set the repeated cells artificially inside the fracture
            # log.debug("set the repeated cells artificially inside the fracture: volume error equal to " * string(length(dup)) * " cells")
            @debug "set the repeated cells artificially inside the fracture: volume error equal to $(length(dup)) cells"
            sgndDist_k[dup] .= -zero_level_set_value
            # fig1 = plot_cells(anularegion, mesh, sgndDist_k, Ribbon, dup, nothing, true)
        else
            recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = false
        end
    end

    if recompute_front
        """
        ##################################################
        #                                                #
        #              RECOMPUTE THE FRONT!              #
        # -because coalescence as been detected          #
        # -because we need to compute LS on more cells   #
        ##################################################
        """
        global_list_of_TIPcells,
        global_list_of_TIPcellsONLY,
        global_list_of_distances,
        global_list_of_angles,
        CellStatusNew,
        global_list_of_newRibbon,
        global_list_of_vertexpositionwithinthecell,
        global_list_of_vertexpositionwithinthecellTIPcellsONLY,
        correct_size_of_pstv_region,
        sgndDist_k, Ffront, number_of_fronts, fronts_dictionary = reconstruct_front_continuous(sgndDist_k,
                                                                                                anularegion,
                                                                                                Ribbon,
                                                                                                eltsChannel,
                                                                                                mesh,
                                                                                                recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                                                                lstTmStp_EltCrack0)
    else
        # find the number of fronts
        number_of_fronts = length(list_of_xintersections_for_all_closed_paths)
        # find the new ribbon cells
        global_list_of_newRibbon = Int64[] # Явно указываем тип
        newRibbon = unique(vec(mesh.NeiElements[global_list_of_TIPcellsONLY, :]))
        temp = sgndDist_k[newRibbon]
        temp[temp .> 0] .= 0 
        nonzero_indices = findall(!iszero, temp)
        newRibbon = newRibbon[nonzero_indices]
        # np.setdiff1d(newRibbon, np.asarray(global_list_of_TIPcellsONLY)) -> setdiff(newRibbon, global_list_of_TIPcellsONLY)
        newRibbon = setdiff(newRibbon, global_list_of_TIPcellsONLY)
        # extend ... tolist() -> append!
        append!(global_list_of_newRibbon, newRibbon)
        correct_size_of_pstv_region = [true, false, false]

        # compute the coordinates for the Ffront variable in the Fracture object
        # for each cell where the front is passing you have to list the coordinates with the intersection with
        # the first edge and the second one
        xinters4all_closed_paths_1 = Vector{Float64}[] # Список векторов
        xinters4all_closed_paths_2 = Vector{Float64}[]
        yinters4all_closed_paths_1 = Vector{Float64}[]
        yinters4all_closed_paths_2 = Vector{Float64}[]
        
        for jj in 1:length(list_of_xintersections_for_all_closed_paths) # 1-based loop
            xintersection = list_of_xintersections_for_all_closed_paths[jj]
            yintersection = list_of_yintersections_for_all_closed_paths[jj]
            push!(xintersection, xintersection[1]) # close the front (1-based)
            push!(yintersection, yintersection[1]) # close the front (1-based)
            push!(xinters4all_closed_paths_1, xintersection[1:end-1])
            push!(xinters4all_closed_paths_2, xintersection[2:end])
            push!(yinters4all_closed_paths_1, yintersection[1:end-1])
            push!(yinters4all_closed_paths_2, yintersection[2:end])
        end
        
        xinters4all_closed_paths_1 = itertools_chain_from_iterable(xinters4all_closed_paths_1)
        xinters4all_closed_paths_2 = itertools_chain_from_iterable(xinters4all_closed_paths_2)
        yinters4all_closed_paths_1 = itertools_chain_from_iterable(yinters4all_closed_paths_1)
        yinters4all_closed_paths_2 = itertools_chain_from_iterable(yinters4all_closed_paths_2)
        
        if length(xinters4all_closed_paths_1) == length(yinters4all_closed_paths_1) == 
        length(xinters4all_closed_paths_2) == length(yinters4all_closed_paths_2)
            Ffront = hcat(xinters4all_closed_paths_1, yinters4all_closed_paths_1, 
                        xinters4all_closed_paths_2, yinters4all_closed_paths_2)
        else
            @warn "Mismatch in lengths of intersection arrays for Ffront"
            Ffront = Array{Float64}(undef, 0, 4)
        end
    end

    if !recompute_front # the following should be executed only if the front has not been recomputed,
        # otherwise fronts_dictionary has already been done

        # construct the informations about the cracks
        if fronts_dictionary["number_of_fronts"] == 2
            if length(fullyfractured) > 0 # divide the fully traversed cells from left to right
                # mesh.CenterCoor[fullyfractured, 1][np.newaxis] -> reshape(mesh.CenterCoor[fullyfractured, 2], 1, :) (1-based, 2nd column)
                # np.column_stack((fronts_dictionary['xint_0'], fronts_dictionary['yint_0'])) -> hcat(fronts_dictionary["xint_0"], fronts_dictionary["yint_0"])
                isfullyfractured0 = ray_tracing_numpy(
                    reshape(mesh.CenterCoor[fullyfractured, 1], 1, :),
                    reshape(mesh.CenterCoor[fullyfractured, 2], 1, :),
                    hcat(fronts_dictionary["xint_0"], fronts_dictionary["yint_0"])
                )
                # np.nonzero(isfullyfractured0) -> findall(!iszero, isfullyfractured0) -> findall(isfullyfractured0 .!= 0)
                nonzero_indices_0 = findall(isfullyfractured0 .!= 0)
                # fullyfractured[np.nonzero(isfullyfractured0)] -> fullyfractured[nonzero_indices_0]
                fullyfractured0 = fullyfractured[nonzero_indices_0]
                # np.setdiff1d(fullyfractured, fullyfractured0) -> setdiff(fullyfractured, fullyfractured0)
                fullyfractured1 = setdiff(fullyfractured, fullyfractured0)
                # .extend(...) -> append!
                append!(fronts_dictionary["TIPcellsANDfullytrav_0"], fullyfractured0)
                append!(fronts_dictionary["TIPcellsANDfullytrav_1"], fullyfractured1)
            end

            # np.where(sgndDist_k <= 0) -> findall(sgndDist_k .<= 0)
            indx = findall(sgndDist_k .<= 0)
            
            crack_cells = ray_tracing_numpy(
                reshape(mesh.CenterCoor[indx, 1], 1, :),
                reshape(mesh.CenterCoor[indx, 2], 1, :),
                hcat(fronts_dictionary["xint_0"], fronts_dictionary["yint_0"])
            )
            #####PLOT TO CHECK THE RESULT
            # plot_ray_tracing_numpy_results(mesh,
            #                                mesh.CenterCoor[indx, 1], # 1-based
            #                                mesh.CenterCoor[indx, 2], # 1-based
            #                                hcat(fronts_dictionary['xint_0'], fronts_dictionary['yint_0']),
            #                                crack_cells)
            #####

            # np.where(crack_cells==1)[0] -> findall(crack_cells .== 1)
            # np.concatenate((A, B)) -> vcat(A, B)
            # np.unique(...) -> unique(...)
            fronts_dictionary["crackcells_0"] = unique(vcat(
                indx[findall(crack_cells .== 1)],
                fronts_dictionary["TIPcellsONLY_0"]
            ))

            crack_cells = ray_tracing_numpy(
                reshape(mesh.CenterCoor[indx, 1], 1, :),
                reshape(mesh.CenterCoor[indx, 2], 1, :),
                hcat(fronts_dictionary["xint_1"], fronts_dictionary["yint_1"])
            )
            #####PLOT TO CHECK THE RESULT
            # plot_ray_tracing_numpy_results(mesh,
            #                                mesh.CenterCoor[indx, 1], # 1-based
            #                                mesh.CenterCoor[indx, 2], # 1-based
            #                                hcat(fronts_dictionary['xint_1'], fronts_dictionary['yint_1']),
            #                                crack_cells)
            #####

            fronts_dictionary["crackcells_1"] = unique(vcat(
                indx[findall(crack_cells .== 1)],
                fronts_dictionary["TIPcellsONLY_1"]
            ))
            
            #####PLOT TO CHECK THE RESULT (закомментировано)
            # newindx=np.concatenate((fronts_dictionary['crackcells_1'],fronts_dictionary['crackcells_0']))
            # crack_cells = np.ones(newindx.size,np.bool_)
            # plot_ray_tracing_numpy_results(mesh,
            #                                mesh.CenterCoor[newindx, 1], # 1-based
            #                                mesh.CenterCoor[newindx, 2], # 1-based
            #                                hcat(vcat(fronts_dictionary['xint_0'],fronts_dictionary['xint_1']),
            #                                     vcat(fronts_dictionary['yint_0'], fronts_dictionary['yint_1'])),
            #                                crack_cells)
            #
            # crack_cells = np.ones(fronts_dictionary['crackcells_0'].size, np.bool_)
            # plot_ray_tracing_numpy_results(mesh,
            #                                mesh.CenterCoor[fronts_dictionary['crackcells_0'], 1], # 1-based
            #                                mesh.CenterCoor[fronts_dictionary['crackcells_0'], 2], # 1-based
            #                                hcat(fronts_dictionary['xint_0'], fronts_dictionary['yint_0']),
            #                                crack_cells)
            # crack_cells = np.ones(fronts_dictionary['crackcells_1'].size, np.bool_)
            # plot_ray_tracing_numpy_results(mesh,
            #                                mesh.CenterCoor[fronts_dictionary['crackcells_1'], 1], # 1-based
            #                                mesh.CenterCoor[fronts_dictionary['crackcells_1'], 2], # 1-based
            #                                hcat(fronts_dictionary['xint_1'], fronts_dictionary['yint_1']),
            #                                crack_cells)
            #####
            
            if lstTmStp_EltCrack0 !== nothing && fronts_dictionary["number_of_fronts"] == 2
                # check if the left fracture is always the same
                # np.in1d(lstTmStp_EltCrack0, fronts_dictionary['crackcells_0']) -> lstTmStp_EltCrack0 .∈ Ref(fronts_dictionary["crackcells_0"])
                # np.ndarray.astype(..., int) -> Int.(...)
                # np.sum(...) > 0 -> sum(...) > 0
                # not (sum(...) > 0) -> !(sum(...) > 0) или sum(...) <= 0
                if !(sum(Int.(lstTmStp_EltCrack0 .∈ Ref(fronts_dictionary["crackcells_0"]))) > 0)
                    temp_crackcells_0 = fronts_dictionary["crackcells_0"]
                    temp_xint_0 = fronts_dictionary["xint_0"]
                    temp_TIPcellsONLY_0 = fronts_dictionary["TIPcellsONLY_0"]
                    temp_TIPcellsANDfullytrav_0 = fronts_dictionary["TIPcellsANDfullytrav_0"]
                    
                    fronts_dictionary["crackcells_0"] = fronts_dictionary["crackcells_1"]
                    fronts_dictionary["xint_0"] = fronts_dictionary["xint_1"]
                    fronts_dictionary["TIPcellsONLY_0"] = fronts_dictionary["TIPcellsONLY_1"]
                    fronts_dictionary["TIPcellsANDfullytrav_0"] = fronts_dictionary["TIPcellsANDfullytrav_1"]
                    
                    fronts_dictionary["crackcells_1"] = temp_crackcells_0
                    fronts_dictionary["xint_1"] = temp_xint_0
                    fronts_dictionary["TIPcellsONLY_1"] = temp_TIPcellsONLY_0
                    fronts_dictionary["TIPcellsANDfullytrav_1"] = temp_TIPcellsANDfullytrav_0
                end
            end
        else
            fronts_dictionary["crackcells_0"] = nothing
            fronts_dictionary["crackcells_1"] = nothing
        end
    end

    # plot_final_reconstruction(mesh,
    #                           list_of_xintersections_for_all_closed_paths,
    #                           list_of_yintersections_for_all_closed_paths,
    #                           anularegion,
    #                           sgndDist_k,
    #                           global_list_of_newRibbon,
    #                           global_list_of_TIPcells,
    #                           list_of_xintersectionsfromzerovertex,
    #                           list_of_yintersectionsfromzerovertex,
    #                           list_of_vertexID, Ribbon)

    return (
        convert(Array, global_list_of_TIPcells), 
        convert(Array, global_list_of_TIPcellsONLY),
        convert(Array, global_list_of_distances),
        convert(Array, global_list_of_angles),
        CellStatusNew, # Array (Matrix) типа Int
        convert(Array, global_list_of_newRibbon),
        convert(Array, global_list_of_vertexpositionwithinthecell),
        convert(Array, global_list_of_vertexpositionwithinthecellTIPcellsONLY),
        correct_size_of_pstv_region, # Vector{Bool}
        sgndDist_k, # Vector{Float64}
        Ffront, # Matrix{Float64}
        number_of_fronts, # Int
        fronts_dictionary # Dict
    )



function UpdateListsFromContinuousFrontRec(
    newRibbon::Vector{Int},
    sgndDist_k::Vector{Float64},
    EltChannel_k::Vector{Int},
    listofTIPcells::Vector{Int},
    listofTIPcellsONLY::Vector{Int},
    mesh::Any
)::Tuple{Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}}

    # the listofTIPcells is not including the fully traversed
    fully_traversed::Vector{Int} = setdiff(listofTIPcells, listofTIPcellsONLY)
    EltChannel_k::Vector{Int} = vcat(EltChannel_k, fully_traversed)
    # EltChannel_k_1 = setdiff(findall(sgndDist_k .< 0) .- 1, listofTIPcellsONLY) # another old way of computing EltChannel_k_1

    EltTip_k::Vector{Int} = listofTIPcellsONLY
    EltCrack_k::Vector{Int} = vcat(EltChannel_k, listofTIPcellsONLY)

    # from utility import plot_as_matrix
    # K = zeros(Float64, mesh.NumberOfElts)
    # K[findall(sgndDist_k .< 0)] .= 1
    # K[listofTIPcells] .= 2
    # plot_as_matrix(K, mesh)

    if length(unique(EltCrack_k)) != length(EltCrack_k)
        # uncomment this to see the source of the error:
        # plot = plot_cell_lists(mesh, listofTIPcellsONLY, fig=nothing, mycolor="b", mymarker=".", shiftx=0.0,
        #                        shifty=0.01, annotate_cellName=false, grid=true)
        # plot = plot_cell_lists(mesh, EltChannel_k, fig=plot, mycolor="g", mymarker="_", shiftx=0.01, shifty=0.01,
        #                        annotate_cellName=false, grid=true)
        message = """FRONT RECONSTRUCTION ERROR: 
the source of this error can be found because of two reasons. 
1)The first reason is that the front is entering more than 1 time the same cell 
2)The second reason is more in depth in how the scheme works.
    If one fracture front is receding because of an artificial deletion of points at the front then
    some of the tip elements they will became channel element of the previous time step. 

>>> You can solve this problem by refining more the mesh. <<<

PS: After removing a point, the LevelSet is recomputed by assuming the distance  
to the reconstructed front.This might result in a numerical recession of the front 
 but at worst it can be detected and solved by spatial or temporal refinement."""
        error(message)
    end

    EltRibbon_k::Vector{Int} = newRibbon

    # Cells status list store the status of all the cells in the domain
    CellStatus_k::Vector{Int} = zeros(Int, mesh.NumberOfElts) 
    CellStatus_k[EltChannel_k] .= 1
    CellStatus_k[EltTip_k] .= 2
    CellStatus_k[EltRibbon_k] .= 3
    # from utility import plot_as_matrix
    # K = zeros(Float64, mesh.NumberOfElts)
    # plot_as_matrix(CellStatus_k, mesh)
    
    return EltChannel_k, EltTip_k, EltCrack_k, EltRibbon_k, CellStatus_k, fully_traversed
end

# check if:
"""
    you_advance_more_than_2_cells(fully_traversed_k, EltTip_last_tmstp, NeiElements, oldfront, newfront, mesh)

The following function is checking if there exist a fully traversed element that is not a neighbour tip cells
at the end of the previous time step. If this is the case then fail the time step and take a smaller one.

# Arguments
- `fully_traversed_k::Vector{Int}`: names of the fully traversed elements at the current time step
- `EltTip_last_tmstp::Vector{Int}`: names of the tip cells at the end of the previous time step
- `NeiElements::Matrix{Int}`: list of neighbour elements for each cell
- `oldfront::Union{Nothing, Matrix{Float64}}`: coordinates defining the front at the end of the previous time step -- used only for plotting
- `newfront::Union{Nothing, Matrix{Float64}}`: coordinates defining the front at the end of the current time step -- used only for plotting
- `mesh::Any`: mesh object -- used only for plotting

# Returns
- `Bool`: True or False
"""
function you_advance_more_than_2_cells(
    fully_traversed_k::Vector{Int},
    EltTip_last_tmstp::Vector{Int},
    NeiElements::Matrix{Int},
    oldfront::Union{Nothing, Matrix{Float64}},
    newfront::Union{Nothing, Matrix{Float64}},
    mesh::Any
)::Bool

    difference = setdiff(fully_traversed_k, unique(NeiElements[EltTip_last_tmstp, :][:]))
    if length(difference) > 0
        # @debug log "$(length(difference))"
        # plot_two_fronts(mesh, newfront=newfront, oldfront=oldfront, fig=nothing, grid=true, cells=difference)
        return true
    else  # == 0
        return false
    end
end



        