# -*- coding: utf-8 -*-
"""
This file is a part of JFrac.
Realization of Pyfrac on Julia language.

"""
module Anisotropy
include("level_set.jl")
include("volume_integral.jl")

using .LevelSet: reconstruct_front_LS_gradient
using .VolumeIntegral: Integral_over_cell

export TI_plain_strain_modulus, projection_from_ribbon, find_angle, construct_polygon, projection_from_ribbon_LS_gradient, get_toughness_from_cellCenter

"""
    projection_from_ribbon(ribbon_elts, channel_elts, mesh, sgnd_dist)

This function finds the projection of the ribbon cell centers onto the fracture front.
It returns the angle inscribed by the perpendiculars drawn from the ribbon cell centers.

# Arguments
- `ribbon_elts::Vector{Int}`: list of ribbon elements
- `channel_elts::Vector{Int}`: list of channel elements
- `mesh::CartesianMesh`: the Cartesian mesh object
- `sgnd_dist::Vector{Float64}`: signed distance function (level set data)

# Returns
- `::Vector{Float64}`: angles (in radians) for each ribbon element
"""
"""
    projection_from_ribbon(ribbon_elts, channel_elts, mesh, sgnd_dist)

This function finds the projection of the ribbon cell centers onto the fracture front.
It returns the angle inscribed by the perpendiculars drawn from the ribbon cell centers.

# Arguments
- `ribbon_elts::Vector{Int}`: list of ribbon elements
- `channel_elts::Vector{Int}`: list of channel elements
- `mesh::CartesianMesh`: the Cartesian mesh object
- `sgnd_dist::Vector{Float64}`: signed distance function (level set data)

# Returns
- `::Vector{Float64}`: angles (in radians) for each ribbon element
"""
function projection_from_ribbon(ribbon_elts::Vector{Int}, channel_elts::Vector{Int}, mesh::CartesianMesh, sgnd_dist::Vector{Float64})::Vector{Float64}

    # reconstruct front to get tip cells from the given level set
    elt_tip, l_tip, alpha_tip, CellStatus = reconstruct_front_LS_gradient(sgnd_dist,
                                                            setdiff(1:mesh.NumberOfElts, channel_elts),
                                                            channel_elts,
                                                            mesh)
    
    # get the filling fraction to find partially filled tip cells
    FillFrac = Integral_over_cell(elt_tip, alpha_tip, l_tip, mesh, "A") / mesh.EltArea

    # taking partially filled as the current tip
    partly_filled = findall(FillFrac .< 0.999999)
    elt_tip = elt_tip[partly_filled]
    l_tip = l_tip[partly_filled]
    alpha_tip = alpha_tip[partly_filled]

    zero_vertex_tip = find_zero_vertex(elt_tip, sgnd_dist, mesh)
    # construct the polygon
    smthed_tip, a, b, c, pnt_lft, pnt_rgt, neig_lft, neig_rgt = construct_polygon(elt_tip,
                                                                                  l_tip,
                                                                                  alpha_tip,
                                                                                  mesh,
                                                                                  zero_vertex_tip)
    if any(isnan, smthed_tip)
        return [NaN]  # Возвращаем вектор, а не скаляр
    end

    zr_vrtx_smthed_tip = find_zero_vertex(smthed_tip, sgnd_dist, mesh)
    alpha = find_angle(ribbon_elts,
                            smthed_tip,
                            zr_vrtx_smthed_tip,
                            a,
                            b,
                            c,
                            pnt_lft[:, 1],
                            pnt_lft[:, 2],
                            pnt_rgt[:, 1],
                            pnt_rgt[:, 2],
                            neig_lft,
                            neig_rgt,
                            mesh)

    return alpha
end

#-----------------------------------------------------------------------------------------------------------------------
"""
    find_angle(elt_ribbon, elt_tip, zr_vrtx_tip, a_tip, b_tip, c_tip, x_lft, y_lft, x_rgt, y_rgt, neig_lft, neig_rgt, mesh)

This function calculates the angle inscribed by the perpendiculars on the given polygon. The polygon is provided
in the form of equations of edges of the polygon (with the form ax+by+c=0) and the left and right points of the
front line in the given tip elements.

# Arguments
- `elt_ribbon::Vector{Int}`: ribbon elements
- `elt_tip::Vector{Int}`: tip elements
- `zr_vrtx_tip::Vector{Int}`: zero vertex tip
- `a_tip::Vector{Float64}`: coefficient a of line equation ax + by + c = 0
- `b_tip::Vector{Float64}`: coefficient b of line equation ax + by + c = 0
- `c_tip::Vector{Float64}`: coefficient c of line equation ax + by + c = 0
- `x_lft::Vector{Float64}`: x-coordinates of left points
- `y_lft::Vector{Float64}`: y-coordinates of left points
- `x_rgt::Vector{Float64}`: x-coordinates of right points
- `y_rgt::Vector{Float64}`: y-coordinates of right points
- `neig_lft::Vector{Int}`: left neighbors
- `neig_rgt::Vector{Int}`: right neighbors
- `mesh::CartesianMesh`: mesh object

# Returns
- `::Vector{Float64}`: angles in radians
"""
function find_angle(elt_ribbon::Vector{Int}, 
                   elt_tip::Vector{Int}, 
                   zr_vrtx_tip::Vector{Int},
                   a_tip::Vector{Float64}, 
                   b_tip::Vector{Float64}, 
                   c_tip::Vector{Float64},
                   x_lft::Vector{Float64}, 
                   y_lft::Vector{Float64},
                   x_rgt::Vector{Float64}, 
                   y_rgt::Vector{Float64},
                   neig_lft::Vector{Int},
                   neig_rgt::Vector{Int},
                   mesh::CartesianMesh)::Vector{Float64}

    closest_tip_cell = zeros(Int, length(elt_ribbon))
    dist_ribbon = zeros(Float64, length(elt_ribbon))
    alpha = zeros(Float64, length(elt_ribbon))
    
    for i in 1:length(elt_ribbon)
        # min dist from the front lines of a ribbon cells
        dist_front_line = zeros(Float64, length(elt_tip))
        point_at_grid_line = zeros(Int, length(elt_tip))

        # loop over tip cells for the current ribbon cell
        for j in 1:length(elt_tip)
            if x_rgt[j] - x_lft[j] == 0
                # if parallel to y-axis
                xx = mesh.CenterCoor[elt_ribbon[i], 1]
                yy = -c_tip[j]
            else
                slope_tip_line = (y_rgt[j] - y_lft[j]) / (x_rgt[j] - x_lft[j])
                m = -1.0 / slope_tip_line # slope perp to the tip line
                intrcpt = mesh.CenterCoor[elt_ribbon[i], 2] - m * mesh.CenterCoor[elt_ribbon[i], 1] #intercept
                # x-coordinate of the point where the perpendicular intersects the drawn perpendicular
                xx = -(intrcpt + c_tip[j]) / (a_tip[j] + m)
                # y-coordinate of the point where the perpendicular intersects the drawn perpendicular
                yy = m * xx + intrcpt
            end

            if x_lft[j] > xx || x_rgt[j] < xx || min(y_lft[j], y_rgt[j]) > yy || max(y_lft[j], y_rgt[j]) < yy
                # if the intersection point is out of the tip cell
                dist_lft_pnt = sqrt((mesh.CenterCoor[elt_ribbon[i], 1] - x_lft[j])^2 + 
                                   (mesh.CenterCoor[elt_ribbon[i], 2] - y_lft[j])^2)
                dist_rgt_pnt = sqrt((mesh.CenterCoor[elt_ribbon[i], 1] - x_rgt[j])^2 + 
                                   (mesh.CenterCoor[elt_ribbon[i], 2] - y_rgt[j])^2)
                # take the distance of either the left or the right point depending upon which is closer to the ribbon
                dist_front_line[j] = min(dist_lft_pnt, dist_rgt_pnt)
                # save which (right of left) point on the front line is closer to the ribbon cell center
                if dist_lft_pnt < dist_rgt_pnt
                    point_at_grid_line[j] = 1
                else
                    point_at_grid_line[j] = 2
                end
            else
                # if the intersection point of the front line and the perpendicular drawn from the zero vertex is the
                # closest point to the riboon cell center
                dist_front_line[j] = abs(mesh.CenterCoor[elt_ribbon[i], 1] * a_tip[j] + 
                                       mesh.CenterCoor[elt_ribbon[i], 2] + c_tip[j]) / 
                                   sqrt(a_tip[j]^2 + 1.0) # distance calculated by
                                                           # min distance to a line
                                                           # from a point formula
            end
        end

        closest_tip_cell[i] = argmin(dist_front_line)

        if point_at_grid_line[closest_tip_cell[i]] == 0
            # if the closest point is the intersection point of the perpendicular
            y = mesh.CenterCoor[elt_ribbon[i], 2]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            # finding angle using arc cosine
            alpha[i] = acos(round(dist_front_line[closest_tip_cell[i]] / 
                                abs(x - mesh.CenterCoor[elt_ribbon[i], 1]), digits=5))
            dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
        elseif point_at_grid_line[closest_tip_cell[i]] == 1
            # if the closest point is the left most point on the front line
            y = mesh.CenterCoor[elt_ribbon[i], 2]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = acos(round(dist_front_line[closest_tip_cell[i]] / 
                                     abs(x - mesh.CenterCoor[elt_ribbon[i], 1]), digits=5))
            x = (-y - c_tip[neig_lft[closest_tip_cell[i]]]) / a_tip[neig_lft[closest_tip_cell[i]]]
            alpha_nei = acos(round(dist_front_line[closest_tip_cell[i]] / 
                                 abs(x - mesh.CenterCoor[elt_ribbon[i], 1]), digits=5))
            alpha[i] = (alpha_closest + alpha_nei) / 2
        elseif point_at_grid_line[closest_tip_cell[i]] == 2
            # if the closest point is the right most point on the front line
            y = mesh.CenterCoor[elt_ribbon[i], 2]
            x = (-y - c_tip[closest_tip_cell[i]]) / a_tip[closest_tip_cell[i]]
            alpha_closest = acos(round(dist_front_line[closest_tip_cell[i]] / 
                                     abs(x - mesh.CenterCoor[elt_ribbon[i], 1]), digits=5))
            x = (-y - c_tip[neig_rgt[closest_tip_cell[i]]]) / a_tip[neig_rgt[closest_tip_cell[i]]]
            alpha_nei = acos(round(dist_front_line[closest_tip_cell[i]] / 
                                 abs(x - mesh.CenterCoor[elt_ribbon[i], 1]), digits=5))
            alpha[i] = (alpha_closest + alpha_nei) / 2
        end

        dist_ribbon[i] = dist_front_line[closest_tip_cell[i]]
    end

    # the code below finds the ribbon cells directly below or above the tip cells with ninety degrees angle and sets
    # them to have ninety degrees angle as well. Similarly, the ribbon cells directly on the left or right of the tip
    # cells with zero degrees angle are set to have zero angles.
    zero_angle = findall(x_lft .== x_rgt)
    for i in 1:length(zero_angle)
        if zr_vrtx_tip[zero_angle[i]] == 0 || zr_vrtx_tip[zero_angle[i]] == 3
            left_in_ribbon = Int[]
            for j in 1:3
                left_in_ribbon = findall(elt_ribbon .== elt_tip[zero_angle[i]] - (j))
                if length(left_in_ribbon) > 0
                    break
                end
            end
            if length(left_in_ribbon) > 0
                alpha[left_in_ribbon] .= 0.0
                dist_ribbon[left_in_ribbon] .= abs(abs(x_rgt[zero_angle[i]]) - 
                                                 abs(mesh.CenterCoor[elt_ribbon[left_in_ribbon], 1]))
            end
        end
        if zr_vrtx_tip[zero_angle[i]] == 1 || zr_vrtx_tip[zero_angle[i]] == 2
            rgt_in_ribbon = Int[]
            for j in 1:3
                rgt_in_ribbon = findall(elt_ribbon .== elt_tip[zero_angle[i]] + (j))
                if length(rgt_in_ribbon) > 0
                    break
                end
            end
            if length(rgt_in_ribbon) > 0
                alpha[rgt_in_ribbon] .= 0.0
                dist_ribbon[rgt_in_ribbon] .= abs(abs(x_rgt[zero_angle[i]]) - 
                                                abs(mesh.CenterCoor[elt_ribbon[rgt_in_ribbon], 1]))
            end
        end
    end

    ninety_angle = findall(y_lft .== y_rgt)
    for i in 1:length(ninety_angle)
        if zr_vrtx_tip[ninety_angle[i]] == 0 || zr_vrtx_tip[ninety_angle[i]] == 1
            btm_in_ribbon = Int[]
            for j in 1:3
                btm_in_ribbon = findall(elt_ribbon .== elt_tip[ninety_angle[i]] - (j) * mesh.nx)
                if length(btm_in_ribbon) > 0
                    break
                end
            end
            if length(btm_in_ribbon) > 0
                alpha[btm_in_ribbon] .= π / 2
                dist_ribbon[btm_in_ribbon] .= abs(abs(y_rgt[ninety_angle[i]]) - 
                                                abs(mesh.CenterCoor[elt_ribbon[btm_in_ribbon], 2]))
            end
        end
        if zr_vrtx_tip[ninety_angle[i]] == 2 || zr_vrtx_tip[ninety_angle[i]] == 3
            top_in_ribbon = Int[]
            for j in 1:3
                top_in_ribbon = findall(elt_ribbon .== elt_tip[ninety_angle[i]] + (j) * mesh.nx)
                if length(top_in_ribbon) > 0
                    break
                end
            end
            if length(top_in_ribbon) > 0
                alpha[top_in_ribbon] .= π / 2
                dist_ribbon[top_in_ribbon] .= abs(abs(y_rgt[ninety_angle[i]]) - 
                                                abs(mesh.CenterCoor[elt_ribbon[top_in_ribbon], 2]))
            end
        end
    end

    return alpha
end

#-----------------------------------------------------------------------------------------------------------------------


"""
    construct_polygon(elt_tip, l_tip, alpha_tip, mesh, zero_vertex_tip)

This function construct a polygon from the given non-continous front. The polygon is constructed by joining the
intersection of the perpendiculars drawn on the front with the front lines. The points closest to each other are
joined and the intersection of the grid lines with these lines are taken as the vertices of the polygon.

# Arguments
- `elt_tip::Vector{Int}`: tip elements
- `l_tip::Vector{Float64}`: tip lengths
- `alpha_tip::Vector{Float64}`: tip angles
- `mesh::CartesianMesh`: mesh object
- `zero_vertex_tip::Vector{Int}`: zero vertex tip

# Returns
- `tip_smoothed::Vector{Int}`: smoothed tip elements
- `smthed_tip_lines_a::Vector{Float64}`: coefficient a of line equations
- `smthed_tip_lines_b::Vector{Float64}`: coefficient b of line equations
- `smthed_tip_lines_c::Vector{Float64}`: coefficient c of line equations
- `smthed_tip_points_left::Matrix{Float64}`: left points of tip lines
- `smthed_tip_points_rgt::Matrix{Float64}`: right points of tip lines
- `tip_lft_neghb::Vector{Int}`: left neighbors
- `tip_rgt_neghb::Vector{Int}`: right neighbors
"""
function construct_polygon(elt_tip::Vector{Int}, 
                         l_tip::Vector{Float64}, 
                         alpha_tip::Vector{Float64},
                         mesh::CartesianMesh, 
                         zero_vertex_tip::Vector{Int})

    slope = zeros(Float64, length(elt_tip))
    pnt_on_line = zeros(Float64, length(elt_tip), 2) # point where the perpendicular drawn on the front intersects the front
    
    # loop over tip cells to find the intersection point
    for i in 1:length(elt_tip)
        if zero_vertex_tip[i] == 0
            # if the perpendicular is drawn from the bottom left vertex
            slope[i] = tan(-(π / 2 - alpha_tip[i])) # slope of a line perpendicular to the front line
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 1] # bottom left vertex
            # coordinates of the intersection point
            pnt_on_line[i, 1] = mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * cos(alpha_tip[i])
            pnt_on_line[i, 2] = mesh.VertexCoor[zr_vrtx_global, 2] + l_tip[i] * sin(alpha_tip[i])
        elseif zero_vertex_tip[i] == 1
            slope[i] = tan(π / 2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 2]
            pnt_on_line[i, 1] = mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * cos(alpha_tip[i])
            pnt_on_line[i, 2] = mesh.VertexCoor[zr_vrtx_global, 2] + l_tip[i] * sin(alpha_tip[i])
        elseif zero_vertex_tip[i] == 2
            slope[i] = tan(-(π / 2 - alpha_tip[i]))
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 3]
            pnt_on_line[i, 1] = mesh.VertexCoor[zr_vrtx_global, 1] - l_tip[i] * cos(alpha_tip[i])
            pnt_on_line[i, 2] = mesh.VertexCoor[zr_vrtx_global, 2] - l_tip[i] * sin(alpha_tip[i])
        elseif zero_vertex_tip[i] == 3
            slope[i] = tan(π / 2 - alpha_tip[i])
            zr_vrtx_global = mesh.Connectivity[elt_tip[i], 4]
            pnt_on_line[i, 1] = mesh.VertexCoor[zr_vrtx_global, 1] + l_tip[i] * cos(alpha_tip[i])
            pnt_on_line[i, 2] = mesh.VertexCoor[zr_vrtx_global, 2] - l_tip[i] * sin(alpha_tip[i])
        end
    end

    # the code below make sure that, for the cells with zero or ninety degrees angle, there are points that are
    # exactly left/right or above/below of each other so that joining them will make a line that is parallel to the x
    # or y axes respectively.
    zero_angle = findall(alpha_tip .== 0.0)
    for i in zero_angle
        if zero_vertex_tip[i] == 0 || zero_vertex_tip[i] == 1
            dist_from_added = sqrt.((pnt_on_line[:, 1] .- pnt_on_line[i, 1]).^2 + 
                                   (pnt_on_line[:, 2] .- pnt_on_line[i, 2] .- mesh.hy).^2)
            closest = argmin(dist_from_added)
            if dist_from_added[closest] < sqrt(mesh.hx^2 + mesh.hy^2) / 10
                pnt_on_line = pnt_on_line[1:end .!= closest, :]
            end
            pnt_on_line = vcat(pnt_on_line, [pnt_on_line[i, 1] pnt_on_line[i, 2] + mesh.hy])
        end
        if zero_vertex_tip[i] == 2 || zero_vertex_tip[i] == 3
            dist_from_added = sqrt.((pnt_on_line[:, 1] .- pnt_on_line[i, 1]).^2 + 
                                   (pnt_on_line[:, 2] .- pnt_on_line[i, 2] .+ mesh.hy).^2)
            closest = argmin(dist_from_added)
            if dist_from_added[closest] < sqrt(mesh.hx^2 + mesh.hy^2) / 10
                pnt_on_line = pnt_on_line[1:end .!= closest, :]
            end
            pnt_on_line = vcat(pnt_on_line, [pnt_on_line[i, 1] pnt_on_line[i, 2] - mesh.hy])
        end
    end

    ninety_angle = findall(alpha_tip .== π / 2)
    for i in ninety_angle
        if zero_vertex_tip[i] == 0 || zero_vertex_tip[i] == 3
            pnt_on_line = vcat(pnt_on_line, [pnt_on_line[i, 1] + mesh.hx pnt_on_line[i, 2]])
        end
        if zero_vertex_tip[i] == 1 || zero_vertex_tip[i] == 2
            pnt_on_line = vcat(pnt_on_line, [pnt_on_line[i, 1] - mesh.hx pnt_on_line[i, 2]])
        end
    end

    grid_lines_x = unique(mesh.VertexCoor[:, 1]) # the x-coordinate of the points on grid lines parallel to y-axis
    grid_lines_y = unique(mesh.VertexCoor[:, 2]) # the y-coordinate of the points on grid lines parallel to x-axis
    polygon = Array{Float64}(undef, 0, 2)

    # closest point algorithm giving the points in order to construct a polygon
    remaining = copy(pnt_on_line) # remaining points to be joined
    pnt_in_order = remaining[1:1, :] # the points of the polygon given in order
    nxt = remaining[1, :]
    remaining = remaining[2:end, :]
    
    while size(remaining, 1) > 0
        dist_from_remnng = sqrt.((remaining[:, 1] .- nxt[1]).^2 + (remaining[:, 2] .- nxt[2]).^2)
        nxt_indx = argmin(dist_from_remnng)
        nxt = remaining[nxt_indx, :]
        remaining = remaining[1:end .!= nxt_indx, :]
        pnt_in_order = vcat(pnt_in_order, nxt')
    end

    # the code below finds the grid lines between two consecutive points. The vertices of the polygon are found by
    # the intersection of the grid lines and the line joining consecutive points (found by the closest point algorithm
    # above).
    i = 1
    while i <= size(pnt_in_order, 1)
        i_next = (i % size(pnt_in_order, 1)) + 1 # to make it cyclic (joining the first point to the last)
        # find the grid lines parallel to y-axis between the closest points under consideration
        if pnt_in_order[i, 1] <= pnt_in_order[i_next, 1]
            grd_lns_btw_pnts_x = findall((pnt_in_order[i_next, 1] .>= grid_lines_x) .& 
                                       (pnt_in_order[i, 1] .< grid_lines_x))
        else
            grd_lns_btw_pnts_x = findall((pnt_in_order[i_next, 1] .<= grid_lines_x) .& 
                                       (pnt_in_order[i, 1] .> grid_lines_x))
        end
        # if there is a grid line between the points
        if length(grd_lns_btw_pnts_x) > 0
            slope_val = (pnt_in_order[i_next, 2] - pnt_in_order[i, 2]) / 
                       (pnt_in_order[i_next, 1] - pnt_in_order[i, 1])
            for j in grd_lns_btw_pnts_x
                x_p = grid_lines_x[j]
                y_p = slope_val * (x_p - pnt_in_order[i_next, 1]) + pnt_in_order[i_next, 2]
                # add the intersection point to the polygon
                polygon = vcat(polygon, [x_p y_p])
            end
        end

        # find the grid lines parallel to x-axis between the closest points under consideration
        if pnt_in_order[i, 2] <= pnt_in_order[i_next, 2]
            grd_lns_btw_pnts_y = findall((pnt_in_order[i_next, 2] .>= grid_lines_y) .& 
                                       (pnt_in_order[i, 2] .< grid_lines_y))
        else
            grd_lns_btw_pnts_y = findall((pnt_in_order[i_next, 2] .<= grid_lines_y) .& 
                                       (pnt_in_order[i, 2] .> grid_lines_y))
        end
        # if there is a grid line between the points
        if length(grd_lns_btw_pnts_y) > 0
            slope_val = (pnt_in_order[i_next, 2] - pnt_in_order[i, 2]) / 
                       (pnt_in_order[i_next, 1] - pnt_in_order[i, 1])
            for j in grd_lns_btw_pnts_y
                y_p = grid_lines_y[j]
                x_p = (y_p - pnt_in_order[i_next, 2]) / slope_val + pnt_in_order[i_next, 1]
                # add the intersection point to the polygon
                polygon = vcat(polygon, [x_p y_p])
            end
        end
        i += 1
    end

    # remove redundant points
    unique_rows = Set{Vector{Float64}}()
    unique_polygon = Array{Float64}(undef, 0, 2)
    for i in 1:size(polygon, 1)
        row = [polygon[i, 1], polygon[i, 2]]
        if !(row in unique_rows)
            push!(unique_rows, row)
            unique_polygon = vcat(unique_polygon, row')
        end
    end
    polygon = unique_polygon

    tip_smoothed = Int[] # the cells containing the edges of polygon (giving the smoothed front)
    smthed_tip_points_left = Array{Float64}(undef, 0, 2) #left points of the tip line in the new tip cells
    smthed_tip_points_rgt = Array{Float64}(undef, 0, 2) #right points of the tip line in the new tip cells

    # loop over the cells of the grid to find the cells containing one of the edges of the polygon
    for i in 1:mesh.NumberOfElts
        # find the vertices of the polygon with x-coordinates greater than or equal to x-coordinate of the bottom left
        # vertex of the cell
        in_cell = polygon[:, 1] .>= mesh.VertexCoor[mesh.Connectivity[i, 1], 1]
        in_cell = in_cell .& (polygon[:, 1] .<= mesh.VertexCoor[mesh.Connectivity[i, 2], 1])
        in_cell = in_cell .& (polygon[:, 2] .>= mesh.VertexCoor[mesh.Connectivity[i, 1], 2])
        in_cell = in_cell .& (polygon[:, 2] .<= mesh.VertexCoor[mesh.Connectivity[i, 4], 2])
        # points of the polygon on the edges of the current cell
        cell_pnt = findall(in_cell)
        if length(cell_pnt) > 2
            # Hack!!! if there is more than two points, find the two furthest
            # return NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN
            dist = (polygon[cell_pnt[1], 1] .- polygon[cell_pnt, 1]).^2 + 
                   (polygon[cell_pnt[1], 2] .- polygon[cell_pnt, 2]).^2
            farthest = argmax(dist)
            to_delete = Int[]
            for m in 2:length(cell_pnt)
                if m != farthest
                    push!(to_delete, cell_pnt[m])
                end
            end
            # delete the extra points from polygon
            polygon = polygon[setdiff(1:size(polygon, 1), to_delete), :]
            # find the two points again
            in_cell = polygon[:, 1] .>= mesh.VertexCoor[mesh.Connectivity[i, 1], 1]
            in_cell = in_cell .& (polygon[:, 1] .<= mesh.VertexCoor[mesh.Connectivity[i, 2], 1])
            in_cell = in_cell .& (polygon[:, 2] .>= mesh.VertexCoor[mesh.Connectivity[i, 1], 2])
            in_cell = in_cell .& (polygon[:, 2] .<= mesh.VertexCoor[mesh.Connectivity[i, 4], 2])
            cell_pnt = findall(in_cell)
        end

        if length(cell_pnt) > 1
            # add the cell to the tip cells
            push!(tip_smoothed, i)
            # add accordingly to the left and right points of the added tip cell
            if polygon[cell_pnt[1], 1] <= polygon[cell_pnt[2], 1]
                smthed_tip_points_left = vcat(smthed_tip_points_left, polygon[cell_pnt[1]:cell_pnt[1], :])
                smthed_tip_points_rgt = vcat(smthed_tip_points_rgt, polygon[cell_pnt[2]:cell_pnt[2], :])
            else
                smthed_tip_points_left = vcat(smthed_tip_points_left, polygon[cell_pnt[2]:cell_pnt[2], :])
                smthed_tip_points_rgt = vcat(smthed_tip_points_rgt, polygon[cell_pnt[1]:cell_pnt[1], :])
            end
        end
    end

    # find the equations(of the form ax+by+c=0) of the front lines in the tip cells
    smthed_tip_lines_slope = (smthed_tip_points_rgt[:, 2] .- smthed_tip_points_left[:, 2]) ./ 
                            (smthed_tip_points_rgt[:, 1] .- smthed_tip_points_left[:, 1])
    smthed_tip_lines_a = -smthed_tip_lines_slope
    smthed_tip_lines_b = ones(Float64, length(tip_smoothed))
    smthed_tip_lines_c = -(smthed_tip_points_rgt[:, 2] .- smthed_tip_lines_slope .* smthed_tip_points_rgt[:, 1])

    # equation of the line with 90 degree angle
    zero_angle = findall(smthed_tip_points_left[:, 1] .== smthed_tip_points_rgt[:, 1])
    smthed_tip_lines_b[zero_angle] .= 0.0
    smthed_tip_lines_a[zero_angle] .= 1.0
    smthed_tip_lines_c[zero_angle] .= -smthed_tip_points_rgt[zero_angle, 1]

    # find the left neighbor of the tip cells in the tip
    tip_lft_neghb = zeros(Int, length(tip_smoothed))
    tip_rgt_neghb = zeros(Int, length(tip_smoothed))
    for i in 1:length(tip_smoothed)
        equal = (smthed_tip_points_rgt[:, 1] .== smthed_tip_points_left[i, 1]) .& 
                (smthed_tip_points_rgt[:, 2] .== smthed_tip_points_left[i, 2])
        left_nei = findall(equal)
        if length(left_nei) != 1
            # Hack!!! find the cell with the same left point
            equal = (smthed_tip_points_left[:, 1] .== smthed_tip_points_left[i, 1]) .& 
                    (smthed_tip_points_left[:, 2] .== smthed_tip_points_left[i, 2])
            left_nei = findall(equal)
            if length(left_nei) == 2
                other_idx = findfirst(x -> x != i, left_nei)
                if other_idx !== nothing
                    tip_lft_neghb[i] = left_nei[other_idx]
                else
                    return NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN
                end
            else
                return NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN
            end
        else
            tip_lft_neghb[i] = left_nei[1]
        end

        # find the eight neighbor of the tip cells in the tip
        equal = (smthed_tip_points_left[:, 1] .== smthed_tip_points_rgt[i, 1]) .& 
                (smthed_tip_points_left[:, 2] .== smthed_tip_points_rgt[i, 2])
        rgt_nei = findall(equal)
        if length(rgt_nei) != 1
            equal = (smthed_tip_points_rgt[:, 1] .== smthed_tip_points_rgt[i, 1]) .& 
                    (smthed_tip_points_rgt[:, 2] .== smthed_tip_points_rgt[i, 2])
            rgt_nei = findall(equal)
            if length(rgt_nei) == 2
                other_idx = findfirst(x -> x != i, rgt_nei)
                if other_idx !== nothing
                    tip_rgt_neghb[i] = rgt_nei[other_idx]
                else
                    return NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN
                end
            else
                return NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN
            end
        else
            tip_rgt_neghb[i] = rgt_nei[1]
        end
    end

    return tip_smoothed, smthed_tip_lines_a, smthed_tip_lines_b, smthed_tip_lines_c, 
           smthed_tip_points_left, smthed_tip_points_rgt, tip_lft_neghb, tip_rgt_neghb
end

#-----------------------------------------------------------------------------------------------------------------------

"""
    projection_from_ribbon_LS_gradient(ribbon_elts, tip_elts, mesh, sgnd_dist)

This function finds the projection of the ribbon cell centers on to the fracture front from the gradient of the
level set. It is returned as the angle inscribed by the perpendiculars drawn on the front from the ribbon cell
centers.

# Arguments
- `ribbon_elts::Vector{Int}`: list of ribbon elements
- `tip_elts::Vector{Int}`: list of tip elements
- `mesh::CartesianMesh`: The cartesian mesh object
- `sgnd_dist::Vector{Float64}`: level set data

# Returns
- `::Vector{Float64}`: the angle inscribed by the perpendiculars drawn on the front from the ribbon cell centers
"""
function projection_from_ribbon_LS_gradient(ribbon_elts::Vector{Int}, 
                                          tip_elts::Vector{Int}, 
                                          mesh::CartesianMesh, 
                                          sgnd_dist::Vector{Float64})::Vector{Float64}

    n_vertex = zeros(Float64, length(tip_elts), 2)
    n_centre = zeros(Float64, length(ribbon_elts), 2)
    Coor_vertex = zeros(Float64, length(tip_elts), 2)
    alpha = zeros(Float64, length(ribbon_elts))

    zero_vertex = find_zero_vertex(tip_elts, sgnd_dist, mesh)
    
    for i in 1:length(tip_elts)
        # neighbors
        #     6     3    7
        #     0    elt   1
        #     4    2     5
        neighbors_tip = zeros(Int, 8)
        neighbors_tip[1:4] = mesh.NeiElements[tip_elts[i], :]
        neighbors_tip[5] = mesh.NeiElements[neighbors_tip[3], 1]
        neighbors_tip[6] = mesh.NeiElements[neighbors_tip[3], 2]
        neighbors_tip[7] = mesh.NeiElements[neighbors_tip[4], 1]
        neighbors_tip[8] = mesh.NeiElements[neighbors_tip[4], 2]

        # Vertex
        #     3         2
        #     0         1
        if zero_vertex[i] == 0
            gradx = -((sgnd_dist[neighbors_tip[1]] + sgnd_dist[neighbors_tip[5]]) / 2 - 
                     (sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[3]]) / 2) / mesh.hx
            grady = ((sgnd_dist[neighbors_tip[1]] + sgnd_dist[tip_elts[i]]) / 2 - 
                    (sgnd_dist[neighbors_tip[5]] + sgnd_dist[neighbors_tip[3]]) / 2) / mesh.hy
            Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] - mesh.hx / 2
            Coor_vertex[i, 2] = mesh.CenterCoor[tip_elts[i], 2] - mesh.hy / 2
        elseif zero_vertex[i] == 1
            gradx = ((sgnd_dist[neighbors_tip[2]] + sgnd_dist[neighbors_tip[6]]) / 2 - 
                    (sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[3]]) / 2) / mesh.hx
            grady = ((sgnd_dist[neighbors_tip[2]] + sgnd_dist[tip_elts[i]]) / 2 - 
                    (sgnd_dist[neighbors_tip[6]] + sgnd_dist[neighbors_tip[3]]) / 2) / mesh.hy
            Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] + mesh.hx / 2
            Coor_vertex[i, 2] = mesh.CenterCoor[tip_elts[i], 2] - mesh.hy / 2
        elseif zero_vertex[i] == 2
            gradx = ((sgnd_dist[neighbors_tip[2]] + sgnd_dist[neighbors_tip[8]]) / 2 - 
                    (sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[4]]) / 2) / mesh.hx
            grady = -((sgnd_dist[neighbors_tip[2]] + sgnd_dist[tip_elts[i]]) / 2 - 
                     (sgnd_dist[neighbors_tip[4]] + sgnd_dist[neighbors_tip[8]]) / 2) / mesh.hy
            Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] + mesh.hx / 2
            Coor_vertex[i, 2] = mesh.CenterCoor[tip_elts[i], 2] + mesh.hy / 2
        elseif zero_vertex[i] == 3
            gradx = -((sgnd_dist[neighbors_tip[7]] + sgnd_dist[neighbors_tip[1]]) / 2 - 
                     (sgnd_dist[tip_elts[i]] + sgnd_dist[neighbors_tip[4]]) / 2) / mesh.hx
            grady = ((sgnd_dist[neighbors_tip[1]] + sgnd_dist[tip_elts[i]]) / 2 - 
                    (sgnd_dist[neighbors_tip[7]] + sgnd_dist[neighbors_tip[4]]) / 2) / mesh.hy
            Coor_vertex[i, 1] = mesh.CenterCoor[tip_elts[i], 1] - mesh.hx / 2
            Coor_vertex[i, 2] = mesh.CenterCoor[tip_elts[i], 2] + mesh.hy / 2
        end
        
        n_vertex[i, 1] = gradx / sqrt(gradx^2 + grady^2)
        n_vertex[i, 2] = grady / sqrt(gradx^2 + grady^2)
    end

    for i in 1:length(ribbon_elts)
        actvElts = findall((2 * abs.(mesh.CenterCoor[ribbon_elts[i], 1] .- Coor_vertex[:, 1]) .- mesh.hx .< mesh.hx / 10) .&
                          (2 * abs.(mesh.CenterCoor[ribbon_elts[i], 2] .- Coor_vertex[:, 2]) .- mesh.hy .< mesh.hy / 10))

        if length(actvElts) > 0
            n_centre[i, 1] = mean(n_vertex[actvElts, 1])
            n_centre[i, 2] = mean(n_vertex[actvElts, 2])
        else
            n_centre[i, 1] = 0.0
            n_centre[i, 2] = 0.0
        end
        
        alpha[i] = abs(asin(n_centre[i, 2]))
    end

    return alpha
end


#-----------------------------------------------------------------------------------------------------------------------


"""
    find_zero_vertex(Elts, level_set, mesh)

This function finds the zero-vertex (the vertex opposite to the propagation direction) from where the perpendicular
is drawn on the front.

# Arguments
- `Elts::Vector{Int}`: the given elements for which the zero-vertex is to be found
- `level_set::Vector{Float64}`: the level set data (distance from front of the elements of the grid)
- `mesh::CartesianMesh`: the mesh given by CartesianMesh object

# Returns
- `::Vector{Int}`: the zero vertex list
"""
function find_zero_vertex(Elts::Vector{Int}, 
                         level_set::Vector{Float64}, 
                         mesh::CartesianMesh)::Vector{Int}

    zero_vertex = zeros(Int, length(Elts))
    
    for i in 1:length(Elts)
        neighbors = mesh.NeiElements[Elts[i], :]
        
        if level_set[neighbors[1]] <= level_set[neighbors[2]] && level_set[neighbors[3]] <= level_set[neighbors[4]]
            zero_vertex[i] = 0
        elseif level_set[neighbors[1]] > level_set[neighbors[2]] && level_set[neighbors[3]] <= level_set[neighbors[4]]
            zero_vertex[i] = 1
        elseif level_set[neighbors[1]] > level_set[neighbors[2]] && level_set[neighbors[3]] > level_set[neighbors[4]]
            zero_vertex[i] = 2
        elseif level_set[neighbors[1]] <= level_set[neighbors[2]] && level_set[neighbors[3]] > level_set[neighbors[4]]
            zero_vertex[i] = 3
        end
    end

    return zero_vertex
end




"""
    get_toughness_from_cellCenter(alpha, sgnd_dist=nothing, elts=nothing, mat_prop=nothing, mesh=nothing)

This function returns the toughness given the angle inscribed from the cell centers on the front. both the cases
of heterogenous or anisotropic toughness are taken care off.

# Arguments
- `alpha::Vector{Float64}`: angles
- `sgnd_dist::Union{Vector{Float64}, Nothing}=nothing`: signed distance function
- `elts::Union{Vector{Int}, Nothing}=nothing`: elements
- `mat_prop::Union{MaterialProperties, Nothing}=nothing`: material properties
- `mesh::Union{CartesianMesh, Nothing}=nothing`: mesh object

# Returns
- `::Vector{Float64}`: toughness values
"""
function get_toughness_from_cellCenter(alpha::Vector{Float64}, 
                                     sgnd_dist::Union{Vector{Float64}, Nothing}=nothing, 
                                     elts::Union{Vector{Int}, Nothing}=nothing, 
                                     mat_prop::Union{MaterialProperties, Nothing}=nothing, 
                                     mesh::Union{CartesianMesh, Nothing}=nothing)::Vector{Float64}

    if mat_prop.anisotropic_K1c
        try
            return mat_prop.K1cFunc.(alpha)
        catch e
            error("For anisotropic toughness, the function taking the angle and returning the toughness is to be provided")
        end
    else
        dist = -sgnd_dist
        x = zeros(Float64, length(elts))
        y = zeros(Float64, length(elts))

        neighbors = mesh.NeiElements[elts, :]
        zero_vertex = find_zero_vertex(elts, sgnd_dist, mesh)
        
        # evaluating the closest tip points
        for i in 1:length(elts)
            if zero_vertex[i] == 0
                x[i] = mesh.CenterCoor[elts[i], 1] + dist[elts[i]] * cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 2] + dist[elts[i]] * sin(alpha[i])
            elseif zero_vertex[i] == 1
                x[i] = mesh.CenterCoor[elts[i], 1] - dist[elts[i]] * cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 2] + dist[elts[i]] * sin(alpha[i])
            elseif zero_vertex[i] == 2
                x[i] = mesh.CenterCoor[elts[i], 1] - dist[elts[i]] * cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 2] - dist[elts[i]] * sin(alpha[i])
            elseif zero_vertex[i] == 3
                x[i] = mesh.CenterCoor[elts[i], 1] + dist[elts[i]] * cos(alpha[i])
                y[i] = mesh.CenterCoor[elts[i], 2] - dist[elts[i]] * sin(alpha[i])
            end

            # assume the angle is zero if the distance of the left and right neighbor is extremely close
            if abs(dist[mesh.NeiElements[elts[i], 1]] / dist[mesh.NeiElements[elts[i], 2]] - 1) < 1e-7
                if sgnd_dist[neighbors[i, 3]] < sgnd_dist[neighbors[i, 4]]
                    x[i] = mesh.CenterCoor[elts[i], 1]
                    y[i] = mesh.CenterCoor[elts[i], 2] + dist[elts[i]]
                elseif sgnd_dist[neighbors[i, 3]] > sgnd_dist[neighbors[i, 4]]
                    x[i] = mesh.CenterCoor[elts[i], 1]
                    y[i] = mesh.CenterCoor[elts[i], 2] - dist[elts[i]]
                end
            end
            
            # assume the angle is 90 degrees if the distance of the bottom and top neighbor is extremely close
            if abs(dist[mesh.NeiElements[elts[i], 3]] / dist[mesh.NeiElements[elts[i], 4]] - 1) < 1e-7
                if sgnd_dist[neighbors[i, 1]] < sgnd_dist[neighbors[i, 2]]
                    x[i] = mesh.CenterCoor[elts[i], 1] + dist[elts[i]]
                    y[i] = mesh.CenterCoor[elts[i], 2]
                elseif sgnd_dist[neighbors[i, 1]] > sgnd_dist[neighbors[i, 2]]
                    x[i] = mesh.CenterCoor[elts[i], 1] - dist[elts[i]]
                    y[i] = mesh.CenterCoor[elts[i], 2]
                end
            end
        end

        # returning the Kprime according to the given function
        K1c = zeros(Float64, length(elts))
        for i in 1:length(elts)
            try
                K1c[i] = mat_prop.K1cFunc(x[i], y[i])
            catch e
                error("For precise space dependant toughness, the function taking the coordinates and returning the toughness is to be provided.")
            end
        end
        return K1c
    end
end


#-----------------------------------------------------------------------------------------------------------------------


"""
    get_toughness_from_zeroVertex(elts, mesh, mat_prop, alpha, l, zero_vrtx)

This function returns the toughness given the angle inscribed from the zero-vertex on the front. both the cases
of heterogenous or anisotropic toughness are taken care off.

# Arguments
- `elts::Vector{Int}`: elements
- `mesh::CartesianMesh`: mesh object
- `mat_prop::MaterialProperties`: material properties
- `alpha::Vector{Float64}`: angles
- `l::Vector{Float64}`: lengths
- `zero_vrtx::Vector{Int}`: zero vertices

# Returns
- `::Vector{Float64}`: toughness values
"""
function get_toughness_from_zeroVertex(elts::Vector{Int}, 
                                     mesh::CartesianMesh, 
                                     mat_prop::MaterialProperties, 
                                     alpha::Vector{Float64}, 
                                     l::Vector{Float64}, 
                                     zero_vrtx::Vector{Int})::Vector{Float64}

    if mat_prop.K1cFunc === nothing
        return mat_prop.K1c[elts]
    end

    if mat_prop.anisotropic_K1c
        return mat_prop.K1cFunc.(alpha)
    else
        x = zeros(Float64, length(elts))
        y = zeros(Float64, length(elts))
        
        for i in 1:length(elts)
            if zero_vrtx[i] == 0
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 1], 1] + l[i] * cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 1], 2] + l[i] * sin(alpha[i])
            elseif zero_vrtx[i] == 1
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 2], 1] - l[i] * cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 2], 2] + l[i] * sin(alpha[i])
            elseif zero_vrtx[i] == 2
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 3], 1] - l[i] * cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 3], 2] - l[i] * sin(alpha[i])
            elseif zero_vrtx[i] == 3
                x[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 4], 1] + l[i] * cos(alpha[i])
                y[i] = mesh.VertexCoor[mesh.Connectivity[elts[i], 4], 2] - l[i] * sin(alpha[i])
            end
        end

        # returning the Kprime according to the given function
        K1c = zeros(Float64, length(elts))
        for i in 1:length(elts)
            K1c[i] = mat_prop.K1cFunc(x[i], y[i])
        end

        return K1c
    end
end


#-----------------------------------------------------------------------------------------------------------------------

"""
    TI_plain_strain_modulus(alpha, Cij)

This function computes the plain strain elasticity modulus in transverse isotropic medium. The modulus is a function
of the orientation of the fracture front with respect to the bedding plane. This functions is used for the tip
inversion and for evaluation of the fracture volume for the case of TI elasticity.

# Arguments
- `alpha::Vector{Float64}`: the angle inscribed by the perpendiculars drawn on the front from the ribbon cell centers
- `Cij::Matrix{Float64}`: the TI stiffness matrix in the canonical basis

# Returns
- `::Vector{Float64}`: plain strain TI elastic modulus
"""
function TI_plain_strain_modulus(alpha::Vector{Float64}, Cij::Matrix{Float64})::Vector{Float64}

    C11 = Cij[1, 1]
    C12 = Cij[1, 2]
    C13 = Cij[1, 3]
    C33 = Cij[3, 3]
    C44 = Cij[4, 4]

    # we use the same notation for the elastic paramateres as S. Fata et al. (2013).
    alphag = (C11 * (C11 - C12) * cos.(alpha) .^ 4 + (C11 * C13 - C12 * (C13 + 2 * C44)) * 
             (cos.(alpha) .* sin.(alpha)) .^ 2 - (C13^2 - C11 * C33 + 2 * C13 * C44) * sin.(alpha) .^ 4 + 
             C11 * C44 * sin.(2 * alpha) .^ 2) ./ (C11 * (C11 - C12) * cos.(alpha) .^ 2 + 
             2 * C11 * C44 * sin.(alpha) .^ 2)

    gammag = ((C11 * cos.(alpha) .^ 4 + 2 * C13 * (cos.(alpha) .* sin.(alpha)) .^ 2 + 
              C33 * sin.(alpha) .^ 4 + C44 * sin.(2 * alpha) .^ 2) / C11) .^ 0.5

    deltag = ((C11 - C12) * (C11 + C12) * cos.(alpha) .^ 4 + 
             2 * (C11 - C12) * C13 * (cos.(alpha) .* sin.(alpha)) .^ 2 + 
             (-C13^2 + C11 * C33) * sin.(alpha) .^ 4 + 
             C11 * C44 * sin.(2 * alpha) .^ 2) ./ (C11 * (2 * (alphag + gammag)) .^ 0.5)

    Eprime = 2 * deltag ./ gammag

    return Eprime
end

end # module
#-----------------------------------------------------------------------------------------------------------------------