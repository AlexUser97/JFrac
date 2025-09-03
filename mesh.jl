# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 11:51:00 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020. All rights reserved.
See the LICENSE.TXT file for more details.
"""

module Mesh
include("visualization.jl")
include("symmetry.jl")

using .Visualization: zoom_factory, to_precision, text3d
using .Symmetry: *

using Logging
using Plots
"""
    CartesianMesh
    A uniform Cartesian mesh centered at (0,0) with domain [-Lx, Lx] × [-Ly, Ly].

    Args:
    - `nx, ny::Int`:                      -- number of elements in x and y directions
    - `Lx, Ly::Float64`:                  -- half-length in x and y directions
    - `symmetric::Bool`:                  -- if true, additional variables (see list of attributes) will be evaluated for
                                             symmetric fracture solver.

    Attributes:
    - `Lx, Ly::Float64`:                  -- length of the domain in x and y directions respectively. The rectangular domain
                                             have a total length of 2*Lx in the x direction and 2*Ly in the y direction. Both
                                             the positive and negative halves are included.
    - `nx, ny::Int`:                      -- number of elements in x and y directions respectively.
    - `hx,hy::Float64`:                   -- grid spacing in x and y directions respectively.
    - `VertexCoor::Matrix{Float64}`:      -- [x,y] Coordinates of the vertices.
    - `CenterCoor::Matrix{Float64}`:      -- [x,y] coordinates of the center of the elements.
    - `NumberOfElts::Int`:                -- total number of elements in the mesh.
    - `EltArea::Float64`:                 -- area of each element.
    - `Connectivity::Matrix{Int}`:        -- connectivity array giving four vertices of an element in the following order
                                             [bottom left, bottom right, top right, top left]
    - `Connectivityelemedges::Matrix{Int}`: connectivity array giving four edges of an element in the following order
                                             [bottom, right, top, left]
    - `Connectivityedgeselem::Matrix{Int}`: connectivity array giving two elements that are sharing an edge
    - `Connectivityedgesnodes::Matrix{Int}`: connectivity array giving two vertices of an edge
    - `Connectivitynodesedges::Matrix{Int}`: connectivity array giving four edges of a node in the following order
                                             [vertical_top, horizotal_left, vertical_bottom, horizotal_right]
    - `Connectivitynodeselem::Matrix{Int}`: connectivity array giving four elements of a node in the following order
                                             [bottom left, bottom right, top right, top left]
    - `NeiElements::Matrix{Int}`:          -- Giving four neighboring elements with the following order:[left, right,
                                             bottom, up].
    - `distCenter::Matrix{Float64}`:       -- the distance of the cells from the center.
    - `CenterElts::Matrix{Int}`:           -- the element in the center (the cell with the injection point).

    - `domainLimits::Matrix{Float64}`:     -- the limits of the domain

    Note:
        The attributes below are only evaluated if symmetric solver is used.

    Attributes:
    - `corresponding::Matrix{Int}`:        -- the index of the corresponding symmetric cells in the set of active cells
                                              (activeSymtrc) for each cell in the mesh.
    - `symmetricElts::Matrix{Int}`:        -- the set of four symmetric cells in the mesh for each of the cell.
    - `activeSymtrc::Matrix{Int}`:         -- the set of cells that are active in the mesh. Only these cells will be solved
                                              and the solution will be replicated in the symmetric cells.
    - `posQdrnt::Matrix{Int}`:             -- the set of elements in the positive quadrant not including the boundaries.
    - `boundary_x::Matrix{Int}`:           -- the elements intersecting the positive x-axis line.
    - `boundary_y::Matrix{Int}`:           -- the elements intersecting the positive y-axis line.
    - `volWeights::Matrix{Float64}`:       -- the weights of the active elements in the volume of the fracture. The cells in the
                                              positive quadrant, the boundaries and the injection cell have the weights of 4, 2
                                              and 1 respectively.

"""
mutable struct CartesianMesh
    
    Lx::Float64
    Ly::Float64
    nx::Int
    ny::Int
    hx::Float64
    hy::Float64
    VertexCoor::Matrix{Float64}
    NumberOfElts::Int
    NumberofNodes::Int
    EltArea::Float64
    domainLimits::Vector{Float64}


    function CartesianMesh(Lx::Float64, Ly::Float64, nx::Int, ny::Int, symmetric::Bool=false)
        
        @info "Creating mesh..." _group="PyFrac.mesh"

        if isa(Lx, Number)
            Lx_val = Float64(Lx)
            xlims = [-Lx, Lx]
        else
            Lx_val = abs(Lx[2] - Lx[1]) / 2
            xlims = [Lx[1], Lx[2]]
        end

        if isa(Ly, Number)
            Ly_val = Float64(Ly)
            ylims = [-Ly, Ly]
        else
            Ly_val = abs(Ly[2] - Ly[1]) / 2
            ylims = [Ly[1], Ly[2]]
        end

        domainLimits = [ylims[1], ylims[2], xlims[1], xlims[2]]

        if nx % 2 == 0
            @warn "Number of elements in x-direction are even. Using $(nx+1) elements to have origin at a cell center..."
            nx += 1
        end

        if ny % 2 == 0
            @warn "Number of elements in y-direction are even. Using $(ny+1) elements to have origin at a cell center..."
            ny += 1
        end

        hx = 2.0 * Lx_val / (nx - 1)
        hy = 2.0 * Ly_val / (ny - 1)


        x = range(domainLimits[3] - hx/2, domainLimits[4] + hx/2, length=nx + 1)
        y = range(domainLimits[1] - hy/2, domainLimits[2] + hy/2, length=ny + 1)
        xv = repeat(x', outer=(ny + 1, 1))
        yv = repeat(y, inner=(ny + 1, 1))

        VertexCoor = hcat(vec(xv), vec(yv))

        NumberofNodes = (nx + 1) * (ny + 1)
        NumberOfElts = nx * ny
        EltArea = hx * hy

        """
        We create a list of cell names thaht are colose to the boundary of the mesh. See the example below.
        In that case the list will contain the elements identified with a x.
        The list of elements will be called Frontlist
        
         _____________________________
        |    |    |    |    |    |    |
        |____|____|____|____|____|____|
        |    | x  |  x |  x |  x |    |
        |____|____|____|____|____|____|
        |    | x  |    |    |  x |    |
        |____|____|____|____|____|____|
        |    | x  |    |    |  x |    |
        |____|____|____|____|____|____|
        |    | x  |  x |  x |  x |    |
        |____|____|____|____|____|____|
        |    |    |    |    |    |    |
        |____|____|____|____|____|____|            
        """
        
        
        Frontlist = Int[]
        append!(Frontlist, (nx+1):(2*nx-1))
        append!(Frontlist, ((ny-3)*nx + nx + 1):((ny-3)*nx + 2*nx - 1))
        for i in 1:(ny-3)
            push!(Frontlist, nx + 1 + i * nx)
            push!(Frontlist, 2 * nx - 2 + i * nx)
        end

        """
         Giving four neighbouring elements in the following order: [left,right,bottom,up]
         ______ ______ _____ 
        |      | top  |     |
        |______|______|_____|
        |left  |  i   |right|
        |______|______|_____|
        |      |bottom|     |
        |______|______|_____|
        """
        NeiElements = zeros(Int, NumberOfElts, 4)
        for i in 1:NumberOfElts
            neighbors = Neighbors(i, nx, ny)
            NeiElements[i, :] = neighbors
        end

        """
         conn is the connectivity array giving four vertices of an element in the following order
         ______ ______ _____ 
        |      |      |     |
        |______4______3_____|
        |      |  i   |     |
        |______1______2_____|
        |      |      |     |
        |______|______|_____|
        """

        """
         connElemEdges is a connectivity array: for each element is listing the name of its 4 edges
         connEdgesElem is a connectivity array: for each edge is listing the name of its 2 neighbouring elements
        """


        #
        # connEdgesNodes is a connectivity array: for each edge is listing the name of its 2 end nodes

        # connNodesElem is a connectivity array: for each node is listing the 4 elements that share that
        # connNodesEdges:
        #            1
        #            |
        #         2__o__4    o is the node and the order in  connNodesEdges is [vertical_top, horizotal_left, vertical_bottom, horizotal_right]
        #            |
        #            3
        
        numberofedges = 2 * nx * ny + nx + ny                         # Edges amount (Peruzzo 2019)

        # Матрицы связности
        conn = Matrix{Int}(undef, NumberOfElts, 4)                    
        booleconnEdgesNodes = zeros(Int, numberofedges, 1)           
        connEdgesNodes = Matrix{Int}(undef, numberofedges, 2)         
        connElemEdges = Matrix{Int}(undef, NumberOfElts, 4)           
        connEdgesElem = fill(typemax(Int), numberofedges, 2)          
        connNodesEdges = fill(typemax(Int), NumberofNodes, 4)         
        connNodesElem = fill(typemax(Int), NumberofNodes, 4)
        
        k = 1
        for j in 1:ny
            for i in 1:nx
                conn[k, 1] = (i + (j-1) * (nx + 1))
                conn[k, 2] = (i + 1) + (j-1) * (nx + 1)
                conn[k, 3] = i + 1 + j * (nx + 1)
                conn[k, 4] = i + j * (nx + 1)
                connElemEdges[k, 1] = ((j-1) * (2 * nx + 1) + i)  # BottomEdge - Peruzzo 2019
                connElemEdges[k, 2] = ((j-1) * (2 * nx + 1) + nx + i + 1)  # RightEdge  - Peruzzo 2019
                connElemEdges[k, 3] = (j * (2 * nx + 1) + i)  # topEdge    - Peruzzo 2019
                connElemEdges[k, 4] = ((j-1) * (2 * nx + 1) + nx + i)  # LeftEdge   - Peruzzo 2019
                connEdgesElem[connElemEdges[k, 1], :] = [k, NeiElements[k, 3]]  # Peruzzo 2019
                connEdgesElem[connElemEdges[k, 2], :] = [k, NeiElements[k, 2]]  # Peruzzo 2019
                connEdgesElem[connElemEdges[k, 3], :] = [k, NeiElements[k, 4]]  # Peruzzo 2019
                connEdgesElem[connElemEdges[k, 4], :] = [k, NeiElements[k, 1]]  # Peruzzo 2019
                # How neighbours are sorted within Nei: [left, right, bottom, up]
                for s in 1:4  # Peruzzo 2019
                    index = connElemEdges[k, s]  # Peruzzo 2019
                    if booleconnEdgesNodes[index] == 0  # Peruzzo 2019
                        booleconnEdgesNodes[index] = 1  # Peruzzo 2019
                        if s < 4  # Peruzzo 2019
                            connEdgesNodes[index, :] = [conn[k, s], conn[k, s + 1]]  # Peruzzo 2019
                        else  # Peruzzo 2019
                            connEdgesNodes[index, :] = [conn[k, s], conn[k, 1]]  # Peruzzo 2019
                    end
                end
                if i == nx || j == ny || i == 1 || j == 1  # start Peruzzo 2019
                    if i == nx && j != ny && i != 1 && j != 1  # right row of cells
                        # for each top left node
                        connNodesEdges[conn[k, 4], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 4], 2] = connElemEdges[k, 4]  # leftedge
                        # BottomEdgeOfTopLeftNeighboursElem
                        connNodesEdges[conn[k, 4], 3] = (j * (2 * nx + 1) + (i - 1))
                        # RightEdgeOfTopLeftNeighboursElem
                        connNodesEdges[conn[k, 4], 4] = (j * (2 * nx + 1) + nx + (i - 1) + 1)
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 2], 2] = connElemEdges[k, 2]  # rightedge
                        # RightEdgeOfBottomNeighboursElem
                        connNodesEdges[conn[k, 2], 3] = ((j - 2) * (2 * nx + 1) + nx + i + 1)
                        connNodesEdges[conn[k, 2], 4] = connElemEdges[k, 1]  # bottomedge #repeated
                        # connNodesElem:
                        #    |   |
                        # ___a___o
                        #    |   |
                        # ___o___b
                        #    |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 2], 1] = NeiElements[k, 3]  # element: bottom left
                        connNodesElem[conn[k, 2], 2] = NeiElements[k, 3]  # element: bottom right #repeated
                        connNodesElem[conn[k, 2], 3] = NeiElements[k, 3]  # element: top right #repeated
                        connNodesElem[conn[k, 2], 4] = k  # element: top left (current k)
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 4], 1] = NeiElements[k, 1]  # element: bottom left
                        connNodesElem[conn[k, 4], 2] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 4], 3] = NeiElements[k, 4]  # element: top right
                        connNodesElem[conn[k, 4], 4] = NeiElements[k - 1, 4]  # element: top left

                    elseif i != nx && j == ny && i != 1 && j != 1  # top row of cells
                        # for each bottom left node
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 1], 2] = connElemEdges[k, 4]  # leftedge
                        # TopEdgeOfBottomLeftNeighboursElem
                        connNodesEdges[conn[k, 1], 3] = (((j - 2) + 1) * (2 * nx + 1) + (i - 1))
                        # RightEdgeOfBottomLeftNeighboursElem
                        connNodesEdges[conn[k, 1], 4] = ((j - 2) * (2 * nx + 1) + nx + (i - 1) + 1)
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 3], 2] = connElemEdges[k, 2]  # rightedge
                        # TopEdgeOfRightNeighboursElem
                        connNodesEdges[conn[k, 3], 3] = (j * (2 * nx + 1) + (i + 1))
                        connNodesEdges[conn[k, 3], 4] = connElemEdges[k, 2]  # rightedge #repeated
                        # connNodesElem:
                        # ___o___b___
                        #    |   |
                        # ___a___o___
                        #    |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 1], 1] = NeiElements[k - 1, 3]  # element: bottom left #repeated
                        connNodesElem[conn[k, 1], 2] = NeiElements[k, 3]  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top right (current k)
                        connNodesElem[conn[k, 1], 4] = NeiElements[k, 1]  # element: top left
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 3], 2] = NeiElements[k, 2]  # element: bottom right
                        connNodesElem[conn[k, 3], 3] = NeiElements[k + 1, 4]  # element: top right
                        connNodesElem[conn[k, 3], 4] = NeiElements[k, 4]  # element: top left

                    elseif i != nx && j != ny && i == 1 && j != 1  # left row of cells
                        # for each bottom right node
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 2], 2] = connElemEdges[k, 2]  # rightedge
                        # TopEdgeOfBottomRightNeighboursElem
                        connNodesEdges[conn[k, 2], 3] = (((j - 2) + 1) * (2 * nx + 1) + (i + 1))
                        # LeftEdgeOfBottomRightNeighboursElem
                        connNodesEdges[conn[k, 2], 4] = ((j - 2) * (2 * nx + 1) + nx + (i + 1))
                        connNodesEdges[conn[k, 4], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 4], 2] = connElemEdges[k, 4]  # leftedge
                        # LeftEdgeOfTopNeighboursElem
                        connNodesEdges[conn[k, 4], 3] = (j * (2 * nx + 1) + nx + i)
                        connNodesEdges[conn[k, 4], 4] = connElemEdges[k, 3]  # topedge #repeated
                        # connNodesElem:
                        # |   |
                        # a___o___
                        # |   |
                        # o___b___
                        # |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 2], 1] = NeiElements[k, 3]  # element: bottom left
                        connNodesElem[conn[k, 2], 2] = NeiElements[k + 1, 3]  # element: bottom right
                        connNodesElem[conn[k, 2], 3] = NeiElements[k, 2]  # element: top right
                        connNodesElem[conn[k, 2], 4] = k  # element: top left  (current k)
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 4], 1] = k  # element: bottom left  #repeated
                        connNodesElem[conn[k, 4], 2] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 4], 3] = NeiElements[k, 4]  # element: top right
                        connNodesElem[conn[k, 4], 4] = NeiElements[k, 4]  # element: top left  #repeated

                    elseif i != nx && j != ny && i != 1 && j == 1  # bottom row of cells
                        # for each top right node
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 3], 2] = connElemEdges[k, 2]  # rightedge
                        # BottomEdgeOfTopRightNeighboursElem
                        connNodesEdges[conn[k, 3], 3] = ((j) * (2 * nx + 1) + (i + 1))
                        # LeftEdgeOfTopRightNeighboursElem
                        connNodesEdges[conn[k, 3], 4] = ((j) * (2 * nx + 1) + nx + (i + 1))
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 1], 2] = connElemEdges[k, 4]  # leftedge
                        # BottomEdgeOfLeftNeighboursElem
                        connNodesEdges[conn[k, 1], 3] = ((j-1) * (2 * nx + 1) + (i - 1))
                        connNodesEdges[conn[k, 1], 4] = connElemEdges[k, 4]  # leftedge # repeated
                        # connNodesElem:
                        #    |   |
                        # ___o___b___
                        #    |   |
                        # ___a___o___
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 1], 1] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 1], 2] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top right (current k)
                        connNodesElem[conn[k, 1], 4] = NeiElements[k, 1]  # element: top left
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 3], 2] = NeiElements[k, 2]  # element: bottom right
                        connNodesElem[conn[k, 3], 3] = NeiElements[k + 1, 2]  # element: top right
                        connNodesElem[conn[k, 3], 4] = NeiElements[k, 4]  # element: top left

                    elseif i == nx && j == ny  # corner cell: top right
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 3], 2] = connElemEdges[k, 2]  # rightedge
                        connNodesEdges[conn[k, 3], 3] = connElemEdges[k, 3]  # topedge   #repeated
                        connNodesEdges[conn[k, 3], 4] = connElemEdges[k, 2]  # rightedge #repeated
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 2], 2] = connElemEdges[k, 2]  # rightedge
                        connNodesEdges[conn[k, 2], 3] = ((j - 2) * (2 * nx + 1) + nx + i + 1)  # RightEdgeBottomCell
                        connNodesEdges[conn[k, 2], 4] = connElemEdges[k, 1]  # bottomedge  #repeated
                        # connNodesElem:
                        # ___o___b
                        #    |   |
                        # ___o___a
                        #    |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 2], 1] = NeiElements[k, 3]  # element: bottom left
                        connNodesElem[conn[k, 2], 2] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 2], 3] = k  # element: top right #repeated
                        connNodesElem[conn[k, 2], 4] = k  # element: top left (current k)
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 3], 2] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 3], 3] = k  # element: top right  #repeated
                        connNodesElem[conn[k, 3], 4] = k  # element: top left  #repeated

                    elseif i == nx && j == 1  # corner cell: bottom right
                        connNodesEdges[conn[k, 2], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 2], 2] = connElemEdges[k, 2]  # rightedge
                        connNodesEdges[conn[k, 2], 3] = connElemEdges[k, 1]  # bottomedge #repeated
                        connNodesEdges[conn[k, 2], 4] = connElemEdges[k, 2]  # rightedge  #repeated
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 1], 2] = connElemEdges[k, 4]  # leftedge
                        connNodesEdges[conn[k, 1], 3] = ((j-1) * (2 * nx + 1) + (i - 1))  # BottomEdgeLeftCell
                        connNodesEdges[conn[k, 1], 4] = connElemEdges[k, 4]  # leftedge  #repeated
                        # connNodesElem:
                        #    |   |
                        # ___o___o
                        #    |   |
                        # ___a___b
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 1], 1] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 1], 2] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top right (current k)
                        connNodesElem[conn[k, 1], 4] = NeiElements[k, 1]  # element: top left
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 2], 1] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 2], 2] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 2], 3] = k  # element: top right  #repeated
                        connNodesElem[conn[k, 2], 4] = k  # element: top left  (current k)

                    elseif i == 1 && j == ny  # corner cell: top left
                        connNodesEdges[conn[k, 4], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 4], 2] = connElemEdges[k, 4]  # leftedge
                        connNodesEdges[conn[k, 4], 3] = connElemEdges[k, 3]  # topedge #repeated
                        connNodesEdges[conn[k, 4], 4] = connElemEdges[k, 4]  # leftedge #repeated
                        connNodesEdges[conn[k, 3], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 3], 2] = connElemEdges[k, 2]  # rightedge
                        connNodesEdges[conn[k, 3], 3] = (j * (2 * nx + 1) + (i + 1))  # TopEdgeRightCell
                        connNodesEdges[conn[k, 3], 4] = connElemEdges[k, 2]  # rightedge #repeated
                        # connNodesElem:
                        # b___a___
                        # |   |
                        # o___o___
                        # |   |
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 3], 1] = k  # element: bottom left (current k)
                        connNodesElem[conn[k, 3], 2] = NeiElements[k, 2]  # element: bottom right
                        connNodesElem[conn[k, 3], 3] = k  # element: top right #repeated
                        connNodesElem[conn[k, 3], 4] = k  # element: top left #repeated
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 4], 1] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 4], 2] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 4], 3] = k  # element: top right  #repeated
                        connNodesElem[conn[k, 4], 4] = k  # element: top left  #repeated

                    elseif i == 1 && j == 1  # corner cell: bottom left
                        connNodesEdges[conn[k, 1], 1] = connElemEdges[k, 1]  # bottomedge
                        connNodesEdges[conn[k, 1], 2] = connElemEdges[k, 4]  # leftedge
                        connNodesEdges[conn[k, 1], 3] = connElemEdges[k, 1]  # bottomedge #repeated
                        connNodesEdges[conn[k, 1], 4] = connElemEdges[k, 4]  # leftedge #repeated
                        connNodesEdges[conn[k, 4], 1] = connElemEdges[k, 3]  # topedge
                        connNodesEdges[conn[k, 4], 2] = connElemEdges[k, 4]  # leftedge
                        connNodesEdges[conn[k, 4], 3] = (j * (2 * nx + 1) + nx + i)  # LeftEdgeTopCell
                        connNodesEdges[conn[k, 4], 4] = connElemEdges[k, 3]  # topedge #repeated
                        # connNodesElem:
                        #
                        # |   |
                        # a___o___
                        # |   |
                        # b___o___
                        #
                        # note: NeiElements(ndarray): [left, right,bottom, up]
                        #
                        # node a (comments with respect to the node)
                        connNodesElem[conn[k, 4], 1] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 4], 2] = k  # element: bottom right (current k)
                        connNodesElem[conn[k, 4], 3] = NeiElements[k, 4]  # element: top right #repeated
                        connNodesElem[conn[k, 4], 4] = k  # element: top left #repeated
                        # node b (comments with respect to the node)
                        connNodesElem[conn[k, 1], 1] = k  # element: bottom left #repeated
                        connNodesElem[conn[k, 1], 2] = k  # element: bottom right #repeated
                        connNodesElem[conn[k, 1], 3] = k  # element: top right  (current k)
                        connNodesElem[conn[k, 1], 4] = k  # element: top left  #repeated
                    end
                else
                    node = conn[k, 2]  # for each bottom right node of the elements not near the mesh boundaryes
                    connNodesEdges[node, 1] = connElemEdges[k, 2]  # rightedge
                    connNodesEdges[node, 2] = connElemEdges[k, 1]  # bottomedge
                    connNodesEdges[node, 3] = connElemEdges[NeiElements[k, 3], 2]  # leftedgeBottomNeighboursElem
                    connNodesEdges[node, 4] = connElemEdges[NeiElements[k + 1, 3], 3]  # bottomedgeLeftNeighboursElem
                    # connNodesElem:
                    # note:  NeiElements(ndarray): [left, right,bottom, up]
                    # o___o___o
                    # | 4k| 3 |     k is the current element
                    # o___x___o     x is the current node
                    # | 1 | 2 |
                    # o___o___o
                    #
                    connNodesElem[node, 1] = NeiElements[k, 3]  # element: bottom left with respect to the node x
                    connNodesElem[node, 2] = NeiElements[k + 1, 3]  # element: bottom right with respect to the node x
                    connNodesElem[node, 3] = NeiElements[k, 2]  # element: top right  with respect to the node x
                    connNodesElem[node, 4] = k  # element: top left (current k) with respect to the node x
                    # end Peruzzo 2019
                end

        k = k + 1

        Connectivity = conn
        Connectivityelemedges = connElemEdges  # Peruzzo 2019
        Connectivityedgeselem = connEdgesElem  # Peruzzo 2019
        Connectivityedgesnodes = connEdgesNodes  # Peruzzo 2019
        Connectivitynodesedges = connNodesEdges  # Peruzzo 2019
        Connectivitynodeselem = connNodesElem  # Peruzzo 2019

        # coordinates of the center of the mesh
        centerMesh = [(domainLimits[3] + domainLimits[4])/2, (domainLimits[2] + domainLimits[1])/2]

        # coordinates of the center of the elements
        CoorMid = Matrix{Float64}(undef, NumberOfElts, 2)
        for e in 1:NumberOfElts
            vertex_indices = conn[e, :]
            vertices = VertexCoor[vertex_indices, :]
            CoorMid[e, :] = vec(mean(vertices, dims=1))
        end
        CenterCoor = CoorMid
        distCenter = sqrt.((CoorMid[:, 1] .- centerMesh[1]) .^ 2 .+ (CoorMid[:, 2] .- centerMesh[2]) .^ 2)
        # the element in the center (used for fluid injection)
        # todo: No it is not necessarily where we inject!
        center_x_condition = findall(abs.(CenterCoor[:, 1] .- centerMesh[1]) .< hx/2)
        center_y_condition = findall(abs.(CenterCoor[:, 2] .- centerMesh[2]) .< hy/2)
        CenterElts = intersect(center_x_condition, center_y_condition)

        if length(CenterElts) != 1
            CenterElts = Int(NumberOfElts / 2)
            @debug "Mesh with no center element. To be looked into"
            # throw(ErrorException("Mesh with no center element. To be looked into"))
        end
        if symmetric
            corresponding = corresponding_elements_in_symmetric(mesh)
            symmetricElts = get_symmetric_elements(mesh, 1:NumberOfElts)
            activeSymtrc, posQdrnt, boundary_x, boundary_y = get_active_symmetric_elements(mesh)
            
            volWeights = fill(4.0f0, length(activeSymtrc))
            volWeights[length(posQdrnt)+1:end-1] .= 2.0f0
            volWeights[end] = 1.0f0
        else
            corresponding = Int[]
            symmetricElts = Matrix{Int}(undef, 0, 0)
            activeSymtrc = Int[]
            posQdrnt = Int[]
            boundary_x = Int[]
            boundary_y = Int[]
            volWeights = Float32[]
        end

        new(Lx_val, Ly_val, nx, ny, hx, hy, NumberOfElts, NumberofNodes, EltArea, domainLimits,
            VertexCoor, CenterCoor, Connectivity, NeiElements, distCenter, CenterElts,
            Connectivityelemedges, Connectivityedgeselem, Connectivityedgesnodes,
            Connectivitynodesedges, Connectivitynodeselem,
            symmetric, corresponding, symmetricElts, activeSymtrc, posQdrnt, boundary_x, boundary_y, volWeights)

    end


    # -----------------------------------------------------------------------------------------------------------------------


    """
    locate_element(mesh, x, y)

    This function gives the cell containing the given coordinates. NaN is returned if the cell is not in the mesh.

    # Arguments
    - `mesh::CartesianMesh`: The mesh object.
    - `x::Float64`: The x coordinate of the given point.
    - `y::Float64`: The y coordinate of the given point.

    # Returns
    - `elt::Union{Int, Float64}`: The element containing the given coordinates, or NaN if outside.
    """
    function locate_element(mesh::CartesianMesh, x::Float64, y::Float64)::Union{Int, Float64}

        if x >= mesh.domainLimits[4] + mesh.hx / 2 || 
        y >= mesh.domainLimits[2] + mesh.hy / 2 ||
        x <= mesh.domainLimits[3] - mesh.hx / 2 || 
        y <= mesh.domainLimits[1] - mesh.hy / 2
            
            @warn "PyFrac.locate_element: Point is outside domain."
            return NaN
        end

        precision = 0.1 * sqrt(eps(Float64))

        cond_x = findall(abs.(mesh.CenterCoor[:, 1] .- x) .< mesh.hx / 2 + precision)
        cond_y = findall(abs.(mesh.CenterCoor[:, 2] .- y) .< mesh.hy / 2 + precision)
        
        candidate_elements = intersect(cond_x, cond_y)

        if length(candidate_elements) == 1
            return candidate_elements[1]
        else
            return NaN
        end
    end

#-----------------------------------------------------------------------------------------------------------------------

    """
        Neighbors(elem, nx, ny)

    Neighbouring elements of an element within the mesh. Boundary elements have themselves as neighbor.

    # Arguments
    - `elem::Int`:         -- element whose neighbor are to be found.
    - `nx::Int`:           -- number of elements in x direction.
    - `ny::Int`:           -- number of elements in y direction.

    # Returns
    - `(left::Int, right::Int, bottom::Int, up::Int)`: A tuple containing the following:

        | left (int)     -- left neighbour.
        | right (int)    -- right neighbour.
        | bottom (int)   -- bottom neighbour.
        | top (int)      -- top neighbour.
    """
    function Neighbors(elem::Int, nx::Int, ny::Int)::Tuple{Int, Int, Int, Int}

        j = div(elem - 1, nx) + 1
        i = mod(elem - 1, nx) + 1

        left = i == 1 ? elem : elem - 1
        right = i == nx ? elem : elem + 1
        bottom = j == 1 ? elem : elem - nx
        top = j == ny ? elem : elem + nx

        return (left, right, bottom, top)
    end


# ----------------------------------------------------------------------------------------------------------------------


    """
    plot(mesh; material_prop=nothing, backGround_param=nothing, fig=nothing, plot_prop=nothing)

    This function plots the 2D mesh. If material properties are given,
    cells will be color-coded according to the parameter specified by backGround_param.

    # Arguments
    - `mesh::CartesianMesh`: mesh object containing fields domainLimits, hx, hy,
    NumberOfElts, VertexCoor, Connectivity.
    - `material_prop::Union{Nothing, MaterialProperties}`: (optional) material properties object.
    - `backGround_param::Union{Nothing, String}`: (optional) name of parameter to color cells by ('sigma0', 'K1c', 'Cl', etc.).
    - `fig`: (optional) existing figure object (Plots.Plot) to overlay.
    - `plot_prop::Union{Nothing, NamedTuple}`: (optional) plot configuration with fields 
    alpha, lineColor, lineWidth, colorMap, meshEdgeColor.

    # Returns
    - Updated or new figure object `fig` (Plots.Plot).
    """
    function plot_mesh(mesh::CartesianMesh; material_prop=nothing, backGround_param=nothing, fig=nothing, plot_prop=nothing)
        
        default_cm = :viridis

        # Create figure if none provided
        if fig === nothing
            fig = Plots.plot()
        end

        # Set axis limits
        xlims = (mesh.domainLimits[3] - mesh.hx/2, mesh.domainLimits[4] + mesh.hx/2)
        ylims = (mesh.domainLimits[1] - mesh.hy/2, mesh.domainLimits[2] + mesh.hy/2)
        Plots.xlims!(fig, xlims...)
        Plots.ylims!(fig, ylims...)

        # Set default plot properties if none provided
        if plot_prop === nothing
            plot_prop = (
                alpha = 0.65,
                lineColor = :gray,
                lineWidth = 0.2,
                colorMap = default_cm,
                meshEdgeColor = :black,
            )
        end

        min_value = 0.0 
        max_value = 1.0 
        parameter_name = ""
        raw_colors = fill(0.5, mesh.NumberOfElts)
        cm = plot_prop.colorMap

        if material_prop !== nothing && backGround_param !== nothing
            min_value, max_value, parameter_name, raw_colors = process_material_prop_for_display(material_prop, backGround_param)
            cm = plot_prop.colorMap
            
            color_range = max_value - min_value
            if color_range == 0
                normalized_colors = fill(0.5, length(raw_colors))
            else
                normalized_colors = (raw_colors .- min_value) / color_range
            end
        else
            normalized_colors = fill(0.5, mesh.NumberOfElts)

        all_shapes = Shape[]
        all_colors = RGB{Float64}[] 
        c_palette = cgrad(cm)

        for i in 1:mesh.NumberOfElts
            verts = reshape(mesh.VertexCoor[mesh.Connectivity[i], :], (4, 2))
            xs, ys = verts[:, 1], verts[:, 2]
            push!(all_shapes, Shape(xs, ys))
            
            color_idx_normalized = normalized_colors[i]
            clamped_idx = clamp(color_idx_normalized, 0.0, 1.0) 
            push!(all_colors, get(c_palette, clamped_idx))
        end

        Plots.plot!(fig, all_shapes, 
            fillcolor = all_colors, 
            fillalpha = plot_prop.alpha, 
            seriestype = :shape,
            linecolor = plot_prop.meshEdgeColor, 
            linewidth = plot_prop.lineWidth,
            label = false,
            aspect_ratio = :equal
        )

        # Add colorbar if coloring applied
        if parameter_name != "" && material_prop !== nothing && backGround_param !== nothing
            dummy_x = [xlims[1], xlims[2]]
            dummy_y = [ylims[1], ylims[2]]
            dummy_z = [min_value, max_value] 
            
            Plots.scatter!(fig, dummy_x, dummy_y, zcolor=dummy_z,
                marker = (0.001, :white, 0),
                linealpha = 0, markeralpha = 0,
                color = cgrad(cm),
                colorbar = :right,
                colorbar_title = parameter_name,
                label = false,
                )
        end

        return fig
    end



#-----------------------------------------------------------------------------------------------------------------------

    def plot_3D(self, material_prop=None, backGround_param=None, fig=None, plot_prop=None):
        """
        This function plots the mesh in 3D. If the material properties is given, the cells will be color coded
        according to the parameter given by the backGround_param argument.

        Args:
            material_prop (MaterialProperties):  -- a MaterialProperties class object
            backGround_param (string):           -- the cells of the grid will be color coded according to the value
                                                    of the parameter given by this argument. Possible options are
                                                    'sigma0' for confining stress, 'K1c' for fracture toughness and
                                                    'Cl' for leak off.
            fig (Figure):                        -- A figure object to superimpose.
            plot_prop (PlotProperties):          -- A PlotProperties object giving the properties to be utilized for
                                                    the plot.

        Returns:
            (Figure):                            -- A Figure object to superimpose.

        """
        log = logging.getLogger('PyFrac.plot3D')
        if backGround_param is not None and material_prop is None:
            raise ValueError("Material properties are required to plot the background parameter.")
        if material_prop is not None and backGround_param is None:
            log.warning("back ground parameter not provided. Plotting confining stress...")
            backGround_param = 'sigma0'

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlim([self.domainLimits[2] * 1.2, self.domainLimits[3] * 1.2])
            ax.set_ylim([self.domainLimits[0] * 1.2, self.domainLimits[1] * 1.2])
            scale = 1.1
            zoom_factory(ax, base_scale=scale)
        else:
            ax = fig.get_axes()[0]

        if plot_prop is None:
            plot_prop = PlotProperties()
        if plot_prop.textSize is None:
            plot_prop.textSize = max(self.Lx / 15, self.Ly / 15)

        log.info("Plotting mesh in 3D...")
        if material_prop is not None and backGround_param is not None:
            min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                        backGround_param)

        # add rectangle for each cell
        for i in range(self.NumberOfElts):
            rgb_col = to_rgb(plot_prop.meshColor)
            if backGround_param is not None:
                face_color = (rgb_col[0] * colors[i], rgb_col[1] * colors[i], rgb_col[2] * colors[i], 0.5)
            else:
                face_color = (rgb_col[0], rgb_col[1], rgb_col[2], 0.5)

            rgb_col = to_rgb(plot_prop.meshEdgeColor)
            edge_color = (rgb_col[0], rgb_col[1], rgb_col[2], 0.2)
            cell = mpatches.Rectangle((self.CenterCoor[i, 0] - self.hx / 2,
                                       self.CenterCoor[i, 1] - self.hy / 2),
                                       self.hx,
                                       self.hy,
                                       ec=edge_color,
                                       fc=face_color)
            ax.add_patch(cell)
            art3d.pathpatch_2d_to_3d(cell)

        if backGround_param is not None and material_prop is not None:
            make_3D_colorbar(self, material_prop, backGround_param, ax, plot_prop)

        self.plot_scale_3d(ax, plot_prop)

        ax.grid(False)
        ax.set_frame_on(False)
        ax.set_axis_off()
        set_aspect_equal_3d(ax)
        return fig


#-----------------------------------------------------------------------------------------------------------------------

    def plot_scale_3d(self, ax, plot_prop):
        """
        This function plots the scale of the fracture by adding lines giving the length dimensions of the fracture.

        """
        log = logging.getLogger('PyFrac.plot_scale_3d')
        log.info("\tPlotting scale...")

        Path = mpath.Path

        rgb_col = to_rgb(plot_prop.meshLabelColor)
        edge_color = (rgb_col[0], rgb_col[1], rgb_col[2], 1.)

        codes = []
        verts = []
        verts_x = np.linspace(self.domainLimits[2], self.domainLimits[3], 7)
        verts_y = np.linspace(self.domainLimits[0], self.domainLimits[1], 7)
        tick_len = max(self.hx / 2, self.hy / 2)
        for i in range(7):
            codes.append(Path.MOVETO)
            elem = self.locate_element(verts_x[i], self.domainLimits[0])
            verts.append((self.CenterCoor[elem, 0], self.domainLimits[0] - self.hy / 2))
            codes.append(Path.LINETO)
            verts.append((self.CenterCoor[elem, 0], self.domainLimits[0] + tick_len))
            x_val = to_precision(np.round(self.CenterCoor[elem, 0], 5), plot_prop.dispPrecision)
            text3d(ax,
                   (self.CenterCoor[elem, 0] - plot_prop.dispPrecision * plot_prop.textSize / 3,
                    self.domainLimits[0] - self.hy / 2 - plot_prop.textSize,
                    0),
                   x_val,
                   zdir="z",
                   size=plot_prop.textSize,
                   usetex=plot_prop.useTex,
                   ec="none",
                   fc=edge_color)

            codes.append(Path.MOVETO)
            elem = self.locate_element(self.domainLimits[2], verts_y[i])
            verts.append((self.domainLimits[2] - self.hx / 2, self.CenterCoor[elem, 1][0]))
            codes.append(Path.LINETO)
            verts.append((self.domainLimits[2] + tick_len, self.CenterCoor[elem, 1][0]))
            y_val = to_precision(np.round(self.CenterCoor[elem, 1], 5), plot_prop.dispPrecision)
            text3d(ax,
                   (self.domainLimits[2] - self.hx / 2 - plot_prop.dispPrecision * plot_prop.textSize,
                    self.CenterCoor[elem, 1] - plot_prop.textSize / 2,
                    0),
                   y_val,
                   zdir="z",
                   size=plot_prop.textSize,
                   usetex=plot_prop.useTex,
                   ec="none",
                   fc=edge_color)

        log.info("\tAdding labels...")
        text3d(ax,
               (0.,
                -self.domainLimits[2] - plot_prop.textSize * 3,
                0),
               'meters',
               zdir="z",
               size=plot_prop.textSize,
               usetex=plot_prop.useTex,
               ec="none",
               fc=edge_color)

        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path,
                                   lw=plot_prop.lineWidth,
                                   facecolor='none',
                                   edgecolor=edge_color)
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch)

#-----------------------------------------------------------------------------------------------------------------------


    def identify_elements(self, elements, fig=None, plot_prop=None, plot_mesh=True, print_number=True):
        """
        This functions identify the given set of elements by highlighting them on the grid. the function plots
        the grid and the given set of elements.

        Args:
            elements (ndarray):             -- the given set of elements to be highlighted.
            fig (Figure):                   -- A figure object to superimpose.
            plot_prop (PlotProperties):     -- A PlotProperties object giving the properties to be utilized for
                                               the plot.
            plot_mesh (bool):               -- if False, grid will not be plotted and only the edges of the given
                                               elements will be plotted.
            print_number (bool):            -- if True, numbers of the cell will also be printed along with outline.

        Returns:
            (Figure):                       -- A Figure object that can be used superimpose further plots.

        """

        if plot_prop is None:
            plot_prop = PlotProperties()

        if plot_mesh:
            fig = self.plot(fig=fig)

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]

        # set the four corners of the rectangular mesh
        ax.set_xlim([self.domainLimits[2] - self.hx / 2, self.domainLimits[3] + self.hx / 2])
        ax.set_ylim([self.domainLimits[0] - self.hy / 2, self.domainLimits[1] + self.hy / 2])

        # add rectangle for each cell
        patch_list = []
        for i in elements:
            polygon = mpatches.Polygon(np.reshape(self.VertexCoor[self.Connectivity[i], :], (4, 2)), True)
            patch_list.append(polygon)

        p = PatchCollection(patch_list,
                            cmap=plot_prop.colorMap,
                            edgecolor=plot_prop.lineColor,
                            linewidth=plot_prop.lineWidth,
                            facecolors='none')
        ax.add_collection(p)

        if print_number:
            # print Element numbers on the plot for elements to be identified
            for i in range(len(elements)):
                ax.text(self.CenterCoor[elements[i], 0] - self.hx / 4, self.CenterCoor[elements[i], 1] -
                        self.hy / 4, repr(elements[i]), fontsize=plot_prop.textSize)

        return fig

#-----------------------------------------------------------------------------------------------------------------------


def make_3D_colorbar(mesh, material_prop, backGround_param, ax, plot_prop):
    """
    This function makes the color bar on 3D mesh plot using rectangular patches with color gradient from gray to the
    color given by the plot properties. The minimum and maximum values are taken from the given parameter in the
    material properties.

    """
    log = logging.getLogger('PyFrac.make_3D_colorbar')
    log.info("\tMaking colorbar...")

    min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                backGround_param)
    rgb_col_mesh = to_rgb(plot_prop.meshEdgeColor)
    edge_color = (rgb_col_mesh[0],
                  rgb_col_mesh[1],
                  rgb_col_mesh[2],
                  0.2)

    color_range = np.linspace(0, 1., 11)
    y = np.linspace(-mesh.Ly, mesh.Ly, 11)
    dy = y[1] - y[0]
    for i in range(11):
        rgb_col = to_rgb(plot_prop.meshColor)
        face_color = (rgb_col[0] * color_range[i],
                      rgb_col[1] * color_range[i],
                      rgb_col[2] * color_range[i],
                      0.5)
        cell = mpatches.Rectangle((mesh.Lx + 4 * mesh.hx,
                                   y[i]),
                                  2 * dy,
                                  dy,
                                  ec=edge_color,
                                  fc=face_color)
        ax.add_patch(cell)
        art3d.pathpatch_2d_to_3d(cell)

    rgb_col_txt = to_rgb(plot_prop.meshLabelColor)
    txt_color = (rgb_col_txt[0],
                 rgb_col_txt[1],
                 rgb_col_txt[2],
                 1.0)
    text3d(ax,
           (mesh.Lx + 4 * mesh.hx, y[9] + 3 * dy, 0),
           parameter,
           zdir="z",
           size=plot_prop.textSize,
           usetex=plot_prop.useTex,
           ec="none",
           fc=txt_color)
    y = [y[0], y[5], y[10]]
    values = np.linspace(min_value, max_value, 11)
    values = [values[0], values[5], values[10]]
    for i in range(3):
        disp_val = to_precision(values[i], plot_prop.dispPrecision)
        text3d(ax,
               (mesh.Lx + 4 * mesh.hx + 2 * dy, y[i] + dy / 2, 0),
               disp_val,
               zdir="z",
               size=plot_prop.textSize,
               usetex=plot_prop.useTex,
               ec="none",
               fc=txt_color)

#-----------------------------------------------------------------------------------------------------------------------


def process_material_prop_for_display(material_prop, backGround_param):
    """
    This function generates the appropriate variables to display the color coded mesh background.

    """

    colors = np.full((len(material_prop.SigmaO),), 0.5)

    if backGround_param in ['confining stress', 'sigma0']:
        max_value = max(material_prop.SigmaO) / 1e6
        min_value = min(material_prop.SigmaO) / 1e6
        if max_value - min_value > 0:
            colors = (material_prop.SigmaO / 1e6 - min_value) / (max_value - min_value)
        parameter = "confining stress ($MPa$)"
    elif backGround_param in ['fracture toughness', 'K1c']:
        max_value = max(material_prop.K1c) / 1e6
        min_value = min(material_prop.K1c) / 1e6
        if max_value - min_value > 0:
            colors = (material_prop.K1c / 1e6 - min_value) / (max_value - min_value)
        parameter = "fracture toughness ($Mpa\sqrt{m}$)"
    elif backGround_param in ['leak off coefficient', 'Cl']:
        max_value = max(material_prop.Cl)
        min_value = min(material_prop.Cl)
        if max_value - min_value > 0:
            colors = (material_prop.Cl - min_value) / (max_value - min_value)
        parameter = "Leak off coefficient"
    elif backGround_param is not None:
        raise ValueError("Back ground color identifier not supported!\n"
                         "Select one of the following:\n"
                         "-- \'confining stress\' or \'sigma0\'\n"
                         "-- \'fracture toughness\' or \'K1c\'\n"
                         "-- \'leak off coefficient\' or \'Cl\'")

    return min_value, max_value, parameter, colors


#-----------------------------------------------------------------------------------------------------------------------

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
