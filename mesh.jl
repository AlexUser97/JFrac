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
    using PyPlot
    using Statistics
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
                @warn "Number of elements in x-direction are even. Using $(nx+1) elements to have origin at a cell center..." _group="PyFrac.mesh"
                nx += 1
            end

            if ny % 2 == 0
                @warn "Number of elements in y-direction are even. Using $(ny+1) elements to have origin at a cell center..." _group="PyFrac.mesh"
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
            connEdgesElem = fill(-1, numberofedges, 2)          
            connNodesEdges = fill(-1, NumberofNodes, 4)         
            connNodesElem = fill(-1, NumberofNodes, 4)
            
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
                CoorMid[e, :] = mean(vertices, dims=1)
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
                @debug "Mesh with no center element. To be looked into" _group="PyFrac.mesh"
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
            
            @warn "PyFrac.locate_element: Point is outside domain." _group="PyFrac.mesh"
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
        plot(self, material_prop=nothing, backGround_param=nothing, fig=nothing, plot_prop=nothing)

        This function plots the mesh in 2D. If the material properties is given, the cells will be color coded
        according to the parameter given by the backGround_param argument.

        # Arguments
        - `material_prop`:           -- a MaterialProperties class object
        - `backGround_param`:        -- the cells of the grid will be color coded according to the value
                                    of the parameter given by this argument. Possible options are
                                    'sigma0' for confining stress, 'K1c' for fracture toughness and
                                    'Cl' for leak off.
        - `fig`:                     -- A figure object to superimpose.
        - `plot_prop`:               -- A PlotProperties object giving the properties to be utilized for
                                    the plot.

        # Returns
        - `(Figure)`:                -- A Figure object to superimpose.
    """
    function plot(self, material_prop=nothing, backGround_param=nothing, fig=nothing, plot_prop=nothing)
        if fig === nothing
            fig, ax = subplots()
        else
            figure(fig.number)
            subplot(111)
            ax = fig.get_axes()[1]
        end

        # set the four corners of the rectangular mesh
        ax.set_xlim([self.domainLimits[3] - self.hx / 2, self.domainLimits[4] + self.hx / 2])
        ax.set_ylim([self.domainLimits[1] - self.hy / 2, self.domainLimits[2] + self.hy / 2])

        # add rectangle for each cell
        patches = []
        for i in 1:self.NumberOfElts
            polygon = Polygon(reshape(self.VertexCoor[self.Connectivity[i], :], (4, 2)), true)
            push!(patches, polygon)
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
            plot_prop.alpha = 0.65
            plot_prop.lineColor = "0.5"
            plot_prop.lineWidth = 0.2
        end

        p = PatchCollection(patches,
                            cmap=plot_prop.colorMap,
                            alpha=plot_prop.alpha,
                            edgecolor=plot_prop.meshEdgeColor,
                            linewidth=plot_prop.lineWidth)

        # applying color according to the prescribed parameter
        if material_prop !== nothing && backGround_param !== nothing
            min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                    backGround_param)
            # plotting color bar
            sm = cm.ScalarMappable(cmap=plot_prop.colorMap,
                                norm=plt.Normalize(vmin=min_value, vmax=max_value))
            sm._A = []
            clr_bar = fig.colorbar(sm, alpha=0.65)
            clr_bar.set_label(parameter)
        else
            colors = fill(0.5, self.NumberOfElts)
        end

        p.set_array(Array(colors))
        ax.add_collection(p)
        axis("equal")

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_3D(self, material_prop=nothing, backGround_param=nothing, fig=nothing, plot_prop=nothing)

        This function plots the mesh in 3D. If the material properties is given, the cells will be color coded
        according to the parameter given by the backGround_param argument.

        # Arguments
        - `material_prop`:           -- a MaterialProperties class object
        - `backGround_param`:        -- the cells of the grid will be color coded according to the value
                                    of the parameter given by this argument. Possible options are
                                    'sigma0' for confining stress, 'K1c' for fracture toughness and
                                    'Cl' for leak off.
        - `fig`:                     -- A figure object to superimpose.
        - `plot_prop`:               -- A PlotProperties object giving the properties to be utilized for
                                    the plot.

        # Returns
        - `(Figure)`:                -- A Figure object to superimpose.
    """
    function plot_3D(self, material_prop=nothing, backGround_param=nothing, fig=nothing, plot_prop=nothing)
        if backGround_param !== nothing && material_prop === nothing
            throw(ArgumentError("Material properties are required to plot the background parameter."))
        end
        if material_prop !== nothing && backGround_param === nothing
            @warn "back ground parameter not provided. Plotting confining stress..." _group="PyFrac.mesh"
            backGround_param = "sigma0"
        end

        if fig === nothing
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.set_xlim([self.domainLimits[3] * 1.2, self.domainLimits[4] * 1.2])
            ax.set_ylim([self.domainLimits[1] * 1.2, self.domainLimits[2] * 1.2])
            scale = 1.1
            zoom_factory(ax, base_scale=scale)
        else
            ax = fig.get_axes()[1]
        end

        if plot_prop === nothing
            plot_prop = PlotProperties()
        end
        if plot_prop.textSize === nothing
            plot_prop.textSize = max(self.Lx / 15, self.Ly / 15)
        end

        @info "Plotting mesh in 3D..." _group="PyFrac.mesh"
        if material_prop !== nothing && backGround_param !== nothing
            min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                    backGround_param)
        end

        # add rectangle for each cell
        for i in 1:self.NumberOfElts
            rgb_col = to_rgb(plot_prop.meshColor)
            if backGround_param !== nothing
                face_color = (rgb_col[1] * colors[i], rgb_col[2] * colors[i], rgb_col[3] * colors[i], 0.5)
            else
                face_color = (rgb_col[1], rgb_col[2], rgb_col[3], 0.5)
            end

            rgb_col = to_rgb(plot_prop.meshEdgeColor)
            edge_color = (rgb_col[1], rgb_col[2], rgb_col[3], 0.2)
            cell = Rectangle((self.CenterCoor[i, 1] - self.hx / 2,
                            self.CenterCoor[i, 2] - self.hy / 2),
                            self.hx,
                            self.hy,
                            ec=edge_color,
                            fc=face_color)
            ax.add_patch(cell)
            art3d.pathpatch_2d_to_3d(cell)
        end

        if backGround_param !== nothing && material_prop !== nothing
            make_3D_colorbar(self, material_prop, backGround_param, ax, plot_prop)
        end

        plot_scale_3d(self, ax, plot_prop)

        ax.grid(false)
        ax.set_frame_on(false)
        ax.set_axis_off()
        set_aspect_equal_3d(ax)
        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        plot_scale_3d(self, ax, plot_prop)

        This function plots the scale of the fracture by adding lines giving the length dimensions of the fracture.
    """
    function plot_scale_3d(self, ax, plot_prop)
        @info "\tPlotting scale..." _group="PyFrac.mesh"

        Path = mpath.Path

        rgb_col = to_rgb(plot_prop.meshLabelColor)
        edge_color = (rgb_col[1], rgb_col[2], rgb_col[3], 1.0)

        codes = []
        verts = []
        verts_x = range(self.domainLimits[3], self.domainLimits[4], length=7)
        verts_y = range(self.domainLimits[1], self.domainLimits[2], length=7)
        tick_len = max(self.hx / 2, self.hy / 2)
        for i in 1:7
            push!(codes, Path.MOVETO)
            elem = locate_element(self, verts_x[i], self.domainLimits[1])
            push!(verts, (self.CenterCoor[elem, 1], self.domainLimits[1] - self.hy / 2))
            push!(codes, Path.LINETO)
            push!(verts, (self.CenterCoor[elem, 1], self.domainLimits[1] + tick_len))
            x_val = to_precision(round(self.CenterCoor[elem, 1], digits=5), plot_prop.dispPrecision)
            text3d(ax,
                (self.CenterCoor[elem, 1] - plot_prop.dispPrecision * plot_prop.textSize / 3,
                    self.domainLimits[1] - self.hy / 2 - plot_prop.textSize,
                    0),
                x_val,
                zdir="z",
                size=plot_prop.textSize,
                usetex=plot_prop.useTex,
                ec="none",
                fc=edge_color)

            push!(codes, Path.MOVETO)
            elem = locate_element(self, self.domainLimits[3], verts_y[i])
            push!(verts, (self.domainLimits[3] - self.hx / 2, self.CenterCoor[elem, 2]))
            push!(codes, Path.LINETO)
            push!(verts, (self.domainLimits[3] + tick_len, self.CenterCoor[elem, 2]))
            y_val = to_precision(round(self.CenterCoor[elem, 2], digits=5), plot_prop.dispPrecision)
            text3d(ax,
                (self.domainLimits[3] - self.hx / 2 - plot_prop.dispPrecision * plot_prop.textSize,
                    self.CenterCoor[elem, 2] - plot_prop.textSize / 2,
                    0),
                y_val,
                zdir="z",
                size=plot_prop.textSize,
                usetex=plot_prop.useTex,
                ec="none",
                fc=edge_color)
        end

        @info "\tAdding labels..." _group="PyFrac.mesh"
        text3d(ax,
            (0.0,
                -self.domainLimits[3] - plot_prop.textSize * 3,
                0),
            "meters",
            zdir="z",
            size=plot_prop.textSize,
            usetex=plot_prop.useTex,
            ec="none",
            fc=edge_color)

        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path,
                                lw=plot_prop.lineWidth,
                                facecolor="none",
                                edgecolor=edge_color)
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch)
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        identify_elements(self, elements, fig=nothing, plot_prop=nothing, plot_mesh=true, print_number=true)

        This functions identify the given set of elements by highlighting them on the grid. the function plots
        the grid and the given set of elements.

        # Arguments
        - `elements`:                -- the given set of elements to be highlighted.
        - `fig`:                     -- A figure object to superimpose.
        - `plot_prop`:               -- A PlotProperties object giving the properties to be utilized for
                                    the plot.
        - `plot_mesh`:               -- if False, grid will not be plotted and only the edges of the given
                                    elements will be plotted.
        - `print_number`:            -- if True, numbers of the cell will also be printed along with outline.

        # Returns
        - `(Figure)`:                -- A Figure object that can be used superimpose further plots.
    """
    function identify_elements(self, elements, fig=nothing, plot_prop=nothing, plot_mesh=true, print_number=true)
        if plot_prop === nothing
            plot_prop = PlotProperties()
        end

        if plot_mesh
            fig = plot(self, fig=fig)
        end

        if fig === nothing
            fig, ax = subplots()
        else
            ax = fig.get_axes()[1]
        end

        # set the four corners of the rectangular mesh
        ax.set_xlim([self.domainLimits[3] - self.hx / 2, self.domainLimits[4] + self.hx / 2])
        ax.set_ylim([self.domainLimits[1] - self.hy / 2, self.domainLimits[2] + self.hy / 2])

        # add rectangle for each cell
        patch_list = []
        for i in elements
            polygon = Polygon(reshape(self.VertexCoor[self.Connectivity[i], :], (4, 2)), true)
            push!(patch_list, polygon)
        end

        p = PatchCollection(patch_list,
                            cmap=plot_prop.colorMap,
                            edgecolor=plot_prop.lineColor,
                            linewidth=plot_prop.lineWidth,
                            facecolors="none")
        ax.add_collection(p)

        if print_number
            # print Element numbers on the plot for elements to be identified
            for i in 1:length(elements)
                ax.text(self.CenterCoor[elements[i], 1] - self.hx / 4, self.CenterCoor[elements[i], 2] -
                        self.hy / 4, string(elements[i]), fontsize=plot_prop.textSize)
            end
        end

        return fig
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        make_3D_colorbar(mesh, material_prop, backGround_param, ax, plot_prop)

        This function makes the color bar on 3D mesh plot using rectangular patches with color gradient from gray to the
        color given by the plot properties. The minimum and maximum values are taken from the given parameter in the
        material properties.
    """
    function make_3D_colorbar(mesh, material_prop, backGround_param, ax, plot_prop)
        @info "\tMaking colorbar..." _group="PyFrac.mesh"

        min_value, max_value, parameter, colors = process_material_prop_for_display(material_prop,
                                                                                backGround_param)
        rgb_col_mesh = to_rgb(plot_prop.meshEdgeColor)
        edge_color = (rgb_col_mesh[1],
                    rgb_col_mesh[2],
                    rgb_col_mesh[3],
                    0.2)

        color_range = range(0, 1.0, length=11)
        y = range(-mesh.Ly, mesh.Ly, length=11)
        dy = y[2] - y[1]
        for i in 1:11
            rgb_col = to_rgb(plot_prop.meshColor)
            face_color = (rgb_col[1] * color_range[i],
                        rgb_col[2] * color_range[i],
                        rgb_col[3] * color_range[i],
                        0.5)
            cell = Rectangle((mesh.Lx + 4 * mesh.hx,
                            y[i]),
                            2 * dy,
                            dy,
                            ec=edge_color,
                            fc=face_color)
            ax.add_patch(cell)
            art3d.pathpatch_2d_to_3d(cell)
        end

        rgb_col_txt = to_rgb(plot_prop.meshLabelColor)
        txt_color = (rgb_col_txt[1],
                    rgb_col_txt[2],
                    rgb_col_txt[3],
                    1.0)
        text3d(ax,
            (mesh.Lx + 4 * mesh.hx, y[10] + 3 * dy, 0),
            parameter,
            zdir="z",
            size=plot_prop.textSize,
            usetex=plot_prop.useTex,
            ec="none",
            fc=txt_color)
        y_selected = [y[1], y[6], y[11]]
        values = range(min_value, max_value, length=11)
        values_selected = [values[1], values[6], values[11]]
        for i in 1:3
            disp_val = to_precision(values_selected[i], plot_prop.dispPrecision)
            text3d(ax,
                (mesh.Lx + 4 * mesh.hx + 2 * dy, y_selected[i] + dy / 2, 0),
                disp_val,
                zdir="z",
                size=plot_prop.textSize,
                usetex=plot_prop.useTex,
                ec="none",
                fc=txt_color)
        end
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        process_material_prop_for_display(material_prop, backGround_param)

        This function generates the appropriate variables to display the color coded mesh background.
    """
    function process_material_prop_for_display(material_prop, backGround_param)
        colors = fill(0.5, length(material_prop.SigmaO))

        if backGround_param in ["confining stress", "sigma0"]
            max_value = maximum(material_prop.SigmaO) / 1e6
            min_value = minimum(material_prop.SigmaO) / 1e6
            if max_value - min_value > 0
                colors = (material_prop.SigmaO / 1e6 - min_value) / (max_value - min_value)
            end
            parameter = "confining stress (\$MPa\$)"
        elseif backGround_param in ["fracture toughness", "K1c"]
            max_value = maximum(material_prop.K1c) / 1e6
            min_value = minimum(material_prop.K1c) / 1e6
            if max_value - min_value > 0
                colors = (material_prop.K1c / 1e6 - min_value) / (max_value - min_value)
            end
            parameter = "fracture toughness (\$Mpa\\sqrt{m}\$)"
        elseif backGround_param in ["leak off coefficient", "Cl"]
            max_value = maximum(material_prop.Cl)
            min_value = minimum(material_prop.Cl)
            if max_value - min_value > 0
                colors = (material_prop.Cl - min_value) / (max_value - min_value)
            end
            parameter = "Leak off coefficient"
        elseif backGround_param !== nothing
            throw(ArgumentError("Back ground color identifier not supported!\n" *
                                "Select one of the following:\n" *
                                "-- 'confining stress' or 'sigma0'\n" *
                                "-- 'fracture toughness' or 'K1c'\n" *
                                "-- 'leak off coefficient' or 'Cl'"))
        end

        return min_value, max_value, parameter, colors
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        set_aspect_equal_3d(ax)

        Fix equal aspect bug for 3D plots.
    """
    function set_aspect_equal_3d(ax)
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        xmean = mean(xlim)
        ymean = mean(ylim)
        zmean = mean(zlim)

        plot_radius = maximum([abs(lim - mean_)
                            for (lims, mean_) in zip((xlim, ylim, zlim), (xmean, ymean, zmean))
                            for lim in lims])

        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    end
end # module Mesh
