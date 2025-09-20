# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""

module LevelSet

    using Logging
    using NLsolve

    export SolveFMM, reconstruct_front, reconstruct_front_LS_gradient, UpdateLists, Eikonal_Res

    """
        SolveFMM(levelSet, EltRibbon, EltChannel, mesh, farAwayPstv, farAwayNgtv)

        Solve Eikonal equation to get level set.

        Arguments:
        - `levelSet::Vector{Float64}`: Level set to be evaluated and updated.
        - `EltRibbon::Vector{Int}`: Cells with given distance from the front.
        - `EltChannel::Vector{Int}`: Cells enclosed by the given cells.
        - `mesh::CartesianMesh`: Mesh object.
        - `farAwayPstv::Vector{Int}`: The cells outwards from ribbon cells for which the distance from front is to be evaluated.
        - `farAwayNgtv::Vector{Int}`: The cells inwards from ribbon cells for which the distance from front is to be evaluated.

        Returns:
        - Nothing. The levelSet is updated in place.
    """
    function SolveFMM(levelSet::Vector{Float64}, 
                    EltRibbon::Vector{Int}, 
                    EltChannel::Vector{Int}, 
                    mesh, 
                    farAwayPstv::Vector{Int}, 
                    farAwayNgtv::Vector{Int})
        
        # For elements radially outward from ribbon cells
        Alive = copy(EltRibbon)
        NarrowBand = copy(EltRibbon)
        FarAway = setdiff(farAwayPstv, NarrowBand)
        
        Alive_status = falses(mesh.NumberOfElts)
        NarrowBand_status = falses(mesh.NumberOfElts)
        FarAway_status = falses(mesh.NumberOfElts)
        Alive_status[Alive] .= true
        NarrowBand_status[NarrowBand] .= true
        FarAway_status[FarAway] .= true
        
        beta = mesh.hx / mesh.hy
        
        while length(NarrowBand) > 0
            # Find element with minimum levelSet value in NarrowBand
            min_idx = argmin(levelSet[NarrowBand])
            Smallest = NarrowBand[min_idx]
            neighbors = mesh.NeiElements[Smallest, :]
            
            for neighbor in neighbors
                if !Alive_status[neighbor]
                    if FarAway_status[neighbor]
                        push!(NarrowBand, neighbor)
                        NarrowBand_status[neighbor] = true
                        # Remove neighbor from FarAway
                        far_idx = findfirst(==(neighbor), FarAway)
                        if far_idx !== nothing
                            deleteat!(FarAway, far_idx)
                        end
                        FarAway_status[neighbor] = false
                    end
                    
                    # Note: Julia uses 1-based indexing, so neighbors indices are 1,2,3,4
                    NeigxMin = min(levelSet[mesh.NeiElements[neighbor, 1]], 
                                levelSet[mesh.NeiElements[neighbor, 2]])
                    NeigyMin = min(levelSet[mesh.NeiElements[neighbor, 3]], 
                                levelSet[mesh.NeiElements[neighbor, 4]])
                    
                    if NeigxMin >= 1e50 && NeigyMin >= 1e50
                        @warn "You are trying to compute the level set in a cell where all the neighbours have infinite distance to the front"
                    end
                    
                    delT = NeigyMin - NeigxMin
                    theta_sq = mesh.hx^2 * (1 + beta^2) - beta^2 * delT^2
                    
                    if theta_sq > 0
                        levelSet[neighbor] = (NeigxMin + beta^2 * NeigyMin + sqrt(theta_sq)) / (1 + beta^2)
                    else
                        # The distance is to be taken from the horizontal or vertical neighbouring cell
                        levelSet[neighbor] = min(NeigyMin + mesh.hy, NeigxMin + mesh.hx)
                    end
                end
            end
            
            push!(Alive, Smallest)
            Alive_status[Smallest] = true
            deleteat!(NarrowBand, min_idx)
            NarrowBand_status[Smallest] = false
        end
        
        # Hack - find out why this is required
        # Handle unevaluated cells using numerical solver
        if length(farAwayPstv) > 0
            unevaluated_mask = levelSet[farAwayPstv] .>= 1e50
            if any(unevaluated_mask)
                unevaluated_indices = findall(unevaluated_mask)
                unevaluated = farAwayPstv[unevaluated_indices]
                
                for i in unevaluated
                    neighbors = mesh.NeiElements[i, :]
                    Eikargs = (levelSet[neighbors[1]], levelSet[neighbors[2]], 
                            levelSet[neighbors[3]], levelSet[neighbors[4]], 
                            1.0, mesh.hx, mesh.hy)
                    guess = maximum(levelSet[neighbors])
                    
                    # Numerical solver using NLsolve
                    function residual_func(Tij)
                        return Eikonal_Res(Tij[1], Eikargs...)
                    end
                    
                    result = nlsolve(residual_func, [guess], autodiff=:forward)
                    if converged(result)
                        levelSet[i] = result.zero[1]
                    else
                        # Fallback: use max neighbor value + mesh size
                        levelSet[i] = guess + min(mesh.hx, mesh.hy)
                    end
                end
            end
        end
        
        # For elements radially inward from ribbon cells
        if length(farAwayNgtv) > 0
            RibbonInwardElts = setdiff(EltChannel, EltRibbon)
            positive_levelSet = fill(1e50, mesh.NumberOfElts)
            positive_levelSet[EltRibbon] = -levelSet[EltRibbon]
            
            Alive = copy(EltRibbon)
            NarrowBand = copy(EltRibbon)
            FarAway = setdiff(farAwayNgtv, NarrowBand)
            
            Alive_status = falses(mesh.NumberOfElts)
            NarrowBand_status = falses(mesh.NumberOfElts)
            FarAway_status = falses(mesh.NumberOfElts)
            Alive_status[Alive] .= true
            NarrowBand_status[NarrowBand] .= true
            FarAway_status[FarAway] .= true
            
            while length(NarrowBand) > 0
                min_idx = argmin(positive_levelSet[NarrowBand])
                Smallest = NarrowBand[min_idx]
                neighbors = mesh.NeiElements[Smallest, :]
                
                for neighbor in neighbors
                    if !Alive_status[neighbor]
                        if FarAway_status[neighbor]
                            push!(NarrowBand, neighbor)
                            NarrowBand_status[neighbor] = true
                            # Remove neighbor from FarAway
                            far_idx = findfirst(==(neighbor), FarAway)
                            if far_idx !== nothing
                                deleteat!(FarAway, far_idx)
                            end
                            FarAway_status[neighbor] = false
                        end
                        
                        NeigxMin = min(positive_levelSet[mesh.NeiElements[neighbor, 1]], 
                                    positive_levelSet[mesh.NeiElements[neighbor, 2]])
                        NeigyMin = min(positive_levelSet[mesh.NeiElements[neighbor, 3]], 
                                    positive_levelSet[mesh.NeiElements[neighbor, 4]])
                        
                        if NeigxMin >= 1e50 && NeigyMin >= 1e50
                            @warn "You are trying to compute the level set in a cell where all the neighbours have infinite distance to the front"
                        end
                        
                        beta = mesh.hx / mesh.hy
                        delT = NeigyMin - NeigxMin
                        theta_sq = mesh.hx^2 * (1 + beta^2) - beta^2 * delT^2
                        
                        if theta_sq > 0
                            positive_levelSet[neighbor] = (NeigxMin + beta^2 * NeigyMin + sqrt(theta_sq)) / (1 + beta^2)
                        else
                            positive_levelSet[neighbor] = min(NeigyMin + mesh.hy, NeigxMin + mesh.hx)
                        end
                    end
                end
                
                push!(Alive, Smallest)
                Alive_status[Smallest] = true
                deleteat!(NarrowBand, min_idx)
                NarrowBand_status[Smallest] = false
            end
            
            # Assigning adjusted value to the level set to be returned
            levelSet[RibbonInwardElts] = -positive_levelSet[RibbonInwardElts]
        end
        
        # Hack - find out why this is required (negative direction)
        # Handle unevaluated cells using numerical solver
        if length(farAwayNgtv) > 0
            unevaluated_mask = abs.(levelSet[farAwayNgtv]) .>= 1e50
            if any(unevaluated_mask)
                unevaluated_indices = findall(unevaluated_mask)
                unevaluated = farAwayNgtv[unevaluated_indices]
                
                for i in unevaluated
                    neighbors = mesh.NeiElements[i, :]
                    Eikargs = (levelSet[neighbors[1]], levelSet[neighbors[2]], 
                            levelSet[neighbors[3]], levelSet[neighbors[4]], 
                            1.0, mesh.hx, mesh.hy)
                    guess = maximum(levelSet[neighbors])
                    
                    # Numerical solver using NLsolve
                    function residual_func(Tij)
                        return Eikonal_Res(Tij[1], Eikargs...)
                    end
                    
                    result = nlsolve(residual_func, [guess], autodiff=:forward)
                    if converged(result)
                        levelSet[i] = result.zero[1]
                    else
                        # Fallback: use max neighbor value + mesh size
                        levelSet[i] = guess + min(mesh.hx, mesh.hy)
                    end
                end
            end
        end
        
        return nothing
    end
    #-----------------------------------------------------------------------------------------------------------------------


    """
        reconstruct_front(dist, bandElts, EltChannel, mesh)

        Track the fracture front, the length of the perpendicular drawn on the fracture and the angle inscribed by the
        perpendicular. The angle is calculated using the formulation given by Pierce and Detournay 2008.

        Arguments:
            dist (Vector{Float64}):         -- the signed distance of the cells from the fracture front.
            bandElts (Vector{Int}):     -- the band of elements to which the search is limited.
            EltChannel (Vector{Int}):   -- list of Channel elements.
            mesh (CartesianMesh):   -- the mesh of the fracture.
    """
    function reconstruct_front(dist::Vector{Float64}, bandElts::Vector{Int}, EltChannel::Vector{Int}, mesh)
        # Elements that are not in channel
        EltRest = setdiff(bandElts, EltChannel)
        ElmntTip = Int[]
        l = Float64[]
        alpha = Float64[]

        for i in 1:length(EltRest)
            neighbors = @view mesh.NeiElements[EltRest[i], :]

            minx = min(dist[neighbors[1]], dist[neighbors[2]]) # Assuming 1-based indexing: 1=left, 2=right
            miny = min(dist[neighbors[3]], dist[neighbors[4]]) # Assuming 3=bottom, 4=top
            # distance of the vertex (zero vertex, i.e. rotated distance) of the current cell from the front
            Pdis = -(minx + miny) / 2

            # if the vertex distance is positive, meaning the fracture has passed the vertex
            if Pdis >= 0
                push!(ElmntTip, EltRest[i])
                push!(l, Pdis)

                # calculate angle imposed by the perpendicular on front (see Peirce & Detournay 2008)
                delDist = miny - minx
                beta = mesh.hx / mesh.hy
                theta_sq = mesh.hx^2 * (1 + beta^2) - beta^2 * delDist^2
                if theta_sq >= 0 # Check to avoid NaN from sqrt
                    theta = sqrt(theta_sq)
                    # angle calculate with inverse of cosine trigonometric function
                    a1 = acos((theta + beta^2 * delDist) / (mesh.hx * (1 + beta^2)))
                    # angle calculate with inverse of sine trigonometric function
                    sinalpha = beta * (theta - delDist) / (mesh.hx * (1 + beta^2))
                    a2 = asin(sinalpha)

                    # !!!Hack. this check of zero or 90 degree angle works better
                    # Note: Julia uses @warn or Logging for warnings, not warnings.filterwarnings
                    if abs(1 - dist[neighbors[1]] / dist[neighbors[2]]) < 1e-5
                        a2 = π / 2
                    elseif abs(1 - dist[neighbors[3]] / dist[neighbors[4]]) < 1e-5
                        a2 = 0.0
                    end

                    #todo hack!!!
                    # checks to remove numerical noise in angle calculation
                    angle_selected = NaN
                    if 0 <= a2 <= π / 2
                        angle_selected = a2
                    elseif 0 <= a1 <= π / 2
                        angle_selected = a1
                    elseif a2 < 0 && a2 > -1e-6
                        angle_selected = 0.0
                    elseif a2 > π / 2 && a2 < π / 2 + 1e-6
                        angle_selected = π / 2
                    elseif a1 < 0 && a1 > -1e-6
                        angle_selected = 0.0
                    elseif a1 > π / 2 && a1 < π / 2 + 1e-6
                        angle_selected = π / 2
                    else
                        if abs(1 - dist[neighbors[1]] / dist[neighbors[2]]) < 0.1
                            angle_selected = π / 2
                        elseif abs(1 - dist[neighbors[3]] / dist[neighbors[4]]) < 0.1
                            angle_selected = 0.0
                        else
                            # alpha remains NaN
                        end
                    end
                    push!(alpha, angle_selected)
                else
                    push!(alpha, NaN) # Handle case where theta_sq is negative
                end
            end
        end

        # Handle NaN values in alpha
        nan_indices = findall(isnan, alpha)
        if !isempty(nan_indices)
            alpha_mesh = fill(NaN, mesh.NumberOfElts)
            alpha_mesh[ElmntTip] = alpha
            for i in nan_indices
                neighbors = @view mesh.NeiElements[ElmntTip[i], :]
                neig_in_tip = intersect(ElmntTip, neighbors)
                alpha_neig = alpha_mesh[neig_in_tip]
                # Remove NaN values from neighbors
                valid_alpha_neig = filter(!isnan, alpha_neig)
                if !isempty(valid_alpha_neig)
                    alpha[i] = mean(valid_alpha_neig)
                end
                # If all neighbors are NaN, alpha[i] remains NaN
            end
        end

        CellStatusNew = zeros(Int, mesh.NumberOfElts)
        CellStatusNew[EltChannel] .= 1
        CellStatusNew[ElmntTip] .= 2

        return ElmntTip, l, alpha, CellStatusNew
    end

    # -----------------------------------------------------------------------------------------------------------------------


    """
        reconstruct_front_LS_gradient(dist, EltBand, EltChannel, mesh)

        Track the fracture front, the length of the perpendicular drawn on the fracture and the angle inscribed by the
        perpendicular. The angle is calculated from the gradient of the level set.

        Arguments:
            dist (Vector{Float64}):         -- the signed distance of the cells from the fracture front.
            EltBand (Vector{Int}):      -- the band of elements to which the search is limited.
            EltChannel (Vector{Int}):   -- list of Channel elements.
            mesh (CartesianMesh):   -- the mesh of the fracture.
    """
    function reconstruct_front_LS_gradient(dist::Vector{Float64}, EltBand::Vector{Int}, EltChannel::Vector{Int}, mesh)
        # Elements that are not in channel
        EltRest = setdiff(EltBand, EltChannel)
        ElmntTip = Int[]
        l = Float64[]
        alpha = Float64[]

        for i in 1:length(EltRest)
            elt_index = EltRest[i]
            neighbors = @view mesh.NeiElements[elt_index, :] # Assuming 1=left, 2=right, 3=bottom, 4=top

            minx = min(dist[neighbors[1]], dist[neighbors[2]])
            miny = min(dist[neighbors[3]], dist[neighbors[4]])
            # distance of the vertex (zero vertex, i.e. rotated distance) of the current cell from the front
            Pdis = -(minx + miny) / 2

            # if the vertex distance is positive, meaning the fracture has passed the vertex
            if Pdis >= 0
                push!(ElmntTip, elt_index)
                push!(l, Pdis)

                # neighbors
                #     6     3    7
                #     0    elt   1
                #     4    2     5
                neighbors_tip = zeros(Int, 8)
                neighbors_tip[1:4] = neighbors # 1-based indexing
                neighbors_tip[5] = mesh.NeiElements[neighbors[3], 1] # neighbor 4's left neighbor
                neighbors_tip[6] = mesh.NeiElements[neighbors[3], 2] # neighbor 4's right neighbor
                neighbors_tip[7] = mesh.NeiElements[neighbors[4], 1] # neighbor 3's left neighbor
                neighbors_tip[8] = mesh.NeiElements[neighbors[4], 2] # neighbor 3's right neighbor

                gradx = 0.0
                grady = 0.0

                # zero Vertex
                #     3         2
                #     0         1
                if dist[neighbors_tip[1]] <= dist[neighbors_tip[2]] && dist[neighbors_tip[3]] <= dist[neighbors_tip[4]]
                    # if zero vertex is 0:
                    gradx = -((dist[neighbors_tip[1]] + dist[neighbors_tip[5]]) / 2 - (dist[elt_index] + dist[neighbors_tip[3]]) / 2) / mesh.hx
                    grady = ((dist[neighbors_tip[1]] + dist[elt_index]) / 2 - (dist[neighbors_tip[5]] + dist[neighbors_tip[3]]) / 2) / mesh.hy

                elseif dist[neighbors_tip[1]] > dist[neighbors_tip[2]] && dist[neighbors_tip[3]] <= dist[neighbors_tip[4]]
                    # if zero vertex is 1:
                    gradx = ((dist[neighbors_tip[2]] + dist[neighbors_tip[6]]) / 2 - (dist[elt_index] + dist[neighbors_tip[3]]) / 2) / mesh.hx
                    grady = ((dist[neighbors_tip[2]] + dist[elt_index]) / 2 - (dist[neighbors_tip[6]] + dist[neighbors_tip[3]]) / 2) / mesh.hy

                elseif dist[neighbors_tip[1]] > dist[neighbors_tip[2]] && dist[neighbors_tip[3]] > dist[neighbors_tip[4]]
                    # if zero vertex is 2:
                    gradx = ((dist[neighbors_tip[2]] + dist[neighbors_tip[8]]) / 2 - (dist[elt_index] + dist[neighbors_tip[4]]) / 2) / mesh.hx
                    grady = -((dist[neighbors_tip[2]] + dist[elt_index]) / 2 - (dist[neighbors_tip[4]] + dist[neighbors_tip[8]]) / 2) / mesh.hy

                elseif dist[neighbors_tip[1]] <= dist[neighbors_tip[2]] && dist[neighbors_tip[3]] > dist[neighbors_tip[4]]
                    # if zero vertex is 3:
                    gradx = -((dist[neighbors_tip[7]] + dist[neighbors_tip[1]]) / 2 - (dist[elt_index] + dist[neighbors_tip[4]]) / 2) / mesh.hx
                    grady = ((dist[neighbors_tip[1]] + dist[elt_index]) / 2 - (dist[neighbors_tip[7]] + dist[neighbors_tip[4]]) / 2) / mesh.hy
                end

                # Calculate angle alpha
                grad_norm = sqrt(gradx^2 + grady^2)
                if grad_norm > 0
                    # Ensure the argument for asin is within [-1, 1] due to potential numerical errors
                    sin_alpha = grady / grad_norm
                    sin_alpha = clamp(sin_alpha, -1.0, 1.0)
                    push!(alpha, abs(asin(sin_alpha)))
                else
                    push!(alpha, 0.0) # or NaN, depending on desired behavior for zero gradient
                end
            end
        end

        CellStatusNew = zeros(Int, mesh.NumberOfElts)
        CellStatusNew[EltChannel] .= 1
        CellStatusNew[ElmntTip] .= 2

        return ElmntTip, l, alpha, CellStatusNew
    end


    # ----------------------------------------------------------------------------------------------------------------------

    """
        UpdateLists(EltsChannel, EltsTipNew, FillFrac, levelSet, mesh)

        This function updates the Element lists, given the element lists from the last time step. EltsTipNew list can have 
        partially filled and fully filled elements. The function updates lists accordingly.

        Arguments:
            EltsChannel (Vector{Int}):      -- channel elements list.
            EltsTipNew (Vector{Int}):       -- list of the new tip elements, including fully filled cells that were tip
                                            cells in the last time step.
            FillFrac (Vector{Float64}):     -- filling fraction of the new tip cells.
            levelSet (Vector{Float64}):     -- current level set.
            mesh (CartesianMesh):           -- the mesh of the fracture.

        Returns:
            - eltsChannel (Vector{Int}):    -- new channel elements list.
            - eltsTip (Vector{Int}):        -- new tip elements list.
            - eltsCrack (Vector{Int}):      -- new crack elements list.
            - eltsRibbon (Vector{Int}):     -- new ribbon elements list.
            - zeroVrtx (Vector{Int}):       -- list specifying the zero vertex of the tip cells. (can have value from 0 to\
                                            3, where 0 signifies bottom left, 1 signifying bottom right, 2 signifying top\
                                            right and 3 signifying top left vertex).
            - CellStatusNew (Vector{Int}):  -- specifies which region each element currently belongs to.
            - newEltChannel (Vector{Int}):  -- list of newly filled elements that become channel elements.
    """
    function UpdateLists(EltsChannel::Vector{Int}, EltsTipNew::Vector{Int}, FillFrac::Vector{Float64}, 
                        levelSet::Vector{Float64}, mesh)
        
        logger = Logging.current_logger()
        # new tip elements contain only the partially filled elements
        # Using a tolerance slightly less than 1.0 as in the Python code
        partially_filled_mask = FillFrac .<= 0.9999 
        eltsTip = EltsTipNew[partially_filled_mask]

        # Tip elements flag to avoid search on each iteration
        inTip = falses(mesh.NumberOfElts)
        inTip[eltsTip] .= true
        i = 1

        # todo: the while below is probably inserting a bug - found it with poor resolution and volume control
        # Note: Be careful with modifying eltsTip while iterating. 
        # Using a while loop that adjusts the index is one way, but a for loop with a separate collection might be safer.
        while i <= length(eltsTip)  # to remove a special case encountered in sharp edges and rectangular cells
            tip_element = eltsTip[i]
            neighbors = @view mesh.NeiElements[tip_element, :] # Assuming order: left, right, bottom, top (1,2,3,4)
            # This is a direct translation of the indexing logic.
            top_left_idx = neighbors[4] - 1 
            
            # Check bounds for top_left_idx before accessing inTip
            if (1 <= neighbors[1] <= mesh.NumberOfElts) && 
            (1 <= neighbors[4] <= mesh.NumberOfElts) && 
            (1 <= top_left_idx <= mesh.NumberOfElts) &&
            inTip[neighbors[1]] && inTip[neighbors[4]] && inTip[top_left_idx]
                
                conjoined = [neighbors[1], neighbors[4], top_left_idx, tip_element]
                # Find index of minimum distance. Assuming mesh.distCenter is a Vector or has linear indexing for elements.
                # If mesh.distCenter is element-wise, this should work. Otherwise, adjust indexing.
                mindist_local_index = argmin(@view mesh.distCenter[conjoined]) # 1-based index within conjoined
                mindist_global_index = conjoined[mindist_local_index]
                
                inTip[mindist_global_index] = false
                # Remove from eltsTip. Find the index in eltsTip.
                remove_index = findfirst(==(mindist_global_index), eltsTip)
                if remove_index !== nothing
                    deleteat!(eltsTip, remove_index)
                    # If we removed an element before or at the current index `i`, we need to adjust `i`
                    if remove_index <= i
                        i -= 1
                    end
                end
            end
            i += 1
        end

        # new channel elements (elements that were tip but are now fully filled)
        newEltChannel = setdiff(EltsTipNew, eltsTip)

        eltsChannel = vcat(EltsChannel, newEltChannel)
        eltsCrack = vcat(eltsChannel, eltsTip)
        eltsRibbon = Int[] # Initialize as empty vector
        zeroVrtx = zeros(Int, length(eltsTip))  # Vertex from where the perpendicular is drawn

        # All the inner cells neighboring tip cells are added to ribbon cells
        for i in 1:length(eltsTip)
            tip_element = eltsTip[i]
            neighbors = @view mesh.NeiElements[tip_element, :] # left, right, bottom, top

            drctx = 0
            drcty = 0

            if levelSet[neighbors[1]] <= levelSet[neighbors[2]] # left <= right
                push!(eltsRibbon, neighbors[1]) # Add left neighbor
                drctx = -1
            else
                push!(eltsRibbon, neighbors[2]) # Add right neighbor
                drctx = 1
            end

            if levelSet[neighbors[3]] <= levelSet[neighbors[4]] # bottom <= top
                push!(eltsRibbon, neighbors[3]) # Add bottom neighbor
                drcty = -1
            else
                push!(eltsRibbon, neighbors[4]) # Add top neighbor
                drcty = 1
            end

            # Assigning zero vertex according to the direction of propagation
            # 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
            if drctx < 0 && drcty < 0
                zeroVrtx[i] = 0
            elseif drctx > 0 && drcty < 0
                zeroVrtx[i] = 1
            elseif drctx < 0 && drcty > 0
                zeroVrtx[i] = 3
            elseif drctx > 0 && drcty > 0
                zeroVrtx[i] = 2
            end
        end

        # Remove tip elements from ribbon elements to ensure they are distinct
        eltsRibbon = setdiff(eltsRibbon, eltsTip)
        
        # Debug check
        if any(levelSet[eltsRibbon] .> 0)
            @debug logger "UpdateLists" "Probably there is a bug here...."
        end

        CellStatusNew = zeros(Int, mesh.NumberOfElts)
        CellStatusNew[eltsChannel] .= 1
        CellStatusNew[eltsTip] .= 2
        CellStatusNew[eltsRibbon] .= 3

        return eltsChannel, eltsTip, eltsCrack, eltsRibbon, zeroVrtx, CellStatusNew, newEltChannel
    end

        # -----------------------------------------------------------------------------------------------------------------------

    """
        Eikonal_Res(Tij, args...)

        Quadratic Eikonal equation residual to be used by numerical root finder.

        Arguments:
        - `Tij::Float64`: The value of the level set function at the current cell.
        - `args::Tuple`: Tuple of parameters `(Tleft, Tright, Tbottom, Ttop, Fij, dx, dy)`.
                        `Tleft`, `Tright`, `Tbottom`, `Ttop` are the level set values of the four neighbors.
                        `Fij` is the right-hand side of the Eikonal equation (usually 1 for distance functions).
                        `dx`, `dy` are the mesh cell dimensions.

        Returns:
        - `Float64`: The residual of the Eikonal equation.
    """
    function Eikonal_Res(Tij::Float64, args...)

        Tleft, Tright, Tbottom, Ttop, Fij, dx, dy = args

        term1 = nanmax([(Tij - Tleft) / dx, 0.0])^2
        term2 = nanmin([(Tright - Tij) / dx, 0.0])^2
        term3 = nanmax([(Tij - Tbottom) / dy, 0.0])^2
        term4 = nanmin([(Ttop - Tij) / dy, 0.0])^2
        
        residual = term1 + term2 + term3 + term4 - Fij^2
        
        return residual
    end

    function nanmin(arr)
        valid_values = filter(!isnan, arr)
        return isempty(valid_values) ? NaN : minimum(valid_values)
    end

    function nanmax(arr)
        valid_values = filter(!isnan, arr)
        return isempty(valid_values) ? NaN : maximum(valid_values)
    end

end # module LevelSet


# -----------------------------------------------------------------------------------------------------------------------