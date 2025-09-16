# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""

module FractureInitialization

    include("level_set.jl")
    include("volume_integral.jl")
    include("symmetry.jl")
    include("continuous_front_reconstruction.jl")

    using .LevelSet: SolveFMM, reconstruct_front, UpdateLists
    using .VolumeIntegral: Integral_over_cell
    using .Symmetry: self_influence
    using .ContinuousFrontReconstruction: reconstruct_front_continuous, UpdateListsFromContinuousFrontRec

    export get_eliptical_survey_cells, get_radial_survey_cells, get_rectangular_survey_cells, generate_footprint, get_width_pressure, g,
        Distance_ellipse, Distance_square, InitializationParameters, check_consistency, Geometry, get_length_dimension, set_length_dimension, get_center, get_survey_points
        


    """
        This function would provide the ribbon of cells on the inside of the perimeter of an ellipse with the given
        lengths of the major and minor axes. A list of all the cells inside the fracture is also provided.

        Arguments:
            mesh (CartesianMesh object): A CartesianMesh class object describing the grid.
            a (Float64): The length of the major axis of the provided ellipse.
            b (Float64): The length of the minor axis of the provided ellipse.
            center (Vector): The coordinates [x, y] of the center point.

        Returns:
            - surv_cells (Vector{Int}) -- the list of cells on the inside of the perimeter of the given ellipse.
            - surv_dist (Vector{Float64}) -- the list of corresponding distances of the surv_cells to the fracture tip.
            - inner_cells (Vector{Int}) -- the list of cells inside the given ellipse.
    """

    function get_eliptical_survey_cells(mesh, a, b, center=nothing)

        
        if center === nothing
            center = [0.0, 0.0]
        end

        # distances of the cell vertices
        dist_vertx = ((mesh.VertexCoor[:, 1] .- center[1]) / a).^2 + ((mesh.VertexCoor[:, 2] .- center[2]) / b).^2 .- 1.0
        
        # vertices that are inside the ellipse
        vertices = dist_vertx[mesh.Connectivity] .< 0

        # cells with all four vertices inside
        log_and = (vertices[:, 1] .& vertices[:, 2]) .& (vertices[:, 3] .& vertices[:, 4])
        inner_cells = findall(log_and)
        
        if length(inner_cells) == 0
            throw(SystemError("The given ellipse is too small compared to mesh!"))
        end

        dist = zeros(Float64, length(inner_cells))
        # get minimum distance from center of the inner cells
        for i in 1:length(inner_cells)
            dist[i] = Distance_ellipse(a,
                                    b,
                                    mesh.CenterCoor[inner_cells[i], 1] - center[1],
                                    mesh.CenterCoor[inner_cells[i], 2] - center[2])
        end

        cell_len = sqrt(mesh.hx * mesh.hx + mesh.hy * mesh.hy)  # one cell diagonal length
        ribbon = findall(dist .<= 2 * cell_len)
        surv_cells = inner_cells[ribbon]
        surv_dist = dist[ribbon]

        return surv_cells, surv_dist, inner_cells
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function would provide the ribbon of cells and their distances to the front on the inside (or outside) of the
        perimeter of a circle with the given radius. A list of all the cells inside the fracture is also provided.

        Arguments:
            mesh (CartesianMesh object): A CartesianMesh class object describing the grid.
            r (Float64): The radius of the circle.
            center (Vector): The coordinates [x, y] of the center point.
            external_crack (Bool): True if you would like the fracture to be an external crack.

        Returns:
            - surv_cells (Vector{Int}) -- the list of cells on the inside of the perimeter of the given circle.
                                            In case of external_crack=true the list of cells outside of the perimeter.
            - surv_dist (Vector{Float64}) -- the list of corresponding distances of the surv_cells to the fracture tip.
            - inner_cells (Vector{Int}) -- the list of cells inside the given circle.
    """

    function get_radial_survey_cells(mesh, r, center=nothing, external_crack=false):

        if center === nothing
            center = [0.0, 0.0]
        end

        # distances of the cell vertices
        dist_vertx = sqrt.((mesh.VertexCoor[:, 1] .- center[1]).^2 + (mesh.VertexCoor[:, 2] .- center[2]).^2) / r .- 1.0

        # vertices that are inside the ellipse
        vertices = dist_vertx[mesh.Connectivity] .<= 0

        # cells with all four vertices inside
        log_and = (vertices[:, 1] .& vertices[:, 2]) .& (vertices[:, 3] .& vertices[:, 4])

        inner_cells = findall(log_and)
        dist = r .- sqrt.((mesh.CenterCoor[inner_cells, 1] .- center[1]).^2 + (mesh.CenterCoor[inner_cells, 2] .- center[2]).^2)

        if length(inner_cells) == 0
            throw(SystemError("The given radius is too small!"))
        end

        cell_len = 2 * sqrt(mesh.hx * mesh.hx + mesh.hy * mesh.hy)  # one cell diagonal length
        ribbon = findall(dist .<= cell_len)
        surv_cells = inner_cells[ribbon]
        surv_dist = dist[ribbon]

        if external_crack
            # vertices that are outside the ellipse
            vertices_out = dist_vertx[mesh.Connectivity] .>= 0

            # cells with all four vertices outside
            log_and_out = (vertices_out[:, 1] .& vertices_out[:, 2]) .& (vertices_out[:, 3] .& vertices_out[:, 4])

            outer_cells = findall(log_and_out)
            dist_outer = -r .+ sqrt.((mesh.CenterCoor[outer_cells, 1] .- center[1]).^2 + (mesh.CenterCoor[outer_cells, 2] .- center[2]).^2)

            # mesh.domainLimits[ bottom top left right ]
            if mesh.domainLimits[1] > center[2] - r  # bottom
                throw(SystemError("The given circle lies outside of the mesh"))
            end
            if mesh.domainLimits[2] < center[2] + r  # top
                throw(SystemError("The given circle lies outside of the mesh"))
            end
            if mesh.domainLimits[3] > center[1] - r  # left
                throw(SystemError("The given circle lies outside of the mesh"))
            end
            if mesh.domainLimits[4] < center[1] + r  # right
                throw(SystemError("The given circle lies outside of the mesh"))
            end

            cell_len = 2 * sqrt(mesh.hx * mesh.hx + mesh.hy * mesh.hy)  # one cell diagonal length
            ribbon = findall(dist_outer .<= cell_len)
            surv_cells = outer_cells[ribbon]
            surv_dist = dist_outer[ribbon]

        end
        
        return surv_cells, surv_dist, inner_cells
    end

    # ----------------------------------------------------------------------------------------------------------------------

    """
        This function would provide the ribbon of cells on the inside of the perimeter of a rectangle with the given
        lengths and height. A list of all the cells inside the fracture is also provided.

        Arguments:
            mesh (CartesianMesh object): A CartesianMesh class object describing the grid.
            length (Float64): The half length of the rectangle.
            height (Float64): The height of the rectangle.
            center (Vector): The coordinates [x, y] of the center point.

        Returns:
            - surv_cells (Vector{Int}) -- the list of cells on the inside of the perimeter of the given rectangle.
            - surv_dist (Vector{Float64}) -- the list of corresponding distances of the surv_cells to the fracture tip.
            - inner_cells (Vector{Int}) -- the list of cells inside the given ellipse.
    """

    function get_rectangular_survey_cells(mesh, length, height, center=nothing)

        if center === nothing
            center = [0.0, 0.0]
        end

        # Find cells inside the rectangle
        x_condition = abs.(mesh.CenterCoor[:, 1] .- center[1]) .< length
        y_condition = abs.(mesh.CenterCoor[:, 2] .- center[2]) .< height / 2
        inner_cells = intersect(findall(x_condition), findall(y_condition))
        
        if length(inner_cells) == 0
            throw(SystemError("The given rectangular region is too small compared to the mesh!"))
        end

        max_x = maximum(mesh.CenterCoor[inner_cells, 1])
        min_x = minimum(mesh.CenterCoor[inner_cells, 1])
        max_y = maximum(mesh.CenterCoor[inner_cells, 2])
        min_y = minimum(mesh.CenterCoor[inner_cells, 2])
        
        epsilon = eps() * 100  # equivalent to 100 * sys.float_info.epsilon
        ribbon_max_x = findall(abs.(mesh.CenterCoor[inner_cells, 1] .- max_x) .< epsilon)
        ribbon_min_x = findall(abs.(mesh.CenterCoor[inner_cells, 1] .- min_x) .< epsilon)
        ribbon_max_y = findall(abs.(mesh.CenterCoor[inner_cells, 2] .- max_y) .< epsilon)
        ribbon_min_y = findall(abs.(mesh.CenterCoor[inner_cells, 2] .- min_y) .< epsilon)

        surv_cells = vcat(inner_cells[ribbon_max_x], inner_cells[ribbon_max_y])
        surv_cells = vcat(surv_cells, inner_cells[ribbon_min_x])
        surv_cells = vcat(surv_cells, inner_cells[ribbon_min_y])
        surv_cells = unique(surv_cells)

        surv_dist = zeros(Float64, length(surv_cells))

        for i in 1:length(surv_cells)
            surv_dist[i] = minimum([length - abs(mesh.CenterCoor[surv_cells[i], 1] - center[1]),
                                height / 2 - abs(mesh.CenterCoor[surv_cells[i], 2] - center[2])])
        end

        return surv_cells, surv_dist, inner_cells
    end

    # ----------------------------------------------------------------------------------------------------------------------

    """
        This function takes the survey cells and their distances from the front and generate the footprint of a fracture
        using the fast marching method.

        Arguments:
            mesh (CartesianMesh): A CartesianMesh class object describing the grid.
            surv_cells (Vector{Int}): List of survey cells from which the distances from front are provided.
            inner_region (Vector{Int}): List of cells enclosed by the survey cells.
            dist_surv_cells (Vector{Float64}): Distances of the provided survey cells from the front.
            projMethod (String): Projection method.

        Returns:
            - EltChannel (Vector{Int}) -- list of cells in the channel region.
            - EltTip (Vector{Int}) -- list of cells in the Tip region.
            - EltCrack (Vector{Int}) -- list of cells in the crack region.
            - EltRibbon (Vector{Int}) -- list of cells in the Ribbon region.
            - ZeroVertex (Vector{Float64}) -- Vertex from which the perpendicular is drawn on the front.
            - CellStatus (Vector{Int}) -- specifies which region each element currently belongs to.
            - l (Vector{Float64}) -- length of perpendicular on the fracture front.
            - alpha (Vector{Float64}) -- angle prescribed by perpendicular on the fracture front.
            - FillF (Vector{Float64}) -- filling fraction of each tip cell.
            - sgndDist (Vector{Float64}) -- signed minimum distance from fracture front.
            - Ffront (Matrix{Float64}) -- front coordinates.
            - number_of_fronts (Int) -- number of fronts.
            - fronts_dictionary (Dict) -- fronts dictionary.
    """

    function generate_footprint(mesh, surv_cells, inner_region, dist_surv_cells, projMethod)

        sgndDist = fill(1e50, mesh.NumberOfElts)
        sgndDist[surv_cells] = -dist_surv_cells

        # rest of the cells outside the survey cell ring
        EltRest = setdiff(1:mesh.NumberOfElts, inner_region)

        # fast marching to get level set
        SolveFMM(sgndDist,
                surv_cells,
                inner_region,
                mesh,
                EltRest,
                inner_region)

        band = 1:mesh.NumberOfElts
        # construct the front
        if projMethod == "LS_continousfront"
            correct_size_of_pstv_region = [false, false, false]
            recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge = false
            
            while !correct_size_of_pstv_region[1]
                # Note: You'll need to implement reconstruct_front_continuous function
                result = reconstruct_front_continuous(sgndDist,
                                                    band,
                                                    surv_cells,
                                                    inner_region,
                                                    mesh,
                                                    recomp_LS_4fullyTravCellsAfterCoalescence_OR_RemovingPtsOnCommonEdge,
                                                    oldfront=nothing)
                
                EltTip_tmp = result[1]
                listofTIPcellsONLY = result[2]
                l_tmp = result[3]
                alpha_tmp = result[4]
                CellStatus = result[5]
                newRibbon = result[6]
                ZeroVertex_with_fully_traversed = result[7]
                ZeroVertex = result[8]
                correct_size_of_pstv_region = result[9]
                sgndDist_k_temp = result[10]
                Ffront = result[11]
                number_of_fronts = result[12]
                fronts_dictionary = result[13]
                
                if correct_size_of_pstv_region[2] || correct_size_of_pstv_region[3]
                    throw(ArgumentError("The mesh is too small for the proposed initiation"))
                end

                if !correct_size_of_pstv_region[1]
                    throw(SystemExit("FRONT RECONSTRUCTION ERROR: it is not possible to initialize the front with the given distances to the front"))
                end
            end
            
            sgndDist = sgndDist_k_temp
            # del correct_size_of_pstv_region - not needed in Julia

        else
            # Note: You'll need to implement reconstruct_front function
            result = reconstruct_front(sgndDist, band, inner_region, mesh)
            EltTip_tmp = result[1]
            l_tmp = result[2]
            alpha_tmp = result[3]
            CSt = result[4]
            Ffront = "It will be computed later by the method process_fracture_front()"
            number_of_fronts = nothing
        end

        # get the filling fraction of the tip cells
        # Note: You'll need to implement Integral_over_cell function
        FillFrac_tmp = Integral_over_cell(EltTip_tmp,
                                        alpha_tmp,
                                        l_tmp,
                                        mesh,
                                        "A") / mesh.EltArea

        # generate cell lists
        if projMethod == "LS_continousfront"
            # Note: You'll need to implement UpdateListsFromContinuousFrontRec function
            result = UpdateListsFromContinuousFrontRec(newRibbon,
                                                    sgndDist,
                                                    inner_region,
                                                    EltTip_tmp,
                                                    listofTIPcellsONLY,
                                                    mesh)
            EltChannel = result[1]
            EltTip = result[2]
            EltCrack = result[3]
            EltRibbon = result[4]
            CellStatus = result[5]
            fully_traversed = result[6]
        else
            # Note: You'll need to implement UpdateLists function
            result = UpdateLists(inner_region,
                            EltTip_tmp,
                            FillFrac_tmp,
                            sgndDist,
                            mesh)
            EltChannel = result[1]
            EltTip = result[2]
            EltCrack = result[3]
            EltRibbon = result[4]
            ZeroVertex = result[5]
            CellStatus = result[6]
            fully_traversed = result[7]
            fronts_dictionary = nothing
            # todo: implement volume control with two different pressures in the fractures in the case of proj_method = 'ILSA_orig'
        end

        # removing fully traversed cells from the tip cells and other lists
        newTip_indices = findall(in.(EltTip_tmp, Ref(EltTip)))
        l = l_tmp[newTip_indices]
        alpha = alpha_tmp[newTip_indices]
        FillFrac = FillFrac_tmp[newTip_indices]

        if length(EltChannel) <= length(EltRibbon)
            throw(SystemExit("No channel elements. The initial radius is probably too small!"))
        end

        return EltChannel, EltTip, EltCrack, EltRibbon, ZeroVertex, CellStatus, l, alpha, FillFrac, sgndDist, Ffront, number_of_fronts, fronts_dictionary
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function calculates the width and pressure depending on the provided data. If only volume is provided, the
        width is calculated as a static fracture with the given footprint. Else, the pressure or width are calculated
        according to the given elasticity matrix.

        Arguments:
            mesh (CartesianMesh): A CartesianMesh class object describing the grid.
            EltCrack (Vector{Int}): List of cells in the crack region.
            EltTip (Vector{Int}): List of cells in the Tip region.
            FillFrac (Vector{Float64}): Filling fraction of each tip cell. Used for correction.
            C (Matrix{Float64}): The elasticity matrix.
            w (Vector{Float64}): The provided width for each cell, can be nothing if not available.
            p (Vector{Float64}): The provided pressure for each cell, can be nothing if not available.
            volume (Float64): The volume of the fracture, can be nothing if not available.
            symmetric (Bool): If true, the fracture will be considered strictly symmetric.
            useBlockToeplizCompression (Bool): Whether to use block Toeplitz compression.
            Eprime (Float64): The plain strain elastic modulus.

        Returns:
            - w_calculated (Vector{Float64}) -- the calculated width.
            - p_calculated (Vector{Float64}) -- the calculated pressure.
    """

    function get_width_pressure(mesh, EltCrack, EltTip, FillFrac, C, w=nothing, p=nothing, volume=nothing, symmetric=false, useBlockToeplizCompression=false, Eprime=nothing)

        if w === nothing && p === nothing && volume === nothing
            throw(ArgumentError("At least one of the three variables w, p and volume has to be provided."))
        end

        if p === nothing
            p_calculated = zeros(Float64, mesh.NumberOfElts)
        elseif !isa(p, Vector)
            p_calculated = zeros(Float64, mesh.NumberOfElts)
            p_calculated[EltCrack] = fill(Float64(p), length(EltCrack))
        else
            p_calculated = p
        end

        if w === nothing
            w_calculated = zeros(Float64, mesh.NumberOfElts)
        elseif !isa(w, Vector)
            w_calculated = zeros(Float64, mesh.NumberOfElts)
            w_calculated[EltCrack] = fill(Float64(w), length(EltCrack))
        else
            w_calculated = w
        end

        if w !== nothing && p !== nothing
            return w_calculated, p_calculated
        end

        if symmetric && !useBlockToeplizCompression
            CrackElts_sym = mesh.corresponding[EltCrack]
            CrackElts_sym = unique(CrackElts_sym)

            EltTip_sym = mesh.corresponding[EltTip]
            EltTip_sym = unique(EltTip_sym)

            FillF_mesh = zeros(Float64, mesh.NumberOfElts)
            FillF_mesh[EltTip] = FillFrac
            FillF_sym = FillF_mesh[mesh.activeSymtrc[EltTip_sym]]
            
            # Note: You'll need to implement self_influence function
            self_infl = self_influence(mesh, Eprime)

            C_EltTip = copy(C[EltTip_sym, EltTip_sym])  # keeping the tip element entries to restore current tip correction

            # filling fraction correction for element in the tip region
            for e in 1:length(EltTip_sym)
                r = FillF_sym[e] - 0.25
                if r < 0.1
                    r = 0.1
                end
                ac = (1 - r) / r
                C[EltTip_sym[e], EltTip_sym[e]] += ac * π / 4.0 * self_infl
            end

            if w === nothing && p !== nothing
                w_sym_EltCrack = C[CrackElts_sym, CrackElts_sym] \ p_calculated[mesh.activeSymtrc[CrackElts_sym]]
                for i in 1:length(w_sym_EltCrack)
                    w_calculated[mesh.symmetricElts[mesh.activeSymtrc[CrackElts_sym[i]]]] = w_sym_EltCrack[i]
                end
            end

            if w !== nothing && p === nothing
                p_sym_EltCrack = C[CrackElts_sym, CrackElts_sym] * w[mesh.activeSymtrc[CrackElts_sym]]
                for i in 1:length(p_sym_EltCrack)
                    p_calculated[mesh.symmetricElts[mesh.activeSymtrc[CrackElts_sym[i]]]] = p_sym_EltCrack[i]
                end
            end

            # calculate the width and pressure by considering fracture as a static fracture.
            if w === nothing && p === nothing
                C_Crack = C[CrackElts_sym, CrackElts_sym]

                A = hcat(C_Crack, -ones(Float64, length(EltCrack)))
                weights = mesh.volWeights[CrackElts_sym]
                weights = vcat(weights, 0.0)
                A = vcat(A, weights')

                b = zeros(Float64, length(EltCrack) + 1)
                b[end] = volume / mesh.EltArea

                sol = A \ b

                w_calculated[EltCrack] = sol[1:length(EltCrack)]
                p_calculated[EltCrack] = sol[end]
            end

            # recover original C (without filling fraction correction)
            C[EltTip_sym, EltTip_sym] = C_EltTip

        elseif useBlockToeplizCompression
            C_Crack = C[EltCrack, EltCrack]
            EltTip_positions = findall(in.(EltCrack, Ref(EltTip)))

            # filling fraction correction for element in the tip region
            r = FillFrac - 0.25
            indx = findall(r .< 0.1)
            r[indx] .= 0.1
            ac = (1 .- r) ./ r
            C_Crack[EltTip_positions, EltTip_positions] = C_Crack[EltTip_positions, EltTip_positions] .* (1.0 .+ ac * π / 4.0)

            if w === nothing && p !== nothing
                w_calculated[EltCrack] = C_Crack \ p_calculated[EltCrack]
            end

            if w !== nothing && p === nothing
                p_calculated[EltCrack] = C_Crack * w[EltCrack]
            end

            # calculate the width and pressure by considering fracture as a static fracture.
            if w === nothing && p === nothing
                A = hcat(C_Crack, -ones(Float64, length(EltCrack)))
                A = vcat(A, ones(Float64, 1, length(EltCrack) + 1))
                A[end, end] = 0.0

                b = zeros(Float64, length(EltCrack) + 1)
                b[end] = volume / mesh.EltArea

                sol = A \ b

                w_calculated[EltCrack] = sol[1:length(EltCrack)]
                p_calculated[EltCrack] = sol[end]
            end

        else
            C_EltTip = copy(C[EltTip, EltTip])  # keeping the tip element entries to restore current tip correction

            # filling fraction correction for element in the tip region
            for e in 1:length(EltTip)
                r = FillFrac[e] - 0.25
                if r < 0.1
                    r = 0.1
                end
                ac = (1 - r) / r
                C[EltTip[e], EltTip[e]] = C[EltTip[e], EltTip[e]] * (1.0 + ac * π / 4.0)
            end

            if w === nothing && p !== nothing
                w_calculated[EltCrack] = C[EltCrack, EltCrack] \ p_calculated[EltCrack]
            end

            if w !== nothing && p === nothing
                p_calculated[EltCrack] = C[EltCrack, EltCrack] * w[EltCrack]
            end

            # calculate the width and pressure by considering fracture as a static fracture.
            if w === nothing && p === nothing
                C_Crack = C[EltCrack, EltCrack]

                A = hcat(C_Crack, -ones(Float64, length(EltCrack)))
                A = vcat(A, ones(Float64, 1, length(EltCrack) + 1))
                A[end, end] = 0.0

                b = zeros(Float64, length(EltCrack) + 1)
                b[end] = volume / mesh.EltArea

                sol = A \ b

                w_calculated[EltCrack] = sol[1:length(EltCrack)]
                p_calculated[EltCrack] = sol[end]
            end

            C[EltTip, EltTip] = C_EltTip
        end

        return w_calculated, p_calculated
    end
    #-----------------------------------------------------------------------------------------------------------------------

    function g(a, b, x0, y0, la)
        return (a * x0 / (a^2 + la))^2 + (b * y0 / (b^2 + la))^2 - 1
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        This function calculates the smallest distance of a point from the given ellipse.

        Arguments:
            a (Float64): The length of the major axis of the ellipse.
            b (Float64): The length of the minor axis of the ellipse.
            x0 (Float64): The x coordinate of the point from which the distance is to be found.
            y0 (Float64): The y coordinate of the point from which the distance is to be found.

        Returns:
            D (Float64): The shortest distance of the point from the ellipse.
    """

    function Distance_ellipse(a, b, x0, y0)


        x0 = abs(x0)
        y0 = abs(y0)
        
        if (x0 < 1e-12 && y0 < 1e-12)
            D = b

        elseif (x0 < 1e-12 && y0 > 0)
            D = abs(y0 - b)

        elseif (y0 < 1e-12 && x0 > 0)
            if (x0 < (a^2 - b^2) / a)
                xellipse = a^2 * x0 / (a^2 - b^2)
                yellipse = b * sqrt(1 - (xellipse / a)^2)
                D = sqrt((x0 - xellipse)^2 + yellipse^2)
            else
                D = abs(x0 - a)
            end

        else
            lamin = -b^2 + b * y0
            lamax = -b^2 + sqrt((a * x0)^2 + (b * y0)^2)

            while (abs(g(a, b, x0, y0, lamin)) > 1e-6 || abs(g(a, b, x0, y0, lamax)) > 1e-6)
                lanew = (lamin + lamax) / 2

                if (g(a, b, x0, y0, lanew) < 0)
                    lamax = lanew
                else
                    lamin = lanew
                end
            end

            la = (lamin + lamax) / 2
            xellipse = a^2 * x0 / (a^2 + la)
            yellipse = b^2 * y0 / (b^2 + la)
            D = sqrt((x0 - xellipse)^2 + (y0 - yellipse)^2)
        end

        return D
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        The shortest distance of a point from a square
    """ 

    function Distance_square(lx, ly, x, y)

        return abs(minimum([lx - x, lx + x, ly - y, ly + y]))
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        This class stores the initialization parameters.

        Arguments:
            geometry (Geometry): Geometry class object describing the geometry of the fracture.
            regime (String): The propagation regime of the fracture. Possible options are the following:
                            - "M"     -- radial fracture in viscosity dominated regime.
                            - "Mt"    -- radial fracture in viscosity dominated regime with leak-off.
                            - "K"     -- radial fracture in toughness dominated regime.
                            - "Kt"    -- radial fracture in toughness dominated regime with leak-off.
                            - "PKN"   -- PKN fracture.
                            - "E_K"   -- elliptical fracture propagating in toughness dominated regime.
                            - "E_E"   -- the elliptical solution with transverse isotropic material properties.
                            - "MDR"   -- viscosity dominated solution for turbulent flow.
            time (Float64): The time since the start of injection.
            width (Vector{Float64}): The initial width of the fracture.
            net_pressure (Union{Float64, Vector{Float64}}): The initial net pressure of the fracture.
            fracture_volume (Float64): Total initial volume of the fracture.
            tip_velocity (Union{Float64, Vector{Float64}}): The velocity of the tip.
            elasticity_matrix (Matrix{Float64}): The BEM elasticity matrix.

        Returns:
            InitializationParameters: An instance of the InitializationParameters class.
    """
    mutable struct InitializationParameters
        geometry
        regime::String
        time::Union{Float64, Nothing}
        width::Union{Vector{Float64}, Nothing}
        netPressure::Union{Union{Float64, Vector{Float64}}, Nothing}
        fractureVolume::Union{Float64, Nothing}
        tipVelocity::Union{Union{Float64, Vector{Float64}}, Nothing}
        C::Union{Matrix{Float64}, Nothing}
        
        function InitializationParameters(; geometry=nothing, regime="M", time=nothing, width=nothing, 
                                    net_pressure=nothing, fracture_volume=nothing, tip_velocity=nothing, 
                                    elasticity_matrix=nothing)
            self = new(geometry, regime, time, width, net_pressure, fracture_volume, tip_velocity, elasticity_matrix)
            check_consistency(self)
            return self
        end
    end


    """
        This function checks if the given parameters are consistent with each other.
    """
    function check_consistency(self::InitializationParameters)

        compatible_regimes = Dict(
            "radial" => ["M", "Mp", "Mt", "K", "Kt", "MDR", "static"],
            "height contained" => ["PKN", "KGD_K", "static"],
            "elliptical" => ["E_E", "E_K", "static"],
            "level set" => ["static"]
        )

        try
            if !(self.regime in compatible_regimes[self.geometry.shape])
                err_string = "Initialization is not supported for the given regime and geometrical shape.\nBelow is " *
                            "the list of compatible regimes and shapes (see documentation for description of " *
                            "the regimes):\n\n"
                for (keys, values) in compatible_regimes
                    err_string = err_string * string(keys) * ":\t" * string(values) * "\n"
                end
                throw(ArgumentError(err_string))
            end
        catch e
            if isa(e, KeyError)
                err_string = "The given geometrical shape is not supported!\nSee the list below for supported shapes:\n"
                for (keys, values) in compatible_regimes
                    err_string = err_string * string(keys) * "\n"
                end
                throw(ArgumentError(err_string))
            else
                rethrow(e)
            end
        end

        errors_analytical = Dict(
            "radial" => "Either time or radius is to be provided for radial fractures!",
            "height containedPKN" => "Either time or length is to be provided for PKN type fractures. The height of the " *
                                "fracture is required in both cases!",
            "height containedKGD_K" => "Either time or length is to be provided for toughness dominated KGD type " *
                                    "fractures. The height of the fracture is required in both cases!",
            "ellipticalE_K" => "Either time or length of minor axis is required to initialize the elliptical " *
                            "fracture in toughness dominated regime!",
            "ellipticalE_E" => "Either time or minor axis length along with the major to minor axes length ratio (gamma) " * 
                            "is to be provided to initialize in transverse isotropic material!"
        )

        errors_static = Dict(
            "radial" => "Radius is to be provided for static radial fractures!",
            "height contained" => "Length and height are required to initialize height contained fractures!",
            "elliptical" => "The length of minor axis and the aspect ratio (Geometry.gamma) is required to initialize the" *
                        " static elliptical fracture!",
            "level set" => "To initialize according to a level set, the survey cells (Geometry.surveyCells) and their " *
                        "distances (Geometry.tipDistances) along with \n the cells enclosed by the survey cells" *
                        " (geometry.innerCells) are required!"
        )

        error = false
        # checks for analytical solutions
        if self.regime != "static"
            if self.time === nothing
                if self.geometry.shape == "radial" && self.geometry.radius === nothing
                    throw(ArgumentError(errors_analytical["radial"]))
                end
                if self.geometry.shape == "height contained"
                    if self.geometry.fractureLength === nothing || self.geometry.fractureHeight === nothing
                        error = true
                    end
                end
                if self.geometry.shape == "elliptical"
                    if self.regime == "E_K" && self.geometry.minorAxis === nothing
                        error = true
                    end
                    if self.regime == "E_E"
                        if self.geometry.minorAxis === nothing || self.geometry.gamma === nothing
                            error = true
                        end
                    end
                end
            else
                if self.geometry.shape == "height contained"
                    if self.geometry.fractureHeight === nothing
                        error = true
                    end
                end
                if self.geometry.shape == "elliptical"
                    if self.regime == "E_E" && self.geometry.gamma === nothing
                        error = true
                    end
                end
            end

            if error
                throw(ArgumentError(errors_analytical[self.geometry.shape * self.regime]))
            end

        # checks for static fracture
        else
            if self.geometry.shape == "radial" && self.geometry.radius === nothing
                error = true
            elseif self.geometry.shape == "height contained"
                if self.geometry.fractureLength === nothing || self.geometry.fractureHeight === nothing
                    error = true
                end
            elseif self.geometry.shape == "elliptical"
                if self.geometry.minorAxis === nothing || self.geometry.gamma === nothing
                    error = true
                end
            elseif self.geometry.shape == "level set"
                if self.geometry.surveyCells === nothing || self.geometry.tipDistances === nothing || 
                            self.geometry.innerCells === nothing
                    error = true
                end
            end

            if error
                throw(ArgumentError(errors_static[self.geometry.shape]))
            end

            if (self.width === nothing && self.netPressure === nothing && self.fractureVolume === nothing) || self.C === nothing
                throw(ArgumentError("The following parameters are required to initialize a static fracture:\n" *
                                "\t\t -- width or net pressure or total volume of the fracture\n" *
                                "\t\t -- the elasticity matrix"))
            end
        end
        
        return nothing
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        This class defines the geometry of the fracture to be initialized.

        Arguments:
            shape (String): String giving the geometrical shape of the fracture. Possible options are:
                        - "radial"
                        - "height contained"
                        - "elliptical"
                        - "level set"
            radius (Float64): The radius of the radial fracture.
            fracture_length (Float64): The half length of the fracture.
            fracture_height (Float64): The height of the height contained fracture.
            minor_axis (Float64): Length of minor axis for elliptical fracture shape.
            gamma (Float64): Ratio of the length of the major axis to the minor axis. It should be more than one.
            survey_cells (Vector{Int}): The cells from which the distances to the fracture tip are provided.
            tip_distances (Vector{Float64}): The minimum distances of the corresponding cells provided in the survey_cells to the tip of the fracture.
            inner_cells (Vector{Int}): The cells enclosed by the cells given in the survey_cells (inclusive).
            center (Vector{Float64}): Location of the center of the geometry.

        Returns:
            Geometry: An instance of the Geometry class.
    """
    mutable struct Geometry
        shape::Union{String, Nothing}
        radius::Union{Float64, Nothing}
        fractureLength::Union{Float64, Nothing}
        fractureHeight::Union{Float64, Nothing}
        minorAxis::Union{Float64, Nothing}
        gamma::Union{Float64, Nothing}
        surveyCells::Union{Vector{Int}, Nothing}
        tipDistances::Union{Vector{Float64}, Nothing}
        innerCells::Union{Vector{Int}, Nothing}
        center::Union{Vector{Float64}, Nothing}
        
        function Geometry(; shape=nothing, radius=nothing, fracture_length=nothing, fracture_height=nothing, 
                        minor_axis=nothing, gamma=nothing, survey_cells=nothing, tip_distances=nothing, 
                        inner_cells=nothing, center=nothing)
            
            if gamma !== nothing
                if gamma < 1.0
                    throw(ArgumentError("The aspect ratio (ratio of the length of major axis to the minor axis) should be more than one"))
                end
            end
            
            return new(shape, radius, fracture_length, fracture_height, minor_axis, gamma, 
                    survey_cells, tip_distances, inner_cells, center)
        end
    end

    # ----------------------------------------------------------------------------------------------------------------------

    """
        Get the length dimension based on the geometry shape.
    """
    function get_length_dimension(self::Geometry)
        if self.shape == "radial"
            length = self.radius
        elseif self.shape == "elliptical"
            length = self.minorAxis
        elseif self.shape == "height contained"
            length = self.fractureLength
        end
        return length
    end

    # ----------------------------------------------------------------------------------------------------------------------
    """
        Set the length dimension based on the geometry shape.
    """
    function set_length_dimension(self::Geometry, length)
        if self.shape == "radial"
            self.radius = length
        elseif self.shape == "elliptical"
            self.minorAxis = length
        elseif self.shape == "height contained"
            self.fractureLength = length
        end
        return nothing
    end

    # ----------------------------------------------------------------------------------------------------------------------
    """
        Get the center of the geometry.
    """
    function get_center(self::Geometry)
        if self.center === nothing
            return [0.0, 0.0]
        else
            return self.center
        end
    end

    # ----------------------------------------------------------------------------------------------------------------------

    """
        This function provides the survey cells, corresponding distances to the front and the enclosed cells for the given geometry.
    """
    function get_survey_points(geometry::Geometry, mesh, source_coord=nothing)

        if geometry.center === nothing
            center = source_coord
        else
            center = geometry.center
        end

        if geometry.shape == "radial"
            if geometry.radius > min(mesh.Lx, mesh.Ly)
                throw(ArgumentError("The radius of the radial fracture is larger than domain!"))
            end
            surv_cells, surv_dist, inner_cells = get_radial_survey_cells(mesh,
                                                                    geometry.radius,
                                                                    center)
        elseif geometry.shape == "elliptical"
            a = geometry.minorAxis * geometry.gamma
            if geometry.minorAxis > mesh.Ly || a > mesh.Lx
                throw(ArgumentError("The axes length of the elliptical fracture is larger than domain!"))
            elseif geometry.minorAxis < 2 * mesh.hy
                throw(ArgumentError("The fracture is very small compared to the mesh cell size!"))
            end
            surv_cells, surv_dist, inner_cells = get_eliptical_survey_cells(mesh,
                                                                        a,
                                                                        geometry.minorAxis,
                                                                        center)
        elseif geometry.shape == "height contained"
            if geometry.fractureLength > mesh.Lx || geometry.fractureHeight > mesh.Ly
                throw(ArgumentError("The fracture is larger than domain!"))
            elseif geometry.fractureLength < 2 * mesh.hx || geometry.fractureHeight < 2 * mesh.hy
                throw(ArgumentError("The fracture is very small compared to the mesh cell size!"))
            end
            surv_cells, surv_dist, inner_cells = get_rectangular_survey_cells(mesh,
                                                                            geometry.fractureLength,
                                                                            geometry.fractureHeight,
                                                                            center)
        elseif geometry.shape == "level set"
            surv_cells = geometry.surveyCells
            surv_dist = geometry.tipDistances
            inner_cells = geometry.innerCells
        else
            throw(ArgumentError("The given footprint shape is not supported!"))
        end

        return surv_cells, surv_dist, inner_cells
    end

end # module FractureInitialization

