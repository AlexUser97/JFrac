# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""
module Symmetry

    using .Mesh: CartesianMesh
    export get_symetric_elements, get_active_symmetric_elements,
        corresponding_elements_in_symmetric, symmetric_elasticity_matrix_from_full,
        load_isotropic_elasticity_matrix_symmetric, self_influence

    """
        get_symetric_elements(mesh, elements)

        This function gives the four symmetric elements in each of the quadrant for the given element list.

        # Arguments
        - `mesh::CartesianMesh`: The mesh object.
        - `elements::Vector{Int}`: The list of elements.

        # Returns
        - `symetric_elts::Matrix{Int}`: The matrix containing four symmetric elements for each input element.
    """
    function get_symetric_elements(mesh::CartesianMesh, elements::Vector{Int})::Matrix{Int}
        symetric_elts = Matrix{Int}(undef, length(elements), 4)
        for i in 1:length(elements)
            i_x = elements[i] % mesh.nx
            i_y = elements[i] ÷ mesh.nx

            symetric_x = mesh.nx - i_x - 1
            symetric_y = mesh.ny - i_y - 1

            symetric_elts[i, :] = [i_y * mesh.nx + i_x,
                                symetric_y * mesh.nx + i_x,
                                i_y * mesh.nx + symetric_x,
                                symetric_y * mesh.nx + symetric_x]
        end

        return symetric_elts
    end

    # -----------------------------------------------------------------------------------------------------------------------

    """
        get_active_symmetric_elements(mesh)

        This functions gives the elements in the first quadrant, including the elements intersecting the x and y
        axes lines.

        # Arguments
        - `mesh::CartesianMesh`: The mesh object.

        # Returns
        - `all_elts::Vector{Int}`: All active symmetric elements.
        - `pos_qdrnt::Vector{Int}`: Elements in the positive quadrant.
        - `boundary_x::Vector{Int}`: Elements intersecting the x-axis.
        - `boundary_y::Vector{Int}`: Elements intersecting the y-axis.
    """
    function get_active_symmetric_elements(mesh::CartesianMesh)
        # elements in the quadrant with positive x and y coordinates
        pos_qdrnt = intersect(
            findall(mesh.CenterCoor[:, 1] .> mesh.hx / 2),
            findall(mesh.CenterCoor[:, 2] .> mesh.hy / 2)
        )

        boundary_x = intersect(
            findall(abs.(mesh.CenterCoor[:, 2]) .< 1e-12),
            findall(mesh.CenterCoor[:, 1] .> mesh.hx / 2)
        )
        
        boundary_y = intersect(
            findall(abs.(mesh.CenterCoor[:, 1]) .< 1e-12),
            findall(mesh.CenterCoor[:, 2] .> mesh.hy / 2)
        )

        all_elts = vcat(pos_qdrnt, boundary_x, boundary_y, mesh.CenterElts[1])

        return all_elts, pos_qdrnt, boundary_x, boundary_y
    end

    # ----------------------------------------------------------------------------------------------------------------------


    """
        corresponding_elements_in_symmetric(mesh)

        This function returns the corresponding elements in symmetric fracture.

        # Arguments
        - `mesh::CartesianMesh`: The mesh object.

        # Returns
        - `correspondence::Vector{Int}`: The correspondence array mapping mesh elements to symmetric elements.
    """
    function corresponding_elements_in_symmetric(mesh::CartesianMesh)::Vector{Int}
        correspondence = Vector{Int}(undef, mesh.NumberOfElts)
        all_elmnts, pos_qdrnt, boundary_x, boundary_y = get_active_symmetric_elements(mesh)

        sym_elts = get_symetric_elements(mesh, pos_qdrnt)
        for i in 1:length(pos_qdrnt)
            correspondence[sym_elts[i]] = i
        end

        sym_bound_x = get_symetric_elements(mesh, boundary_x)
        for i in 1:length(boundary_x)
            correspondence[sym_bound_x[i]] = i + length(pos_qdrnt)
        end

        sym_bound_y = get_symetric_elements(mesh, boundary_y)
        for i in 1:length(boundary_y)
            correspondence[sym_bound_y[i]] = i + length(pos_qdrnt) + length(boundary_x)
        end

        correspondence[mesh.CenterElts[1]] = length(pos_qdrnt) + length(boundary_x) + length(boundary_y)

        return correspondence
    end


    #-----------------------------------------------------------------------------------------------------------------------

    """
        symmetric_elasticity_matrix_from_full(C, mesh)

        Create a symmetric elasticity matrix from a full elasticity matrix.

        # Arguments
        - `C::Matrix{Float32}`: The full elasticity matrix.
        - `mesh::CartesianMesh`: The mesh object.

        # Returns
        - `C_sym::Matrix{Float32}`: The symmetric elasticity matrix.
    """
    function symmetric_elasticity_matrix_from_full(C::Matrix{Float32}, mesh::CartesianMesh)::Matrix{Float32}
        all_elmnts, pos_qdrnt, boundary_x, boundary_y = get_active_symmetric_elements(mesh)

        no_elements = length(pos_qdrnt) + length(boundary_x) + length(boundary_y) + 1
        C_sym = Matrix{Float32}(undef, no_elements, no_elements)

        indx_boun_x = length(pos_qdrnt)
        indx_boun_y = indx_boun_x + length(boundary_x)
        indx_cntr_elm = indx_boun_y + length(boundary_y)

        sym_elements = get_symetric_elements(mesh, pos_qdrnt)
        sym_elem_xbound = get_symetric_elements(mesh, boundary_x)
        sym_elem_ybound = get_symetric_elements(mesh, boundary_y)

        # influence on elements
        for i in 1:length(pos_qdrnt)
            C_sym[i, 1:indx_boun_x] = C[pos_qdrnt[i], sym_elements[:, 1]] + 
                                    C[pos_qdrnt[i], sym_elements[:, 2]] + 
                                    C[pos_qdrnt[i], sym_elements[:, 3]] + 
                                    C[pos_qdrnt[i], sym_elements[:, 4]]

            C_sym[i, indx_boun_x+1:indx_boun_y] = C[pos_qdrnt[i], sym_elem_xbound[:, 1]] + 
                                                C[pos_qdrnt[i], sym_elem_xbound[:, 4]]

            C_sym[i, indx_boun_y+1:indx_cntr_elm] = C[pos_qdrnt[i], sym_elem_ybound[:, 1]] + 
                                                    C[pos_qdrnt[i], sym_elem_ybound[:, 2]]
        end

        C_sym[1:indx_boun_x, end] = C[pos_qdrnt, mesh.CenterElts[1]]

        # influence on x boundary elements
        for i in 1:length(boundary_x)
            C_sym[i + indx_boun_x, 1:indx_boun_x] = C[boundary_x[i], sym_elements[:, 1]] + 
                                                    C[boundary_x[i], sym_elements[:, 2]] + 
                                                    C[boundary_x[i], sym_elements[:, 3]] + 
                                                    C[boundary_x[i], sym_elements[:, 4]]

            C_sym[i + indx_boun_x, indx_boun_x+1:indx_boun_y] = C[boundary_x[i], sym_elem_xbound[:, 1]] + 
                                                                C[boundary_x[i], sym_elem_xbound[:, 4]]

            C_sym[i + indx_boun_x, indx_boun_y+1:indx_cntr_elm] = C[boundary_x[i], sym_elem_ybound[:, 1]] + 
                                                                C[boundary_x[i], sym_elem_ybound[:, 2]]
        end

        C_sym[indx_boun_x+1:indx_boun_y, end] = C[boundary_x, mesh.CenterElts[1]]

        # influence on y boundary elements
        for i in 1:length(boundary_y)
            C_sym[i + indx_boun_y, 1:indx_boun_x] = C[boundary_y[i], sym_elements[:, 1]] + 
                                                    C[boundary_y[i], sym_elements[:, 2]] + 
                                                    C[boundary_y[i], sym_elements[:, 3]] + 
                                                    C[boundary_y[i], sym_elements[:, 4]]

            C_sym[i + indx_boun_y, indx_boun_x+1:indx_boun_y] = C[boundary_y[i], sym_elem_xbound[:, 1]] + 
                                                                C[boundary_y[i], sym_elem_xbound[:, 4]]

            C_sym[i + indx_boun_y, indx_boun_y+1:indx_cntr_elm] = C[boundary_y[i], sym_elem_ybound[:, 1]] + 
                                                                C[boundary_y[i], sym_elem_ybound[:, 2]]
        end

        C_sym[indx_boun_y+1:indx_cntr_elm, end] = C[boundary_y, mesh.CenterElts[1]]

        # influence on center element
        C_sym[end, 1:length(pos_qdrnt)] = C[mesh.CenterElts[1], sym_elements[:, 1]] + 
                                        C[mesh.CenterElts[1], sym_elements[:, 2]] + 
                                        C[mesh.CenterElts[1], sym_elements[:, 3]] + 
                                        C[mesh.CenterElts[1], sym_elements[:, 4]]

        C_sym[end, indx_boun_x+1:indx_boun_y] = C[mesh.CenterElts[1], sym_elem_xbound[:, 1]] + 
                                                C[mesh.CenterElts[1], sym_elem_xbound[:, 4]]

        C_sym[end, indx_boun_y+1:indx_cntr_elm] = C[mesh.CenterElts[1], sym_elem_ybound[:, 1]] + 
                                                C[mesh.CenterElts[1], sym_elem_ybound[:, 2]]

        C_sym[end, end] = C[mesh.CenterElts[1], mesh.CenterElts[1]]

        return C_sym
    end


    """
        load_isotropic_elasticity_matrix_symmetric(mesh, Ep)

        Evaluate the elasticity matrix for the whole mesh.

        # Arguments
        - `mesh::CartesianMesh`:    -- a mesh object describing the domain.
        - `Ep::Float64`:            -- plain strain modulus.

        # Returns
        - `C_sym::Matrix{Float32}`: -- the elasticity matrix for a symmetric fracture.
    """
    function load_isotropic_elasticity_matrix_symmetric(mesh::CartesianMesh, Ep::Float64)::Matrix{Float32}
        all_elmnts, pos_qdrnt, boundary_x, boundary_y = get_active_symmetric_elements(mesh)

        no_elements = length(pos_qdrnt) + length(boundary_x) + length(boundary_y) + 1
        C_sym = Matrix{Float32}(undef, no_elements, no_elements)

        indx_boun_x = length(pos_qdrnt)
        indx_boun_y = indx_boun_x + length(boundary_x)
        indx_cntr_elm = indx_boun_y + length(boundary_y)

        sym_elements = get_symetric_elements(mesh, pos_qdrnt)
        sym_elem_xbound = get_symetric_elements(mesh, boundary_x)
        sym_elem_ybound = get_symetric_elements(mesh, boundary_y)

        a = mesh.hx / 2.0
        b = mesh.hy / 2.0

        # influence on elements
        for i in 1:length(pos_qdrnt)
            x = mesh.CenterCoor[pos_qdrnt[i], 1] - mesh.CenterCoor[:, 1]
            y = mesh.CenterCoor[pos_qdrnt[i], 2] - mesh.CenterCoor[:, 2]

            C_i = (Ep / (8.0 * π)) * (sqrt.((a - x).^2 + (b - y).^2) ./ ((a - x) * (b - y)) + 
                sqrt.((a + x).^2 + (b - y).^2) ./ ((a + x) * (b - y)) + 
                sqrt.((a - x).^2 + (b + y).^2) ./ ((a - x) * (b + y)) + 
                sqrt.((a + x).^2 + (b + y).^2) ./ ((a + x) * (b + y)))

            C_sym[i, 1:indx_boun_x] = C_i[sym_elements[:, 1]] + 
                                    C_i[sym_elements[:, 2]] + 
                                    C_i[sym_elements[:, 3]] + 
                                    C_i[sym_elements[:, 4]]

            C_sym[i, indx_boun_x+1:indx_boun_y] = C_i[sym_elem_xbound[:, 1]] + 
                                                C_i[sym_elem_xbound[:, 4]]

            C_sym[i, indx_boun_y+1:indx_cntr_elm] = C_i[sym_elem_ybound[:, 1]] + 
                                                    C_i[sym_elem_ybound[:, 2]]

            C_sym[i, end] = C_i[mesh.CenterElts[1]]
        end

        # influence on x boundary elements
        for i in 1:length(boundary_x)
            x = mesh.CenterCoor[boundary_x[i], 1] - mesh.CenterCoor[:, 1]
            y = mesh.CenterCoor[boundary_x[i], 2] - mesh.CenterCoor[:, 2]

            C_i = (Ep / (8.0 * π)) * (sqrt.((a - x).^2 + (b - y).^2) ./ ((a - x) * (b - y)) + 
                sqrt.((a + x).^2 + (b - y).^2) ./ ((a + x) * (b - y)) + 
                sqrt.((a - x).^2 + (b + y).^2) ./ ((a - x) * (b + y)) + 
                sqrt.((a + x).^2 + (b + y).^2) ./ ((a + x) * (b + y)))

            C_sym[i + indx_boun_x, 1:indx_boun_x] = C_i[sym_elements[:, 1]] + 
                                                    C_i[sym_elements[:, 2]] + 
                                                    C_i[sym_elements[:, 3]] + 
                                                    C_i[sym_elements[:, 4]]

            C_sym[i + indx_boun_x, indx_boun_x+1:indx_boun_y] = C_i[sym_elem_xbound[:, 1]] + 
                                                                C_i[sym_elem_xbound[:, 4]]

            C_sym[i + indx_boun_x, indx_boun_y+1:indx_cntr_elm] = C_i[sym_elem_ybound[:, 1]] + 
                                                                C_i[sym_elem_ybound[:, 2]]

            C_sym[indx_boun_x + i, end] = C_i[mesh.CenterElts[1]]
        end

        # influence on y boundary elements
        for i in 1:length(boundary_y)
            x = mesh.CenterCoor[boundary_y[i], 1] - mesh.CenterCoor[:, 1]
            y = mesh.CenterCoor[boundary_y[i], 2] - mesh.CenterCoor[:, 2]

            C_i = (Ep / (8.0 * π)) * (sqrt.((a - x).^2 + (b - y).^2) ./ ((a - x) * (b - y)) + 
                sqrt.((a + x).^2 + (b - y).^2) ./ ((a + x) * (b - y)) + 
                sqrt.((a - x).^2 + (b + y).^2) ./ ((a - x) * (b + y)) + 
                sqrt.((a + x).^2 + (b + y).^2) ./ ((a + x) * (b + y)))

            C_sym[i + indx_boun_y, 1:indx_boun_x] = C_i[sym_elements[:, 1]] + 
                                                    C_i[sym_elements[:, 2]] + 
                                                    C_i[sym_elements[:, 3]] + 
                                                    C_i[sym_elements[:, 4]]

            C_sym[i + indx_boun_y, indx_boun_x+1:indx_boun_y] = C_i[sym_elem_xbound[:, 1]] + 
                                                                C_i[sym_elem_xbound[:, 4]]

            C_sym[i + indx_boun_y, indx_boun_y+1:indx_cntr_elm] = C_i[sym_elem_ybound[:, 1]] + 
                                                                C_i[sym_elem_ybound[:, 2]]

            C_sym[indx_boun_y + i, end] = C_i[mesh.CenterElts[1]]
        end

        # influence on center element
        x = mesh.CenterCoor[mesh.CenterElts[1], 1] - mesh.CenterCoor[:, 1]
        y = mesh.CenterCoor[mesh.CenterElts[1], 2] - mesh.CenterCoor[:, 2]

        C_i = (Ep / (8.0 * π)) * (sqrt.((a - x).^2 + (b - y).^2) ./ ((a - x) * (b - y)) + 
            sqrt.((a + x).^2 + (b - y).^2) ./ ((a + x) * (b - y)) + 
            sqrt.((a - x).^2 + (b + y).^2) ./ ((a - x) * (b + y)) + 
            sqrt.((a + x).^2 + (b + y).^2) ./ ((a + x) * (b + y)))

        C_sym[end, 1:length(pos_qdrnt)] = C_i[sym_elements[:, 1]] + 
                                        C_i[sym_elements[:, 2]] + 
                                        C_i[sym_elements[:, 3]] + 
                                        C_i[sym_elements[:, 4]]

        C_sym[end, indx_boun_x+1:indx_boun_y] = C_i[sym_elem_xbound[:, 1]] + 
                                            C_i[sym_elem_xbound[:, 4]]

        C_sym[end, indx_boun_y+1:indx_cntr_elm] = C_i[sym_elem_ybound[:, 1]] + 
                                                C_i[sym_elem_ybound[:, 2]]

        C_sym[end, end] = C_i[mesh.CenterElts[1]]

        return C_sym
    end

    #-----------------------------------------------------------------------------------------------------------------------

    """
        self_influence(mesh, Ep)

        Calculate the self influence coefficient for the mesh elements.

        # Arguments
        - `mesh::CartesianMesh`: The mesh object containing element dimensions.
        - `Ep::Float64`: The plane strain modulus.

        # Returns
        - `Float64`: The self influence coefficient.
    """
    function self_influence(mesh::CartesianMesh, Ep::Float64)::Float64
        a = mesh.hx / 2.0
        b = mesh.hy / 2.0
        
        return Ep / (2.0 * π) * sqrt(a^2 + b^2) / (a * b)
    end
end # module Symmetry