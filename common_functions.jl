module CommonFunctions

    export locate_element

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
    function locate_element(mesh, x::Float64, y::Float64)::Union{Int, Float64}

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
end