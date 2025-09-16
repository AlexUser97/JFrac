# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac elasticity module on Julia language.
"""

module Elasticity

    using LinearAlgebra
    using Logging
    using JSON
    using Serialization
    using FilePathsBase: joinpath
    using Base.Filesystem: isfile

    include("mesh.jl")
    using .Mesh: CartesianMesh

    export load_isotropic_elasticity_matrix, load_isotropic_elasticity_matrix_toepliz, 
        get_Cij_Matrix, load_TI_elasticity_matrix, load_elasticity_matrix, mapping_old_indexes

    """
        load_isotropic_elasticity_matrix(Mesh, Ep)

        Evaluate the elasticity matrix for the whole mesh.

        # Arguments
        - `Mesh::CartesianMesh`: a mesh object describing the domain.
        - `Ep::Float64`: plain strain modulus.

        # Returns
        - `Matrix{Float32}`: the elasticity matrix.
    """
    function load_isotropic_elasticity_matrix(Mesh::CartesianMesh, Ep::Float64)::Matrix{Float32}
        """
        a and b are the half breadth and height of a cell
        ___________________________________
        |           |           |           |
        |           |           |           |
        |     .     |     .     |     .     |
        |           |           |           |
        |___________|___________|___________|
        |           |     ^     |           |
        |           |   b |     |           |
        |     .     |     .<--->|     .     |
        |           |        a  |           |
        |___________|___________|___________|
        |           |           |           |
        |           |           |           |
        |     .     |     .     |     .     |
        |           |           |           |
        |___________|___________|___________|
        
        """

        a = Mesh.hx / 2.0
        b = Mesh.hy / 2.0
        Ne = Mesh.NumberOfElts

        C = Matrix{Float32}(undef, Ne, Ne)

        for i in 1:Ne
            x = Mesh.CenterCoor[i, 1] - Mesh.CenterCoor[:, 1]
            y = Mesh.CenterCoor[i, 2] - Mesh.CenterCoor[:, 2]

            C[i, :] = (Ep / (8.0 * π)) * (
                sqrt.((a - x).^2 + (b - y).^2) ./ ((a - x) * (b - y)) +
                sqrt.((a + x).^2 + (b - y).^2) ./ ((a + x) * (b - y)) +
                sqrt.((a - x).^2 + (b + y).^2) ./ ((a - x) * (b + y)) +
                sqrt.((a + x).^2 + (b + y).^2) ./ ((a + x) * (b + y))
            )
        end

        return C
    end

    """
    Class for loading isotropic elasticity matrix in Toeplitz format.
    """
    mutable struct load_isotropic_elasticity_matrix_toepliz
        Ep::Float64
        const_val::Float64
        a::Float64
        b::Float64
        nx::Int
        C_toeplotz_coe::Vector{Float32}
    end

    """
        load_isotropic_elasticity_matrix_toepliz(Mesh, Ep)

        Constructor for the load_isotropic_elasticity_matrix_toepliz class.

        # Arguments
        - `Mesh::CartesianMesh`: a mesh object describing the domain.
        - `Ep::Float64`: plain strain modulus.

        # Returns
        - `load_isotropic_elasticity_matrix_toepliz`: initialized object.
    """
    function load_isotropic_elasticity_matrix_toepliz(Mesh::CartesianMesh, Ep::Float64)
        const_val = (Ep / (8.0 * π))
        obj = load_isotropic_elasticity_matrix_toepliz(Ep, const_val, 0.0, 0.0, 0, Float32[])
        reload!(obj, Mesh)
        return obj
    end

    """
        reload!(self, Mesh)

        Reload the Toeplitz coefficients based on the mesh.

        # Arguments
        - `self::load_isotropic_elasticity_matrix_toepliz`: the object instance.
        - `Mesh::CartesianMesh`: a mesh object describing the domain.
    """
    function reload!(self::load_isotropic_elasticity_matrix_toepliz, Mesh::CartesianMesh)
        hx = Mesh.hx
        hy = Mesh.hy
        a = hx / 2.0
        b = hy / 2.0
        nx = Mesh.nx
        ny = Mesh.ny
        self.a = a
        self.b = b
        self.nx = nx
        const_val = self.const_val

        """
        Let us make some definitions:
        cartesian mesh             := a structured rectangular mesh of (nx,ny) cells of rectangular shape
            
                                            |<------------nx----------->|
                                        _    ___ ___ ___ ___ ___ ___ ___
                                        |   | . | . | . | . | . | . | . |
                                        |   |___|___|___|___|___|___|___|
                                        ny  | . | . | . | . | . | . | . |  
                                        |   |___|___|___|___|___|___|___|   y
                                        |   | . | . | . | . | . | . | . |   |
                                        -   |___|___|___|___|___|___|___|   |____x  
                                    
                                    the cell centers are marked by .
        
        set of unique distances    := given a set of cells in a cartesian mesh, consider the set of unique distances 
                                    between any pair of cell centers.
        set of unique coefficients := given a set of unique distances then consider the interaction coefficients
                                    obtained from them
                                    
        C_toeplotz_coe             := An array of size (nx*ny), populated with the unique coefficients. 
        
        Mathematically speaking:
        for i in (0,ny) and j in (0,nx) take the set of combinations (i,j) such that [i^2 y^2 + j^2 x^2]^1/2 is unique
        """
        C_toeplotz_coe = Vector{Float32}(undef, ny * nx)
        xindrange = collect(0:nx-1)
        xrange = xindrange * hx
        
        for i in 0:ny-1
            y = i * hy
            amx = a - xrange
            apx = a + xrange
            bmy = b - y
            bpy = b + y
            start_idx = i * nx + 1
            end_idx = (i + 1) * nx
            C_toeplotz_coe[start_idx:end_idx] = const_val * (
                sqrt.((amx.^2) + (bmy^2)) ./ (amx * bmy) +
                sqrt.((apx.^2) + (bmy^2)) ./ (apx * bmy) +
                sqrt.((amx.^2) + (bpy^2)) ./ (amx * bpy) +
                sqrt.((apx.^2) + (bpy^2)) ./ (apx * bpy)
            )
        end
        
        self.C_toeplotz_coe = C_toeplotz_coe
        return nothing
    end

    """
        getindex(self, elementsXY)

        Get submatrix of C based on element indices.

        # Arguments
        - `self::load_isotropic_elasticity_matrix_toepliz`: the object instance.
        - `elementsXY::Tuple{Vector{Int}, Vector{Int}}`: tuple of (elemY, elemX) indices.

        # Returns
        - `Matrix{Float32}`: submatrix of C.
    """
    function Base.getindex(self::load_isotropic_elasticity_matrix_toepliz, elementsXY::Tuple{Vector{Int}, Vector{Int}})
        """
        critical call: it should be as fast as possible
        :param elemX: (numpy array) columns to take
        :param elemY: (numpy array) rows to take
        :return: submatrix of C
        """
        
        elemX = elementsXY[2] # Julia uses 1-based indexing
        elemY = elementsXY[1] # Julia uses 1-based indexing
        dimX = length(elemX)  # number of elements to consider on x axis
        dimY = length(elemY)  # number of elements to consider on y axis

        if dimX == 0 || dimY == 0
            return Matrix{Float32}(undef, dimY, dimX)
        else
            nx = self.nx  # number of element in x direction in the global mesh
            C_sub = Matrix{Float32}(undef, dimY, dimX)  # submatrix of C
            localC_toeplotz_coe = copy(self.C_toeplotz_coe)  # local access is faster
            
            if dimX != dimY
                iY = fld.(elemY .- 1, nx) .+ 1  # fld is floor division, adjust for 1-based indexing
                jY = (elemY .- 1) .% nx .+ 1    # modulo operation, adjust for 1-based indexing
                iX = fld.(elemX .- 1, nx) .+ 1
                jX = (elemX .- 1) .% nx .+ 1
                
                # strategy 1
                for iter1 in 1:dimY
                    i1 = iY[iter1]
                    j1 = jY[iter1]
                    indices = abs.(j1 .- jX) + nx * abs.(i1 .- iX) .+ 1  # adjust for 1-based indexing
                    C_sub[iter1, :] = localC_toeplotz_coe[indices]
                end
                return C_sub

            elseif dimX == dimY && elemY == elemX
                i = fld.(elemX .- 1, nx) .+ 1
                j = (elemX .- 1) .% nx .+ 1
                
                # strategy 1
                for iter1 in 1:dimX
                    i1 = i[iter1]
                    j1 = j[iter1]
                    indices = abs.(j .- j1) + nx * abs.(i .- i1) .+ 1  # adjust for 1-based indexing
                    C_sub[iter1, :] = localC_toeplotz_coe[indices]
                end
                return C_sub

            else
                iY = fld.(elemY .- 1, nx) .+ 1
                jY = (elemY .- 1) .% nx .+ 1
                iX = fld.(elemX .- 1, nx) .+ 1
                jX = (elemX .- 1) .% nx .+ 1
                
                # strategy 1
                for iter1 in 1:dimY
                    i1 = iY[iter1]
                    j1 = jY[iter1]
                    indices = abs.(j1 .- jX) + nx * abs.(i1 .- iX) .+ 1  # adjust for 1-based indexing
                    C_sub[iter1, :] = localC_toeplotz_coe[indices]
                end
                return C_sub
            end
        end
    end

    """
        get_Cij_Matrix(youngs_mod, nu)

        Calculate the Cij matrix for transversely isotropic materials.

        # Arguments
        - `youngs_mod::Float64`: Young's modulus.
        - `nu::Float64`: Poisson's ratio.

        # Returns
        - `Matrix{Float64}`: the Cij matrix.
    """
    function get_Cij_Matrix(youngs_mod::Float64, nu::Float64)::Matrix{Float64}
        k = youngs_mod / (3 * (1 - 2 * nu))
        la = (3 * k * (3 * k - youngs_mod)) / (9 * k - youngs_mod)
        mu = 3 / 2 * (k - la)

        Cij = zeros(Float64, 6, 6)
        Cij[1, 1] = (la + 2 * mu) * (1 + 0.00007)
        Cij[1, 3] = la * (1 + 0.00005)
        Cij[3, 3] = (la + 2 * mu) * (1 + 0.00009)
        Cij[4, 4] = mu * (1 + 0.00001)
        Cij[6, 6] = mu * (1 + 0.00003) 
        Cij[1, 2] = Cij[1, 1] - 2 * Cij[6, 6]

        # Make symmetric
        Cij[2, 1] = Cij[1, 2]
        Cij[3, 1] = Cij[1, 3]
        Cij[2, 2] = Cij[1, 1]
        Cij[2, 3] = Cij[1, 3]
        Cij[3, 2] = Cij[1, 3]
        Cij[5, 5] = Cij[4, 4]

        return Cij
    end

    """
        load_TI_elasticity_matrix(Mesh, mat_prop, sim_prop)

        Create the elasticity matrix for transversely isotropic materials.

        # Arguments
        - `Mesh::CartesianMesh`: a mesh object describing the domain.
        - `mat_prop`: the MaterialProperties object giving the material properties.
        - `sim_prop`: the SimulationProperties object giving the numerical parameters.

        # Returns
        - `Matrix{Float64}`: the elasticity matrix.
    """
    function load_TI_elasticity_matrix(Mesh::CartesianMesh, mat_prop, sim_prop)::Matrix{Float64}
        log = Logging.current_logger()
        @info "Writing parameters to a file..." _group="JFrac.load_TI_elasticity_matrix"
        
        data = Dict(
            "Solid parameters" => Dict(
                "C11" => mat_prop.Cij[1, 1],
                "C12" => mat_prop.Cij[1, 2],
                "C13" => mat_prop.Cij[1, 3],
                "C33" => mat_prop.Cij[3, 3],
                "C44" => mat_prop.Cij[4, 4]
            ),
            "Mesh" => Dict(
                "L1" => Mesh.Lx,
                "L3" => Mesh.Ly,
                "n1" => Mesh.nx,
                "n3" => Mesh.ny
            )
        )

        @info "Writing parameters to a file..." _group="JFrac.load_TI_elasticity_matrix"
        
        curr_directory = pwd()
        cd(sim_prop.TI_KernelExecPath)
        
        try
            open("stiffness_matrix.json", "w") do f
                JSON.print(f, data, 3)
            end

            suffix = ""
            if Sys.iswindows()
            else
                suffix = "./"
            end

            @info "running C++ process..." _group="JFrac.load_TI_elasticity_matrix"
            run(`$suffix TI_elasticity_kernel`)

            @info "Reading global TI elasticity matrix..." _group="JFrac.load_TI_elasticity_matrix"
            
            n_elements = data["Mesh"]["n1"] * data["Mesh"]["n3"]
            C = Matrix{Float64}(undef, n_elements, n_elements)
            
            open("StrainResult.bin", "r") do file
                total_elements = n_elements * n_elements
                raw_data = read(file, total_elements * sizeof(Float64))
                C_vec = reinterpret(Float64, raw_data)
                C[:] = reshape(C_vec, (n_elements, n_elements))
            end
            
            return C
            
        catch e
            if isa(e, SystemError) && !isfile("StrainResult.bin")
                error("file not found")
            else
                rethrow(e)
            end
        finally
            cd(curr_directory)
        end
    end

    """
        load_elasticity_matrix(Mesh, EPrime)

        The function loads the elasticity matrix from the saved file. If the loaded matrix is not compatible with respect
        to the current mesh or plain strain modulus, the compatible matrix is computed and saved in a file.

        # Arguments
        - `Mesh::CartesianMesh`: a mesh object describing the domain.
        - `EPrime::Float64`: plain strain modulus.

        # Returns
        - `Matrix`: the elasticity matrix.
    """
    function load_elasticity_matrix(Mesh::CartesianMesh, EPrime::Float64)
        log = Logging.current_logger()
        @info "Reading global elasticity matrix..." _group="JFrac.load_elasticity_matrix"
        
        try
            file = open("CMatrix", "r")
            (C, MeshLoaded, EPrimeLoaded) = deserialize(file)
            close(file)
            
            if (Mesh.nx, Mesh.ny, Mesh.Lx, Mesh.Ly, EPrime) == (MeshLoaded.nx, MeshLoaded.ny, MeshLoaded.Lx, MeshLoaded.Ly, EPrimeLoaded)
                return C
            else
                @warn "The loaded matrix is not correct with respect to the current mesh or the current plain strain modulus.\nMaking global matrix..." _group="JFrac.load_elasticity_matrix"
                C = load_isotropic_elasticity_matrix(Mesh, EPrime)
                Elast = (C, Mesh, EPrime)
                
                file = open("CMatrix", "w")
                serialize(file, Elast)
                close(file)
                
                @info "Done!" _group="JFrac.load_elasticity_matrix"
                return C
            end
        catch e
            if isa(e, SystemError) && !isfile("CMatrix")
                @error "file not found\nBuilding the global elasticity matrix..." _group="JFrac.load_elasticity_matrix"
                C = load_isotropic_elasticity_matrix(Mesh, EPrime)
                Elast = (C, Mesh, EPrime)
                
                file = open("CMatrix", "w")
                serialize(file, Elast)
                close(file)
                
                @info "Done!" _group="JFrac.load_elasticity_matrix"
                return C
            else
                rethrow(e)
            end
        end
    end

    """
        mapping_old_indexes(new_mesh, mesh, direction=nothing)

        Function to get the mapping of the indexes.

        # Arguments
        - `new_mesh::CartesianMesh`: the new mesh object.
        - `mesh::CartesianMesh`: the old mesh object.
        - `direction::Union{String, Nothing}`: direction of mesh extension.

        # Returns
        - `Vector{Int}`: array of new indexes.
    """
    function mapping_old_indexes(new_mesh::CartesianMesh, mesh::CartesianMesh, direction::Union{String, Nothing}=nothing)::Vector{Int}
        dne = (new_mesh.NumberOfElts - mesh.NumberOfElts)
        dnx = (new_mesh.nx - mesh.nx)
        dny = (new_mesh.ny - mesh.ny)

        old_indexes = collect(0:mesh.NumberOfElts-1) .+ 1  # Convert to 1-based indexing

        if direction == "top"
            new_indexes = old_indexes
        elseif direction == "bottom"
            new_indexes = old_indexes + dne
        elseif direction == "left"
            new_indexes = old_indexes + (fld.(old_indexes .- 1, mesh.nx) .+ 1) * dnx
        elseif direction == "right"
            new_indexes = old_indexes + fld.(old_indexes .- 1, mesh.nx) * dnx
        elseif direction == "horizontal"
            new_indexes = old_indexes + (fld.(old_indexes .- 1, mesh.nx) .+ 1/2) * dnx
        elseif direction == "vertical"
            new_indexes = old_indexes + dne / 2
        else
            new_indexes = old_indexes + 1/2 * dny * new_mesh.nx + (fld.(old_indexes .- 1, mesh.nx) .+ 1/2) * dnx
        end

        return convert(Vector{Int}, round.(new_indexes))
    end


end # module Elasticity