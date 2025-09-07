# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""
module Properties
using Dates
using Logging
using LoggingExtras

include("mesh.jl")
include("labels.jl")
using .Mesh: CartesianMesh, locate_element 

export MaterialProperties, FluidProperties, InjectionProperties, LoadingProperties, SimulationProperties,
       set_logging_to_file, set_tipAsymptote, get_tipAsymptote, set_viscousInjection, get_viscousInjection, set_volumeControl,
       get_volumeControl, set_dryCrack_mechLoading, get_dryCrack_mechLoading, set_outputFolder, get_outputFolder, set_solTimeSeries,
       get_solTimeSeries, set_simulation_name, get_simulation_name, get_timeStamp, get_time_step_prefactor,
       set_mesh_extension_direction, get_mesh_extension_direction, set_mesh_extension_factor, get_mesh_extension_factor, 
       IterationProperties, PlotProperties, LabelProperties


"""
to_rgb(color)

Convert a color specification to an RGB tuple (r, g, b) with values in range [0, 1].

# Arguments
- `color::Union{String, Tuple}`: Color specification (name, hex, or RGB tuple)

# Returns
- `Tuple{Float64, Float64, Float64}`: RGB values in range [0, 1]
"""
function to_rgb(color::Union{String, Tuple})
    if isa(color, Tuple)
        if length(color) >= 3
            return (Float64(color[1]), Float64(color[2]), Float64(color[3]))
        else
            throw(ArgumentError("RGB tuple must have at least 3 elements"))
        end
    end
    
    # Обработка строковых представлений цветов
    color_name = lowercase(color)
    
    # Основные цвета CSS/XKCD
    color_map = Dict(
        "black" => (0.0, 0.0, 0.0),
        "white" => (1.0, 1.0, 1.0),
        "red" => (1.0, 0.0, 0.0),
        "green" => (0.0, 1.0, 0.0),
        "blue" => (0.0, 0.0, 1.0),
        "yellow" => (1.0, 1.0, 0.0),
        "cyan" => (0.0, 1.0, 1.0),
        "magenta" => (1.0, 0.0, 1.0),
        "orange" => (1.0, 0.647, 0.0),
        "purple" => (0.5, 0.0, 0.5),
        "brown" => (0.647, 0.165, 0.165),
        "pink" => (1.0, 0.753, 0.796),
        "gray" => (0.5, 0.5, 0.5),
        "grey" => (0.5, 0.5, 0.5),
        "firebrick" => (0.698, 0.133, 0.133),
        "olivedrab" => (0.419, 0.557, 0.137),
        "royalblue" => (0.255, 0.412, 0.882),
        "deeppink" => (1.0, 0.078, 0.576),
        "darkmagenta" => (0.545, 0.0, 0.545),
        "yellowgreen" => (0.604, 0.804, 0.196),
        "0.5" => (0.5, 0.5, 0.5),  # Серый цвет как в matplotlib
        "r" => (1.0, 0.0, 0.0)     # Красный как в matplotlib
    )
    
    # Обработка шестнадцатеричных цветов
    if color_name[1] == '#'
        hex_color = color_name[2:end]
        if length(hex_color) == 6
            r = parse(Int, hex_color[1:2], base=16) / 255.0
            g = parse(Int, hex_color[3:4], base=16) / 255.0
            b = parse(Int, hex_color[5:6], base=16) / 255.0
            return (r, g, b)
        end
    end
    
    if haskey(color_map, color_name)
        return color_map[color_name]
    else
        @warn "Unknown color: $color. Using black as default."
        return (0.0, 0.0, 0.0)
    end
end


"""
    MaterialProperties

Class defining the Material properties of the solid.

# Arguments
- `Mesh::CartesianMesh`:           -- the CartesianMesh object describing the mesh.
- `Eprime::Float64`:               -- plain strain modulus.
- `toughness::Float64`:            -- Linear-Elastic Plane-Strain Fracture Toughness.
- `Carters_coef::Float64`:         -- Carter's leak off coefficient.
- `confining_stress::Float64`:     -- in-situ confining stress field normal to fracture surface.
- `grain_size::Float64`:           -- the grain size of the rock; used to calculate the relative roughness.
- `K1c_func::Function`:            -- the function giving the toughness on the domain. It takes one argument
                                      (angle) in case of anisotropic toughness and two arguments (x, y) in case
                                      of heterogeneous toughness. The function is also used to get the
                                      toughness if the domain is re-meshed.
- `anisotropic_K1c::Bool`:         -- flag to specify if the fracture toughness is anisotropic.
- `confining_stress_func::Function`:-- the function giving the in-situ stress on the domain. It should takes
                                      two arguments (x, y) to give the stress on these coordinates. It is also
                                      used to get the stress if the domain is re-meshed.
- `Carters_coef_func::Function`:   -- the function giving the in Carter's leak off coefficient on the domain.
                                      It should takes two arguments (x, y) to give the coefficient on these
                                      coordinates. It is also used to get the leak off coefficient if the
                                      domain is re-meshed.
- `TI_elasticity::Bool`:           -- if True, the medium is elastic transverse isotropic.
- `Cij::Matrix{Float64}`:          -- the transverse isotropic stiffness matrix (in the canonical basis); needs to
                                      be provided if TI_elasticity=True.
- `free_surf::Bool`:               -- the free surface flag. True if the effect of free surface is to be taken
                                      into account.
- `free_surf_depth::Float64`:      -- the depth of the fracture from the free surface.
- `TI_plane_angle::Float64`:       -- the angle of the plane of the fracture with respect to the free surface.
- `minimum_width::Float64`:        -- minimum width corresponding to the asperity of the material.
- `pore_pressure::Float64`:        -- the pore pressure in the medium.

# Attributes
- `Eprime::Float64`:           -- plain strain modulus.
- `K1c::Vector{Float64}`:      -- Linear-Elastic Plane-Strain Toughness for each cell.
- `Kprime::Vector{Float64}`:   -- 4*(2/pi)^0.5 * K1c.
- `Cl::Float64`:               -- Carter's leak off coefficient.
- `Cprime::Vector{Float64}`:   -- 2 * Carter's leak off coefficient.
- `SigmaO::Vector{Float64}`:   -- in-situ confining stress field normal to fracture surface.
- `grainSize::Float64`:        -- the grain size of the rock; used to calculate the relative roughness.
- `anisotropic_K1c::Bool`:     -- if True, the toughness is considered anisotropic.
- `Kc1::Float64`:              -- the fracture toughness along the x-axis, in case it is anisotropic.
- `TI_elasticity::Bool`:       -- the flag specifying transverse isotropic elasticity.
- `Cij::Matrix{Float64}`:      -- the transverse isotropic stiffness matrix (in the canonical basis).
- `freeSurf::Bool`:            -- if True, the effect of free surface is to be taken into account.
- `FreeSurfDepth::Float64`:    -- the depth of the fracture from the free surface.
- `TI_PlaneAngle::Float64`:    -- the angle of the plane of the fracture with respect to the free surface.
- `K1cFunc::Function`:         -- the function giving the toughness on the domain. It takes one argument (angle) in
                                  case of anisotropic toughness and two arguments (x, y) in case of heterogeneous
                                  toughness. The function is also used to get the toughness if the domain is
                                  re-meshed.
- `SigmaOFunc::Function`:      -- the function giving the in-situ stress on the domain. It should takes two
                                  arguments(x, y) to give the stress on these coordinates. It is also used to get
                                  the confining stress if the domain is re-meshed.
- `ClFunc::Function`:          -- the function giving the in Carter's leak off coefficient on the domain. It should
                                  takes two arguments (x, y) to give the coefficient on these coordinates. It is
                                  also used to get the leak off coefficient if the domain is re-meshed.
- `wc::Float64`:               -- minimum width corresponding to the asperity of the material.
- `porePressure::Float64`:     -- the pore pressure in the medium.
"""

mutable struct MaterialProperties

    Eprime::Float64
    K1c::Vector{Float64}
    Kprime::Vector{Float64}
    Cl::Float64
    Cprime::Vector{Float64}
    SigmaO::Vector{Float64}
    grainSize::Float64
    anisotropic_K1c::Bool
    Kc1::Union{Float64, Nothing}
    TI_elasticity::Bool
    Cij::Union{Matrix{Float64}, Nothing}
    free_surf::Bool
    free_surf_depth::Float64
    TI_PlaneAngle::Float64
    K1cFunc::Union{Function, Nothing}
    SigmaOFunc::Union{Function, Nothing}
    ClFunc::Union{Function, Nothing}
    wc::Float64
    porePressure::Float64
end

"""
    MaterialProperties(Mesh, Eprime, toughness=0.0, Carters_coef=0.0, confining_stress=0.0, grain_size=0.0, 
                      K1c_func=nothing, anisotropic_K1c=false, confining_stress_func=nothing, 
                      Carters_coef_func=nothing, TI_elasticity=false, Cij=nothing, free_surf=false, 
                      free_surf_depth=1.e300, TI_plane_angle=0.0, minimum_width=1e-10, pore_pressure=-1.e100)

The constructor function for MaterialProperties.
"""
function MaterialProperties(Mesh::CartesianMesh, Eprime::Float64, toughness::Float64=0.0, 
                          Carters_coef::Float64=0.0, confining_stress::Float64=0.0, grain_size::Float64=0.0,
                          K1c_func::Union{Function, Nothing}=nothing, anisotropic_K1c::Bool=false,
                          confining_stress_func::Union{Function, Nothing}=nothing,
                          Carters_coef_func::Union{Function, Nothing}=nothing, TI_elasticity::Bool=false,
                          Cij::Union{Matrix{Float64}, Nothing}=nothing, free_surf::Bool=false,
                          free_surf_depth::Float64=1.e300, TI_plane_angle::Float64=0.0,
                          minimum_width::Float64=1e-10, pore_pressure::Float64=-1.e100)

        Eprime_val = Eprime

        if toughness != 0.0
            K1c_val = fill(toughness, Mesh.NumberOfElts)
            Kprime_val = sqrt(32.0 / π) * toughness * ones(Float64, Mesh.NumberOfElts)
        else
            K1c_val = zeros(Float64, Mesh.NumberOfElts)
            Kprime_val = zeros(Float64, Mesh.NumberOfElts)
        end

        Cl_val = Carters_coef
        Cprime_val = 2.0 * Carters_coef * ones(Float64, Mesh.NumberOfElts)

        SigmaO_val = confining_stress * ones(Float64, Mesh.NumberOfElts)
        Kc1_val = nothing
        if anisotropic_K1c
            try
                Kc1_val = K1c_func(0.0)
            catch
                error("The given K1c function is not correct for anisotropic case! It should take one" *
                  " argument, i.e. the angle and return a toughness value.")
            end
        end
        if K1c_func !== nothing && !anisotropic_K1c
            try
                K1c_func(0.0, 0.0)
            catch
                error("The given K1c function is not correct! It should take two arguments, " *
                    "i.e. the x and y coordinates of a point and return the toughness at this point.")
            end
        end
        if TI_elasticity || free_surf
            if Cij !== nothing
                if size(Cij) != (6, 6)
                    error("Cij matrix is not a 6x6 array!")
                end
            else
                error("Cij matrix is not provided!")
            end
        end
        if free_surf
            if free_surf_depth == 1.e300
                error("Depth from free surface is to be provided.")
            elseif Cij === nothing
                error("The stiffness matrix (in the canonical basis) is to be provided")
            end
        end
        mat_prop = MaterialProperties(
            Eprime_val,
            K1c_val,
            Kprime_val,
            Cl_val,
            Cprime_val,
            SigmaO_val,
            grain_size,
            anisotropic_K1c,
            Kc1_val,
            TI_elasticity,
            Cij,
            free_surf,
            free_surf_depth,
            TI_plane_angle,
            K1c_func,
            confining_stress_func,
            Carters_coef_func,
            minimum_width,
            pore_pressure
        )
        # Override with values evaluated by given functions if provided
        if (K1c_func !== nothing) || (confining_stress_func !== nothing) || (Carters_coef_func !== nothing)
            remesh!(mat_prop, Mesh)
        end
        
        return mat_prop
    end


    # ------------------------------------------------------------------------------------------------------------------

    """
        remesh!(self, mesh)

    This function evaluates the toughness, confining stress and leak off coefficient on the given mesh using the
    functions provided in the MaterialProperties object. It should be evaluated each time re-meshing is done.

    # Arguments
    - `self::MaterialProperties`:   -- the MaterialProperties object.
    - `mesh::CartesianMesh`:        -- the CartesianMesh object describing the new mesh.
    """
    function remesh!(self::MaterialProperties, mesh::CartesianMesh)
        if self.K1cFunc !== nothing && !self.anisotropic_K1c
            self.K1c = Vector{Float64}(undef, mesh.NumberOfElts)
            for i in 1:mesh.NumberOfElts
                self.K1c[i] = self.K1cFunc(mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2])
            end
            self.Kprime = self.K1c * sqrt(32.0 / π)
        elseif self.K1cFunc !== nothing && self.anisotropic_K1c
            self.K1c = Vector{Float64}(undef, mesh.NumberOfElts)
            for i in 1:mesh.NumberOfElts
                self.K1c[i] = self.K1cFunc(π / 2)
            end
            self.Kprime = self.K1c * sqrt(32.0 / π)
        else
            self.Kprime = fill(self.Kprime[1], mesh.NumberOfElts)
        end

        if self.SigmaOFunc !== nothing
            self.SigmaO = Vector{Float64}(undef, mesh.NumberOfElts)
            for i in 1:mesh.NumberOfElts
                self.SigmaO[i] = self.SigmaOFunc(mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2])
            end
        else
            self.SigmaO = fill(self.SigmaO[1], mesh.NumberOfElts)
        end

        if self.ClFunc !== nothing
            self.Cl = Vector{Float64}(undef, mesh.NumberOfElts)
            self.Cprime = Vector{Float64}(undef, mesh.NumberOfElts)
            for i in 1:mesh.NumberOfElts
                self.Cl[i] = self.ClFunc(mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2])
            end
            self.Cprime = 2.0 * self.Cl
        else
            self.Cprime = fill(self.Cprime[1], mesh.NumberOfElts)
        end
    end


# -----------------------------------------------------------------------------------------------------------------------

"""
    FluidProperties

Class defining the fluid properties.

# Arguments
- `viscosity::Float64`:     -- viscosity of the fluid.
- `density::Float64`:       -- density of the fluid.
- `rheology::String`:       -- string specifying rheology of the fluid. Possible options:
                              - "Newtonian"
                              - "Herschel-Bulkley" or "HBF"
                              - "power-law" or "PLF"
- `turbulence::Bool`:       -- turbulence flag. If true, turbulence will be taken into account.
- `compressibility::Float64`: -- the compressibility of the fluid.
- `n::Float64`:             -- flow index of the Herschel-Bulkey fluid.
- `k::Float64`:             -- consistency index of the Herschel-Bulkey fluid.
- `T0::Float64`:            -- yield stress of the Herschel-Bulkey fluid.

# Attributes
- `viscosity::Float64`:     -- Viscosity of the fluid (note its different from local viscosity, see
                               fracture class for local viscosity).
- `muPrime::Float64`:       -- 12 * viscosity (parallel plates viscosity factor).
- `rheology::String`:       -- string specifying rheology of the fluid. Possible options:
                              - "Newtonian"
                              - "Herschel-Bulkley" or "HBF"
                              - "power-law" or "PLF"
- `density::Float64`:       -- density of the fluid.
- `turbulence::Bool`:       -- turbulence flag. If true, turbulence will be taken into account.
- `compressibility::Float64`: -- the compressibility of the fluid.
- `n::Float64`:             -- flow index of the Herschel-Bulkey fluid.
- `k::Float64`:             -- consistency index of the Herschel-Bulkey fluid.
- `T0::Float64`:            -- yield stress of the Herschel-Bulkey fluid.
- `Mprime::Float64`:        -- 2^(n + 1) * (2 * n + 1)^n / n^n  * k
- `var1::Float64`:          -- some variables depending upon n. Saved to avoid recomputation
- `var2::Float64`:          -- some variables depending upon n. Saved to avoid recomputation
- `var3::Float64`:          -- some variables depending upon n. Saved to avoid recomputation
- `var4::Float64`:          -- some variables depending upon n. Saved to avoid recomputation
- `var5::Float64`:          -- some variables depending upon n. Saved to avoid recomputation
"""
mutable struct FluidProperties
    viscosity::Float64
    muPrime::Float64
    rheology::String
    density::Float64
    turbulence::Bool
    compressibility::Float64
    n::Union{Float64, Nothing}
    k::Union{Float64, Nothing}
    T0::Union{Float64, Nothing}
    Mprime::Union{Float64, Nothing}
    var1::Union{Float64, Nothing}
    var2::Union{Float64, Nothing}
    var3::Union{Float64, Nothing}
    var4::Union{Float64, Nothing}
    var5::Union{Float64, Nothing}
end

"""
    FluidProperties(viscosity, density=1000.0, rheology="Newtonian", turbulence=false, 
                   compressibility=0.0, n=nothing, k=nothing, T0=nothing)

Constructor function for FluidProperties.
"""
function FluidProperties(viscosity::Float64, density::Float64=1000.0, rheology::String="Newtonian", turbulence::Bool=false, 
                        compressibility::Float64=0.0, n::Union{Float64, Nothing}=nothing, k::Union{Float64, Nothing}=nothing, T0::Union{Float64, Nothing}=nothing)
    
    # Process viscosity - ВСЕГДА КОНСТАНТА
    viscosity_val = viscosity
    muPrime_val = 12.0 * viscosity_val  # the geometric viscosity in the parallel plate solution
    
    # Check rheology options
    rheologyOptions = ["Newtonian", "Herschel-Bulkley", "HBF", "power-law", "PLF"]
    if rheology in rheologyOptions
        rheology_val = rheology
        
        # Initialize variables that might not be set
        n_val = n
        k_val = k
        T0_val = T0
        Mprime_val = nothing
        var1_val = nothing
        var2_val = nothing
        var3_val = nothing
        var4_val = nothing
        var5_val = nothing
        
        if rheology in ["Herschel-Bulkley", "HBF"]
            if n === nothing || k === nothing || T0 === nothing
                error("n (flow index), k(consistency index) and T0 (yield stress) are required for a Herschel-Bulkley type fluid!")
            end
            n_val = n
            k_val = k
            T0_val = T0
            Mprime_val = 2.0^(n + 1) * (2 * n + 1)^n / n^n * k
            var1_val = Mprime_val^(-1.0 / n)
            var2_val = 1.0 / n - 1.0
            var3_val = 2.0 + 1.0 / n
            var4_val = 1.0 + 1.0 / n
            var5_val = n / (n + 1.0)
        elseif rheology in ["power-law", "PLF"]
            if n === nothing || k === nothing
                error("n (flow index) and k(consistency index) are required for a power-law type fluid!")
            end
            n_val = n
            k_val = k
            T0_val = 0.0
            Mprime_val = 2.0^(n + 1) * (2 * n + 1)^n / n^n * k
        end
        
        # Create FluidProperties object
        return FluidProperties(
            viscosity_val,
            muPrime_val,
            rheology_val,
            density,
            turbulence,
            compressibility,
            n_val,
            k_val,
            T0_val,
            Mprime_val,
            var1_val,
            var2_val,
            var3_val,
            var4_val,
            var5_val
        )
    else
        error("Invalid input for fluid rheology. Possible options: " * string(rheologyOptions))
    end
end


# ----------------------------------------------------------------------------------------------------------------------


"""
    InjectionProperties

Class defining the injection parameters.

# Arguments
- `rate::Matrix{Float64}`:         -- array specifying the time series (row 1) and the corresponding injection
                                     rates (row 2). The times are instant where the injection rate changes.
                                     
                                     Attention:
                                        The first time should be zero. The corresponding injection rate would
                                        be taken for initialization of the fracture with an analytical solution,
                                        if required.
- `mesh::CartesianMesh`:           -- the CartesianMesh object defining mesh.
- `source_coordinates::Vector{Float64}`: -- list or Vector with a length of 2, specifying the x and y coordinates
                                     of the injection point. Not used if source_loc_func is provided (See below).
- `source_loc_func::Function`:     -- the source location function is used to get the elements in which the fluid is
                                     injected. It should take the x and y coordinates and return True or False
                                     depending upon if the source is present on these coordinates. This function is
                                     evaluated at each of the cell centre coordinates to determine if the cell is
                                     a source element. It should have to arguments (x, y) and return True or False.
                                     It is also called upon re-meshing to get the source elements on the coarse
                                     mesh.
- `sink_loc_func::Function`:       -- the sink location function is used to get the elements where there is a fixed rate
                                     sink. It should take the x and y coordinates and return True or False
                                     depending upon if the sink is present on these coordinates. This function is
                                     evaluated at each of the cell centre coordinates to determine if the cell is
                                     a sink element. It should have to arguments (x, y) and return True or False.
                                     It is also called upon re-meshing to get the source elements on the coarse
                                     mesh.
- `sink_vel_func::Function`:       -- this function gives the sink velocity at the given (x, y) point.
- `model_inj_line::Bool`:          -- flag to model injection line.
- `il_compressibility::Float64`:   -- injection line compressibility.
- `il_volume::Float64`:            -- injection line volume.
- `perforation_friction::Float64`: -- perforation friction.
- `initial_pressure::Float64`:     -- initial pressure of the injection line.
- `rate_delayed_second_injpoint::Matrix{Float64}`: -- rate for delayed second injection point.
- `delayed_second_injpoint_loc::Vector{Float64}`: -- coordinates of delayed second injection point.
- `initial_rate_delayed_second_injpoint::Float64`: -- initial rate for delayed second injection point.
- `rate_delayed_inj_pt_func::Function`: -- function for delayed injection point rate.
- `delayed_second_injpoint_loc_func::Function`: -- function for delayed second injection point location.

# Attributes
- `injectionRate::Matrix{Float64}`: -- array specifying the time series (row 1) and the corresponding injection
                                     rates (row 2). The time series provide the time when the injection rate
                                     changes.
- `sourceCoordinates::Vector{Float64}`: -- array with a single row and two columns specifying the x and y coordinate
                                     of the injection point coordinates. If there are more than one source elements,
                                     the average is taken to get an estimate injection cell at the center.
- `sourceElem::Vector{Int}`:       -- the element(s) where the fluid is injected in the cartesian mesh.
- `sourceLocFunc::Union{Function, Nothing}`: -- the source location function is used to get the elements in which the fluid is
                                     injected. It should take the x and y coordinates and return True or False
                                     depending upon if the source is present on these coordinates. This function is
                                     evaluated at each of the cell centre coordinates to determine if the cell is
                                     a source element. It should have to arguments (x, y) and return True or False.
                                     It is also called upon re-meshing to get the source elements on the coarse
                                     mesh.
- `sinkLocFunc::Union{Function, Nothing}`: -- see description of arguments.
- `sinkVelFunc::Union{Function, Nothing}`: -- see description of arguments.
- `sinkElem::Vector{Int}`:         -- sink elements.
- `sinkVel::Vector{Float64}`:      -- sink velocities.
- `modelInjLine::Bool`:            -- flag to model injection line.
- `ILCompressibility::Float64`:    -- injection line compressibility.
- `ILVolume::Float64`:             -- injection line volume.
- `perforationFriction::Float64`:  -- perforation friction.
- `initPressure::Float64`:         -- initial pressure of the injection line.
- `injectionRate_delayed_second_injpoint::Float64`: -- rate for delayed second injection point.
- `injectionTime_delayed_second_injpoint::Float64`: -- time for delayed second injection point.
- `rate_delayed_inj_pt_func::Union{Function, Nothing}`: -- function for delayed injection point rate.
- `delayed_second_injpoint_Coordinates::Vector{Float64}`: -- coordinates of delayed second injection point.
- `delayed_second_injpoint_elem::Union{Vector{Int}, Nothing}`: -- elements for delayed second injection point.
- `delayed_second_injpoint_loc_func::Union{Function, Nothing}`: -- function for delayed second injection point location.
- `init_rate_delayed_second_injpoint::Float64`: -- initial rate for delayed second injection point.
"""
mutable struct InjectionProperties
    injectionRate::Matrix{Float64}
    sourceCoordinates::Vector{Float64}
    sourceElem::Vector{Int}
    sourceLocFunc::Union{Function, Nothing}
    sinkLocFunc::Union{Function, Nothing}
    sinkVelFunc::Union{Function, Nothing}
    sinkElem::Vector{Int}
    sinkVel::Vector{Float64}
    modelInjLine::Bool
    ILCompressibility::Union{Float64, Nothing}
    ILVolume::Union{Float64, Nothing}
    perforationFriction::Union{Float64, Nothing}
    initPressure::Union{Float64, Nothing}
    injectionRate_delayed_second_injpoint::Union{Float64, Nothing}
    injectionTime_delayed_second_injpoint::Union{Float64, Nothing}
    rate_delayed_inj_pt_func::Union{Function, Nothing}
    delayed_second_injpoint_Coordinates::Union{Vector{Float64}, Nothing}
    delayed_second_injpoint_elem::Union{Vector{Int}, Nothing}
    delayed_second_injpoint_loc_func::Union{Function, Nothing}
    init_rate_delayed_second_injpoint::Float64
end

"""
    InjectionProperties(rate, mesh, source_coordinates=nothing, source_loc_func=nothing, sink_loc_func=nothing,
                       sink_vel_func=nothing, model_inj_line=false, il_compressibility=nothing, il_volume=nothing,
                       perforation_friction=nothing, initial_pressure=nothing, rate_delayed_second_injpoint=nothing,
                       delayed_second_injpoint_loc=nothing, initial_rate_delayed_second_injpoint=nothing,
                       rate_delayed_inj_pt_func=nothing, delayed_second_injpoint_loc_func=nothing, check_cell_vertices=false)

The constructor of the InjectionProperties class.
ATTENTION: check_cell_vertices is new function.
"""
function InjectionProperties(rate::Union{Matrix{Float64}, Float64}, mesh::CartesianMesh, 
                           source_coordinates::Union{Vector{Float64}, Nothing}=nothing, 
                           source_loc_func::Union{Function, Nothing}=nothing, 
                           sink_loc_func::Union{Function, Nothing}=nothing,
                           sink_vel_func::Union{Function, Nothing}=nothing, 
                           model_inj_line::Bool=false, 
                           il_compressibility::Union{Float64, Nothing}=nothing, 
                           il_volume::Union{Float64, Nothing}=nothing,
                           perforation_friction::Union{Float64, Nothing}=nothing, 
                           initial_pressure::Union{Float64, Nothing}=nothing, 
                           rate_delayed_second_injpoint::Union{Matrix{Float64}, Nothing}=nothing,
                           delayed_second_injpoint_loc::Union{Vector{Float64}, Nothing}=nothing, 
                           initial_rate_delayed_second_injpoint::Union{Float64, Nothing}=nothing,
                           rate_delayed_inj_pt_func::Union{Function, Nothing}=nothing, 
                           delayed_second_injpoint_loc_func::Union{Function, Nothing}=nothing, 
                           check_cell_vertices::Bool=false)
    
    # Initialize variables that might not be set
    injectionRate_val = Matrix{Float64}(undef, 0, 0)
    injectionRate_delayed_second_injpoint_val = nothing
    injectionTime_delayed_second_injpoint_val = nothing
    rate_delayed_inj_pt_func_val = rate_delayed_inj_pt_func
    delayed_second_injpoint_Coordinates_val = nothing
    delayed_second_injpoint_elem_val = nothing
    delayed_second_injpoint_loc_func_val = delayed_second_injpoint_loc_func
    init_rate_delayed_second_injpoint_val = 0.0
    
    # check if the rate is provided otherwise throw an error
    if typeof(rate) == Matrix{Float64}
        if size(rate, 1) != 2
            error("Invalid injection rate. The list should have 2 rows (to specify time and" *
                  " corresponding injection rate) for each entry")
        elseif rate[1, 1] != 0.0
            error("The injection rate should start from zero second i.e. rate[1, 1] should" *
                  " be zero.")
        else
            injectionRate_val = rate
        end
    else
        injectionRate_val = [0.0 rate]
    end

    if rate_delayed_second_injpoint !== nothing && rate_delayed_inj_pt_func === nothing
        if typeof(rate_delayed_second_injpoint) == Matrix{Float64}
            if size(rate_delayed_second_injpoint, 1) != 2  # todo: check the condition
                error("Invalid injection rate of the delayed injection point. The list should have 2 rows (to specify time and" *
                      " corresponding injection rate) for each entry")
            elseif rate_delayed_second_injpoint[1, 1] == 0.0  # todo: check the condition
                error("The injection rate of the delayed injection point should start from a time >0 second i.e. rate[1, 1] should" *
                      " be nonzero.")
            end
            injectionRate_delayed_second_injpoint_val = rate_delayed_second_injpoint[2, 1]
            injectionTime_delayed_second_injpoint_val = rate_delayed_second_injpoint[1, 1]
            rate_delayed_inj_pt_func_val = nothing
        else
            error("Bad specification of the delayed injection point" *
                  " it should be a Matrix{Float64} prescribing the initial nonzero time of injection and the rate.")
        end
    else
        rate_delayed_inj_pt_func_val = rate_delayed_inj_pt_func
    end

    if initial_rate_delayed_second_injpoint !== nothing
        init_rate_delayed_second_injpoint_val = Float64(initial_rate_delayed_second_injpoint)
    else
        init_rate_delayed_second_injpoint_val = 0.0
    end

    sourceElem_val = Int[]
    
    if delayed_second_injpoint_loc !== nothing
        if typeof(delayed_second_injpoint_loc) == Vector{Float64}
            delayed_second_injpoint_Coordinates_val = delayed_second_injpoint_loc
            delayed_second_injpoint_elem_val = [locate_element(mesh, delayed_second_injpoint_Coordinates_val[1],
                                                              delayed_second_injpoint_Coordinates_val[2])]
            delayed_second_injpoint_loc_func_val = nothing
        else
            error("Bad specification of the delayed injection point" *
                  " it should be a Vector{Float64} prescribing the coordinates of the point.")
        end
    elseif delayed_second_injpoint_loc_func !== nothing
        delayed_second_injpoint_loc_func_val = delayed_second_injpoint_loc_func
        if length(sourceElem_val) == 0
            sourceElem_val = Int[]
            delayed_second_injpoint_elem_val = Int[]
        end
        for i in 1:mesh.NumberOfElts
            if delayed_second_injpoint_loc_func(mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2], mesh.hx, mesh.hy)
                push!(sourceElem_val, i)
                push!(delayed_second_injpoint_elem_val, i)
            end
        end
    else
        delayed_second_injpoint_elem_val = nothing
        delayed_second_injpoint_loc_func_val = nothing
    end

    if source_loc_func === nothing
        sourceCoordinates_val = Vector{Float64}(undef, 2)
        if source_coordinates !== nothing
            if length(source_coordinates) == 2
                @info "Setting the source coordinates to the closest cell center..."
                sourceCoordinates_val = source_coordinates
            else
                # error
                error("Invalid source coordinates. Correct format: a list or numpy array with a single" *
                      " row and two columns to \n specify x and y coordinate of the source e.g." *
                      " [x_coordinate, y_coordinate]")
            end
        else
            sourceCoordinates_val = [0.0, 0.0]
        end

        sourceElem_single = locate_element(mesh, sourceCoordinates_val[1], sourceCoordinates_val[2])
        if isnan(sourceElem_single)
            error("The given source location is out of the mesh!")
        end
        sourceElem_val = [sourceElem_single]
        sourceCoordinates_val = mesh.CenterCoor[sourceElem_single, :]
        @info "Injection point: (x, y) = ($(mesh.CenterCoor[sourceElem_single, 1]), $(mesh.CenterCoor[sourceElem_single, 2]))"
        sourceLocFunc_val = nothing
    else
        sourceLocFunc_val = source_loc_func
        sourceElem_val = Int[]
        for i in 1:mesh.NumberOfElts
            inside = false
            if check_cell_vertices
                # Check all vertices
                vertices = mesh.VertexCoor[mesh.Connectivity[i, :], :]
                for v in 1:size(vertices, 1)
                    if source_loc_func(vertices[v, 1], vertices[v, 2])
                        inside = true
                        break
                    end
                end
            else
                # Previous logic
                inside = source_loc_func(mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2])
            end
            if inside
                push!(sourceElem_val, i)
            end
        end
    end
    
    if delayed_second_injpoint_elem_val !== nothing && !all(elem in sourceElem_val for elem in delayed_second_injpoint_elem_val)
        error("The delayed injection points elements are not contained in the list of all the injection elements")
    end

    if length(sourceElem_val) == 0
        error("No source element found!")
    end
    
    sourceCoordinates_val = [mean(mesh.CenterCoor[sourceElem_val, 1]),
                            mean(mesh.CenterCoor[sourceElem_val, 2])]

    sinkLocFunc_val = sink_loc_func
    sinkVelFunc_val = sink_vel_func
    sinkElem_val = Int[]
    sinkVel_val = Float64[]
    
    if sink_loc_func !== nothing
        if sink_vel_func === nothing
            error("Sink velocity function is required for sink elements!")
        end

        for i in 1:mesh.NumberOfElts
            if sink_loc_func(mesh.CenterCoor[i, 1], mesh.CenterCoor[i, 2])
                push!(sinkElem_val, i)
            end
        end

        sinkVel_val = Vector{Float64}(undef, length(sinkElem_val))
        for i in 1:length(sinkElem_val)
            sinkVel_val[i] = sink_vel_func(mesh.CenterCoor[sinkElem_val[i], 1],
                                          mesh.CenterCoor[sinkElem_val[i], 2])
        end
    end

    modelInjLine_val = model_inj_line
    ILCompressibility_val = nothing
    ILVolume_val = nothing
    perforationFriction_val = nothing
    initPressure_val = nothing
    
    if model_inj_line
        if il_compressibility !== nothing
            ILCompressibility_val = il_compressibility
        else
            error("Injection line compressibility is required!")
        end
        if il_volume !== nothing
            ILVolume_val = il_volume
        else
            error("Injection line volume is required!")
        end
        if perforation_friction !== nothing
            perforationFriction_val = perforation_friction
        else
            error("Perforation friction is required!")
        end
        if initial_pressure !== nothing
            initPressure_val = initial_pressure
        else
            error("initial pressure of the injection line is required!")
        end
    end
    
    # Create InjectionProperties object
    return InjectionProperties(
        injectionRate_val,
        sourceCoordinates_val,
        sourceElem_val,
        sourceLocFunc_val,
        sinkLocFunc_val,
        sinkVelFunc_val,
        sinkElem_val,
        sinkVel_val,
        modelInjLine_val,
        ILCompressibility_val,
        ILVolume_val,
        perforationFriction_val,
        initPressure_val,
        injectionRate_delayed_second_injpoint_val,
        injectionTime_delayed_second_injpoint_val,
        rate_delayed_inj_pt_func_val,
        delayed_second_injpoint_Coordinates_val,
        delayed_second_injpoint_elem_val,
        delayed_second_injpoint_loc_func_val,
        init_rate_delayed_second_injpoint_val,
        check_cell_vertices
    )
    end

    # -------------------------------------------------------------------------------------------------------------------

    """
        get_injection_rate(self, tm, frac)

    This function gives the current injection rate at all of the cells in the domain.

    # Arguments
    - `self::InjectionProperties`: -- the InjectionProperties object.
    - `tm::Float64`:               -- the time at which the injection rate is required.
    - `frac::Fracture`:            -- the Fracture object containing the mesh and the current fracture elements.

    # Returns
    - `Qin::Vector{Float64}`:      -- a vector of the size of the mesh with injection rates in each of the cell.
    """
    function get_injection_rate(self::InjectionProperties, tm::Float64, frac::Fracture)::Vector{Float64}
        Qin = zeros(Float64, frac.mesh.NumberOfElts)
        indxCurTime = maximum(findall(tm .>= self.injectionRate[1, :]))
        currentRate = self.injectionRate[2, indxCurTime]  # current injection rate
        currentSource = intersect(self.sourceElem, frac.EltChannel)
        if length(currentSource) > 0
            Qin[currentSource] .= currentRate / length(currentSource)
        end

        return Qin
    end

    # -------------------------------------------------------------------------------------------------------------------

    """
    remesh!(self, new_mesh, old_mesh)

    This function is called every time the domain is remeshed.

    # Arguments
    - `self::InjectionProperties`: -- the InjectionProperties object.
    - `new_mesh::CartesianMesh`:   -- the CartesianMesh object describing the new coarse mesh.
    - `old_mesh::CartesianMesh`:   -- the CartesianMesh object describing the old mesh.
    """
    function remesh!(self::InjectionProperties, new_mesh::CartesianMesh, old_mesh::CartesianMesh)
        # update source elements according to the new mesh.
        if self.sourceLocFunc === nothing
            actv_cells = Set{Int}()
            for i in self.sourceElem
                push!(actv_cells, Int(locate_element(new_mesh, old_mesh.CenterCoor[i, 1], old_mesh.CenterCoor[i, 2])))
            end
            self.sourceElem = collect(actv_cells)
        else
            self.sourceElem = Int[]
            for i in 1:new_mesh.NumberOfElts
                if self.sourceLocFunc(new_mesh.CenterCoor[i, 1], new_mesh.CenterCoor[i, 2])
                    push!(self.sourceElem, i)
                end
            end
        end

        self.sourceCoordinates = Float64[]
        source_coords_x = Float64[]
        source_coords_y = Float64[]
        for elem in self.sourceElem
            push!(source_coords_x, new_mesh.CenterCoor[elem, 1])
            push!(source_coords_y, new_mesh.CenterCoor[elem, 2])
        end
        self.sourceCoordinates = [mean(source_coords_x), mean(source_coords_y)]

        if self.delayed_second_injpoint_loc_func !== nothing
            if length(self.sourceElem) == 0
                self.sourceElem = Int[]
                self.delayed_second_injpoint_elem = Int[]
            end
            for i in 1:new_mesh.NumberOfElts
                if self.delayed_second_injpoint_loc_func(new_mesh.CenterCoor[i, 1], new_mesh.CenterCoor[i, 2],
                                                    new_mesh.hx, new_mesh.hy)
                    push!(self.sourceElem, i)
                    push!(self.delayed_second_injpoint_elem, i)
                end
            end
        else
            self.delayed_second_injpoint_elem = nothing
        end

        if self.sinkLocFunc !== nothing
            self.sinkElem = Int[]
            for i in 1:new_mesh.NumberOfElts
                if self.sinkLocFunc(new_mesh.CenterCoor[i, 1], new_mesh.CenterCoor[i, 2])
                    push!(self.sinkElem, i)
                end
            end

            self.sinkVel = Vector{Float64}(undef, length(self.sinkElem))
            for i in 1:length(self.sinkElem)
                self.sinkVel[i] = self.sinkVelFunc(new_mesh.CenterCoor[self.sinkElem[i], 1],
                                                new_mesh.CenterCoor[self.sinkElem[i], 2])
            end
        end
    end
end


# ----------------------------------------------------------------------------------------------------------------------


"""
    LoadingProperties

Class defining the mechanical loading properties.

# Attributes
- `EltLoaded::Vector{Int}`: -- array of elements that are loaded.
- `displRate::Float64`:     -- the rate at which the elements in the EltLoaded list are displaced due to the
                               applied mechanical loading.
"""
mutable struct LoadingProperties
    EltLoaded::Vector{Int}
    displRate::Float64
end

"""
    LoadingProperties(displ_rate=0.0, loaded_elts=nothing)

The constructor of the LoadingProperties class.

# Arguments
- `displ_rate::Float64`:    -- the rate at which the elements in the EltLoaded list are displaced due to the
                               applied mechanical loading.
- `loaded_elts::Vector{Int}`: -- array of elements that are loaded.
"""
function LoadingProperties(displ_rate::Float64=0.0, loaded_elts::Union{Vector{Int}, Nothing}=nothing)
    if loaded_elts !== nothing
        EltLoaded_val = loaded_elts
    else
        error("The loaded elements should be given in the form a Vector{Int} of integers.")
    end
    
    return LoadingProperties(EltLoaded_val, displ_rate)
end


# ----------------------------------------------------------------------------------------------------------------------

"""
    SimulationProperties

Class defining the simulation properties.

# Arguments
- `address::Union{String, Nothing}`: -- the folder where the simulation parameters file is located. The file must be
                                        named 'simul_param'. For the description of the arguments and there default
                                        values, see the module default_parameters.

# Attributes
- `tolFractFront::Float64`:       -- tolerance for the fracture front loop.
- `toleranceEHL::Float64`:        -- tolerance for the Elastohydrodynamic solver.
- `toleranceVStagnant::Float64`:  -- tolerance on the velocity to decide if a cell is stagnant.
- `toleranceProjection::Float64`: -- tolerance for projection iteration for anisotropic case
- `maxFrontItrs::Int`:            -- maximum iterations to for the fracture front loop.
- `maxSolverItrs::Int`:           -- maximum iterations for the EHL iterative solver (Picard-Newton hybrid) in this
                                        case.
- `maxProjItrs::Int`:             -- maximum iterations for the loop to find projection on the front from ribbon.
- `tmStpPrefactor::Float64`:      -- factor for time-step adaptivity.
- `maxTimeSteps::Int`:            -- maximum number of time steps.
- `finalTime::Union{Float64, Nothing}`: -- time where the simulation ends.
- `timeStepLimit::Union{Float64, Nothing}`: -- limit above which time step will not exceed.
- `fixedTmStp::Union{Matrix{Float64}, Float64, Nothing}`: -- a float or an array giving the fixed time step. The array should have two rows,
                                        with the first row giving the time at which the time step would change and the
                                        second row giving the corresponding time step. If None is given as time step,
                                        appropriate time step would be calculated.
- `maxReattempts::Int`:           -- maximum number of reattempts in case of failure of a time step. A smaller
                                        time step will be attempted the given number of times.
- `reAttemptFactor::Float64`:     -- the factor by which time step is reduced on reattempts.
- `plotFigure::Bool`:             -- flag specifying to plot fracture trace after the given time period.
- `saveToDisk::Bool`:             -- flag specifying to save fracture to dist after the given time period.
- `plotAnalytical::Bool`:         -- if true, analytical solution will also be plotted along with the computed
                                       solution.
- `analyticalSol::Union{String, Nothing}`: -- the analytical solution of the radial fracture to be plotted on the fracture.
                                        Possible options:
                                            - K  (toughness dominated regime, without leak off)
                                            - Kt (toughness dominated regime , with leak off)
                                            - M  (viscosity dominated regime, without leak off)
                                            - Mt (viscosity dominated regime , with leak off)
                                            - E  (elliptical, toughness dominated without leak off)
- `bckColor::Union{String, Nothing}`: -- the string specifying the parameter according to which the background of the
                                        domain is color coded. Possible options:
                                            - sigma0 or confining stress
                                            - K1c or fracture toughness
                                            - Cl or leak-off coefficient
- `plotTimePeriod::Union{Float64, Nothing}`: -- the time period after which the figures are plotted during simulation.
- `blockFigure::Bool`:            -- if True, the plotted figure(s) will be blocked after every time they are
                                        plotted. The simulation will advance when any key will be pressed from keyboard.
- `plotTSJump::Int`:              -- the number of time steps after which the variables given in plotVar attribute
                                        are plotted. E.g. a value of 4 will result in plotting every four time steps.
- `plotVar::Vector{String}`:      -- a list of variable(s) to be plotted during simulation. The time / time steps
                                        after which the output is done can be controlled with a number of parameters (
                                        see above).
- `saveTimePeriod::Union{Float64, Nothing}`: -- the time period after which the results are saved to disk during simulation.
- `saveTSJump::Int`:              -- the number of time steps after which the results are saved to disk, e.g. a value
                                        of 4 will result in plotting every four time steps.
- `elastohydrSolver::String`:     -- the type of solver to solve the elasto-hydrodynamic system. At the moment, two
                                        main solvers can be specified.
                                            - 'implicit_Picard'
                                            - 'implicit_Anderson'
                                            - 'RKL2'
- `substitutePressure::Bool`:     -- a flag specifying the solver to be used. If True, the pressure will be
                                        substituted in the channel elements (see Zia and Lecampion, 2019).
- `solveDeltaP::Bool`:            -- a flag specifying the solver to be used. If True, the change in pressure,
                                        instead of pressure will be solved in the tip cells and the cells where the
                                        width constraint is active (see Zia and Lecampion, 2019).
- `solveStagnantTip::Bool`:       -- if True, the stagnant tip cells will also be solved for width. This may result
                                        in more stable pressure as the elasticity equation will also be solved in those
                                        cells.
- `solveTipCorrRib::Bool`:        -- if True, the tip cells corresponding to the closed ribbon cells will also be
                                        considered as closed and the width will be imposed on them.
- `solveSparse::Union{Bool, Nothing}`: -- if True, the fluid conductivity matrix will be made with sparse matrix.
- `saveRegime::Bool`:             -- if True, the regime of the propagation as observed in the ribbon cell (see Zia
                                        and Lecampion 2018, IJF) will be saved.
- `verbositylevel::String`:       -- the level of details about the ongoing simulation to be written on the log file
                                        (currently the levels 'debug', 'info', 'warning' and 'error' are supported).
- `log2file::Bool`:               -- True if you want to log to a file, otherwise set it to false
- `enableRemeshing::Bool`:        -- if True, the computational domain will be compressed by the factor given by
                                        by the variable remeshFactor after the fracture front reaches the end of the
                                        domain.
- `remeshFactor::Float64`:        -- the factor by which the domain is compressed on re-meshing.
- `meshExtension::Vector{Bool}`:  -- an array of booleans defining if the mesh should be extended in the given
                                        direction or if it should get compressed. The distribution is bottom, top,
                                        left, right
- `meshExtensionFactor::Vector{Float64}`: -- factor by which the current mesh is extended in the extension direction
- `meshExtensionAllDir::Bool`:    -- allow the mesh to extend in all directions
- `frontAdvancing::String`:       -- The type of front advancing to be done. Possible options are:
                                            - 'explicit'
                                            - 'predictor-corrector'
                                            - 'implicit'
- `gravity::Bool`:                -- if True, the effect of gravity will be taken into account.
- `collectPerfData::Bool`:        -- if True, the performance data will be collected in the form of a tree.
- `paramFromTip::Bool`:           -- if True, the space dependant parameters such as toughness and leak-off
                                        coefficients will be taken from the tip by projections instead of taking them
                                        from the ribbon cell center. The numerical scheme as a result will become
                                        unstable due to the complexities in finding the projection
- `saveReynNumb::Bool`:           -- if True, the Reynold's number at each edge of the cells inside the fracture
                                        will be saved.
- `saveFluidFlux::Bool`:          -- if True, the fluid flux at each edge of the cells inside the fracture
                                        will be saved.
- `saveFluidVel::Bool`:           -- if True, the fluid velocity at each edge of the cells inside the fracture
                                        will be saved.
- `saveEffVisc::Bool`:            -- if True, the Newtonian equivalent viscosity of the non-Newtonian fluid will
                                        be saved.
- `saveG::Bool`:                  -- if True, the coefficient G (see Zia et al. 2020) would be saved for the non-Newtonian fluid
- `TI_KernelExecPath::String`:    -- the folder containing the executable to calculate transverse isotropic
                                       kernel or kernel with free surface.
- `explicitProjection::Bool`:     -- if True, direction from last time step will be used to evaluate TI parameters.
- `symmetric::Bool`:              -- if True, the four quadrant of the domain will be considered symmetric and only
                                        one will be solved for. The rest will be replaced by its reflection along the x
                                        and y axes.

                                        Attention:
                                            The symmetric fracture is only implemented in the toughness dominated case.
                                            Use full domain if viscous fluid is injected.
- `enableGPU::Bool`:              -- if True, the dense matrix vector product for the RKL scheme would be done using
                                        GPU. If False, multithreaded dot product implemented in the explicit_RKL module
                                        will be used to do it.
- `nThreads::Int`:                -- The number of threads to be used for the dense matrix dot product in the RKL
                                        scheme. By default set to 4.
- `projMethod::String`:           -- the method by which the angle prescribed by the projections onto the front
                                        are evaluated. Possible options are:
                                            - 'ILSA_orig' (the method described in the original ILSA scheme).
                                            - 'LS_grad' (using gradient of the level set).
- `height::Union{Float64, Nothing}`: -- this parameters is only used in the case of height contained hydraulic fracture
                                       plots, e.g. to plot analytical solutions of PKN and KGD fractures.
- `aspectRatio::Union{Float64, Nothing}`: -- this parameters is only used in the case of elliptical hydraulic fracture
                                       plots, e.g. to plot analytical solutions of anisotropic toughness or TI
                                       elasticity.
- `maxReattemptsFracAdvMore2Cells::Int`: -- number of time reduction that are made if the fracture is advancing more than two cells (e.g. because of an heterogeneity)
- `forceTmStpLmtANDLmtAdvTo2cells::Bool`: -- this will force the contemporaneity of timeStepLimit and limitAdancementTo2cells

# Attention:
# These attributes below are private:

- `__outputAddress::Union{String, Nothing}`: -- disk address of the files to be saved. If not given, a new
                                            ./Data/"tim stamp" folder will be automatically created.
- `__outputFolder::String`:       -- the full path to the output folder.
- `__solTimeSeries::Union{Vector{Float64}, Nothing}`: -- time series where the solution is required. The time stepping would
                                            be adjusted to get solution exactly at the given times.
- `__dryCrack_mechLoading::Bool`: -- if True, the mechanical loading solver will be used.
- `__viscousInjection::Bool`:     -- if True, the the solver will also take the fluid viscosity into
                                            account.
- `__volumeControl::Bool`:        -- if True, the the volume control solver will be used.
- `__simName::Union{String, Nothing}`: -- the name of the simulation.
- `__timeStamp::String`:          -- the time at which the simulation properties was created.
- `__tipAsymptote::String`:       -- the tip asymptote to be used. Can be modified with the provided function.
"""
mutable struct SimulationProperties
    # tolerances
    tolFractFront::Float64
    toleranceEHL::Float64
    toleranceProjection::Float64
    toleranceVStagnant::Float64
    HershBulkEpsilon::Float64
    HershBulkGmin::Float64

    # max iterations
    maxFrontItrs::Int
    maxSolverItrs::Int
    maxProjItrs::Int

    # time and time stepping
    maxTimeSteps::Int
    tmStpPrefactor::Float64
    finalTime::Union{Float64, Nothing}
    __solTimeSeries::Union{Vector{Float64}, Nothing}
    timeStepLimit::Union{Float64, Nothing}
    fixedTmStp::Union{Matrix{Float64}, Float64, Nothing}

    # time step re-attempt
    maxReattempts::Int
    reAttemptFactor::Float64
    maxReattemptsFracAdvMore2Cells::Int

    # output parameters
    plotFigure::Bool
    plotAnalytical::Bool
    analyticalSol::Union{String, Nothing}
    __simName::Union{String, Nothing}
    __outputAddress::Union{String, Nothing}
    __outputFolder::String
    saveToDisk::Bool
    bckColor::Union{String, Nothing}
    blockFigure::Bool
    plotTSJump::Int
    plotTimePeriod::Union{Float64, Nothing}
    plotVar::Vector{String}
    saveTSJump::Int
    saveTimePeriod::Union{Float64, Nothing}
    plotATsolTimeSeries::Bool

    # solver type
    elastohydrSolver::String
    Anderson_parameter::Int
    relaxation_factor::Float64
    __dryCrack_mechLoading::Bool
    __viscousInjection::Bool
    __volumeControl::Bool
    substitutePressure::Bool
    solveDeltaP::Bool
    solveStagnantTip::Bool
    solveTipCorrRib::Bool
    solveSparse::Union{Bool, Nothing}

    # miscellaneous
    useBlockToeplizCompression::Bool
    verbositylevel::String
    log2file::Bool
    __tipAsymptote::String
    saveRegime::Bool
    enableRemeshing::Bool
    remeshFactor::Float64

    meshExtension::Vector{Bool}
    meshExtensionFactor::Vector{Float64}
    meshExtensionAllDir::Bool
    maxElementIn::Float64
    maxCellSize::Float64
    meshReductionFactor::Float64
    meshReductionPossible::Bool
    limitAdancementTo2cells::Bool
    forceTmStpLmtANDLmtAdvTo2cells::Bool
    frontAdvancing::String
    collectPerfData::Bool
    paramFromTip::Bool
    saveReynNumb::Bool
    gravity::Bool
    TI_KernelExecPath::String
    saveFluidFlux::Bool
    saveFluidVel::Bool
    saveFluidFluxAsVector::Bool
    saveFluidVelAsVector::Bool
    saveEffVisc::Bool
    saveStatisticsPostCoalescence::Bool
    saveYieldRatio::Bool
    saveG::Bool
    explicitProjection::Bool
    symmetric::Bool
    doublefracture::Bool
    projMethod::String
    enableGPU::Bool
    nThreads::Int

    # fracture geometry to calculate analytical solution for plotting
    height::Union{Float64, Nothing}
    aspectRatio::Union{Float64, Nothing}

    # parameter deciding to save the leak-off tip parameter
    saveChi::Bool

    # roughness parameters
    roughness_model::Union{String, Nothing}
    roughness_sigma::Union{Float64, Nothing}

    __timeStamp::String
end

"""
    SimulationProperties(address=nothing)

The constructor of the SimulationProperties class. See documentation of the class.
"""
function SimulationProperties(address::Union{String, Nothing}=nothing)

    param_module = if address === nothing
        default_parameters
    else
        user_module = Module(:UserSimParams)
        full_path = joinpath(address, "simul_param.jl") 
        try
            Base.include(user_module, full_path)
            user_module
        catch e
            @error "Failed to load parameters from $full_path" exception=(e, catch_backtrace())
            @info "Falling back to default parameters"
            default_parameters
        end
    end

    timestamp = string(Dates.now())

    sim_prop = SimulationProperties(
        # tolerances
        param_module.toleranceFractureFront,
        param_module.toleranceEHL,
        param_module.tol_projection,
        param_module.toleranceVStagnant,
        param_module.Hersh_Bulk_epsilon,
        param_module.Hersh_Bulk_Gmin,

        # max iterations
        param_module.max_front_itrs,
        param_module.max_solver_itrs,
        param_module.max_proj_Itrs,

        # time and time stepping
        param_module.maximum_steps,
        param_module.tmStp_prefactor,
        param_module.final_time, # Union{Float64, Nothing}
        param_module.req_sol_at, # Union{Vector{Float64}, Nothing}
        param_module.timeStep_limit, # Union{Float64, Nothing}
        param_module.fixed_time_step, # Union{Matrix{Float64}, Float64, Nothing}

        # time step re-attempt
        param_module.max_reattemps,
        param_module.reattempt_factor,
        param_module.max_reattemps_FracAdvMore2Cells,

        # output parameters
        param_module.plot_figure,
        param_module.plot_analytical,
        param_module.analytical_sol, # Union{String, Nothing}
        param_module.sim_name, # Union{String, Nothing}
        param_module.output_folder, # Union{String, Nothing}
        "",
        param_module.save_to_disk,
        param_module.bck_color, # Union{String, Nothing}
        param_module.block_figure,
        param_module.plot_TS_jump,
        param_module.plot_time_period, # Union{Float64, Nothing}
        param_module.plot_var,
        param_module.save_TS_jump,
        param_module.save_time_period, # Union{Float64, Nothing}
        param_module.plot_at_sol_time_series,

        # solver type
        param_module.elastohydr_solver,
        param_module.m_Anderson,
        param_module.relaxation_param,
        param_module.mech_loading,
        param_module.viscous_injection,
        param_module.volume_control,
        param_module.substitute_pressure,
        param_module.solve_deltaP,
        param_module.solve_stagnant_tip,
        param_module.solve_tip_corr_rib,
        param_module.solve_sparse, # Union{Bool, Nothing}

        # miscellaneous
        param_module.use_block_toepliz_compression,
        param_module.verbosity_level,
        param_module.log_to_file,
        param_module.tip_asymptote,
        param_module.save_regime,
        param_module.enable_remeshing,
        param_module.remesh_factor,

        param_module.mesh_extension_direction, # Vector{Bool}
        param_module.mesh_extension_factor, # Vector{Float64}
        param_module.mesh_extension_all_sides,
        Inf, # maxElementIn
        Inf, # maxCellSize
        param_module.mesh_reduction_factor,
        true, # meshReductionPossible
        param_module.limit_Adancement_To_2_cells,
        param_module.force_time_step_limit_and_max_adv_to_2_cells,
        param_module.front_advancing,
        param_module.collect_perf_data,
        param_module.param_from_tip,
        param_module.save_ReyNumb,
        param_module.gravity,
        param_module.TI_Kernel_exec_path,
        param_module.save_fluid_flux,
        param_module.save_fluid_vel,
        param_module.save_fluid_flux_as_vector,
        param_module.save_fluid_vel_as_vector,
        param_module.save_effective_viscosity,
        param_module.save_statistics_post_coalescence,
        param_module.save_yield_ratio,
        param_module.save_G,
        param_module.explicit_projection,
        param_module.symmetric,
        param_module.double_fracture_vol_contr,
        param_module.proj_method,
        param_module.enable_GPU,
        param_module.n_threads,

        # fracture geometry
        param_module.height, # Union{Float64, Nothing}
        param_module.aspect_ratio, # Union{Float64, Nothing}

        # parameter deciding to save the leak-off tip parameter
        param_module.save_chi,

        # roughness parameters
        param_module.roughness_model, # Union{String, Nothing}
        param_module.roughness_sigma, # Union{Float64, Nothing}
        
        timestamp # __timeStamp
    )

    # --- Validation ---
    # Валидация fixedTmStp
    if sim_prop.fixedTmStp !== nothing && isa(sim_prop.fixedTmStp, Matrix{Float64})
        if size(sim_prop.fixedTmStp, 1) != 2
            error("Invalid fixed time step. The list should have 2 rows (to specify time and " *
                  "corresponding time step) for each entry")
        end
    end

    # Валидация paramFromTip
    if sim_prop.paramFromTip
        error("Parameters from tip not yet supported!")
    end

    # Валидация projMethod
    if !(sim_prop.projMethod in ["ILSA_orig", "LS_grad", "LS_continousfront"])
        error("Projection method is not recognised!")
    end

    # Валидация doublefracture/projMethod
    if sim_prop.projMethod != "LS_continousfront" && sim_prop.doublefracture
        error("You set the option doublefracture=True but\n " *
              "The volume control solver has been implemented \n" *
              "only with the option projMethod==LS_continousfront activated")
    end

    return sim_prop
end

# ----------------------------------------------------------------------------------------------------------------------
function set_logging_to_file(self::SimulationProperties, address::String, verbosity_level::Union{String, Nothing}=nothing)

    level_str = something(verbosity_level, self.verbositylevel)
    
    file_level = if level_str == "debug"
        Logging.Debug
    elseif level_str == "info"
        Logging.Info
    elseif level_str == "warning" 
        Logging.Warn
    elseif level_str == "error"
        Logging.Error
    elseif level_str == "critical"
        Logging.Error
    else
        @warn "Unknown verbosity level '$level_str', defaulting to Info level."
        Logging.Info
    end


    slash = Sys.iswindows() ? "\\" : "/"
    log_dir = endswith(address, slash) ? address : address * slash
    log_file_path = log_dir * "PyFrac_log.txt"
    
    try
        log_io = open(log_file_path, "w")
        file_logger = FileLogger(log_io)
        pyfrac_filtered_logger = EarlyFilteredLogger(
            (log_args) -> begin
                group_name = getfield(log_args.group, :name)
                startswith(string(group_name), "PyFrac")
            end,
            file_logger
        )
        
        min_level_logger = MinLevelLogger(file_level, pyfrac_filtered_logger)
        current_console_logger = global_logger()
        tee_logger = TeeLogger(current_console_logger, min_level_logger)
        global_logger(tee_logger)
        self.log2file = true
        @info "Log file set up" _group="PyFrac.general"
        
    catch e
        @error "Failed to set up log file at $log_file_path" exception=(e, catch_backtrace()) _group="PyFrac.general"
        self.log2file = false
    end
    
    return nothing
end

# ----------------------------------------------------------------------------------------------------------------------
# setter and getter functions

"""
    set_tipAsymptote(self, tip_asymptote)

The function to set up the tip asymptote.

# Arguments
- `tip_asymptote::String`: propagation regime. possible options are:
    - K  (toughness dominated regime, without leak off)
    - M  (viscosity dominated regime, without leak off)
    - Mt (viscosity dominated regime , with leak off)
    - U  (Universal regime accommodating viscosity, toughness and leak off (see Donstov and Pierce, 2017), 0 order)
    - U1  (Universal regime accommodating viscosity, toughness and leak off (see Donstov and Pierce, 2017), delta correction)
    - MK (viscosity to toughness transition regime)
    - MDR (Maximum drag reduction asymptote, see Lecampion & Zia 2019)
    - M_MDR (Maximum drag reduction asymptote in viscosity storage regime, see Lecampion & Zia 2019)
    - HBF or HBF_aprox (Herschel-Bulkley fluid, see Bessmertnykh and Dontsov 2019; the tip volume is evaluated with a fast approximation)
    - HBF_num_quad (Herschel-Bulkley fluid, see Bessmertnykh and Dontsov 2019; the tip volume is evaluated with numerical quadrature of the approximate function, which makes it very slow)
    - PLF or PLF_aprox (power law fluid, see Dontsov and Kresse 2017; the tip volume is evaluated with a fast approximation)
    - PLF_num_quad (power law fluid, see Dontsov and Kresse 2017; the tip volume is evaluated with numerical quadrature of the approximate function, which makes it very slow)
    - PLF_M (power law fluid in viscosity storage regime; see Desroche et al.)

# Returns
- `Nothing`
"""
function set_tipAsymptote(self::SimulationProperties, tip_asymptote::String)
    tipAssymptOptions = ["K", "M", "Mt", "U", "U1", "MK", "MDR", "M_MDR", "HBF", "HBF_aprox",
                         "HBF_num_quad", "PLF", "PLF_aprox", "PLF_num_quad", "PLF_M"]
    if tip_asymptote in tipAssymptOptions  # check if tip asymptote matches any option
        self.__tipAsymptote = tip_asymptote
    else  # error
        error("Invalid tip asymptote. Possible options: " * string(tipAssymptOptions))
    end
    
    return nothing
end

function get_tipAsymptote(self::SimulationProperties)::String
    return self.__tipAsymptote
end

function set_viscousInjection(self::SimulationProperties, visc_injection::Bool)
    self.__viscousInjection = visc_injection
    if visc_injection
        self.__volumeControl = false
        self.__dryCrack_mechLoading = false
    end
    return nothing
end

function get_viscousInjection(self::SimulationProperties)::Bool
    return self.__viscousInjection
end

function set_volumeControl(self::SimulationProperties, vol_control::Bool)
    self.__volumeControl = vol_control
    if vol_control
        self.__viscousInjection = false
        self.__dryCrack_mechLoading = false
    end
    return nothing
end

function get_volumeControl(self::SimulationProperties)::Bool
    return self.__volumeControl
end

function set_dryCrack_mechLoading(self::SimulationProperties, mech_loading::Bool)
    self.__dryCrack_mechLoading = mech_loading
    if mech_loading
        self.__viscousInjection = false
        self.__volumeControl = false
    end
    return nothing
end

function get_dryCrack_mechLoading(self::SimulationProperties)::Bool
    return self.__dryCrack_mechLoading
end

function set_outputFolder(self::SimulationProperties, output_address::Union{String, Nothing})
    # check operating system to get appropriate slash in the address
    slash = Sys.iswindows() ? "\\" : "/"
    
    if output_address !== nothing
        self.saveToDisk = true
        
        if output_address[end] == slash
            output_address = output_address[1:end-1]
        end
        
        self.__outputAddress = output_address
        self.__outputFolder = output_address * slash * get_simulation_name(self) * slash
    else
        self.__outputAddress = output_address
        self.__outputFolder = "." * slash * "_simulation_data_PyFrac" * slash * get_simulation_name(self) * slash
    end
    
    return nothing
end

function get_outputFolder(self::SimulationProperties)::String
    return self.__outputFolder
end

function set_solTimeSeries(self::SimulationProperties, sol_t_srs::Union{Matrix{Float64}, Nothing})
    if sol_t_srs === nothing
        self.__solTimeSeries = nothing
    elseif typeof(sol_t_srs) == Matrix{Float64}
        self.__solTimeSeries = sol_t_srs
    else
        error("The given solution time series is not a Matrix{Float64}!")
    end
    return nothing
end

function get_solTimeSeries(self::SimulationProperties)::Union{Matrix{Float64}, Nothing}
    return self.__solTimeSeries
end

"""
    set_simulation_name(self, simul_name)

Set the simulation name.

# Arguments
- `simul_name::Union{String, Nothing}`: The simulation name.
"""
function set_simulation_name(self::SimulationProperties, simul_name::Union{String, Nothing})
    time_stmp = Dates.format(Dates.now(), "yyyy-mm-dd__HH_MM_SS")
    
    if simul_name === nothing
        self.__simName = "simulation" * "__" * time_stmp
    else
        if typeof(simul_name) != String
            error("The given simulation name is not a string")
        else
            self.__simName = simul_name
        end
    end
    
    self.__timeStamp = string(Dates.now())
    
    try
        set_outputFolder(self, self.__outputAddress)
    catch e
        # Pass silently if __outputAddress is not yet defined
    end
    
    return nothing
end

function get_simulation_name(self::SimulationProperties)::String
    return self.__simName
end

function get_timeStamp(self::SimulationProperties)::String
    return self.__timeStamp
end

function get_time_step_prefactor(self::SimulationProperties, t::Float64)::Float64
    if typeof(self.tmStpPrefactor) == Matrix{Float64}
        if size(self.tmStpPrefactor, 1) == 2
            indxCurTime = maximum(findall(t .>= self.tmStpPrefactor[1, :]))
            return self.tmStpPrefactor[2, indxCurTime]  # current pre-factor
        else
            error("The time step pre-factor array should have two rows, where each column giving the " *
                  "time at which the pre-factor is changed in the first row, and the corresponding " *
                  "pre-factor in the second row.")
        end
    else
        return Float64(self.tmStpPrefactor)
    end
end

"""
    set_mesh_extension_direction(self, direction)

The function to set up in which directions the mesh should be extended.

# Arguments
- `direction::Vector{String}`: Direction where the mesh should be extended:
    - "top" (mesh extension towards positive y)
    - "bottom" (mesh extension towards negative y)
    - "left" (mesh extension towards negative x)
    - "right" (mesh extension towards positive x)
    - "vertical" (mesh extension up and down)
    - "horizontal" (mesh extension left and right)
    - "all" (extend the mesh in all directions)
"""
function set_mesh_extension_direction(self::SimulationProperties, direction::Vector{String})
    for i in direction
        if i == "vertical"
            self.meshExtension[1:2] = [true, true]
        elseif i == "horizontal"
            self.meshExtension[3:4] = [true, true]
        elseif i == "top"
            self.meshExtension[2] = true
        elseif i == "bottom"
            self.meshExtension[1] = true
        elseif i == "left"
            self.meshExtension[3] = true
        elseif i == "right"
            self.meshExtension[4] = true
        elseif i == "all"
            self.meshExtension = [true, true, true, true]
        else
            error("Invalid mesh extension definition Possible options: top, bottom, left, right, vertical, " *
                  "horizontal or all")
        end
    end
    return nothing
end

function get_mesh_extension_direction(self::SimulationProperties)::Vector{Bool}
    return self.meshExtension
end

"""
    set_mesh_extension_factor(self, ext_factor)

The function to set up the factor deciding on the number of elements to add in the corresponding direction.

# Arguments
- `ext_factor::Union{Float64, Vector{Float64}}`: The factor either given:
    - a float: all directions extend by the same amount
    - a list with two entries: the first gives the factor in x the second in y direction
    - a list with four entries: the entries respectively correspond to 'left', 'right', 'bottom', 'top'.
"""
function set_mesh_extension_factor(self::SimulationProperties, ext_factor::Union{Float64, Vector{Float64}})
    if typeof(ext_factor) == Float64
        self.meshExtensionFactor = [ext_factor, ext_factor, ext_factor, ext_factor]
    elseif length(ext_factor) == 2
        self.meshExtensionFactor = [ext_factor[2], ext_factor[2], ext_factor[1], ext_factor[1]]
    elseif length(ext_factor) == 4
        self.meshExtensionFactor = [ext_factor[3], ext_factor[4], ext_factor[1], ext_factor[2]]
    else
        error("The given form of the factor is not supported. Either give a common factor (Float64) a " *
              "vector of two entries (x and y direction) or a vector of four entries " *
              "['bottom', 'top', 'left', 'right']")
    end
    return nothing
end

function get_mesh_extension_factor(self::SimulationProperties)::Vector{Float64}
    return self.meshExtensionFactor
end

# -----------------------------------------------------------------------------------------------------------------------

"""
    IterationProperties

This class stores performance data in the form of a tree.

# Arguments
- `itr_type::String`: -- currently, the following iterations are profiled:
                         - "time step"
                         - "time step attempt"
                         - "same front"
                         - "extended front"
                         - "tip inversion"
                         - "tip width"
                         - "nonlinear system solve"
                         - "width constraint iteration"
                         - "linear system solve"
                         - "Brent method"
"""
mutable struct IterationProperties
    iterations::Int
    norm::Union{Float64, Nothing}
    itrType::String
    time::Union{Float64, Nothing}
    CpuTime_start::Float64
    CpuTime_end::Union{Float64, Nothing}
    status::Union{Bool, Nothing}
    failure_cause::Union{String, Nothing}
    NumbOfElts::Union{Int, Nothing}
    
    # sub-iterations data
    attempts_data::Vector{IterationProperties}
    sameFront_data::Vector{IterationProperties}
    extendedFront_data::Vector{IterationProperties}
    tipInv_data::Vector{IterationProperties}
    tipWidth_data::Vector{IterationProperties}
    nonLinSolve_data::Vector{IterationProperties}
    brentMethod_data::Vector{IterationProperties}
    widthConstraintItr_data::Vector{IterationProperties}
    linearSolve_data::Vector{IterationProperties}
    RKL_data::Vector{IterationProperties}
end

"""
    IterationProperties(itr_type="not initialized")

Constructor for IterationProperties.
"""
function IterationProperties(itr_type::String="not initialized")
    # Initialize all sub-iteration data vectors as empty
    attempts_data = IterationProperties[]
    sameFront_data = IterationProperties[]
    extendedFront_data = IterationProperties[]
    tipInv_data = IterationProperties[]
    tipWidth_data = IterationProperties[]
    nonLinSolve_data = IterationProperties[]
    brentMethod_data = IterationProperties[]
    widthConstraintItr_data = IterationProperties[]
    linearSolve_data = IterationProperties[]
    RKL_data = IterationProperties[]
    
    # Initialize sub-iterations data based on itr_type
    if itr_type == "time step"
        attempts_data = IterationProperties[]
    elseif itr_type == "time step attempt"
        sameFront_data = IterationProperties[]
        extendedFront_data = IterationProperties[]
    elseif itr_type == "same front"
        nonLinSolve_data = IterationProperties[]
    elseif itr_type == "extended front"
        tipInv_data = IterationProperties[]
        tipWidth_data = IterationProperties[]
        nonLinSolve_data = IterationProperties[]
    elseif itr_type == "tip inversion"
        brentMethod_data = IterationProperties[]
    elseif itr_type == "tip width"
        brentMethod_data = IterationProperties[]
    elseif itr_type == "nonlinear system solve"
        widthConstraintItr_data = IterationProperties[]
    elseif itr_type == "width constraint iteration"
        linearSolve_data = IterationProperties[]
        RKL_data = IterationProperties[]
    elseif itr_type == "linear system solve"
        # No sub-iteration data
    elseif itr_type == "Brent method"
        # No sub-iteration data
    else
        error("The given iteration type is not supported!")
    end
    
    return IterationProperties(
        0,  # iterations
        nothing,  # norm
        itr_type,  # itrType
        nothing,  # time
        Dates.time(),
        nothing,  # CpuTime_end
        nothing,  # status
        nothing,  # failure_cause
        nothing,  # NumbOfElts
        attempts_data,
        sameFront_data,
        extendedFront_data,
        tipInv_data,
        tipWidth_data,
        nonLinSolve_data,
        brentMethod_data,
        widthConstraintItr_data,
        linearSolve_data,
        RKL_data
    )
end

# -----------------------------------------------------------------------------------------------------------------------

"""
    instrument_start(itr_type, perfNode)

Start instrumenting a performance node.

# Arguments
- `itr_type::String`: The type of iteration.
- `perfNode::Union{IterationProperties, Nothing}`: The parent performance node.

# Returns
- `Union{IterationProperties, Nothing}`: The new performance node or nothing.
"""
function instrument_start(itr_type::String, perfNode::Union{IterationProperties, Nothing})
    if perfNode !== nothing
        perfNode_return = IterationProperties(itr_type)
    else
        perfNode_return = nothing
    end
    
    return perfNode_return
end

"""
    instrument_close(perfNode, perfNode_subItr, norm, n_elem, status, fail_cause, simulated_time)

Close instrumenting a performance node.

# Arguments
- `perfNode::IterationProperties`: The parent performance node.
- `perfNode_subItr::IterationProperties`: The sub-iteration performance node.
- `norm::Float64`: The norm value.
- `n_elem::Int`: The number of elements.
- `status::Bool`: The status.
- `fail_cause::String`: The failure cause.
- `simulated_time::Float64`: The simulated time.
"""
function instrument_close(perfNode::IterationProperties, perfNode_subItr::IterationProperties, 
                         norm::Float64, n_elem::Int, status::Bool, fail_cause::String, simulated_time::Float64)
    perfNode_subItr.CpuTime_end = Dates.time()
    perfNode_subItr.NumbOfElts = n_elem
    perfNode.iterations += 1
    perfNode_subItr.norm = norm
    perfNode_subItr.status = status
    perfNode_subItr.time = simulated_time
    if !status
        perfNode_subItr.failure_cause = fail_cause
    end
    
    # Add the sub-iteration data to the appropriate parent vector
    if perfNode.itrType == "time step" && perfNode_subItr.itrType == "time step attempt"
        push!(perfNode.attempts_data, perfNode_subItr)
    elseif perfNode.itrType == "time step attempt" && perfNode_subItr.itrType == "same front"
        push!(perfNode.sameFront_data, perfNode_subItr)
    elseif perfNode.itrType == "time step attempt" && perfNode_subItr.itrType == "extended front"
        push!(perfNode.extendedFront_data, perfNode_subItr)
    elseif perfNode.itrType == "same front" && perfNode_subItr.itrType == "nonlinear system solve"
        push!(perfNode.nonLinSolve_data, perfNode_subItr)
    elseif perfNode.itrType == "extended front" && perfNode_subItr.itrType == "tip inversion"
        push!(perfNode.tipInv_data, perfNode_subItr)
    elseif perfNode.itrType == "extended front" && perfNode_subItr.itrType == "tip width"
        push!(perfNode.tipWidth_data, perfNode_subItr)
    elseif perfNode.itrType == "extended front" && perfNode_subItr.itrType == "nonlinear system solve"
        push!(perfNode.nonLinSolve_data, perfNode_subItr)
    elseif perfNode.itrType == "tip inversion" && perfNode_subItr.itrType == "Brent method"
        push!(perfNode.brentMethod_data, perfNode_subItr)
    elseif perfNode.itrType == "tip width" && perfNode_subItr.itrType == "Brent method"
        push!(perfNode.brentMethod_data, perfNode_subItr)
    elseif perfNode.itrType == "nonlinear system solve" && perfNode_subItr.itrType == "width constraint iteration"
        push!(perfNode.widthConstraintItr_data, perfNode_subItr)
    elseif perfNode.itrType == "width constraint iteration" && perfNode_subItr.itrType == "linear system solve"
        push!(perfNode.linearSolve_data, perfNode_subItr)
    elseif perfNode.itrType == "width constraint iteration" && perfNode_subItr.itrType == "RKL"
        push!(perfNode.RKL_data, perfNode_subItr)
    end
    
    return nothing
end
# -----------------------------------------------------------------------------------------------------------------------

"""
    PlotProperties

This class stores the parameters used for plotting of the post-processed results.
"""
mutable struct PlotProperties
    lineStyle::String
    lineWidth::Float64
    lineColor::Union{String, Tuple, Nothing}
    colorMap::Union{String, Nothing}
    lineColorAnal::String
    lineStyleAnal::String
    lineWidthAnal::Union{Float64, Nothing}
    textSize::Union{Float64, Nothing}
    dispPrecision::Int
    meshColor::String
    meshEdgeColor::String
    meshLabelColor::String
    interpolation::String
    alpha::Float64
    graphScaling::String
    plotLegend::Bool
    PlotFP_Time::Bool
    useTex::Bool
    colorMaps::Vector{String}
    colorsList::Vector{String}
end

"""
    PlotProperties(; color_map=nothing, line_color=nothing, line_style="-", line_width=1.0, line_style_anal="--",
                   line_color_anal="r", interpolation="none", alpha=0.8, line_width_anal=nothing, text_size=nothing,
                   disp_precision=3, mesh_color="yellowgreen", mesh_edge_color="grey", mesh_label_color="black",
                   graph_scaling="linear", color_maps=nothing, colors_list=nothing, plot_legend=true, plot_FP_time=true,
                   use_tex=false)

Constructor for PlotProperties.
"""
function PlotProperties(color_map::Union{String, Nothing}=nothing, 
                       line_color::Union{String, Tuple, Nothing}=nothing, 
                       line_style::String="-", 
                       line_width::Float64=1.0, 
                       line_style_anal::String="--",
                       line_color_anal::String="r", 
                       interpolation::String="none", 
                       alpha::Float64=0.8, 
                       line_width_anal::Union{Float64, Nothing}=nothing, 
                       text_size::Union{Float64, Nothing}=nothing,
                       disp_precision::Int=3, 
                       mesh_color::String="yellowgreen", 
                       mesh_edge_color::String="grey", 
                       mesh_label_color::String="black",
                       graph_scaling::String="linear", 
                       color_maps::Union{Vector{String}, Nothing}=nothing, 
                       colors_list::Union{Vector{String}, Nothing}=nothing, 
                       plot_legend::Bool=true, 
                       plot_FP_time::Bool=true,
                       use_tex::Bool=false)
    
    # Handle color_maps
    if color_maps === nothing
        colorMaps_val = ["cool", "Wistia", "summer", "autumn"]
    else
        colorMaps_val = color_maps
    end
    
    # Handle colors_list
    if colors_list === nothing
        colorsList_val = ["black", "firebrick", "olivedrab", "royalblue", "deeppink", "darkmagenta"]
    else
        colorsList_val = colors_list
    end
    
    # Handle lineColor
    if line_color === nothing
        lineColor_val = to_rgb(colorsList_val[1])
    else
        lineColor_val = line_color
        colorsList_val = [string(line_color)]
    end
    
    # Handle colorMap
    if color_map === nothing
        colorMap_val = colorMaps_val[1]
    else
        colorMap_val = color_map
        colorMaps_val = [color_map]
    end
    
    return PlotProperties(
        line_style,
        line_width,
        lineColor_val,
        colorMap_val,
        line_color_anal,
        line_style_anal,
        line_width_anal,
        text_size,
        disp_precision,
        mesh_color,
        mesh_edge_color,
        mesh_label_color,
        interpolation,
        alpha,
        graph_scaling,
        plot_legend,
        plot_FP_time,
        use_tex,
        colorMaps_val,
        colorsList_val
    )
end
# -----------------------------------------------------------------------------------------------------------------------

"""
    LabelProperties

This class stores the labels of a plot figure.
"""
mutable struct LabelProperties
    yLabel::String
    xLabel::String
    zLabel::String
    colorbarLabel::String
    units::String
    unitConversion::Float64
    figLabel::String
    legend::String
    useLatex::Bool
end

"""
    LabelProperties(variable, data_subset="whole mesh", projection="2D", use_latex=true)

Constructor for LabelProperties.

# Arguments
- `variable::String`: The variable to be plotted.
- `data_subset::String`: The data subset to be plotted. Options are:
                         - "whole mesh" or "wm"
                         - "slice" or "s"  
                         - "point" or "p"
- `projection::String`: The type of projection. Options are:
                        - "2D_clrmap", "2D_contours", "2D_vectorfield", "3D", "2D" for 2D projections
                        - "1D" for 1D projection
- `use_latex::Bool`: Whether to use LaTeX formatting.
"""
function LabelProperties(variable::String, data_subset::String="whole mesh", projection::String="2D", use_latex::Bool=true)
    if !(variable in supported_variables)
        error(err_msg_variable)
    end
    
    if variable in unidimensional_variables
        projection = "1D"
    end
    
    yLabel_val = ""
    xLabel_val = ""
    zLabel_val = ""
    colorbarLabel_val = ""
    units_val = ""
    unitConversion_val = 0.0
    figLabel_val = ""
    legend_val = ""
    useLatex_val = use_latex
    
    if data_subset in ("whole mesh", "wm")
        if projection in ("2D_clrmap", "2D_contours", "2D_vectorfield", "3D", "2D")
            yLabel_val = "meters"
            xLabel_val = "meters"
            zLabel_val = var_labels[variable] * units[variable]
        elseif projection == "1D"
            xLabel_val = "time (s)"
            yLabel_val = var_labels[variable] * units[variable]
        end
    elseif data_subset in ("slice", "s")
        if occursin("2D", projection)
            xLabel_val = "meters"
            yLabel_val = var_labels[variable] * units[variable]
        elseif projection == "3D"
            yLabel_val = "meters"
            xLabel_val = "meters"
            zLabel_val = var_labels[variable] * units[variable]
        end
    elseif data_subset in ("point", "p")
        xLabel_val = "time (s)"
        yLabel_val = var_labels[variable] * units[variable]
    end
    
    colorbarLabel_val = var_labels[variable] * units[variable]
    units_val = units[variable]
    unitConversion_val = unit_conversion[variable]
    figLabel_val = Fig_labels[variable]
    legend_val = var_labels[variable]
    useLatex_val = use_latex
    
    return LabelProperties(
        yLabel_val,
        xLabel_val,
        zLabel_val,
        colorbarLabel_val,
        units_val,
        unitConversion_val,
        figLabel_val,
        legend_val,
        useLatex_val
    )
end

end

