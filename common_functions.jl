module CommonFunctions

    export locate_element, logging_level

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
            
            @warn "Point is outside domain." _group="JFrac.locate_element"
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

    """
        logging_level(logging_level_string)

        This function returns the pertinent logging level based on the string received as input.

        # Arguments
        - `logging_level_string::String`: string that defines the level of logging:
            - 'debug' - Detailed information, typically of interest only when diagnosing problems.
            - 'info' - Confirmation that things are working as expected.
            - 'warning' - An indication that something unexpected happened, or indicative of some problem in the near future.
            - 'error' - Due to a more serious problem, the software has not been able to perform some function.

        # Returns
        - Logging level code (e.g., Logging.Debug).
    """
    
    function logging_level(logging_level_string::String)::LogLevel
        level_map = Dict{String, LogLevel}(
            "debug" => Logging.Debug,
            "Debug" => Logging.Debug,
            "DEBUG" => Logging.Debug,
            "info" => Logging.Info,
            "Info" => Logging.Info,
            "INFO" => Logging.Info,
            "warning" => Logging.Warn,
            "Warning" => Logging.Warn,
            "WARNING" => Logging.Warn,
            "error" => Logging.Error,
            "Error" => Logging.Error,
            "ERROR" => Logging.Error,
        )

        if haskey(level_map, logging_level_string)
            return level_map[logging_level_string]
        else
            error("Options are: debug, info, warning, error, critical. Got: $logging_level_string")
        end
    end

    macro custom_log(level, msg, group)
        msg = esc(msg)
        group = esc(group)

        quote
            log_level = try
                logging_level(string($level))
            catch e
                @error "Unknown log level: $($level)" _group = $group
                return
            end

            if log_level == Logging.Debug
                @debug $msg _group = $group
            elseif log_level == Logging.Info
                @info $msg _group = $group
            elseif log_level == Logging.Warn
                @warn $msg _group = $group
            elseif log_level == Logging.Error
                @error $msg _group = $group
            end
        end
    end


end