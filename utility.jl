# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""

module Utility

    include("tip_inversion.jl")
    using .TipInversion: TipAsymInversion
    include("common_functions.jl")
    using .CommonFunctions: logging_level
    using JLD2
    using Logging
    using LoggingExtras
    using PyPlot
    using FilePathsBase: joinpath
    using Base.Filesystem: readdir, endswith

    export plot_as_matrix, read_fracture, save_images_to_video, setup_logging_to_console, nanmin, nanmax

    """
        plot_as_matrix(data, mesh, fig=nothing)

        Plot data as a matrix using the given mesh.

        # Arguments
        - `data::Vector{Float64}`: The data to be plotted.
        - `mesh::CartesianMesh`: The mesh object.
        - `fig::Union{PyObject, Nothing}`: The figure object (optional).

        # Returns
        - `PyObject`: The figure object.
    """
    function plot_as_matrix(data::Vector{Float64}, mesh, fig=nothing)
        if fig === nothing
            fig = PyPlot.figure()
        end
        ax = PyPlot.matplotlib.pyplot.gca()
        
        ReMesh = reshape(data, (mesh.nx, mesh.ny))'
        cax = PyPlot.matshow(ReMesh)
        PyPlot.colorbar(cax)
        PyPlot.show()
        
        return fig
    end
    #-----------------------------------------------------------------------------------------------------------------------
    """
        read_fracture(filename)

        Read a fracture object from a JLD2 file.

        # Arguments
        - `filename::String`: The name of the .jld2 file to read from.

        # Returns
        - The loaded fracture object.
    """
    function read_fracture(filename::String)
        if !endswith(filename, ".jld2")
            filename = filename * ".jld2"
        end
        return JLD2.load(filename)
    end
    #-----------------------------------------------------------------------------------------------------------------------
    """
        save_images_to_video(image_folder, video_name="movie")

        Save .png images from a folder to a video file using FFMPEG.

        # Arguments
        - `image_folder::String`: The folder containing .png images.
        - `video_name::String`: The name of the output video file (default: "movie").
                            The function will ensure it has a common video extension like .mp4 or .avi.
    """
    function save_images_to_video(image_folder::String, video_name::String="movie")
        
        log = "JFrac.save_image_to_video"
        
        if !occursin(r"\.(mp4|avi|mov|mkv)$", video_name)
            video_name = video_name * ".mp4"
        end

        images = filter(f -> endswith(f, ".png"), readdir(image_folder))
        sort!(images)

        if isempty(images)
            @warn "No .png images found in folder $image_folder" _group = log
            return
        end
        first_image_path = joinpath(image_folder, images[1])

        if !isfile(first_image_path)
            @error "First image file not found: $first_image_path" _group = log
            return
        end


        temp_list_file = tempname() * ".txt"
        try
            open(temp_list_file, "w") do f
                for img in images
                    println(f, "file '$(joinpath(image_folder, img))'")
                end
            end

            # FFMPEG
            # -r 1: input shots frequency 1 shot/sec
            # -f concat -safe 0: use file-list
            # -c:v libx264: use codec H.264
            # -pix_fmt yuv420p: pixel format
            # -y: rewrite output file
            cmd = `$(
                FFMPEG.exe(
                    "-r", "1",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", temp_list_file,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    video_name
                )
            )`
            @debug "Running FFMPEG command: $cmd" _group = log
            run(cmd)

            @info "Video saved as $video_name" _group = log

        catch e
            @error "Failed to create video" exception=(e, catch_backtrace()) _group = log
        finally
            if isfile(temp_list_file)
                rm(temp_list_file, force=true)
            end
        end
    end
    #-----------------------------------------------------------------------------------------------------------------------

    """
        setup_logging_to_console(verbosity_level="debug")

        This function sets up the log to the console.
        Note: from any module in the code you can use the logging capabilities. You just have to:

        1) import the Logging module
        using Logging

        2) use the logging macros with a _group, such as 'JFrac.general' or 'JFrac.frontrec'
        @debug "debug message" _group="JFrac.frontrec"
        @info "info message" _group="JFrac.general"
        @warn "warn message" _group="JFrac.whatever"
        @error "error message" _group="JFrac.error"

        # Arguments
        - `verbosity_level::String`: string that defines the level of logging concerning the console:
            - 'debug'    - Detailed information, typically of interest only when diagnosing problems.
            - 'info'     - Confirmation that things are working as expected.
            - 'warning'  - An indication that something unexpected happened, or indicative of some problem in the near future.
            - 'error'    - Due to a more serious problem, the software has not been able to perform some function.
    """
    function setup_logging_to_console(verbosity_level::String = "debug")

        console_lvl = logging_level(verbosity_level)

        formatter(log_args) = begin
            level_str = rpad(string(log_args.level), 8)
            message_str = log_args.message
            return "$level_str:     $message_str"
        end

        console_logger = FormatLogger(formatter) do io, args
            println(io, args.message)
        end

        jfrac_filter(log_args) = startswith(get(log_args.group, :name, ""), "JFrac")
        level_logger = MinLevelLogger(console_logger, console_lvl)
        filtered_logger = EarlyFilteredLogger(jfrac_filter, level_logger)
        global_logger(filtered_logger)
        @info "Console logger set up correctly" _group = "JFrac.general"

        return nothing
    end
    

    function nanmin(arr)
        valid_values = filter(!isnan, arr)
        return isempty(valid_values) ? NaN : minimum(valid_values)
    end

    function nanmax(arr)
        valid_values = filter(!isnan, arr)
        return isempty(valid_values) ? NaN : maximum(valid_values)
    end

end # module Utility