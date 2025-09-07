# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""

module PostprocessPerformance

using Logging
using FilePathsBase: joinpath
using Base.Filesystem: readdir, isfile
using Serialization
using PyPlot
using Dates
using RegularExpressions

include("properties.jl")
include("visualization.jl")
using .Properties: PlotProperties
using .Visualization: plot_variable_vs_time

export load_performance_data, get_performance_variable, plot_performance, print_performance_data

const slash = Sys.iswindows() ? "\\" : "/"

"""
    load_performance_data(address, sim_name="simulation")

This function loads the performance data in the given simulation. If no simulation name is provided, the most
recent simulation will be loaded.

# Arguments
- `address::String`: the disk address where the simulation results are saved
- `sim_name::String`: the name of the simulation

# Returns
- `perf_data::Vector`: the loaded performance data in the form of a list of IterationProperties objects.
"""
function load_performance_data(address::Union{String, Nothing}, sim_name::String="simulation")
    @info "---loading performance data---" _group="JFrac.load_performace_data"

    if address === nothing
        address = "." * slash * "_simulation_data_PyFrac"
    end

    if address[end] != slash[end]
        address = address * slash
    end

    sim_full_name = ""
    if occursin(r"\d+-\d+-\d+__\d+_\d+_\d+", sim_name[end-19:end])
        sim_full_name = sim_name
    else
        simulations = readdir(address)
        time_stamps = String[]
        for i in simulations
            if occursin(Regex(sim_name * "__\\d+-\\d+-\\d+__\\d+_\\d+_\\d+"), i)
                push!(time_stamps, i[end-19:end])
            end
        end
        if length(time_stamps) == 0
            error("Simulation not found! The address might be incorrect.")
        end

        Tmst_sorted = sort(time_stamps)
        sim_full_name = sim_name * "__" * Tmst_sorted[end]
    end

    filename = joinpath(address, sim_full_name, "perf_data.dat")
    perf_data = nothing
    
    try
        open(filename, "r") do inp
            perf_data = deserialize(inp)
        end
    catch e
        if isa(e, SystemError) && !isfile(filename)
            error("Performance data not found! Check if it's saving is enabled in simulation properties.")
        else
            rethrow(e)
        end
    end

    return perf_data
end

"""
    get_performance_variable(perf_data, iteration, variable)

This function gets the required variable from the specified iteration.

# Arguments
- `perf_data::Vector`: the loaded performance data in the form of a list of IterationProperties objects.
- `iteration::String`: the type of iteration.
- `variable::String`: the name of the variable to be retrieved.

# Returns
- `Tuple{Vector, Vector, Vector}`: (var_list, time_list, N_list) - the loaded variable, corresponding times, and element counts.
"""
function get_performance_variable(perf_data::Vector, iteration::String, variable::String)
    var_list = Any[]
    time_list = Float64[]
    N_list = Int[]

    function append_variable(Iteration, variable)
        push!(var_list, getproperty(Iteration, Symbol(variable)))
        push!(time_list, Iteration.time)
        push!(N_list, Iteration.NumbOfElts)
    end

    for (i_node, node) in enumerate(perf_data)
        if iteration == "time step"
            append_variable(node, variable)
        else
            for (i_TS_attempt, TS_attempt) in enumerate(node.attempts_data)
                if iteration == "time step attempt"
                    append_variable(TS_attempt, variable)
                else
                    for (i_sameFP_inj, sameFP_inj) in enumerate(TS_attempt.sameFront_data)
                        if iteration == "same front"
                            append_variable(sameFP_inj, variable)
                        else
                            for (i_nonLinSolve, nonLinSolve_itr) in enumerate(sameFP_inj.nonLinSolve_data)
                                if iteration == "nonlinear system solve"
                                    append_variable(nonLinSolve_itr, variable)
                                else
                                    for (i_widthConstraint, widthConstraint_Itr) in enumerate(nonLinSolve_itr.widthConstraintItr_data)
                                        if iteration == "width constraint iteration"
                                            append_variable(widthConstraint_Itr, variable)
                                        else
                                            for (i_linearSolve, linearSolve_Itr) in enumerate(widthConstraint_Itr.linearSolve_data)
                                                if iteration == "linear system solve"
                                                    append_variable(linearSolve_Itr, variable)
                                                end
                                            end
                                            for (i_RKLSolve, RKLSolve_Itr) in enumerate(widthConstraint_Itr.RKL_data)
                                                if iteration == "RKL time step"
                                                    append_variable(RKLSolve_Itr, variable)
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end

                    for (i_extFP_inj, extFP_inj) in enumerate(TS_attempt.extendedFront_data)
                        if iteration == "extended front"
                            append_variable(extFP_inj, variable)
                        else
                            for (i_tipInv_itr, tipInv_itr) in enumerate(extFP_inj.tipInv_data)
                                if iteration == "tip inversion"
                                    append_variable(tipInv_itr, variable)
                                else
                                    for (i_brentq_itr, brentq_itr) in enumerate(tipInv_itr.brentMethod_data)
                                        if iteration == "Brent method"
                                            append_variable(brentq_itr, variable)
                                        end
                                    end
                                end
                            end

                            for (i_tipWidth_itr, tipWidth_itr) in enumerate(extFP_inj.tipWidth_data)
                                if iteration == "tip width"
                                    append_variable(tipWidth_itr, variable)
                                else
                                    for (i_brentq_itr, brentq_itr) in enumerate(tipWidth_itr.brentMethod_data)
                                        if iteration == "Brent method"
                                            append_variable(brentq_itr, variable)
                                        end
                                    end
                                end
                            end

                            for (i_nonLinSolve, nonLinSolve_itr) in enumerate(extFP_inj.nonLinSolve_data)
                                if iteration == "nonlinear system solve"
                                    append_variable(nonLinSolve_itr, variable)
                                else
                                    for (i_widthConstraint, widthConstraint_Itr) in enumerate(nonLinSolve_itr.widthConstraintItr_data)
                                        if iteration == "width constraint iteration"
                                            append_variable(widthConstraint_Itr, variable)
                                        else
                                            for (i_linearSolve, linearSolve_Itr) in enumerate(widthConstraint_Itr.linearSolve_data)
                                                if iteration == "linear system solve"
                                                    append_variable(linearSolve_Itr, variable)
                                                end
                                            end
                                            for (i_RKLSolve, RKLSolve_Itr) in enumerate(widthConstraint_Itr.RKL_data)
                                                if iteration == "RKL time step"
                                                    append_variable(RKLSolve_Itr, variable)
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return var_list, time_list, N_list
end

"""
    plot_performance(address, variable, sim_name="simulation", fig=nothing, plot_prop=nothing, plot_vs_N=false)

This function plot the performance variable from the given simulation.

# Arguments
- `address::String`: the disk location where the results of the simulation were saved.
- `variable::String`: the name of the variable to be plotted.
- `sim_name::String`: the name of the simulation.
- `fig`: a figure to superimpose on
- `plot_prop`: a PlotProperties object
- `plot_vs_N::Bool`: if true, a plot of the variable versus the number of cells will also be plotted.

# Returns
- `fig`: the figure object
"""
function plot_performance(address::String, variable::String, sim_name::String="simulation", 
                         fig=nothing, plot_prop=nothing, plot_vs_N::Bool=false)
    perf_data = load_performance_data(address, sim_name)

    var_list = nothing
    time_list = nothing
    N_list = nothing

    if variable in ["time step attempts"]
        var_list, time_list, N_list = get_performance_variable(perf_data, "time step", "iterations")
    elseif variable in ["fracture front iterations"]
        var_list, time_list, N_list = get_performance_variable(perf_data, "time step attempt", "iterations")
    elseif variable in ["tip inversion iterations"]
        var_list, time_list, N_list = get_performance_variable(perf_data, "Brent method", "iterations")
    elseif variable in ["width constraint iterations"]
        var_list, time_list, N_list = get_performance_variable(perf_data, "nonlinear system solve", "iterations")
    elseif variable in ["Picard iterations"]
        var_list, time_list, N_list = get_performance_variable(perf_data, "width constraint iteration", "iterations")
    elseif variable in ["RKL substeps"]
        var_list, time_list, N_list = get_performance_variable(perf_data, "RKL time step", "iterations")
    elseif variable in ["CPU time: time steps"]
        t_start_list, time_list, N_list = get_performance_variable(perf_data, "time step", "CpuTime_start")
        t_end_list, time_list, N_list = get_performance_variable(perf_data, "time step", "CpuTime_end")
        var_list = [i - j for (i, j) in zip(t_end_list, t_start_list)]
    elseif variable in ["CPU time: time step attempts"]
        t_start_list, time_list, N_list = get_performance_variable(perf_data, "time step attempt", "CpuTime_start")
        t_end_list, time_list, N_list = get_performance_variable(perf_data, "time step attempt", "CpuTime_end")
        var_list = [i - j for (i, j) in zip(t_end_list, t_start_list)]
    else
        error("Cannot recognize the required variable.")
    end

    var_list_np = convert(Vector{Float64}, var_list)
    time_list_np = convert(Vector{Float64}, time_list)
    N_list_np = convert(Vector{Int}, N_list)

    if plot_prop === nothing
        plot_prop = (line_style=".",)
    end

    if plot_vs_N
        if fig === nothing
            fig = PyPlot.figure()
        end
        ax = PyPlot.gca()
        ax.plot(N_list_np, var_list_np, plot_prop.line_style, label="fracture front iterations")
        ax.set_ylabel(variable)
        ax.set_xlabel("number of elements")
    else
        if fig === nothing
            fig = PyPlot.figure()
        end
        ax = PyPlot.gca()
        ax.plot(time_list_np, var_list_np, plot_prop.line_style, label=variable)
        ax.set_ylabel(variable)
        ax.set_xlabel("time")
    end

    return fig
end

"""
    print_performance_data(address, sim_name=nothing)

This function generate a file with details of all the iterations and the data collected regarding their preformance

# Arguments
- `address::String`: the disk location where the results of the simulation were saved.
- `sim_name::Union{String, Nothing}`: the name of the simulation.
"""
function print_performance_data(address::String, sim_name::Union{String, Nothing}=nothing)
    @info "---saving iterations data---" _group="JFrac.print_performance_data"
    
    perf_data = load_performance_data(address, sim_name)

    f = open("performance_data.txt", "w+")

    function print_non_linear_system_performance(iteration_prop, tabs)
        write(f, tabs * "--->Non linear system solve" * '\n')
        write(f, tabs * "\tnumber of width constraint iterations to solve non linear system = " * repr(iteration_prop.iterations) * '\n')
        write(f, tabs * "\tCPU time taken: " * repr(iteration_prop.CpuTime_end - iteration_prop.CpuTime_start) * " seconds" * '\n')
        write(f, tabs * "\tnorm for the iteration = " * repr(iteration_prop.norm) * '\n')

        for (i_widthConstraint, widthConstraint_Itr) in enumerate(iteration_prop.widthConstraintItr_data)
            write(f, tabs * "\t--->width constraint iteration " * repr(i_widthConstraint + 1) * '\n')
            write(f, tabs * "\t\tnumber of linear system solved for the Picard iteration = " * repr(widthConstraint_Itr.iterations) * '\n')
            write(f, tabs * "\t\tCPU time taken: " * repr(widthConstraint_Itr.CpuTime_end - widthConstraint_Itr.CpuTime_start) * " seconds" * '\n')
            write(f, tabs * "\t\tnorm for the iteration = " * repr(widthConstraint_Itr.norm) * '\n')

            for (i_linearSolve, linearSolve_Itr) in enumerate(widthConstraint_Itr.linearSolve_data)
                write(f, tabs * "\t\t--->linear system solve: iteration no " * repr(i_linearSolve + 1) * '\n')
                write(f, tabs * "\t\t\tsub-iteration data not collected" * '\n')
                write(f, tabs * "\t\t\tCPU time taken: " * repr(linearSolve_Itr.CpuTime_end - linearSolve_Itr.CpuTime_start) * " seconds" * '\n')
                write(f, tabs * "\t\t\tnorm of the iteration = " * repr(linearSolve_Itr.norm) * '\n')
            end
        end
    end

    for (i_node, node) in enumerate(perf_data)
        status = node.status ? "successful" : "unsuccessful"
        write(f, "time step status = " * status * '\n')
        write(f, "number of attempts = " * repr(node.iterations) * '\n')

        for (i_TS_attempt, TS_attempt) in enumerate(node.attempts_data)
            write(f, "--->attempt number: " * repr(i_TS_attempt + 1) * '\n')
            write(f, "\tattempt to advance to: " * repr(TS_attempt.ti