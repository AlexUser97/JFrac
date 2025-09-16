# -*- coding: utf-8 -*-
"""
This file is a part of JFrac.
Realization of Pyfrac on Julia language.
This file defines the default simulation parameters for PyFrac.jl.

"""

module DefaultParameters

    export toleranceFractureFront, toleranceEHL, tol_projection, toleranceVStagnant,
           Hersh_Bulk_epsilon, Hersh_Bulk_Gmin, max_front_itrs, max_solver_itrs,
           max_proj_Itrs, tmStp_prefactor, req_sol_at, final_time, maximum_steps,
           timeStep_limit, fixed_time_step, max_reattemps, reattempt_factor,
           plot_figure, plot_analytical, analytical_sol, bck_color, sim_name,
           block_figure, plot_var, plot_proj, plot_time_period, plot_TS_jump,
           plot_at_sol_time_series, verbosity_level, output_folder, save_to_disk,
           save_time_period, save_TS_jump, save_chi, save_regime, save_ReyNumb,
           save_fluid_flux, save_fluid_vel, save_fluid_flux_as_vector,
           save_fluid_vel_as_vector, save_effective_viscosity, save_yield_ratio,
           save_statistics_post_coalescence, save_G, collect_perf_data, log_to_file,
           elastohydr_solver, m_Anderson, relaxation_param, mech_loading,
           volume_control, double_fracture_vol_contr, viscous_injection,
           substitute_pressure, solve_deltaP, solve_stagnant_tip, solve_tip_corr_rib,
           solve_sparse, tip_asymptote, gravity, TI_Kernel_exec_path, symmetric,
           enable_GPU, n_threads, use_block_toepliz_compression, proj_method,
           explicit_projection, front_advancing, param_from_tip,
           limit_Adancement_To_2_cells, force_time_step_limit_and_max_adv_to_2_cells,
           max_reattemps_FracAdvMore2Cells, mesh_extension_direction,
           mesh_extension_factor, mesh_extension_all_sides, mesh_reduction_factor,
           enable_remeshing, remesh_factor, height, aspect_ratio, roughness_model,
           roughness_sigma


    # tolerances
    const toleranceFractureFront = 1.0e-3         # tolerance for the fracture front position solver.
    const toleranceEHL = 1.0e-4                   # tolerance for the elastohydrodynamic system solver.
    const tol_projection = 2.5e-3                 # tolerance for the toughness iteration.
    const toleranceVStagnant = 1e-6               # tolerance on the velocity to decide if a cell is stagnant.
    const Hersh_Bulk_epsilon = 1e-3               # for Herschel Bulkley fluid; the value where the coefficient G is regularized.
    const Hersh_Bulk_Gmin = 1e-5                  # for Herschel Bulkley fluid; the min value of the coefficient G.

    # max iterations
    const max_front_itrs = 25                     # maximum iterations for the fracture front.
    const max_solver_itrs = 100                   # maximum iterations for the elastohydrodynamic solver.
    const max_proj_Itrs = 10                      # maximum projection iterations.

    # time and time stepping
    const tmStp_prefactor = 0.5                   # time step prefactor(pf) to determine the time step(dt = pf*min(dx, dy)/max(v).
    const req_sol_at = nothing                    # times at which the solution is required.
    const final_time = nothing                    # time to stop the propagation.
    const maximum_steps = 5000                    # maximum time steps.
    const timeStep_limit = nothing                # limit for the time step.
    const fixed_time_step = nothing               # constant time step.

    # time step re-attempt
    const max_reattemps = 20                      # maximum reattempts in case of time step failure.
    const reattempt_factor = 0.5                  # the factor by which time step is reduced on reattempts.
    const max_reattemps_FracAdvMore2Cells = 50    # number of time reduction that are made if the fracture is advancing more than two cells.

    # output
    const plot_figure = true                      # if True, figure will be plotted after the given time period.
    const plot_analytical = false                 # if True, analytical solution will also be plotted.
    const analytical_sol = nothing                # the analytical solution to be plotted.
    const bck_color = nothing                     # the parameter according to which background is color coded (see class doc).
    const sim_name = nothing                      # name given to the simulation.
    const block_figure = false                    # if true, the simulation will proceed after the figure is closed.
    const plot_var = ["w"]                        # the list of variables to be plotted during simulation.
    const plot_proj = "2D_clrmap"                 # projection to be plotted with.
    const plot_time_period = nothing              # the time period after which the variables given in plot_var are plotted.
    const plot_TS_jump = 1                        # the number of time steps after which the given variables are plotted.
    const plot_at_sol_time_series = true          # plot when the time is in the requested time series
    const verbosity_level = "debug"               # the level of details about the ongoing simulation to be written to the log file ('debug','info','warning','error','critical').

    # Saving options
    const output_folder = nothing                 # the address to save the output data.
    const save_to_disk = true                     # if True, fracture will be saved after the given time period.
    const save_time_period = nothing              # the time period after which the output is saved to disk.
    const save_TS_jump = 1                        # the number of time steps after which the output is saved to disk.
    const save_chi = false                        # Question if we save the tip asymptotics leak-off parameter (Tip leak-off parameter)
    const save_regime = true                      # if True, the regime of the ribbon cells will also be saved.
    const save_ReyNumb = false                    # if True, the Reynold's number at each edge will be saved.
    const save_fluid_flux = false                 # if True, the fluid flux at each edge will be saved.
    const save_fluid_vel = false                  # if True, the fluid vel at each edge will be saved.
    const save_fluid_flux_as_vector = false       # if True, the fluid flux at each edge will be saved as vector, i.e. with two components.
    const save_fluid_vel_as_vector = false        # if True, the fluid vel at each edge will be saved as vector, i.e. with two components.
    const save_effective_viscosity = false        # if True, the Newtonian equivalent viscosity of the non-Newtonian fluid will be saved.
    const save_yield_ratio = false                # if True, the ratio of the height of fluid column yielded to total width will be saved.
    const save_statistics_post_coalescence = false # if True, the statistics post coalescence of two fractures are saved to json file
    const save_G = false                          # if True, the prefactor G, giving the effect of yield stress will be saved.
    const collect_perf_data = false               # if True, performance data will be collected in the form of a tree.
    const log_to_file = true                      # set it True or False depending if you would like to log the messages to a log file

    # type of solver
    const elastohydr_solver = "implicit_Anderson" # set the elasto-hydrodynamic system solver to implicit with Anderson iteration.
    const m_Anderson = 4                          # number of previous solutions to take into account in the Anderson scheme
    const relaxation_param = 1.0                  # parameter defining the under-relaxation performed (default is not relaxed)
    const mech_loading = false                    # if True, the mechanical loading solver will be used.
    const volume_control = false                  # if True, the volume control solver will be used.
    const double_fracture_vol_contr = false       # enable the volume control solver for two fractures
    const viscous_injection = true                # if True, the viscous fluid solver solver will be used.
    const substitute_pressure = true              # if True, the pressure will be substituted with width to make the EHL system.
    const solve_deltaP = true                     # if True, the change in pressure, instead of pressure will be solved.
    const solve_stagnant_tip = false              # if True, stagnant tip cells will also be solved for
    const solve_tip_corr_rib = true               # if True, the corresponding tip cells to closed ribbon cells will be solved.
    const solve_sparse = nothing                  # if True, the fluid conductivity matrix will be made with sparse matrix.

    # miscellaneous
    const tip_asymptote = "U1"                    # the tip_asymptote to be used (see class documentation for details).
    const gravity = false                         # if True, the effect of gravity will be taken into account.
    const TI_Kernel_exec_path = "../TI_Kernel/build" # the folder containing the executable to calculate TI elasticity matrix.

    # performances and memory savings
    const symmetric = false                       # if True, only positive quarter of the cartesian coordinates will be solved.
    const enable_GPU = false                      # if True, GPU will be use to do the dense matrix vector product.
    const n_threads = 4                           # setting the number of threads for multi-threaded dot product for RKL scheme.
    const use_block_toepliz_compression = false   # if True, only the unique coeff. of the elasticity matrix will be saved. It saves memory but it does more operations per time step.

    #Front advancement
    const proj_method = "LS_continousfront"       # set the method to evaluate projection on front to the original ILSA method.
    const explicit_projection = false             # if True, direction from last time step will be used to evaluate TI parameters.
    const front_advancing = "predictor-corrector" # possible options include 'implicit', 'explicit' and 'predictor-corrector'.
    const param_from_tip = false                  # set the space dependant tip parameters to be taken from ribbon cell.
    const limit_Adancement_To_2_cells = false     # limit the timestep in such a way that the front will advance less than 2 cells in a row
    const force_time_step_limit_and_max_adv_to_2_cells = false # this will force the contemporaneity of timeStepLimit and limitAdancementTo2cells

    # Mesh extension
    const mesh_extension_direction = fill(false, 4)  # Per default the mesh is not extended in any direction
    const mesh_extension_factor = [2.0, 2.0, 2.0, 2.0] # How many total elements we will have in this direction
    const mesh_extension_all_sides = false        # To allow the fracture to extend in all directions simultaneously
    const mesh_reduction_factor = 2.0             # the factor by which we reduce the number of elements

    # Remeshing
    const enable_remeshing = true                 # if true, computation domain will be remeshed after reaching end of the domain.
    const remesh_factor = 2.0                     # the factor by which the mesh is compressed.

    # fracture geometry
    const height = nothing                        # fracture height to calculate the analytical solution for PKN or KGD geometry.
    const aspect_ratio = nothing                  # fracture aspect ratio to calculate the analytical solution for TI case.

    # roughness parameters
    const roughness_model = nothing
    const roughness_sigma = nothing


end