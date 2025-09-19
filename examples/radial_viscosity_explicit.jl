# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""


using Logging
using PyPlot 

include(joinpath(@__DIR__, "..", "JFrac.jl"))

using .JFrac.Mesh: CartesianMesh
using .JFrac.Properties: MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
using .JFrac.FractureModule: Fracture
using .JFrac.ControllerModule: Controller
using .JFrac.Initialization: Geometry, InitializationParameters
using .JFrac.Utility: setup_logging_to_console 
# 
# Импорт модуля визуализации
using .JFrac.Visualization: load_fractures, get_fracture_variable, plot_fracture_list,
                            plot_analytical_solution, plot_fracture_list_at_point,
                            plot_analytical_solution_at_point, plot_fracture_list_slice

# Настройка уровня логирования в консоль
setup_logging_to_console(verbosity_level="info") # Реализация зависит от вашей функции

println("Creating mesh...")
Lx, Ly = 0.3, 0.3
nx, ny = 41, 41
Mesh = CartesianMesh(Lx, Ly, nx, ny)

# свойства твердого тела
nu = 0.4                            # Коэффициент Пуассона
youngs_mod = 3.3e10                 # Модуль Юнга
Eprime = youngs_mod / (1 - nu^2)    # Модуль плоской деформации
K_Ic = 0.5                          # Устойчивость трещины

# свойства материала
Solid = MaterialProperties(Mesh, Eprime, K_Ic)

# параметры закачки
Q0 = 0.001  # расход закачки
Injection = InjectionProperties(Q0, Mesh)

# свойства жидкости
Fluid = FluidProperties(viscosity=1.1e-3)

# свойства симуляции
println("Setting simulation properties...")
simulProp = SimulationProperties()
simulProp.finalTime = 1e5                           # время остановки симуляции
simulProp.set_tipAsymptote("M")                     # асимптота вершины по вязкости
simulProp.frontAdvancing = "explicit"               # явное отслеживание фронта
simulProp.saveTSJump, simulProp.plotTSJump = 5, 5   # сохранять и рисовать каждые 5 шагов
simulProp.set_outputFolder("./Data/M_radial_explicit") # путь сохранения файлов

# параметры инициализации
println("Initializing fracture...")
Fr_geometry = Geometry("radial", radius=0.1)
init_param = InitializationParameters(Fr_geometry, regime="M")

# создание объекта трещины
Fr = Fracture(Mesh, init_param, Solid, Fluid, Injection, simulProp)

# создание контроллера
controller = Controller(Fr, Solid, Fluid, Injection, simulProp)

# --- Запуск симуляции ---
println("Running simulation...")
controller.run()
println("Simulation completed.")

# --- Визуализация результатов ---
# Проверка, запущен ли скрипт как часть пакетной обработки
# В Julia можно проверить существование файла
# if !isfile("./batch_run.txt") # Визуализируем только для отдельных примеров

#     println("Loading and plotting results...")

#     # загрузка результатов симуляции
#     Fr_list, properties = load_fractures(address="./Data/M_radial_explicit") # загрузить все трещины
#     time_srs = get_fracture_variable(Fr_list, variable="time")               # список времен

#     # --- Построение графиков ---

#     # 1. Построение графика среднего радиуса трещины
#     plot_prop_R = PlotProperties() # Предполагается, что PlotProperties доступен из JFrac.Visualization или JFrac.Properties
#     plot_prop_R.lineStyle = "."               # стиль линии - точки
#     plot_prop_R.graphScaling = "loglog"       # логарифмический масштаб
#     Fig_R = plot_fracture_list(Fr_list, variable="d_mean", plot_prop=plot_prop_R)
    
#     # наложение аналитического решения для радиуса
#     Fig_R = plot_analytical_solution("M", "d_mean", Solid, Injection, fluid_prop=Fluid,
#                                      time_srs=time_srs, fig=Fig_R)

#     # 2. Построение графика раскрытия в центре
#     plot_prop_w = PlotProperties()
#     plot_prop_w.lineStyle = "."               # стиль линии - точки
#     plot_prop_w.graphScaling = "loglog"       # логарифмический масштаб
#     Fig_w = plot_fracture_list_at_point(Fr_list, variable="w", plot_prop=plot_prop_w)
    
#     # наложение аналитического решения для раскрытия в центре
#     Fig_w = plot_analytical_solution_at_point("M", "w", Solid, Injection,
#                                               fluid_prop=Fluid, time_srs=time_srs, fig=Fig_w)

#     # 3. Построение отпечатков трещин для конкретных моментов времени
#     specific_times = [2.0, 200.0, 5000.0, 30000.0, 100000.0]
#     Fr_list_snapshots, properties_snapshots = load_fractures(address="./Data/M_radial_explicit",
#                                                              time_srs=specific_times)
#     time_srs_snapshots = get_fracture_variable(Fr_list_snapshots, variable="time")

#     # отпечаток трещины (сетка + фронт)
#     Fig_FP = plot_fracture_list(Fr_list_snapshots, variable="mesh", projection="2D")
#     Fig_FP = plot_fracture_list(Fr_list_snapshots, variable="footprint", projection="2D", fig=Fig_FP)
#     # наложение аналитического отпечатка
#     Fig_FP = plot_analytical_solution("M", "footprint", Solid, Injection, fluid_prop=Fluid,
#                                       time_srs=time_srs_snapshots, projection="2D", fig=Fig_FP)

#     # 4. Построение среза раскрытия
#     # Создаем массив для возврата крайних точек (аналог np.empty)
#     ext_pnts = Array{Float64}(undef, 2, 2) 
#     Fig_WS = plot_fracture_list_slice(Fr_list_snapshots, variable="w", projection="2D",
#                                       plot_cell_center=true, extreme_points=ext_pnts)
#     # наложение аналитического среза
#     Fig_WS = plot_analytical_solution_slice("M", "w", Solid, Injection, fluid_prop=Fluid,
#                                             fig=Fig_WS, time_srs=time_srs_snapshots,
#                                             point1=ext_pnts[1, :], point2=ext_pnts[2, :]) # Передаем строки как векторы

#     # 5. Построение 3D визуализации
#     Fig_Fr = plot_fracture_list(Fr_list_snapshots, variable="mesh", projection="3D")
#     Fig_Fr = plot_fracture_list(Fr_list_snapshots, variable="width", projection="3D", fig=Fig_Fr)
#     Fig_Fr = plot_fracture_list(Fr_list_snapshots, variable="footprint", projection="3D", fig=Fig_Fr)

#     # Отображение всех графиков
#     println("Displaying plots...")
#     PyPlot.show(block=true)
#     println("Plots displayed.")

# else
#     println("Batch run detected. Skipping visualization.")
# end

println("Script finished.")
