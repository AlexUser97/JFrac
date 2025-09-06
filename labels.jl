# -*- coding: utf-8 -*-
"""
This file is part of JFrac.
Realization of Pyfrac on Julia language.

"""

module Labels

export Fig_labels, var_labels, units, unit_conversion, supported_variables,
       unidimensional_variables, bidimensional_variables, required_string,
       err_msg_variable, supported_projections, suitable_elements, err_var_not_saved, TS_errorMessages

using OrderedCollections

const Fig_labels = OrderedDict{String, String}([
    "t" => "Time",
    "time" => "Time",
    "w" => "Fracture Width",
    "width" => "Fracture Width",
    "pf" => "Fluid Pressure",
    "fluid pressure" => "Fluid Pressure",
    "pn" => "net Pressure",
    "net pressure" => "net Pressure",
    "front velocity" => "Front Velocity",
    "v" => "Front Velocity",
    "Reynolds number" => "Reynold's number",
    "Re" => "Reynold's number",
    "dpdx" => "horizontal pressure gradient",
    "pressure gradient x" => "horizontal pressure gradient",
    "dpdy" => "vertical pressure gradient",
    "pressure gradient y" => "vertical pressure gradient",
    "fluid flux" => "Fluid Flux",
    "ff" => "Fluid Flux",
    "fluid velocity" => "Fluid Velocity",
    "fv" => "Fluid Velocity",
    "fluid velocity as vector field" => "Fluid Velocity",
    "fvvf" => "Fluid Velocity",
    "fluid flux as vector field" => "Fluid Flux",
    "ffvf" => "Fluid Flux",
    "front_dist_min" => "Closest Distance to Front",
    "d_min" => "Closest Distance to Front",
    "front_dist_max" => "Farthest Distance to Front",
    "d_max" => "Farthest Distance to Front",
    "front_dist_mean" => "Mean Distance to Front",
    "d_mean" => "Mean Distance to Front",
    "V" => "Total Volume",
    "volume" => "Total Volume",
    "lk" => "Leak off",
    "leak off" => "Leak off",
    "lkt" => "Total Leaked of Volume",
    "leaked off total" => "Total Leaked of Volume",
    "ar" => "Aspect Ratio",
    "aspect ratio" => "Aspect Ratio",
    "efficiency" => "Fracture Efficiency",
    "ef" => "Fracture Efficiency",
    "mesh" => "Mesh",
    "footprint" => "Fracture Footprint",
    "surface" => "Fracture Surface",
    "chi" => "Tip leak-off parameter",
    "regime" => "Propagation Regime",
    "source elements" => "Source Elements",
    "se" => "Source Elements",
    "effective viscosity" => "Effective Viscosity",
    "ev" => "Effective Viscosity",
    "prefactor G" => "G",
    "G" => "G",
    "injection line pressure" => "Injection Line Pressure",
    "ilp" => "Injection Line Pressure",
    "injection rate" => "Injection Rate",
    "ir" => "Injection Rate",
    "total injection rate" => "Total Injection Rate",
    "tir" => "Total Injection Rate"
])

const var_labels = OrderedDict{String, String}([
    "t" => "time",
    "time" => "time",
    "w" => "width",
    "width" => "width",
    "pf" => "pressure",
    "fluid pressure" => "pressure",
    "pn" => "pressure",
    "net pressure" => "pressure",
    "front velocity" => "front velocity",
    "v" => "Front Velocity",
    "Reynolds number" => "Reynold's number",
    "Re" => "Reynold's number",
    "dpdx" => "horizontal pressure gradient",
    "pressure gradient x" => "horizontal pressure gradient",
    "dpdy" => "vertical pressure gradient",
    "pressure gradient y" => "vertical pressure gradient",
    "fluid flux" => "fluid flux",
    "ff" => "fluid flux",
    "fluid velocity" => "fluid velocity",
    "fv" => "fluid velocity",
    "fluid velocity as vector field" => "fluid velocity",
    "fvvf" => "fluid velocity",
    "fluid flux as vector field" => "fluid flux",
    "ffvf" => "fluid flux",
    "front_dist_min" => "\$R_{min}\$",
    "d_min" => "\$R_{min}\$",
    "front_dist_max" => "\$R_{max}\$",
    "d_max" => "\$R_{max}\$",
    "front_dist_mean" => "\$R_{mean}\$",
    "d_mean" => "\$R_{mean}\$",
    "V" => "total volume",
    "volume" => "total volume",
    "lk" => "leak off",
    "leak off" => "leak off",
    "lkt" => "total leaked off volume",
    "leaked off total" => "total leaked off volume",
    "ar" => "aspect ratio",
    "aspect ratio" => "aspect ratio",
    "efficiency" => "fracture efficiency",
    "ef" => "fracture efficiency",
    "mesh" => "",
    "footprint" => "",
    "surface" => "",
    "chi" => "",
    "regime" => "regime",
    "source elements" => "source elements",
    "se" => "source elements",
    "effective viscosity" => "effective viscosity",
    "ev" => "effective viscosity",
    "prefactor G" => "G",
    "G" => "G",
    "injection line pressure" => "injection line pressure",
    "ilp" => "injection line pressure",
    "injection rate" => "injection rate",
    "ir" => "injection rate",
    "total injection rate" => "total injection rate",
    "tir" => "total injection rate"
])

const units = OrderedDict{String, String}([
    "t" => "(\\\$s\\\$)",
    "time" => "(\\\$s\\\$)",
    "w" => " (\\\$mm\\\$)",
    "width" => " (\\\$mm\\\$)",
    "pf" => " (\\\$MPa\\\$)",
    "fluid pressure" => " (\\\$MPa\\\$)",
    "pn" => " (\\\$MPa\\\$)",
    "net pressure" => " (\\\$MPa\\\$)",
    "front velocity" => " (\\\$m/s\\\$)",
    "v" => " (\\\$m/s\\\$)",
    "Reynolds number" => "",
    "Re" => "",
    "dpdx" => "(\\\$MPa/m\\\$)",
    "pressure gradient x" => "(\\\$MPa/m\\\$)",
    "dpdy" => "(\\\$MPa/m\\\$)",
    "pressure gradient y" => "(\\\$MPa/m\\\$)",
    "fluid flux" => " (\\\$m^2/s\\\$)",
    "ff" => " (\\\$m^2/s\\\$)",
    "fluid velocity" => " (\\\$m/s\\\$)",
    "fv" => " (\\\$m/s\\\$)",
    "fluid velocity as vector field" => " (\\\$m/s\\\$)",
    "fvvf" => " (\\\$m/s\\\$)",
    "fluid flux as vector field" => " (\\\$m^2/s\\\$)",
    "ffvf" => " (\\\$m^2/s\\\$)",
    "front_dist_min" => " (\\\$meters\\\$)",
    "d_min" => " (\\\$meters\\\$)",
    "front_dist_max" => " (\\\$meters\\\$)",
    "d_max" => " (\\\$meters\\\$)",
    "front_dist_mean" => " (\\\$meters\\\$)",
    "d_mean" => " (\\\$meters\\\$)",
    "V" => " \\$m^3\\\$",
    "volume" => " \\$m^3\\\$",
    "lk" => " \\$m^3\\\$",
    "leak off" => " \\$m^3\\\$",
    "lkt" => " \\$m^3\\\$",
    "leaked off total" => " \\$m^3\\\$",
    "ar" => "",
    "aspect ratio" => "",
    "efficiency" => "",
    "ef" => "",
    "mesh" => "",
    "footprint" => "",
    "surface" => " (\\\$mm\\\$)",
    "chi" => "",
    "regime" => "",
    "source elements" => "",
    "se" => "",
    "effective viscosity" => "(\\\$Pa\\\\cdot s\\\$)",
    "ev" => "(\\\$Pa\\\\cdot s\\\$)",
    "prefactor G" => "",
    "G" => "",
    "injection line pressure" => " MPa",
    "ilp" => " MPa",
    "injection rate" => " \\$m^3/s\\\$",
    "ir" => " \\$m^3/s\\\$",
    "total injection rate" => " \\$m^3/s\\\$",
    "tir" => " \\$m^3/s\\\$"
])

const unit_conversion = OrderedDict{String, Union{Float64, Nothing}}([
    "t" => 1.0,
    "time" => 1.0,
    "w" => 1.e-3,
    "width" => 1.e-3,
    "pf" => 1.e6,
    "fluid pressure" => 1.e6,
    "pn" => 1.e6,
    "net pressure" => 1.e6,
    "front velocity" => 1.0,
    "v" => 1.0,
    "Reynolds number" => 1.0,
    "Re" => 1.0,
    "dpdx" => 1.e6,
    "pressure gradient x" => 1.e6,
    "dpdy" => 1.e6,
    "pressure gradient y" => 1.e6,
    "fluid flux" => 1.0,
    "ff" => 1.0,
    "fluid velocity" => 1.0,
    "fv" => 1.0,
    "fluid velocity as vector field" => 1.0,
    "fvvf" => 1.0,
    "fluid flux as vector field" => 1.0,
    "ffvf" => 1.0,
    "front_dist_min" => 1.0,
    "d_min" => 1.0,
    "front_dist_max" => 1.0,
    "d_max" => 1.0,
    "front_dist_mean" => 1.0,
    "d_mean" => 1.0,
    "V" => 1.0,
    "volume" => 1.0,
    "lk" => 1.0,
    "leak off" => 1.0,
    "lkt" => 1.0,
    "leaked off total" => 1.0,
    "ar" => 1.0,
    "aspect ratio" => 1.0,
    "efficiency" => 100.0,
    "ef" => 100.0,
    "mesh" => nothing,
    "footprint" => nothing,
    "surface" => 1.e-3,
    "chi" => 1.0,
    "regime" => 1.0,
    "source elements" => 1.0,
    "se" => 1.0,
    "effective viscosity" => 1.0,
    "ev" => 1.0,
    "prefactor G" => 1.0,
    "G" => 1.0,
    "injection line pressure" => 1e6,
    "ilp" => 1e6,
    "injection rate" => 1.0,
    "ir" => 1.0,
    "total injection rate" => 1.0,
    "tir" => 1.0
])

const supported_variables = [
    "w", "width", "pf", "fluid pressure", "pn", "net pressure",
    "front velocity", "v", "Reynolds number", "Re", "fluid flux", "ff",
    "fluid velocity", "fv",
    "dpdx", "pressure gradient x", "dpdy", "pressure gradient y",
    "fluid velocity as vector field","fvvf","fluid flux as vector field","ffvf",
    "front_dist_min", "d_min",
    "front_dist_max", "d_max", "front_dist_mean",
    "d_mean", "mesh", "footprint", "t", "time", "volume",
    "V", "lk", "leak off", "lkt", "leaked off total",
    "ar", "aspect ratio", "efficiency", "ef", "surface", "front intercepts", "fi",
    "chi", "regime", "source elements", "se", "effective viscosity", "ev",
    "prefactor G", "G", "injection line pressure", "ilp", "injection rate", "ir",
    "total injection rate", "tir"
]

const unidimensional_variables = [
    "time", "t", "front_dist_min", "d_min", "front_dist_max",
    "d_max", "V", "volume", "front_dist_mean", "d_mean",
    "efficiency", "ef", "aspect ratio", "ar", "lkt", "leaked off total",
    "injection line pressure", "ilp", "total injection rate", "tir"
]

const bidimensional_variables = [
    "w", "width", "pf", "fluid pressure", "pn", "net pressure",
    "dpdx", "pressure gradient x", "dpdy", "pressure gradient y",
    "front velocity", "v", "Reynolds number", "Re", "fluid flux", "ff",
    "fluid velocity", "fv", "fluid velocity as vector field","fvvf","fluid flux as vector field","ffvf",
    "fluid velocity", "fv", "lk", "leak off", "surface", "front intercepts", "fi",
    "chi", "regime", "effective viscosity", "ev", "prefactor G", "G",
    "injection rate", "ir"
]

const required_string = OrderedDict{String, String}([
    "t" => "100000",
    "time" => "100000",
    "w" => "000100",
    "width" => "000100",
    "pn" => "001000",
    "net pressure" => "001000",
    "front velocity" => "000010",
    "v" => "000010",
    "front_dist_min" => "010000",
    "d_min" => "010000",
    "front_dist_max" => "010000",
    "d_max" => "010000",
    "front_dist_mean" => "010000",
    "d_mean" => "010000",
    "radius" => "010000",
    "r" => "010000"
])

const err_msg_variable = """Given variable is not supported. Select one of the following:
-- 'w' or 'width'
-- 'pf' or 'fluid pressure'
-- 'pn' or 'net pressure'
-- 'Re' or 'Reynolds number'
-- 'v' or 'front velocity'
-- 'ff' or 'fluid flux'
-- 'fv' or 'fluid velocity'
-- 'd_min' or 'front_dist_min'
-- 'd_max' or 'front_dist_max'
-- 'd_mean' or 'front_dist_mean'
-- 'mesh'
-- 'footprint'
-- 'V' or 'volume'
-- 'lk' or 'leak off'
-- 'lkt' or 'leaked off total'
-- 'ar' or 'aspect ratio'
-- 'ef' or 'efficiency'
-- 'surface'
-- 'chi'
-- 'regime'
-- 'se' or 'source elements'
-- 'ev' or 'effective viscosity'
-- 'prefactor G' or 'G'
-- 'injection line pressure' or 'ilp'
-- 'injection rate' or 'ir'
-- 'total injection rate' or 'tir'
"""

const supported_projections = OrderedDict{String, Vector{String}}([
    "w" => ["2D_clrmap", "2D_contours", "3D"],
    "width" => ["2D_clrmap", "2D_contours", "3D"],
    "pf" => ["2D_clrmap", "2D_contours", "3D"],
    "fluid pressure" => ["2D_clrmap", "2D_contours", "3D"],
    "pn" => ["2D_clrmap", "2D_contours", "3D"],
    "net pressure" => ["2D_clrmap", "2D_contours", "3D"],
    "front velocity" => ["2D_clrmap", "2D_contours"],
    "v" => ["2D_clrmap", "2D_contours"],
    "Reynolds number" => ["2D_clrmap", "2D_contours", "3D"],
    "Re" => ["2D_clrmap", "2D_contours", "3D"],
    "dpdx" => ["2D_clrmap", "2D_contours"],
    "pressure gradient x" => ["2D_clrmap", "2D_contours"],
    "dpdy" => ["2D_clrmap", "2D_contours"],
    "pressure gradient y" => ["2D_clrmap", "2D_contours"],
    "fluid flux" => ["2D_clrmap", "2D_contours", "3D"],
    "ff" => ["2D_clrmap", "2D_contours", "3D"],
    "fluid velocity" => ["2D_clrmap", "2D_contours", "3D"],
    "fv" => ["2D_clrmap", "2D_contours", "3D"],
    "fluid flux as vector field" => ["2D_vectorfield"],
    "ffvf" => ["2D_vectorfield"],
    "fluid velocity as vector field" => ["2D_vectorfield"],
    "fvvf" => ["2D_vectorfield"],
    "front_dist_min" => ["1D"],
    "d_min" => ["1D"],
    "front_dist_max" => ["1D"],
    "d_max" => ["1D"],
    "front_dist_mean" => ["1D"],
    "d_mean" => ["1D"],
    "mesh" => ["2D", "3D"],
    "footprint" => ["2D", "3D"],
    "t" => ["1D"],
    "time" => ["1D"],
    "volume" => ["1D"],
    "V" => ["1D"],
    "lk" => ["2D_clrmap", "2D_contours", "3D"],
    "leak off" => ["2D_clrmap", "2D_contours", "3D"],
    "lkt" => ["1D"],
    "leaked off total" => ["1D"],
    "ar" => ["1D"],
    "aspect ratio" => ["1D"],
    "efficiency" => ["1D"],
    "ef" => ["1D"],
    "surface" => ["3D"],
    "chi" => ["2D_clrmap", "2D_contours"],
    "regime" => ["2D_clrmap", "2D_contours", "3D"],
    "source elements" => ["2D_clrmap", "2D_contours", "3D"],
    "se" => ["2D_clrmap", "2D_contours", "3D"],
    "effective viscosity" => ["2D_clrmap", "2D_contours", "3D"],
    "ev" => ["2D_clrmap", "2D_contours", "3D"],
    "prefactor G" => ["2D_clrmap", "2D_contours", "3D"],
    "G" => ["2D_clrmap", "2D_contours", "3D"],
    "injection line pressure" => ["1D"],
    "ilp" => ["1D"],
    "injection rate" => ["2D_clrmap", "2D_contours", "3D"],
    "ir" => ["2D_clrmap", "2D_contours", "3D"],
    "total injection rate" => ["1D"],
    "tir" => ["1D"]
])

const suitable_elements = OrderedDict{String, Union{String, Nothing}}([
    "w" => "crack",
    "width" => "crack",
    "pf" => "crack",
    "fluid pressure" => "crack",
    "pn" => "crack",
    "net pressure" => "crack",
    "front velocity" => "crack",
    "v" => "crack",
    "Reynolds number" => "channel",
    "Re" => "channel",
    "fluid flux" => "channel",
    "ff" => "channel",
    "fluid velocity" => "channel",
    "fv" => "channel",
    "fluid flux as vector field" => "channel",
    "ffvf" => "channel",
    "fluid velocity as vector field" => "channel",
    "fvvf" => "channel",
    "front_dist_min" => nothing,
    "d_min" => nothing,
    "front_dist_max" => nothing,
    "d_max" => nothing,
    "front_dist_mean" => nothing,
    "d_mean" => nothing,
    "mesh" => nothing,
    "footprint" => nothing,
    "t" => nothing,
    "time" => nothing,
    "volume" => nothing,
    "V" => nothing,
    "lk" => "crack",
    "leak off" => "crack",
    "lkt" => nothing,
    "leaked off total" => nothing,
    "ar" => nothing,
    "aspect ratio" => nothing,
    "efficiency" => nothing,
    "ef" => nothing,
    "surface" => "crack",
    "chi" => "crack",
    "regime" => "crack",
    "source elements" => "crack",
    "se" => "crack",
    "effective viscosity" => "channel",
    "ev" => "channel",
    "prefactor G" => "channel",
    "G" => "channel",
    "injection line pressure" => nothing,
    "ilp" => nothing,
    "injection rate" => "crack",
    "ir" => "crack",
    "total injection rate" => nothing,
    "tir" => nothing
])

const err_var_not_saved = """The required variable is not available. Probably, saving of the variable was not
enabled during the simulation. Enable saving it through simulation properties."""

const TS_errorMessages = [
    "Propagation not attempted!",                                                     #0
    "Time step successful!",                                                          #1
    "Evaluated level set is not valid!",                                              #2
    "Front is not tracked correctly!",                                                #3
    "Evaluated tip volume is not valid!",                                             #4
    "Solution obtained from the elastohydrodynamic solver is not valid!",             #5
    "Did not converge after max iterations!",                                         #6
    "Tip inversion is not correct!",                                                  #7
    "Ribbon element not found in the enclosure of the tip cell!",                     #8
    "Filling fraction not correct!",                                                  #9
    "Toughness iteration did not converge!",                                          #10
    "projection could not be found!",                                                 #11
    "Reached end of grid!",                                                           #12
    "Leak off can't be evaluated!",                                                   #13
    "fracture fully closed",                                                          #14
    "iterating more is not leading the iterations on the front position to converge!",#15
    "maximum number of elements in the crack reached!"                                #16
]

end # module Labels