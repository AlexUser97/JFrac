module JFrac

    include("default_parameters.jl")
    include("fluid_model.jl")
    include("labels.jl")
    include("mesh.jl")
    include("symmetry.jl")
    
    include("anisotropy.jl")
    include("continuous_front_reconstruction.jl")
    include("controller.jl")
    include("elasticity.jl")
    include("elastohydrodynamic_solver.jl")
    include("explicit_RKL.jl")
    include("fracture.jl")
    include("fracture_initialization.jl")
    include("HF_reference_solutions.jl")
    include("level_set.jl")
    include("mesh.jl")
    include("postprocess_fracture.jl")
    include("postprocess_performance.jl")
    include("properties.jl")
    include("time_step_solution.jl")
    include("tip_inversion.jl")
    include("utility.jl")
    include("visualization.jl")
    include("volume_integral.jl")

    export Anisotropy, ContinuousFrontReconstruction, Controller, DefaultParameters, Elasticity, ElastohydrodynamicSolver,
           ExplicitRKL, FluidModel, FractureModule, FractureInitialization, HFReferenceSolutions, Labels, LevelSet, Mesh, PostprocessFracture, PostprocessPerformance,
           TimeStepSolution, TipInversion, Utility, Visualization, VolumeIntegral, Properties, Symmetry

end # module