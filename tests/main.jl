include("visualization.jl")
include("symmetry.jl")
include("mesh.jl")

# Пример: применяем все преобразования к координате 5
result = Mesh.apply_all_transformations(5)
println(result)  # Выведет: 5 (5*2 + (-5) = 5)