module Symmetry
    # Применяет симметрию к координате (например, отражение)
    function apply_symmetry(coord::Int)::Int
        return -coord  # Отражение относительно нуля
    end

    # Применяет симметрию и увеличивает масштаб
    function apply_symmetry_with_zoom(coord::Int)::Int
        return Visualization.apply_zoom(coord) + apply_symmetry(coord)
    end
end
