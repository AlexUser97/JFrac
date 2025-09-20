module Visualization
    function apply_zoom(coord::Int)::Int
        return coord * 2
    end

    function apply_zoom_with_symmetry(coord::Int)::Int
        return apply_zoom(coord) + Symmetry.apply_symmetry(coord)
    end
end