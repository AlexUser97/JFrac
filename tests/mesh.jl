module Mesh
    function apply_all_transformations(coord)
        if coord === nothing
            return nothing
        else
            return Visualization.apply_zoom(coord) + Symmetry.apply_symmetry(coord)
        end
    end
end