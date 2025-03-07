import numpy as np
import warnings

def particle_slice(particle_data, point, normal, plane_thickness):

    # Check if the particles file has points
    if particle_data.n_points == 0:
        warnings.warn("cannot slice empty particles file")
        return

    # Make normal a unit vector
    normal = normal / np.linalg.norm(normal)

    bottom_plane = (np.dot(normal, particle_data.points.T) -
                    np.dot(normal, point) -
                    plane_thickness / 2
                    <= 0)

    top_plane = (np.dot(normal, particle_data.points.T) -
                    np.dot(normal, point) +
                    plane_thickness / 2
                    >= 0)


    slice_boolean_mask = bottom_plane & top_plane

    particle_slice_column = (
        f"particle_slice_p{''.join(str(p_i) for p_i in point)}"
        f"_n{''.join(str(n_i) for n_i in normal)}"
    )

    particle_data[particle_slice_column] = slice_boolean_mask.astype(int)

    return particle_data, particle_slice_column