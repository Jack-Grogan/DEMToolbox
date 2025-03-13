import numpy as np
import warnings

from ..classes.slice_class import Slice

def particle_slice(particle_data, point, normal, plane_thickness, slice_column=None):
    """Identify particles that lie within a planar slice."

    Identify particles that lie within a planar slice defined by a 
    point and normal vector.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    point : list
        The point on the plane as [x, y, z].
    normal : list
        The normal vector to the plane as [x, y, z].
    plane_thickness : int or float
        The thickness of the plane.
    slice_column : str, optional
        The name of the slice column in the particle data,
        by default None. If None, the column name will be
        "particle_slice_p{point}_n{normal}".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the slice column added.
    particle_slice : Slice
        A slice object containing the slice column, the number of
        particles in the slice, the normal vector, the point on the
        plane and the plane thickness

    Raises
    ------
    ValueError
        If point or normal are not 3 element lists.
    ValueError
        If plane_thickness is not an integer or float.
    UserWarning
        If the particle data has no points return unedited
        particle data and None for column
    """
    if particle_data.n_points == 0:
        warnings.warn("cannot slice empty particles file", UserWarning)
        particle_slice = Slice(slice_column, 0, 0, 
                               normal, point, plane_thickness)
        return particle_data, particle_slice
    
    if len(point) != 3 or len(normal) != 3:
        raise ValueError("point and normal must be 3 element lists")
    
    if not isinstance(plane_thickness, (int, float)):
        raise ValueError("plane_thickness must be an integer or float")

    if slice_column is None:
        slice_column = (
            f"particle_slice_p{''.join(str(p_i) for p_i in point)}"
            f"_n{''.join(str(n_i) for n_i in normal)}"
        )

    # Make normal a unit vector
    normal = normal / np.linalg.norm(normal)

    bottom_plane = (np.dot(normal, particle_data.points.T) 
                    - np.dot(normal, point)
                    - plane_thickness / 2
                    <= 0)
    
    top_plane = (np.dot(normal, particle_data.points.T) 
                 - np.dot(normal, point) 
                 + plane_thickness / 2
                 >= 0)

    slice_boolean_mask = bottom_plane & top_plane
    particle_data[slice_column] = slice_boolean_mask.astype(int)

    n_inside_particles = np.sum(slice_boolean_mask)
    n_outside_particles = np.sum(~slice_boolean_mask)

    particle_slice = Slice(slice_column, n_inside_particles, 
                           n_outside_particles, normal, point, 
                           plane_thickness)
    
    return particle_data, particle_slice