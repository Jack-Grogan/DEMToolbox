import numpy as np
import warnings

from ..particle_sampling.sample_2d_slice import sample_2d_slice

def velocity_vector_field(particle_data, container_data, point, vector_1, 
                          vector_2, plane_thickness, resolution,
                          bounds=None,
                          sample_column=None, 
                          velocity_column="v",
                          append_column="mean_resolved_velocity",
                          particle_id_column="id"):
    """Calculate the velocity vector field of a 2D slice of particles.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    container_data : vtkPolyData
        The container vtk.
    point : list
        A point on the plane as [x, y, z].
    vector_1 : list
        The first sample vector to split the particles along.
    vector_2 : list
        The second sample vector to split the particles along.
    plane_thickness : int or float
        The thickness of the plane.
    resolution : list
        The resolution of the 2D sample space in the form [m, n].
    bounds : list, optional
        The bounds of the sample space in the form [vec_1_lower_bound,
        vec_1_upper_bound, vec_2_lower_bound, vec_2_upper_bound].
        If None, the bounds will be determined from the container data,
        by default None.
    sample_column : str, optional
        The name of the samples column to append to the particle data,
        by default None. If None, the column name will be
        "particle_slice_p{point}_n{normal}".
    velocity_column : str, optional
        The name of the velocity column in the particle data, by default "v".
    append_column : str, optional
        The name of the column to append to the particle data, by default
        "mean_resolved_velocity".
    particle_id_column : str, optional
        The name of the particle id column in the particle data, by
        default "id".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the mean resolved velocity column added.
    velocity_vectors : tuple
        An array of the mean resolved velocity vectors for particles in the
        sample space.
    samples : ParticleSamples
        The samples object containing the sample information that was
        used to calculate the velocity vector field. if no particles
        are in the sample space, the samples object will be None.


    Raises
    ------
    ValueError
        If vectors are not 3 element lists.
    ValueError
        If point is not a 3 element list.
    ValueError
        If resolution is not a 2 element list of integers.
    ValueError
        If resolution is less than or equal to 0.
    ValueError
        If plane_thickness is not an integer or float.
    ValueError
        If plane_thickness is less than or equal to 0.
    VaueError
        If vectors are not orthogonal.
    UserWarning
        If the particle data has no points return unedited particle data
        and NaN array for the velocity vectors.
    UserWarning
        If the container data has no points return unedited particle data
        and NaN array for the velocity vectors.
    """
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        velocity_vectors = np.zeros((resolution[1], resolution[0], 2))
        velocity_vectors[:] = np.nan
        return particle_data, velocity_vectors, None
    
    if container_data.n_points == 0:
        warnings.warn("Cannot sample empty container file.", UserWarning)
        velocity_vectors = np.zeros((resolution[1], resolution[0], 2))
        velocity_vectors[:] = np.nan
        return particle_data, velocity_vectors, None
    
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    particle_data, samples = sample_2d_slice(particle_data, 
                                        container_data, 
                                        point, 
                                        vector_1, 
                                        vector_2,
                                        plane_thickness, 
                                        resolution,
                                        bounds=bounds,
                                        append_column=sample_column,
                                        particle_id_column=particle_id_column
                                        )
    
    cell_velocity = np.zeros((particle_data.n_points, 3))
    cell_velocity[:] = np.nan

    velocity_vectors = np.zeros((resolution[1] * resolution[0], 2))
    velocity_vectors[:] = np.nan
    
    for ids in samples.occupied_cells:
        sample_boolean_mask = particle_data[samples.name] == ids

        particle_velocities = particle_data.point_data[velocity_column][sample_boolean_mask]
        mean_velocity_vector = np.mean(particle_velocities, axis=0)
        mean_res_vec_1_vel = np.dot(mean_velocity_vector, vector_1)
        mean_res_vec_2_vel = np.dot(mean_velocity_vector, vector_2)

        resolved_velocity_vector = (mean_res_vec_1_vel 
                                    * vector_1 
                                    + mean_res_vec_2_vel 
                                    * vector_2)

        velocity_vectors[ids] = np.array([mean_res_vec_1_vel, 
                                        mean_res_vec_2_vel])
        
        cell_velocity[sample_boolean_mask] = resolved_velocity_vector

    velocity_vectors = velocity_vectors.reshape(resolution[1],
                                                resolution[0], 2)

    particle_data[append_column] = cell_velocity

    return particle_data, velocity_vectors, samples