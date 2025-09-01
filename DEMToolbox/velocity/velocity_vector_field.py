import numpy as np
import warnings

from ..particle_sampling.sample_2d_slice import sample_2d_slice

def velocity_vector_field(particle_data, bounds, point, vector_1, 
                          vector_2, plane_thickness, resolution,
                          sample_column=None, 
                          velocity_column="v",
                          append_column="mean_resolved_velocity",
                          particle_id_column="id"):
    """Calculate the velocity vector field of a 2D slice of particles.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    bounds : list, np.ndarray or vtkPolyData
        If a list or np.ndarray bounds of the sample space in the form 
        [vec_1_lower_bound, vec_1_upper_bound, vec_2_lower_bound, 
        vec_2_upper_bound]. If a vtk, the bounds will be determined from 
        the vtk's bounds.
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
    occupancy : np.ndarray
        An array of the number of particles in each sample
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
        occupancy = np.zeros((resolution[1], resolution[0]))
        return particle_data, velocity_vectors, occupancy, None
    
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    particle_data, samples = sample_2d_slice(particle_data, 
                                        bounds, 
                                        point, 
                                        vector_1, 
                                        vector_2,
                                        plane_thickness, 
                                        resolution,
                                        append_column=sample_column,
                                        particle_id_column=particle_id_column
                                        )

    
    cell_ids = particle_data[samples.name].astype(int)
    velocities = particle_data.point_data[velocity_column]

    # Filter out -1 IDs (out of mesh elements)
    valid_mask = cell_ids != -1
    cell_ids_valid = cell_ids[valid_mask]
    velocities_valid = velocities[valid_mask]

    # Compute sum of velocities in each cell
    sum_vel = np.zeros((resolution[1] * resolution[0], 3))
    np.add.at(sum_vel, cell_ids_valid, velocities_valid)

    mean_vel = np.zeros_like(sum_vel)
    mean_vel[samples.occupied_cells] = ( 
                    sum_vel[samples.occupied_cells] 
                    / samples.particles[samples.occupied_cells, None]
                    )

    # Project onto vector_1 and vector_2
    mean_res_vec_1 = mean_vel @ vector_1
    mean_res_vec_2 = mean_vel @ vector_2

    resolved_velocity_vector = (
        mean_res_vec_1[:, None] * vector_1 +
        mean_res_vec_2[:, None] * vector_2
    )

    # Store results
    velocity_vectors = np.column_stack([mean_res_vec_1, mean_res_vec_2])
    cell_velocity = resolved_velocity_vector[cell_ids]

    # Set invalid cells to NaN
    cell_velocity[~valid_mask] = [np.nan, np.nan, np.nan]

    velocity_vectors = velocity_vectors.reshape(resolution[1],
                                                resolution[0], 2)

    particle_data[append_column] = cell_velocity
    occupancy = samples.particles.reshape(resolution[1], 
                                          resolution[0])

    return particle_data, velocity_vectors, occupancy, samples