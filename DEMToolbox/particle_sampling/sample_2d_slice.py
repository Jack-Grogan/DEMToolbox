import numpy as np
import warnings

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_2d_slice(particle_data, 
                    bounds, 
                    point, 
                    vector_1, 
                    vector_2,
                    plane_thickness, 
                    resolution, 
                    append_column=None, 
                    particle_id_column="id"):
    """Split the particles into samples split along a 2D slice.

    Split the particles into samples split along a 2D slice defined by a
    point and two orthogonal vectors.

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
    append_column : str, optional
        The name of the samples column to append to the particle data,
        by default None. If None, the column name will be
        "particle_slice_p{point}_n{normal}".
    particle_id_column : str, optional
        The name of the particle id column in the particle data, by
        default "id".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the samples column added.
    samples : ParticleSamples
        A ParticleSamples object containing the samples column name,
        the particle ids and their corresponding sample ids stored in a
        ParticleAttribute object, a list of sample elements, a list of
        occupied sample elements, a list of the number of particles in
        the sample elements, the number of particles in the sample
        elements, the number of sampled and unsampled particles.

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
    ValueError
        If vectors are not orthogonal.
    UserWarning
        If the particle data has no points return unedited particle
        data and an empty samples object.
    UserWarning
        If the container data has no points return unedited particle
        data and an empty samples object
    """
    if len(vector_1) != 3 or len(vector_2) != 3:
        raise ValueError("Vectors must be 3 element lists.")
    
    if len(point) != 3:
        raise ValueError("Point must be a 3 element list.")
    
    if len(resolution) != 2:
        raise ValueError("Resolution must be a 2 element list.")
    
    if not all(isinstance(i, int) for i in resolution):
        raise ValueError("Resolution must be an integer.")

    if any(i <= 0 for i in resolution):
        raise ValueError("Resolution must be greater than 0.")

    if not isinstance(plane_thickness, (int, float)):
        raise ValueError("Plane_thickness must be an integer or float.")
    
    if plane_thickness <= 0:
        raise ValueError("Plane_thickness must be greater than 0.")

    # Check the vectors are orthogonal
    dot_product = np.dot(vector_1, vector_2)
    if dot_product != 0:
        raise ValueError("Sample vectors must be orthogonal to each other.")
    
    # Make all vectors unit vectors
    vector_1 =  np.asarray(vector_1) / np.linalg.norm(vector_1)
    vector_2 =  np.asarray(vector_2) / np.linalg.norm(vector_2)
    normal = np.cross(vector_1, vector_2)
    normal = normal / np.linalg.norm(normal)

    if append_column is None:
        append_column = (
            f"particle_slice_p{''.join(str(p_i) for p_i in point)}"
            f"_n{''.join(str(n_i) for n_i in normal)}"
        )
        
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return (particle_data, samples)
    
    if isinstance(bounds, list or np.ndarray):
        
        if len(bounds) != 4:
            raise ValueError("Bounds must be a list of 4 elements.")
        
        if not all(isinstance(i, (int, float)) for i in bounds):
            raise ValueError("Bounds must be a list of integers or floats.")
        
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Bounds must be in the form "
                             "[vec_1_lower_bound, vec_1_upper_bound, "
                             "vec_2_lower_bound, vec_2_upper_bound].")
        
        vec_1_sample_bounds = np.linspace(bounds[0], bounds[1],
                                          resolution[0] + 1)
        vec_2_sample_bounds = np.linspace(bounds[2], bounds[3],
                                          resolution[1] + 1)
    
    else:
        if bounds.n_points == 0:
            warnings.warn("Cannot sample empty container file.", UserWarning)
            sample_attribute = ParticleAttribute(particle_id_column, 
                                            append_column,
                                            np.empty((0, 2)))
            samples = ParticleSamples(
                append_column, sample_attribute, [], [], [], 0, 0)
            return (particle_data, samples)

        # Resolve the particles along the vectors
        resolved_particles_vec_1 = np.dot(particle_data.points, 
                                        vector_1)
        resolved_particles_vec_2 = np.dot(particle_data.points, 
                                        vector_2)

        # Resolve container along the vectors
        resolved_container_vec_1 = np.dot(bounds.points, 
                                        vector_1)
        resolved_container_vec_2 = np.dot(bounds.points, 
                                        vector_2)

        # Define the sample bounds linearly along the resolved container
        vec_1_sample_bounds = np.linspace(min(resolved_container_vec_1),
                                          max(resolved_container_vec_1),
                                          resolution[0] + 1)
        vec_2_sample_bounds = np.linspace(min(resolved_container_vec_2),
                                          max(resolved_container_vec_2),
                                          resolution[1] + 1)
        
    # Define the slice boolean mask
    bottom_plane = (np.dot(normal, particle_data.points.T) 
                - np.dot(normal, point)
                - plane_thickness / 2
                <= 0)
    
    top_plane = (np.dot(normal, particle_data.points.T) 
                 - np.dot(normal, point) 
                 + plane_thickness / 2
                 >= 0)

    slice_boolean_mask = bottom_plane & top_plane

    # Bin index for each particle in each dimension
    i = np.digitize(resolved_particles_vec_2, vec_2_sample_bounds) - 1
    j = np.digitize(resolved_particles_vec_1, vec_1_sample_bounds) - 1

    # Mask out particles outside the valid bin range
    valid_mask = (
        (j >= 0) & (j < len(vec_1_sample_bounds) - 1) &
        (i >= 0) & (i < len(vec_2_sample_bounds) - 1) &
        slice_boolean_mask
    )

    i, j = i[valid_mask], j[valid_mask]

    # Assign unique sample ids to each particle
    sample_id = np.ravel_multi_index((i, j),
                            (len(vec_2_sample_bounds) - 1,
                             len(vec_1_sample_bounds) - 1)).astype(int)

    # Fill sample_elements array (preallocate first)
    sample_elements = np.full(len(resolved_particles_vec_1), -1, dtype=int)
    sample_elements[valid_mask] = sample_id

    # Count particles per cell
    cells = np.arange(resolution[0] * resolution[1], dtype=int)
    occupied_cells, counts = np.unique(sample_id, return_counts=True)
    occupied_cells = occupied_cells[occupied_cells != -1]
    cell_particles = np.zeros(resolution[0] * resolution[1], dtype=int)
    np.add.at(cell_particles, occupied_cells, counts)

    # Add the sample elements to the particle data
    particle_data[append_column] = sample_elements

    # Create an array of particle ids and their corresponding sample ids
    sample_data = np.array([particle_data["id"], sample_elements]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                         append_column,
                                         sample_data)

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = sum((sample_elements) == -1)
    n_sampled_particles = len(sample_elements) - n_unsampled_particles

    samples = ParticleSamples(append_column, 
                              sample_attribute,
                              cells,
                              occupied_cells, 
                              cell_particles, 
                              n_sampled_particles, 
                              n_unsampled_particles, 
                              )
    
    return (particle_data, samples)