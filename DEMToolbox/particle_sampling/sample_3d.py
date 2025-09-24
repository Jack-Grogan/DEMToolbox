import numpy as np
import warnings
import pyvista as pv

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_3d(particle_data, 
              bounds, 
              vector_1,  
              vector_2, 
              vector_3, 
              resolution, 
              append_column="3D_samples",
              particle_id_column="id"):
    """Split the particles into samples split along 3 dimensions.

    Split the particles into a 3D sample space defined by three 
    orthogonal vectors and a resolution in each vector direction.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    bounds : list, np.ndarray or vtkPolyData
        If a list or np.ndarray bounds of the sample space in the form 
        [x_min, x_max, y_min, y_max, z_min, z_max]. If a vtk, the bounds 
        will be determined from the vtk's bounds.
    vector_1 : list
        The first sample vector to split the particles along.
    vector_2 : list
        The second sample vector to split the particles along.
    vector_3 : list
        The third sample vector to split the particles along.
    resolution : list
        The resolution of the 3D sample space in the form [m, n, o].
    append_column : str, optional
        The name of the samples column to append to the particle data, 
        by default "3D_samples".
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
        If resolution is not a 3 element list of integers.
    ValueError
        If resolution is less than or equal to 0.
    ValueError
        If vectors are not orthogonal.
    UserWarning
        If the particle data has no points return unedited particle
        data and an empty samples object.
    ValueError
        If bounds is not a 6 element list or np.ndarray of integers or 
        floats or a vtkPolyData.
    ValueError
        If bounds list does not contains elements not of the order
        [x_min, x_max, y_min, y_max, z_min, z_max].
    UserWarning
        If bounds is a vtkPolyData and has no points return unedited
        particle data and an empty samples object.
    """
    if len(vector_1) != 3 or len(vector_2) != 3 or len(vector_3) != 3:
        raise ValueError("Vectors must be 3 element lists.")
    
    if len(resolution) != 3:
        raise ValueError("Resolution must be a 3 element list.")
    
    if not all(isinstance(i, int) for i in resolution):
        raise ValueError("Resolution must be a list of integers.")
    
    if any(i <= 0 for i in resolution):
        raise ValueError("Resolution must be greater than 0 in all "
                         "dimensions.")
    
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return (particle_data, samples)
    
    # Check the sample vectors are orthogonal
    dot_product_1 = np.dot(vector_1, vector_2)
    dot_product_2 = np.dot(vector_1, vector_3)
    dot_product_3 = np.dot(vector_2, vector_3)
    
    if not np.allclose([dot_product_1, dot_product_2, dot_product_3], 0):
        raise ValueError("Sample vectors must be orthogonal to each other.")
    
    # Normalise the vectors
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)
    vector_3 = vector_3 / np.linalg.norm(vector_3)

    resolved_particles_vec_1 = np.dot(particle_data.points, vector_1)
    resolved_particles_vec_2 = np.dot(particle_data.points, vector_2)
    resolved_particles_vec_3 = np.dot(particle_data.points, vector_3)

    if isinstance(bounds, list) or isinstance(bounds, np.ndarray):
        if len(bounds) != 6:
            raise ValueError("Bounds must be a list of 6 elements: "
                             "[x_min, x_max, y_min, y_max, z_min, z_max].")
        
        if not all(isinstance(i, (int, float)) for i in bounds):
            raise ValueError("Bounds must be a list of integers or floats.")
        
        # Apply the bounds to the particle data
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Bounds are not valid. Ensure that for each "
                             "dimension min < max.")
        
        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])
        
        min_bound_vec_1 = np.min(corners @ vector_1)
        max_bound_vec_1 = np.max(corners @ vector_1)

        min_bound_vec_2 = np.min(corners @ vector_2)
        max_bound_vec_2 = np.max(corners @ vector_2)

        min_bound_vec_3 = np.min(corners @ vector_3)
        max_bound_vec_3 = np.max(corners @ vector_3)

        if min_bound_vec_1 > max_bound_vec_1:
            min_bound_vec_1, max_bound_vec_1 = max_bound_vec_1, min_bound_vec_1

        if min_bound_vec_2 > max_bound_vec_2:
            min_bound_vec_2, max_bound_vec_2 = max_bound_vec_2, min_bound_vec_2

        if min_bound_vec_3 > max_bound_vec_3:
            min_bound_vec_3, max_bound_vec_3 = max_bound_vec_3, min_bound_vec_3          

        vec_1_sample_bounds = np.linspace(min_bound_vec_1,
                                          max_bound_vec_1,
                                          resolution[0] + 1)
        vec_2_sample_bounds = np.linspace(min_bound_vec_2,
                                          max_bound_vec_2,
                                          resolution[1] + 1)
        vec_3_sample_bounds = np.linspace(min_bound_vec_3,
                                          max_bound_vec_3,
                                          resolution[2] + 1)

    elif isinstance(bounds, pv.PolyData):

        if bounds.n_points == 0:
            warnings.warn("Cannot sample with empty bounds vtk file. "
                          "Returning unedited particle data.", 
                          UserWarning)
            sample_attribute = ParticleAttribute(particle_id_column, 
                                            append_column,
                                            np.empty((0, 2)))
            samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
            return (particle_data, samples)
        
        resolved_bounds_vec_1 = np.dot(bounds.points, vector_1)
        resolved_bounds_vec_2 = np.dot(bounds.points, vector_2)
        resolved_bounds_vec_3 = np.dot(bounds.points, vector_3)

        # Define the sample bounds linearly along the resolved vtk data
        vec_1_sample_bounds = np.linspace(min(resolved_bounds_vec_1),
                                        max(resolved_bounds_vec_1),
                                        resolution[0] + 1)
        vec_2_sample_bounds = np.linspace(min(resolved_bounds_vec_2),
                                        max(resolved_bounds_vec_2),
                                        resolution[1] + 1)
        vec_3_sample_bounds = np.linspace(min(resolved_bounds_vec_3),
                                        max(resolved_bounds_vec_3),
                                        resolution[2] + 1)
    else:
        raise ValueError("Bounds must be a list of 6 elements or a "
                         "vtkPolyData.")

    i = np.digitize(resolved_particles_vec_3, vec_3_sample_bounds) - 1
    j = np.digitize(resolved_particles_vec_2, vec_2_sample_bounds) - 1
    k = np.digitize(resolved_particles_vec_1, vec_1_sample_bounds) - 1

    # Filter out particles outside the valid ranges
    mask = (
        (i >= 0) & (i < len(vec_3_sample_bounds) - 1) &
        (j >= 0) & (j < len(vec_2_sample_bounds) - 1) &
        (k >= 0) & (k < len(vec_1_sample_bounds) - 1)
    )
    i, j, k = i[mask], j[mask], k[mask] 

    sample_id = np.ravel_multi_index((i, j, k),
                                    (len(vec_3_sample_bounds)-1,
                                    len(vec_2_sample_bounds)-1,
                                    len(vec_1_sample_bounds)-1)).astype(int)
          
    cells = np.arange(resolution[0] * resolution[1] * resolution[2], dtype=int)
    occupied_cells, counts = np.unique(sample_id, return_counts=True)
    occupied_cells = occupied_cells[occupied_cells != -1]
    cell_particles = np.zeros_like(cells, dtype=int)

    np.add.at(cell_particles, occupied_cells, counts)

    sample_elements = np.full(particle_data.n_points, -1, dtype=int)
    sample_elements[mask] = sample_id

    # Add the sample elements to the particle data
    particle_data[append_column] = sample_elements

    # Create a DataFrame for the sample elements
    sample_data = np.array([particle_data["id"], sample_elements]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                         append_column,
                                         sample_data)

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = np.sum(sample_elements == -1)
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