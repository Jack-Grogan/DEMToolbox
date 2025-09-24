import numpy as np
import warnings 
import pyvista as pv

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_1d(particle_data, 
              bounds,
              vector, 
              resolution,
              append_column="1D_samples",
              particle_id_column="id"):
    """Split the particles into samples split along 1 dimension.

    Split the particles into a 1D samples along a user defined vector to
    generate a number of sample locations defined by the resolution.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    bounds : list, np.ndarray or vtkPolyData
        If a list or np.ndarray bounds of the sample space in the form 
        [x_min, x_max, y_min, y_max, z_min, z_max]. If a vtk, the bounds 
        will be determined from the vtk's bounds.
    vector : list
        The sample vector to split the particles along as [x, y, z].
    resolution : int
        The resolution of the 1D sample along the specified vector.
    append_column : str, optional
        The name of the samples column to append to the particle data, 
        by default "1D_samples".
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
        If vector is not a 3 element list.
    ValueError
        If resolution is not an integer.
    ValueError
        If resolution is less than or equal to 0.
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
    if len(vector) != 3:
        raise ValueError("Vector must be a 3 element list.")
    
    if not isinstance(resolution, int):
        raise ValueError("Resolution must be an integer.")
    
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")
    
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return (particle_data, samples)
    
    # Normalise the vector
    vector = vector / np.linalg.norm(vector)

    resolved_particles = np.dot(particle_data.points, vector)

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
        
        min_bound_vec = np.min(corners @ vector)
        max_bound_vec = np.max(corners @ vector)

        if min_bound_vec > max_bound_vec:
            min_bound_vec, max_bound_vec = max_bound_vec, min_bound_vec

        sample_bounds = np.linspace(min_bound_vec,
                                    max_bound_vec, 
                                    resolution + 1)

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
        
        resolved_bounds = np.dot(bounds.points, vector)

        # Define the sample bounds linearly along the resolved vtk data
        sample_bounds = np.linspace(min(resolved_bounds),
                                    max(resolved_bounds), 
                                    resolution + 1)
        
    else:
        raise ValueError("Bounds must be a list or array of 6 elements "
                         "or a vtkPolyData.")
    
    i = np.digitize(resolved_particles, sample_bounds) - 1

    # Mask out particles outside the valid bin range
    valid_mask = (i >= 0) & (i < len(sample_bounds) - 1)
    i = i[valid_mask]

    # Assign unique sample ids to each particle
    sample_id = i
    sample_elements = np.full(len(resolved_particles), -1, dtype=int)
    sample_elements[valid_mask] = sample_id

    # Count particles per cell
    n_bins = len(sample_bounds) - 1
    cells = np.arange(n_bins, dtype=int)
    occupied_cells, counts = np.unique(sample_id, return_counts=True)
    occupied_cells = occupied_cells[occupied_cells != -1]
    cell_particles = np.zeros(n_bins, dtype=int)
    np.add.at(cell_particles, occupied_cells, counts)

    # Add the sample column to the particle data
    particle_data[append_column] = sample_elements

    # Create an array of particle ids and their corresponding sample ids
    sample_data = np.array([particle_data["id"], sample_elements]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                         append_column,
                                         sample_data)

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = np.sum(sample_elements == -1)
    n_sampled_particles = len(sample_elements) - n_unsampled_particles

    # Create the ParticleSamples object to store data about the samples
    samples = ParticleSamples(append_column, 
                              sample_attribute,
                              cells, 
                              occupied_cells, 
                              cell_particles, 
                              n_sampled_particles, 
                              n_unsampled_particles, 
                              )

    return (particle_data, samples)