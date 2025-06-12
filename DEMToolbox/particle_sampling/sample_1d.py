import numpy as np
import warnings 

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_1d(particle_data, 
              container_data,
              vector, 
              resolution,
              append_column="1D_samples",
              particle_id_column="id"):
    """Split the particles into samples split along 1 dimension.

    Split the particles into a 1D samples along a user defined vector to
    generate a number of sample locations in the container defined by 
    the resolution.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    container_data : vtkPolyData
        The container vtk.
    vector : list
        The sample vector to split the particles along as [x, y, z].
    resolution : int
        The resolution of the 1D sample.
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
    UserWarning
        If the container data has no points return unedited particle
        data and an empty samples object.
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
    
    if container_data.n_points == 0:
        warnings.warn("Cannot sample empty container file.", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return (particle_data, samples)
    
    # Normalise the vector
    vector = vector / np.linalg.norm(vector)

    # Resolve the particles and container along the vector
    resolved_particles = np.dot(particle_data.points, vector)
    resolved_container = np.dot(container_data.points, vector)

    # Define the sample bounds linearly along the resolved container
    sample_bounds = np.linspace(min(resolved_container),
                                max(resolved_container), 
                                resolution + 1)
    
    # Create the empty sample elements array
    sample_elements = np.empty(particle_data.n_points)
    sample_elements[:] = np.nan
    
    cells = []
    occupied_cells = []
    cell_particles = []

    sample_id = int(0)
    for i in range(len(sample_bounds) - 1):
        
        above_lower = resolved_particles >= sample_bounds[i]
        below_upper = resolved_particles < sample_bounds[i+1]

        # Boolean array of particles in the sample element
        sample_element = above_lower & below_upper

        # Assign the sample element id to the particles in the sample
        sample_elements[sample_element] = int(sample_id)

        cells.append(sample_id)
        cell_particles.append(sum(sample_element))
        if sum(sample_element) > 0:
            occupied_cells.append(sample_id)

        sample_id += int(1)

    # Add the sample column to the particle data
    particle_data[append_column] = sample_elements

    # Create an array of particle ids and their corresponding sample ids
    sample_data = np.array([particle_data["id"], sample_elements]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                         append_column,
                                         sample_data)

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = sum(np.isnan(sample_elements))
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