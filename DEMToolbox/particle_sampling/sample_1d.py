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