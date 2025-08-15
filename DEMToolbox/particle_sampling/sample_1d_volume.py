import numpy as np
import warnings

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_1d_volume(particle_data, 
                     sample_vector, 
                     resolution=2, 
                     append_column=None,
                     particle_id_column="id"):
    """Sample the particles into equal volume samples along a specified vector.

    This function samples the particles into equal volume samples along a
    specified vector. The vector is normalised and the particles are split
    into n_samples defined by the resolution parameter. The particles are
    sampled along the vector based on their volume. The resulting particle 
    data will have a new column added with the sample class for each 
    particle, and a ParticleSamples object will be returned containing
    the sample information.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk containing the particles to be sampled.
    sample_vector : list or np.ndarray
        The vector along which to sample the particles, specified as a
        3-element list [x, y, z].
    resolution : int, optional
        The number of samples to create along the specified vector, by
        default 2. Must be greater than or equal to 2 and less than or
        equal to the number of particles in the particle data.
    append_column : str, optional
        The name of the samples column to append to the particle data.
        If None, a default name based on the sample vector will be used.
        Default is None.
    particle_id_column : str, optional
        The name of the particle id column in the particle data, by
        default "id".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the attribute column added.
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
        If resolution is less than 2.
    ValueError
        If sample_vector is not a 3-element list.
    ValueError
        If resolution is not an integer.
    ValueError
        If resolution is greater than the number of particles in the
        particle data.
    UserWarning
        If the particle data has no points, a warning is issued and
        an empty ParticleSamples object is returned.
    """
    if len(sample_vector) != 3:
        raise ValueError("sample_vector must be a 3 element list.")
    
    if resolution < 2:
        raise ValueError("resolution must be greater than or equal to 2.")
    
    if not np.issubdtype(type(resolution), np.integer):
        raise ValueError("resolution must be an integer.")
    
    sample_vector = np.asarray(sample_vector)
    sample_vector = sample_vector/np.linalg.norm(sample_vector)
    if append_column is None:
        append_column = (f"{sample_vector[0]}_{sample_vector[1]}_"
                            f"{sample_vector[2]}_volume_sample")
        
    if particle_data.n_points == 0:
        warnings.warn("Cannot split empty particles file", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                             append_column,
                                             np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return particle_data, samples
    
    if resolution > particle_data.n_points:
        raise ValueError("resolution must be less than or equal to the "
                         "number of particles in the particle data.")
    
    # Calculate equal volume samples along the specified vector
    resolved_points = np.dot(particle_data.points, sample_vector)

    sorted_indices = np.argsort(resolved_points)

    if particle_data.point_data.get("volume") is None:
        volume = 4/3 * np.pi * (particle_data.point_data["radius"] ** 3)
        particle_data.point_data["volume"] = volume

    sorted_volume = particle_data.point_data["volume"][sorted_indices]
    cumulative_volume = np.cumsum(sorted_volume)
    total_volume = np.sum(sorted_volume)
    target_volume = total_volume / resolution

    samples_column = np.zeros(particle_data.n_points, dtype=int)

    # Assign the last sample to all remaining particles
    samples_column[:] = resolution - 1

    # loop through the samples and assign the class
    current_index = 0
    for i in range(resolution - 1):
        split_index = np.searchsorted(cumulative_volume, 
                                        (i + 1) * target_volume)
        samples_column[sorted_indices[current_index:split_index + 1]] = i
        current_index = split_index

    # Add the samples column to the particle data
    particle_data[append_column] = samples_column

    # Create an array of particle ids and their corresponding sample ids
    sample_data = np.array([particle_data.point_data["id"], samples_column]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                         append_column,
                                         sample_data)

    samples = ParticleSamples(append_column,
                              sample_attribute,
                              list(range(resolution)),
                              list(range(resolution)),
                              [np.sum(samples_column == i) for i in range(resolution)],
                              particle_data.n_points,
                              0,
                              )

    return particle_data, samples