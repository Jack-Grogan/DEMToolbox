import numpy as np
import warnings

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_2d(particle_data, 
              container_data, 
              vector_1, 
              vector_2, 
              resolution, 
              append_column="2D_samples", 
              particle_id_column="id"):
    """Split the particles into samples split along 2 dimensions.

    Split the particles into a 2D sample space defined by two orthogonal
    vectors and a resolution in each vector direction.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    container_data : vtkPolyData
        The container vtk.
    vector_1 : list
        The first sample vector to split the particles along.
    vector_2 : list
        The second sample vector to split the particles along.
    resolution : list
        The resolution of the 2D sample space in the form [m, n].
    append_column : str, optional
        The name of the samples column to append to the particle data, 
        by default "2D_samples".
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
        If resolution is not a 2 element list of integers.
    ValueError
        If resolution is less than or equal to 0.
    ValueError
        If vectors are not orthogonal.
    UserWarning
        If the particle data has no points return unedited particle
        data and an empty samples object.
    UserWarning
        If the container data has no points return unedited particle
        data and an empty samples object.
    """
    if len(vector_1) != 3 or len(vector_2) != 3:
        raise ValueError("Vectors must be 3 element lists.")
    
    if len(resolution) != 2:
        raise ValueError("Resolution must be a 2 element list.")
    
    if not all(isinstance(i, int) for i in resolution):
        raise ValueError("Resolution must be an integer.")

    if any(i <= 0 for i in resolution):
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
    
    # Normalise the vectors
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    # Check the vectors are orthogonal
    dot_product = np.dot(vector_1, vector_2)
    if dot_product != 0:
        raise ValueError("Sample vectors must be orthogonal to each other.")

    # Resolve the particles and container along the vectors
    resolved_particles_vec_1 = np.dot(particle_data.points,  vector_1)
    resolved_particles_vec_2 = np.dot(particle_data.points,  vector_2)
    resolved_container_vec_1 = np.dot(container_data.points, vector_1)
    resolved_container_vec_2 = np.dot(container_data.points, vector_2)

    # Define the sample bounds linearly along the resolved container
    vec_1_sample_bounds = np.linspace(min(resolved_container_vec_1),
                                    max(resolved_container_vec_1),
                                    resolution[0] + 1)
    vec_2_sample_bounds = np.linspace(min(resolved_container_vec_2),
                                    max(resolved_container_vec_2),
                                    resolution[1] + 1)

    # Create the empty sample elements array
    sample_elements = np.empty(particle_data.n_points)
    sample_elements[:] = np.nan

    cells = []
    occupied_cells = []
    cell_particles = []

    sample_id = int(0)
    for i in range(len(vec_2_sample_bounds) - 1):

        above_lower_vec_2 = (resolved_particles_vec_2
                              >= vec_2_sample_bounds[i])
        below_upper_vec_2 = (resolved_particles_vec_2 
                             < vec_2_sample_bounds[i+1])

        for j in range(len(vec_1_sample_bounds) - 1):

            above_lower_vec_1 = (resolved_particles_vec_1 
                                 >= vec_1_sample_bounds[j])
            below_upper_vec_1 = (resolved_particles_vec_1 
                                 < vec_1_sample_bounds[j+1])

            # Boolean array of particles in the sample element
            sample_element = ((above_lower_vec_1 & below_upper_vec_1) 
                            & (above_lower_vec_2 & below_upper_vec_2))

            # Assign the sample element id to the particles in the sample
            sample_elements[sample_element] = int(sample_id)
            cells.append(sample_id)
            cell_particles.append(sum(sample_element))
            if sum(sample_element) > 0:
                occupied_cells.append(sample_id)

            sample_id += int(1)

    # Add the sample elements to the particle data
    particle_data[append_column] = sample_elements

    # Create a DataFrame for the sample elements
    sample_data = np.array([particle_data["id"], sample_elements]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        sample_data)

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = sum(np.isnan(sample_elements))
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