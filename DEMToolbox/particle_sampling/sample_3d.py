import numpy as np
import pandas as pd
import warnings

from ..classes.particle_samples import ParticleSamples

def sample_3d(particle_data, container_data, vector_1,  vector_2, vector_3, 
              resolution, append_column="3D_samples"):
    """Split the particles into samples split along 3 dimensions.

    Split the particles into a 3D sample space defined by three 
    orthogonal vectors and a resolution in each vector direction.

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
    vector_3 : list
        The third sample vector to split the particles along.
    resolution : list
        The resolution of the 3D sample space in the form [m, n, o].
    append_column : str, optional
        The name of the samples column to append to the particle data, 
        by default "3D_samples".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the samples column added.
    samples : ParticleSamples
        A ParticleSamples object containing the samples column name, a 
        list of sample elements, a list of occupied sample elements, 
        a list of the number of particles in the sample elements, the
        number of particles in the sample elements, the number of 
        sampled and unsampled particles and a dataframe containing the
        sample id, lower bound, upper bound and number of particles in
        the sample element.

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
    UserWarning
        If the container data has no points return unedited particle
        data and an empty samples object.
    """
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        samples = ParticleSamples(append_column, [], [], [], 0, 0)
        return (particle_data, samples)
    
    if container_data.n_points == 0:
        warnings.warn("Cannot sample empty container file.", UserWarning)
        samples = ParticleSamples(append_column, [], [], [], 0, 0)
        return (particle_data, samples)
    
    if len(vector_1) != 3 or len(vector_2) != 3 or len(vector_3) != 3:
        raise ValueError("Vectors must be 3 element lists.")
    
    if len(resolution) != 3:
        raise ValueError("Resolution must be a 3 element list.")
    
    if not all(isinstance(i, int) for i in resolution):
        raise ValueError("Resolution must be a list of integers.")
    
    if any(i <= 0 for i in resolution):
        raise ValueError("Resolution must be greater than.")
    
    # Normalise the vectors
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)
    vector_3 = vector_3 / np.linalg.norm(vector_3)

    # Check the sample vectors are orthogonal
    dot_product_1 = np.dot(vector_1, vector_2)
    dot_product_2 = np.dot(vector_1, vector_3)
    dot_product_3 = np.dot(vector_2, vector_3)
    if dot_product_1 != 0 or dot_product_2 != 0 or dot_product_3 != 0:
        raise ValueError("Sample vectors must be orthogonal to each other.")
    
    # Resolve the particles and container along the vectors
    resolved_particles_vec_1 = np.dot(particle_data.points, vector_1)
    resolved_particles_vec_2 = np.dot(particle_data.points, vector_2)
    resolved_particles_vec_3 = np.dot(particle_data.points, vector_3)
    resolved_container_vec_1 = np.dot(container_data.points, vector_1)
    resolved_container_vec_2 = np.dot(container_data.points, vector_2)
    resolved_container_vec_3 = np.dot(container_data.points, vector_3)

    # Define the sample bounds linearly along the resolved container
    vec_1_sample_bounds = np.linspace(min(resolved_container_vec_1),
                                      max(resolved_container_vec_1),
                                      resolution[0] + 1)
    vec_2_sample_bounds = np.linspace(min(resolved_container_vec_2),
                                      max(resolved_container_vec_2),
                                      resolution[1] + 1)
    vec_3_sample_bounds = np.linspace(min(resolved_container_vec_3),
                                      max(resolved_container_vec_3),
                                      resolution[2] + 1)
    
    # Create the empty sample elements array
    sample_elements = np.empty(particle_data.n_points)
    sample_elements[:] = np.nan

    cells = []
    occupied_cells = []
    sample_data = []
    cell_particles = []

    sample_id = int(0)
    for i in range(len(vec_3_sample_bounds) - 1):
        
        above_lower_vec_3 = (resolved_particles_vec_3 
                             >= vec_3_sample_bounds[i])
        below_upper_vec_3 = (resolved_particles_vec_3 
                             < vec_3_sample_bounds[i+1])

        for j in range(len(vec_2_sample_bounds) - 1):

            above_lower_vec_2 = (resolved_particles_vec_2 
                                 >= vec_2_sample_bounds[j])
            below_upper_vec_2 = (resolved_particles_vec_2 
                                 < vec_2_sample_bounds[j+1])

            for k in range(len(vec_1_sample_bounds) - 1):

                above_lower_vec_1 = (resolved_particles_vec_1 
                                     >= vec_1_sample_bounds[k])
                below_upper_vec_1 = (resolved_particles_vec_1 
                                     < vec_1_sample_bounds[k+1])

                # Boolean array of particles in the sample element
                sample_element = ((above_lower_vec_1 & below_upper_vec_1) 
                                & (above_lower_vec_2 & below_upper_vec_2) 
                                & (above_lower_vec_3 & below_upper_vec_3))

                # Assign the sample element id to the particles 
                # in the sample element
                sample_elements[sample_element] = int(sample_id)
                cells.append(sample_id)
                cell_particles.append(sum(sample_element))
                if sum(sample_element) > 0:
                    occupied_cells.append(sample_id)

                # Store the sample element id, bounds and number of particles
                sample_data.append(
                    (sample_id, vec_1_sample_bounds[k], 
                     vec_1_sample_bounds[k+1], vec_2_sample_bounds[j],
                     vec_2_sample_bounds[j+1], vec_3_sample_bounds[i],
                     vec_3_sample_bounds[i+1], sum(sample_element))
                     )
                
                sample_id += int(1)

    sample_df = pd.DataFrame(sample_data, 
                             columns=["sample id", "vec_1_lower_bound",
                                      "vec_1_upper_bound", "vec_2_lower_bound",
                                      "vec_2_upper_bound", "vec_3_lower_bound",
                                      "vec_3_upper_bound", "n_particles"],
                            )

    # Add the sample elements to the particle data
    particle_data[append_column] = sample_elements

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = sum(np.isnan(sample_elements))
    n_sampled_particles = len(sample_elements) - n_unsampled_particles

    samples = ParticleSamples(append_column, cells, occupied_cells, 
                           cell_particles, n_sampled_particles, 
                           n_unsampled_particles, sample_df)

    return (particle_data, samples)