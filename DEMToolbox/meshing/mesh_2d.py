import numpy as np
import pandas as pd
import warnings

def mesh_particles_2d(particle_data, container_data, vector_1, 
                      vector_2, resolution, column_name="2D_mesh"):
    """Split the particles into a 2D mesh.

    Split the particles into a n x m mesh elements linearly along the
    mesh vectors. The mesh vectors must be orthogonal to each other.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    container_data : vtkPolyData
        The container vtk.
    vector_1 : list
        The first mesh vector to split the particles along.
    vector_2 : list
        The second mesh vector to split the particles along.
    resolution : list
        The resolution of the 2D mesh.
    column_name : str, optional
        The name of the mesh column in the particle data,
        by default "2D_mesh".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the mesh column added.
    column_name : str
        The name of the mesh column in the particle data.
    mesh_df : pd.DataFrame
        A dataframe containing the mesh id, lower bound, upper bound
        and number of particles in the mesh element.
    in_mesh_particles : int
        The number of particles in the mesh elements.
    out_of_mesh_particles : int
        The number of particles out of the mesh elements.

    Raises
    ------
    ValueError
        If vector_1 or vector_2 are not 3 element lists.
    ValueError
        If resolution is not a 2 element list of integers.
    ValueError
        If resolution is less than or equal to 0.
    ValueError
        If the mesh vectors are not orthogonal to each other.
    UserWarning
        If the particle data has no points return unedited
        particle data an empty mesh dataframe and nan for
        in_mesh_particles and out_of_mesh_particles.
    UserWarning
        If the container data has no points return unedited
        particle data an empty mesh dataframe and nan for
        in_mesh_particles and out_of_mesh_particles.
    """
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "vec_1_lower_bound",
                                        "vec_1_upper_bound", 
                                        "vec_2_lower_bound",
                                        "vec_2_upper_bound",
                                        "n_particles"])
        
        return particle_data, mesh_df, np.nan, np.nan
    
    if container_data.n_points == 0:
        warnings.warn("cannot mesh empty container file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "vec_1_lower_bound",
                                        "vec_1_upper_bound", 
                                        "vec_2_lower_bound",
                                        "vec_2_upper_bound",
                                        "n_particles"])
        
        return particle_data, mesh_df, np.nan, np.nan
    
    if len(vector_1) != 3 or len(vector_2) != 3:
        raise ValueError("vectors must be 3 element lists")
    
    if len(resolution) != 2:
        raise ValueError("resolution must be a 2 element list")
    
    if not all(isinstance(i, int) for i in resolution):
        raise ValueError("resolution must be an integer")

    if any(i <= 0 for i in resolution):
        raise ValueError("resolution must be greater than 0")
    
    # Normalise the vectors
    normalised_vector_1 = vector_1 / np.linalg.norm(vector_1)
    normalised_vector_2 = vector_2 / np.linalg.norm(vector_2)

    # Check the vectors are orthogonal
    dot_product = np.dot(normalised_vector_1, normalised_vector_2)
    if dot_product != 0:
        raise ValueError("mesh vectors must be orthogonal to each other")

    # Resolve the particles and container along the vectors
    resolved_particles_vec_1 = np.dot(particle_data.points, 
                                      normalised_vector_1)
    resolved_particles_vec_2 = np.dot(particle_data.points, 
                                      normalised_vector_2)
    resolved_container_vec_1 = np.dot(container_data.points, 
                                      normalised_vector_1)
    resolved_container_vec_2 = np.dot(container_data.points, 
                                      normalised_vector_2)

    # Define the mesh bounds linearly along the resolved container
    vec_1_mesh_bounds = np.linspace(min(resolved_container_vec_1),
                                    max(resolved_container_vec_1),
                                    resolution[0] + 1)
    vec_2_mesh_bounds = np.linspace(min(resolved_container_vec_2),
                                    max(resolved_container_vec_2),
                                    resolution[1] + 1)

    # Create the empty mesh elements array
    mesh_elements = np.empty_like(particle_data.points)
    mesh_elements[:] = np.nan

    mesh_data = []

    mesh_id = 0
    for i in range(len(vec_2_mesh_bounds) - 1):

        above_lower_vec_2 = (resolved_particles_vec_2 
                             >= vec_2_mesh_bounds[i])
        below_upper_vec_2 = (resolved_particles_vec_2 
                             < vec_2_mesh_bounds[i+1])

        for j in range(len(vec_1_mesh_bounds) - 1):

            above_lower_vec_1 = (resolved_particles_vec_1 
                                 >= vec_1_mesh_bounds[j])
            below_upper_vec_1 = (resolved_particles_vec_1 
                                 < vec_1_mesh_bounds[j+1])

            # Boolean array of particles in the mesh element
            mesh_element = ((above_lower_vec_1 & below_upper_vec_1) 
                            & (above_lower_vec_2 & below_upper_vec_2))

            # Assign the mesh element id to the particles in the mesh
            mesh_elements[mesh_element] = mesh_id

            # Store the mesh element id, bounds and number of particles
            mesh_data.append((mesh_id, vec_1_mesh_bounds[j], 
                            vec_1_mesh_bounds[j+1], vec_2_mesh_bounds[i], 
                            vec_2_mesh_bounds[i+1], sum(mesh_element)))
            
            mesh_id += 1

    mesh_df = pd.DataFrame(mesh_data, columns=["mesh id", "vec_1_lower_bound",
                                               "vec_1_upper_bound",
                                               "vec_2_lower_bound",
                                               "vec_2_upper_bound",
                                               "n_particles"])

    # Add the mesh elements to the particle data
    particle_data[column_name] = mesh_elements

    # Count the number of particles in and out of the mesh elements
    out_of_mesh_particles = sum(np.isnan(mesh_elements))
    in_mesh_particles = len(mesh_elements) - out_of_mesh_particles

    return (particle_data, column_name, mesh_df,
            in_mesh_particles, out_of_mesh_particles)