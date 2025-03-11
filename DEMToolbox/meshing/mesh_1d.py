import numpy as np
import warnings 
import pandas as pd

def mesh_particles_1d(particle_data, container_data, vector, 
                      resolution, mesh_column="1D_mesh"):
    """Split the particles into a 1D mesh.

    Split the particles into a n mesh elements linearly along a the mesh 
    vector.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    container_data : vtkPolyData
        The container vtk.
    vector : list
        The mesh vector to split the particles along.
    resolution : int
        The resolution of the 1D mesh.
    mesh_column : str, optional
        The name of the mesh column in the particle data,
          by default "1D_mesh".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the mesh column added.
    mesh_attributes : tuple
        A tuple containing the mesh column, a dataframe containing the
        mesh id, lower bound, upper bound and number of particles in the
        mesh element, the number of particles in the mesh elements and
        the number of particles out of the mesh elements.
    
    Raises
    ------
    ValueError
        If vector is not a 3 element list.
    ValueError
        If resolution is not an integer.
    ValueError
        If resolution is less than or equal to 0.
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
        mesh_df = pd.DataFrame(columns=["mesh id", "lower_bound", 
                                        "upper_bound", "n_particles"])
        return (particle_data, (mesh_column, mesh_df, np.nan, np.nan))
    
    if container_data.n_points == 0:
        warnings.warn("cannot mesh empty container file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "lower_bound",
                                         "upper_bound", "n_particles"])
        return (particle_data, (mesh_column, mesh_df, np.nan, np.nan))
    
    if len(vector) != 3:
        raise ValueError("vector must be a 3 element list")
    
    if not isinstance(resolution, int):
        raise ValueError("resolution must be an integer")
    
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0")
    
    # Normalise the vector
    normalised_vector = vector / np.linalg.norm(vector)

    # Resolve the particles and container along the vector
    resolved_particles = np.dot(particle_data.points, 
                                normalised_vector)
    resolved_container = np.dot(container_data.points, 
                                normalised_vector)

    # Define the mesh bounds linearly along the resolved container
    mesh_bounds = np.linspace(min(resolved_container),
                              max(resolved_container), 
                              resolution + 1)
    
    # Create the empty mesh elements array
    mesh_elements = np.empty(particle_data.n_points)
    mesh_elements[:] = np.nan

    mesh_data = []
    
    mesh_id = int(0)
    for i in range(len(mesh_bounds) - 1):
        
        above_lower = resolved_particles >= mesh_bounds[i]
        below_upper = resolved_particles < mesh_bounds[i+1]

        # Boolean array of particles in the mesh element
        mesh_element = above_lower & below_upper

        # Assign the mesh element id to the particles in the mesh
        mesh_elements[mesh_element] = int(mesh_id)

        # Store the mesh element id, bounds and number of particles
        mesh_data.append((mesh_id, mesh_bounds[i], 
                          mesh_bounds[i+1], sum(mesh_element)))
        
        mesh_id += int(1)

    mesh_df = pd.DataFrame(mesh_data, columns=["mesh id", "lower_bound",
                                               "upper_bound", "n_particles"])

    # Add the mesh column to the particle data
    particle_data[mesh_column] = mesh_elements

    # Count the number of particles in and out of the mesh elements
    out_of_mesh_particles = sum(np.isnan(mesh_elements))
    in_mesh_particles = len(mesh_elements) - out_of_mesh_particles

    mesh_attributes = (mesh_column, mesh_df, 
                       in_mesh_particles, out_of_mesh_particles)

    return (particle_data, mesh_attributes)