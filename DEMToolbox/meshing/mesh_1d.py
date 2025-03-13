import numpy as np
import warnings 
import pandas as pd

from ..classes.mesh_class import Mesh

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
    mesh : Mesh
        A mesh object containing the mesh column, a list of mesh 
        elements, a list of occupied mesh elements, the number of 
        particles in the mesh elements, the number of particles out 
        of the mesh elements and a dataframe containing the mesh id, 
        lower bound, upper bound and number of particles in the mesh 
        element.

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
        n_meshed_particles and n_unmeshed_particles.
    UserWarning
        If the container data has no points return unedited
        particle data an empty mesh dataframe and nan for
        n_meshed_particles and n_unmeshed_particles.
    """
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "lower_bound", 
                                        "upper_bound", "n_particles"])
        mesh = Mesh(mesh_column, [], [], 0, 0, mesh_df)
        return (particle_data, mesh)
    
    if container_data.n_points == 0:
        warnings.warn("cannot mesh empty container file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "lower_bound",
                                         "upper_bound", "n_particles"])
        
        mesh = Mesh(mesh_column, [], [], 0, 0, mesh_df)
        return (particle_data, mesh)
    
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
    
    cells = []
    occupied_cells = []
    mesh_data = []

    mesh_id = int(0)
    for i in range(len(mesh_bounds) - 1):
        
        above_lower = resolved_particles >= mesh_bounds[i]
        below_upper = resolved_particles < mesh_bounds[i+1]

        # Boolean array of particles in the mesh element
        mesh_element = above_lower & below_upper

        # Assign the mesh element id to the particles in the mesh
        mesh_elements[mesh_element] = int(mesh_id)

        cells.append(mesh_id)
        if sum(mesh_element) > 0:
            occupied_cells.append(mesh_id)

        # Store the mesh element id, bounds and number of particles
        mesh_data.append((mesh_id, mesh_bounds[i], 
                          mesh_bounds[i+1], sum(mesh_element)))
        
        mesh_id += int(1)

    mesh_df = pd.DataFrame(mesh_data, columns=["mesh id", "lower_bound",
                                               "upper_bound", "n_particles"])

    # Add the mesh column to the particle data
    particle_data[mesh_column] = mesh_elements

    # Count the number of particles in and out of the mesh elements
    n_unmeshed_particles = sum(np.isnan(mesh_elements))
    n_meshed_particles = len(mesh_elements) - n_unmeshed_particles

    mesh = Mesh(mesh_column, cells, occupied_cells, n_meshed_particles,
                n_unmeshed_particles, mesh_df)

    return (particle_data, mesh)