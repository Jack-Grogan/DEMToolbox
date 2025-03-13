import numpy as np
import pandas as pd
import warnings

from ..classes.mesh_class import Mesh

def mesh_particles_3d(particle_data, container_data, vector_1, 
                      vector_2, vector_3, resolution, mesh_column="3D_mesh"):
    """Split the particles into a 3D Cartesian mesh.

    Split the particles into a n x m x o mesh elements linearly along the
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
    vector_3 : list
        The third mesh vector to split the particles along.
    resolution : list
        The resolution of the 3D mesh.
    mesh_column : str, optional
        The name of the mesh column in the particle data,
        by default "3D_mesh".
        
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
        If vector_1, vector_2 or vector_3 are not 3 element lists.
    ValueError
        If resolution is not a 3 element list of integers.
    ValueError
        If resolution is less than or equal to 0.
    ValueError
        If the mesh vectors are not orthogonal to each other.
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
        mesh_df = pd.DataFrame(columns=["mesh id", "vec_1_lower_bound",
                                        "vec_1_upper_bound", 
                                        "vec_2_lower_bound",
                                        "vec_2_upper_bound",
                                        "vec_3_lower_bound",
                                        "vec_3_upper_bound",
                                        "n_particles"])
        mesh = Mesh(mesh_column, [], [], 0, 0, mesh_df)
        return (particle_data, mesh)
    
    if container_data.n_points == 0:
        warnings.warn("cannot mesh empty container file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "vec_1_lower_bound",
                                        "vec_1_upper_bound", 
                                        "vec_2_lower_bound",
                                        "vec_2_upper_bound",
                                        "vec_3_lower_bound",
                                        "vec_3_upper_bound",
                                        "n_particles"])
        mesh = Mesh(mesh_column, [], [], 0, 0, mesh_df)
        return (particle_data, mesh)
    
    if len(vector_1) != 3 or len(vector_2) != 3 or len(vector_3) != 3:
        raise ValueError("vectors must be 3 element lists")
    
    if len(resolution) != 3:
        raise ValueError("resolution must be a 3 element list")
    
    if not all(isinstance(i, int) for i in resolution):
        raise ValueError("resolution must be a list of integers")
    
    if any(i <= 0 for i in resolution):
        raise ValueError("resolution must be greater than")
    
    # Normalise the vectors
    normalised_vector_1 = vector_1 / np.linalg.norm(vector_1)
    normalised_vector_2 = vector_2 / np.linalg.norm(vector_2)
    normalised_vector_3 = vector_3 / np.linalg.norm(vector_3)

    # Check the mesh vectors are orthogonal
    dot_product_1 = np.dot(normalised_vector_1, normalised_vector_2)
    dot_product_2 = np.dot(normalised_vector_1, normalised_vector_3)
    dot_product_3 = np.dot(normalised_vector_2, normalised_vector_3)
    if dot_product_1 != 0 or dot_product_2 != 0 or dot_product_3 != 0:
        raise ValueError("mesh vectors must be orthogonal to each other")
    
    # Resolve the particles and container along the vectors
    resolved_particles_vec_1 = np.dot(particle_data.points, 
                                      normalised_vector_1)
    resolved_particles_vec_2 = np.dot(particle_data.points, 
                                      normalised_vector_2)
    resolved_particles_vec_3 = np.dot(particle_data.points, 
                                      normalised_vector_3)
    resolved_container_vec_1 = np.dot(container_data.points, 
                                      normalised_vector_1)
    resolved_container_vec_2 = np.dot(container_data.points, 
                                      normalised_vector_2)
    resolved_container_vec_3 = np.dot(container_data.points, 
                                      normalised_vector_3)

    # Define the mesh bounds linearly along the resolved container
    vec_1_mesh_bounds = np.linspace(min(resolved_container_vec_1),
                                    max(resolved_container_vec_1),
                                    resolution[0] + 1)
    vec_2_mesh_bounds = np.linspace(min(resolved_container_vec_2),
                                    max(resolved_container_vec_2),
                                    resolution[1] + 1)
    vec_3_mesh_bounds = np.linspace(min(resolved_container_vec_3),
                                    max(resolved_container_vec_3),
                                    resolution[2] + 1)
    
    # Create the empty mesh elements array
    mesh_elements = np.empty(particle_data.n_points)
    mesh_elements[:] = np.nan

    cells = []
    occupied_cells = []
    mesh_data = []

    mesh_id = int(0)
    for i in range(len(vec_3_mesh_bounds) - 1):
        
        above_lower_vec_3 = (resolved_particles_vec_3 
                             >= vec_3_mesh_bounds[i])
        below_upper_vec_3 = (resolved_particles_vec_3 
                             < vec_3_mesh_bounds[i+1])

        for j in range(len(vec_2_mesh_bounds) - 1):

            above_lower_vec_2 = (resolved_particles_vec_2 
                                 >= vec_2_mesh_bounds[j])
            below_upper_vec_2 = (resolved_particles_vec_2 
                                 < vec_2_mesh_bounds[j+1])

            for k in range(len(vec_1_mesh_bounds) - 1):

                above_lower_vec_1 = (resolved_particles_vec_1 
                                     >= vec_1_mesh_bounds[k])
                below_upper_vec_1 = (resolved_particles_vec_1 
                                     < vec_1_mesh_bounds[k+1])

                # Boolean array of particles in the mesh element
                mesh_element = ((above_lower_vec_1 & below_upper_vec_1) 
                                & (above_lower_vec_2 & below_upper_vec_2) 
                                & (above_lower_vec_3 & below_upper_vec_3))

                # Assign the mesh element id to the particles 
                # in the mesh element
                mesh_elements[mesh_element] = int(mesh_id)

                cells.append(mesh_id)
                if sum(mesh_element) > 0:
                    occupied_cells.append(mesh_id)

                # Store the mesh element id, bounds and number of particles
                mesh_data.append((mesh_id, vec_1_mesh_bounds[k],
                                vec_1_mesh_bounds[k+1], vec_2_mesh_bounds[j],
                                vec_2_mesh_bounds[j+1], vec_3_mesh_bounds[i],
                                vec_3_mesh_bounds[i+1], sum(mesh_element)))
                
                mesh_id += int(1)

    mesh_df = pd.DataFrame(mesh_data, columns=["mesh id", "vec_1_lower_bound",
                                               "vec_1_upper_bound",
                                               "vec_2_lower_bound",
                                               "vec_2_upper_bound",
                                               "vec_3_lower_bound",
                                               "vec_3_upper_bound",
                                               "n_particles"])

    # Add the mesh elements to the particle data
    particle_data[mesh_column] = mesh_elements

    # Count the number of particles in and out of the mesh elements
    n_unmeshed_particles = sum(np.isnan(mesh_elements))
    n_meshed_particles = len(mesh_elements) - n_unmeshed_particles

    mesh = Mesh(mesh_column, cells, occupied_cells, n_meshed_particles,
                n_unmeshed_particles, mesh_df)

    return (particle_data, mesh)