import numpy as np
import warnings
from collections import namedtuple

def mesh_particles_1d(particle_data, cylinder_data, mesh_vec, mesh_resolution_1d):
        
    # Check if the particles file has points
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file")
        return (), "1D_mesh", np.nan, np.nan
    
    normalised_mesh_vec = mesh_vec / np.linalg.norm(mesh_vec)

    resolved_particles = np.dot(particle_data.points, normalised_mesh_vec)
    resolved_cylinder = np.dot(cylinder_data.points, normalised_mesh_vec)

    mesh_bounds = np.linspace(min(resolved_cylinder), max(resolved_cylinder), 
                                mesh_resolution_1d + 1)
    
    # sim.mesh_bounds_1d = mesh_bounds

    mesh_elements = np.zeros(len(particle_data.points))
    mesh_elements[:] = np.nan

    counter = 0
    
    id_named_tuple = namedtuple("id_named_tuple", 
                                ["id", "dim_1_lower_bound", "dim_1_upper_bound"])
    id_bounds = []

    for i in range(len(mesh_bounds) - 1):
        
        above_lower = resolved_particles >= mesh_bounds[i]
        below_upper = resolved_particles < mesh_bounds[i+1]

        mesh_element = above_lower & below_upper

        mesh_elements[mesh_element] = counter
        id_bounds.append(id_named_tuple(counter, mesh_bounds[i], mesh_bounds[i+1]))

        counter += 1

    mesh_column = "1D_mesh"
    particle_data[mesh_column] = mesh_elements

    out_of_mesh_particles = sum(np.isnan(mesh_elements))
    in_mesh_particles = len(mesh_elements) - out_of_mesh_particles

    return (particle_data, cylinder_data, id_bounds, mesh_column, 
            in_mesh_particles, out_of_mesh_particles)