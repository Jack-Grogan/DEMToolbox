import numpy as np
import warnings
from collections import namedtuple

def mesh_particles_3d(particle_data, cylinder_data, mesh_vec_x, mesh_vec_y, mesh_vec_z, mesh_resolution_3d):

    # Check if the particles file has points
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file")
        return (), "3D_mesh", np.nan, np.nan
    
    normalised_mesh_vec_x = mesh_vec_x / np.linalg.norm(mesh_vec_x)
    normalised_mesh_vec_y = mesh_vec_y / np.linalg.norm(mesh_vec_y)
    normalised_mesh_vec_z = mesh_vec_z / np.linalg.norm(mesh_vec_z)

    dot_product_1 = np.dot(normalised_mesh_vec_x, normalised_mesh_vec_y)
    dot_product_2 = np.dot(normalised_mesh_vec_x, normalised_mesh_vec_z)
    dot_product_3 = np.dot(normalised_mesh_vec_y, normalised_mesh_vec_z)

    if dot_product_1 != 0 or dot_product_2 != 0 or dot_product_3 != 0:
        raise ValueError("mesh vectors are not orthogonal to each other")
    
    resolved_particles_dim_1 = np.dot(particle_data.points, normalised_mesh_vec_x)
    resolved_particles_dim_2 = np.dot(particle_data.points, normalised_mesh_vec_y)
    resolved_particles_dim_3 = np.dot(particle_data.points, normalised_mesh_vec_z)

    resolved_cylinder_dim_1 = np.dot(cylinder_data.points, normalised_mesh_vec_x)
    resolved_cylinder_dim_2 = np.dot(cylinder_data.points, normalised_mesh_vec_y)
    resolved_cylinder_dim_3 = np.dot(cylinder_data.points, normalised_mesh_vec_z)

    dim_1_mesh_bounds = np.linspace(min(resolved_cylinder_dim_1),
                                    max(resolved_cylinder_dim_1),
                                    mesh_resolution_3d[0] + 1)
    
    dim_2_mesh_bounds = np.linspace(min(resolved_cylinder_dim_2),
                                    max(resolved_cylinder_dim_2),
                                    mesh_resolution_3d[1] + 1)
    
    dim_3_mesh_bounds = np.linspace(min(resolved_cylinder_dim_3),
                                    max(resolved_cylinder_dim_3),
                                    mesh_resolution_3d[2] + 1)
    
    # sim.mesh_bounds_3d = (dim_1_mesh_bounds, dim_2_mesh_bounds, dim_3_mesh_bounds)

    mesh_elements = np.zeros(len(particle_data.points))
    mesh_elements[:] = np.nan

    counter = 0

    id_named_tuple = namedtuple("id_named_tuple",
                                ["id", "dim_1_lower_bound", "dim_1_upper_bound",
                                "dim_2_lower_bound", "dim_2_upper_bound",
                                "dim_3_lower_bound", "dim_3_upper_bound"])
    
    id_bounds = []

    for i in range(len(dim_3_mesh_bounds) - 1):
        
        above_lower_dim_3 = resolved_particles_dim_3 >= dim_3_mesh_bounds[i]
        below_upper_dim_3 = resolved_particles_dim_3 < dim_3_mesh_bounds[i+1]

        for j in range(len(dim_2_mesh_bounds) - 1):

            above_lower_dim_2 = resolved_particles_dim_2 >= dim_2_mesh_bounds[j]
            below_upper_dim_2 = resolved_particles_dim_2 < dim_2_mesh_bounds[j+1]

            for k in range(len(dim_1_mesh_bounds) - 1):

                above_lower_dim_1 = resolved_particles_dim_1 >= dim_1_mesh_bounds[k]
                below_upper_dim_1 = resolved_particles_dim_1 < dim_1_mesh_bounds[k+1]

                mesh_element = ((above_lower_dim_1 & below_upper_dim_1) & 
                                (above_lower_dim_2 & below_upper_dim_2) & 
                                (above_lower_dim_3 & below_upper_dim_3))

                mesh_elements[mesh_element] = counter
                id_bounds.append(id_named_tuple(counter, dim_1_mesh_bounds[k], 
                                                dim_1_mesh_bounds[k+1],
                                                dim_2_mesh_bounds[j], dim_2_mesh_bounds[j+1],
                                                dim_3_mesh_bounds[i], dim_3_mesh_bounds[i+1]))

                counter += 1

    mesh_column = "3D_mesh"
    particle_data[mesh_column] = mesh_elements

    out_of_mesh_particles = sum(np.isnan(mesh_elements))
    in_mesh_particles = len(mesh_elements) - out_of_mesh_particles

    return (particle_data, cylinder_data, id_bounds, mesh_column, 
            in_mesh_particles, out_of_mesh_particles)