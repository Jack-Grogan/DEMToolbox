import numpy as np
import warnings

from DEMToolbox.meshing import mesh_particles_2d
from DEMToolbox.meshing import particle_slice

def velocity_vector_field(particle_data, cylinder_data, point, mesh_vec_x, mesh_vec_y, 
                          mesh_resolution_2d, plane_thickness):
    
    mesh_vec_x = np.asarray(mesh_vec_x)
    mesh_vec_y = np.asarray(mesh_vec_y)

    normal = np.cross(mesh_vec_x, mesh_vec_y)

    # Make all vectors unit vectors
    mesh_vec_x = mesh_vec_x / np.linalg.norm(mesh_vec_x)
    mesh_vec_y = mesh_vec_y / np.linalg.norm(mesh_vec_y)
    normal = normal / np.linalg.norm(normal)

    # Check if the particles file has points
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file")

        velocity_vectors = np.zeros((mesh_resolution_2d[0], mesh_resolution_2d[1], 2))
        velocity_vectors[:] = np.nan
        velocity_mag = np.zeros((mesh_resolution_2d[0], mesh_resolution_2d[1]))
        velocity_mag[:] = np.nan
        
        return velocity_vectors, velocity_mag
    
    particle_data, _, _, mesh_column, _, _ = mesh_particles_2d(particle_data, 
                                                                             cylinder_data, 
                                                                             mesh_vec_x, 
                                                                             mesh_vec_y, 
                                                                             mesh_resolution_2d)
        
    mesh = particle_data[mesh_column]

    particle_data, particle_slice_column = particle_slice(particle_data, point, 
                                                                         normal, plane_thickness)
    
    p_slice = particle_data[particle_slice_column]

    n_mesh_elements = (mesh_resolution_2d[0] * mesh_resolution_2d[1])

    mesh_id_booleans = []
    for ids in range(n_mesh_elements):
        mesh_boolean_mask = mesh == ids
        mesh_id_booleans.append(mesh_boolean_mask)

    velocity_vectors = np.zeros((n_mesh_elements, 2))
    velocity_mag = np.zeros(n_mesh_elements)
    cell_velocity = np.zeros((particle_data.n_points, 3))

    velocity_vectors[:] = np.nan
    velocity_mag[:] = np.nan
    cell_velocity[:] = np.nan

    # Loop through the mesh elements
    for i, mesh_element in enumerate(mesh_id_booleans):

        mesh_particles = (p_slice & mesh_element).astype(bool)

        if sum(mesh_particles) > 10:

            particle_velocities = particle_data.point_data["v"][mesh_particles]
            mean_velocity_vector = np.mean(particle_velocities, axis=0)

            mean_resolved_vec_1_velocity = np.dot(mean_velocity_vector, mesh_vec_x)
            mean_resolved_vec_2_velocity = np.dot(mean_velocity_vector, mesh_vec_y)

            resolved_velocity_vector = (mean_resolved_vec_1_velocity * mesh_vec_x +
                                        mean_resolved_vec_2_velocity * mesh_vec_y)

            velocity_vectors[i] = np.array((mean_resolved_vec_1_velocity, mean_resolved_vec_2_velocity))
            velocity_mag[i] = np.linalg.norm([mean_resolved_vec_1_velocity, mean_resolved_vec_2_velocity])
            cell_velocity[mesh_particles] = resolved_velocity_vector


    velocity_mag = velocity_mag.reshape(mesh_resolution_2d[0], mesh_resolution_2d[1])
    velocity_vectors = velocity_vectors.reshape(mesh_resolution_2d[0], mesh_resolution_2d[1], 2)

    for i, velocity_vector in enumerate(velocity_vectors):
        for j, (x_vector, y_vector) in enumerate(velocity_vector):
            velocity_vectors[i, j][0] = (x_vector / np.linalg.norm([x_vector, y_vector]))
            velocity_vectors[i, j][1] = (y_vector / np.linalg.norm([x_vector, y_vector]))


    velocity_mag = np.flipud(velocity_mag)
    velocity_vectors = np.flipud(velocity_vectors)

    particle_data["mean_resolved_velocity"] = cell_velocity

    return particle_data, velocity_vectors, velocity_mag