import numpy as np
import warnings

from DEMToolbox.meshing import mesh_particles_2d
from DEMToolbox.meshing import particle_slice

def velocity_vector_field(particle_data, container_data, point, mesh_vec_x, 
                          mesh_vec_y, resolution, plane_thickness, 
                          mesh_column="2D_mesh", slice_column=None,
                          velocity_column="mean_resolved_velocity"):
    """Calculate the velocity vector field of a 2D mesh."

    Calculate the velocity vector field of particle that lie within a
    planar slice defined by a point a normal vector and a plane 
    thickness.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    container_data : vtkPolyData
        The container vtk.
    point : list
        The point on the plane as [x, y, z].
    mesh_vec_x : list
        The first mesh vector to split the particles along. Corespomds
        to the columns of the velocity field and velocity magnitude 
        arrays.
    mesh_vec_y : list
        The second mesh vector to split the particles along. Corespomds
        to the rows of the velocity field and velocity magnitude arrays.
    resolution : list
        The resolution of the 2D mesh in the form [m, n].
    plane_thickness : int or float 
        The thickness of the plane.
    mesh_column : str, optional
        The name of the mesh column in the particle data,
        by default "2D_mesh".
    slice_column : str, optional
        The name of the slice column in the particle data,
        by default None. If None, the column name will be
        "particle_slice_p{point}_n{normal}".
    velocity_column : str, optional
        The name of the velocity column in the particle data,
        by default "mean_resolved_velocity".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the velocity column added.
    mesh_column : str
        The name of the mesh column in the particle data.
    slice_column : str
        The name of the slice column in the particle data.
    velocity_column : str
        The name of the velocity column in the particle data.
    velocity_vectors : np.ndarray
        The velocity vectors of the mesh elements in the form
        [n, m, 2]. n is the number of rows and m is the number
        of columns. n = 0 represents the top row and m = 0 
        represents the left column.
    velocity_mag : np.ndarray
        The magnitude of the velocity vectors of the mesh elements
        in the form [n, m]. n is the number of rows and m is the
        number of columns. n = 0 represents the top row and m = 0
        represents the left column.
    """
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file", UserWarning)

        velocity_vectors = np.zeros((resolution[1], resolution[0], 2))
        velocity_vectors[:] = np.nan
        velocity_mag = np.zeros((resolution[1], resolution[0]))
        velocity_mag[:] = np.nan
        
        return (particle_data, (mesh_column, slice_column, velocity_column,
                velocity_vectors, velocity_mag))

    mesh_vec_x = np.asarray(mesh_vec_x)
    mesh_vec_y = np.asarray(mesh_vec_y)
    normal = np.cross(mesh_vec_x, mesh_vec_y)

    # Make all vectors unit vectors
    mesh_vec_x = mesh_vec_x / np.linalg.norm(mesh_vec_x)
    mesh_vec_y = mesh_vec_y / np.linalg.norm(mesh_vec_y)
    normal = normal / np.linalg.norm(normal)

    # Mesh the particles in 2D
    particle_data, mesh_output = mesh_particles_2d(particle_data,
                                                  container_data, 
                                                  mesh_vec_x, 
                                                  mesh_vec_y, 
                                                  resolution)
    mesh_column = mesh_output[0]
    mesh = particle_data[mesh_column]

    # Slice the particles
    particle_data, slice_column = particle_slice(particle_data, 
                                                 point, 
                                                 normal, 
                                                 plane_thickness,
                                                 slice_column)
    p_slice = particle_data[slice_column]

    # Get the mesh ids and the number of mesh elements

    n_mesh_elements = (resolution[0] * resolution[1])

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

    for i, mesh_element in enumerate(mesh_id_booleans):

        mesh_particles = (p_slice & mesh_element).astype(bool)

        if sum(mesh_particles) > 10:

            particle_velocities = particle_data.point_data["v"][mesh_particles]
            mean_velocity_vector = np.mean(particle_velocities, axis=0)

            mean_resolved_vec_1_velocity = np.dot(mean_velocity_vector, 
                                                  mesh_vec_x)
            mean_resolved_vec_2_velocity = np.dot(mean_velocity_vector, 
                                                  mesh_vec_y)

            resolved_velocity_vector = (mean_resolved_vec_1_velocity 
                                        * mesh_vec_x 
                                        + mean_resolved_vec_2_velocity 
                                        * mesh_vec_y)

            velocity_vectors[i] = np.array([mean_resolved_vec_1_velocity,
                                             mean_resolved_vec_2_velocity])
            velocity_mag[i] = np.linalg.norm([mean_resolved_vec_1_velocity, 
                                              mean_resolved_vec_2_velocity])
            cell_velocity[mesh_particles] = resolved_velocity_vector

    velocity_mag = velocity_mag.reshape(resolution[1], resolution[0])
    velocity_vectors = velocity_vectors.reshape(resolution[1], 
                                                resolution[0], 2)

    for i, velocity_vector in enumerate(velocity_vectors):
        for j, (x_vector, y_vector) in enumerate(velocity_vector):
            velocity_vectors[i, j][0] = (x_vector 
                                         / np.linalg.norm([x_vector, 
                                                           y_vector]))
            velocity_vectors[i, j][1] = (y_vector
                                         / np.linalg.norm([x_vector, 
                                                           y_vector]))

    # velocity_mag = np.flipud(velocity_mag)
    # velocity_vectors = np.flipud(velocity_vectors)
    particle_data["mean_resolved_velocity"] = cell_velocity

    return (particle_data, (mesh_column, slice_column, velocity_column, 
            velocity_vectors, velocity_mag))