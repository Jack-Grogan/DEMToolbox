import numpy as np
import warnings

from ..meshing.mesh_2d import mesh_particles_2d
from ..meshing.slice import particle_slice
from ..classes.velocity_class import VelocityField

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
    velocity_field : VelocityField
        A velocity field object containing the mesh column, the slice
        column, the velocity column, the velocity vectors and the
        velocity magnitude.

    Raises
    ------
    ValueError
        If point, mesh_vec_x or mesh_vec_y are not 3 element lists.
    ValueError
        If resolution is not a 2 element list.
    ValueError
        If plane_thickness is not an integer or float.
    UserWarning
        If the particle data has no points return unedited
        particle data and nan matrices for velocity vectors and velocity
        magnitude.
    """
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file", UserWarning)

        velocity_vectors = np.zeros((resolution[1], resolution[0], 2))
        velocity_vectors[:] = np.nan
        velocity_mag = np.zeros((resolution[1], resolution[0]))
        velocity_mag[:] = np.nan

        velocity_field = VelocityField(mesh_column, slice_column, 
                                       velocity_column, velocity_vectors,
                                       velocity_mag)
        
        return (particle_data, velocity_field)

    mesh_vec_x = np.asarray(mesh_vec_x)
    mesh_vec_y = np.asarray(mesh_vec_y)
    normal = np.cross(mesh_vec_x, mesh_vec_y)

    # Make all vectors unit vectors
    mesh_vec_x = mesh_vec_x / np.linalg.norm(mesh_vec_x)
    mesh_vec_y = mesh_vec_y / np.linalg.norm(mesh_vec_y)
    normal = normal / np.linalg.norm(normal)

    # Mesh the particles in 2D
    particle_data, mesh_2d = mesh_particles_2d(particle_data,
                                            container_data, 
                                            mesh_vec_x, 
                                            mesh_vec_y, 
                                            resolution)
    mesh = particle_data[mesh_2d.name]

    # Slice the particles
    particle_data, p_slice = particle_slice(particle_data, 
                                                   point, 
                                                   normal, 
                                                   plane_thickness,
                                                   slice_column)

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

        mesh_particles = (particle_data[p_slice.name] & mesh_element).astype(bool)

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

    velocity_field = VelocityField(mesh_column, slice_column,
                                   velocity_column, velocity_vectors,
                                   velocity_mag)

    return (particle_data, velocity_field)