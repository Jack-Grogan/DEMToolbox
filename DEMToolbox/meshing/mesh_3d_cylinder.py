import numpy as np
import warnings
from collections import namedtuple

def mesh_particles_3d_cylinder(particle_data, cylinder_data, mesh_resolution_3d, 
                                mesh_constant="volume", start_rotation=0):

    # Check if the particles file has points
    if particle_data.n_points == 0:
        warnings.warn("cannot mesh empty particles file")
        return (), "3D_cylinder_mesh", np.nan, np.nan

    # Perform checks on the input variables
    if len(mesh_resolution_3d) != 3:
        raise ValueError("mesh_resolution_3d must be a list of 3 integers")

    if not all([isinstance(i, int) for i in mesh_resolution_3d]):
        raise ValueError("mesh_resolution_3d must be a list of 3 integers")

    if not isinstance(start_rotation, (int, float)):
        raise ValueError("start_rotation must be an integer or float")

    # specify lacey mesh resolution
    ang_mesh = mesh_resolution_3d[0]
    rad_mesh = mesh_resolution_3d[1]
    z_mesh = mesh_resolution_3d[2]

    # sim.mesh_resolution_3d = mesh_resolution_3d

    # determine the radius of the cylinder mesh
    x_radii = abs(cylinder_data.bounds[1] - cylinder_data.bounds[0])/2
    y_radii = abs(cylinder_data.bounds[3] - cylinder_data.bounds[2])/2

    # calculate the radial increments of the lacey meshing depending on a chosen constant
    if mesh_constant == "radius":
        radial_mesh_boundaries = np.linspace(0, max(x_radii, y_radii), rad_mesh + 1)

    elif mesh_constant == "volume":
        max_radii_squared = max(x_radii, y_radii)**2
        radial_mesh_boundaries = np.sqrt(
            np.linspace(0, max_radii_squared, rad_mesh + 1)
            )
        
    else:
        raise("Invalid mesh constant")

    # calculate linearly spaced z mesh boundaries
    z_mesh_boundaries = np.linspace(cylinder_data.bounds[4], 
                                    cylinder_data.bounds[5], 
                                    z_mesh + 1)

    # calculate linearly spaced angular mesh boundaries
    angular_mesh_boundaries = np.linspace(0, 2*np.pi, ang_mesh + 1)

    # sim.mesh_bounds_3d_cylinder = (angular_mesh_boundaries, 
    #                                 radial_mesh_boundaries, z_mesh_boundaries)

    x_center = cylinder_data.center[0]
    y_center = cylinder_data.center[1]

    # calculate particle z positions
    particle_z = particle_data.points[:,2]

    # calculate particle radial positions
    particle_radii = np.sqrt(
        (particle_data.points[:, 0] - x_center)**2
        + (particle_data.points[:, 1] - y_center)**2
        )

    # calculate particle angular positions for x rotate by pi/6
    resolved_angular_data = (
        np.arctan2(
            (particle_data.points[:,1] - y_center),
            (particle_data.points[:,0] - x_center)
        ) +
        np.pi + start_rotation
    ) % (2*np.pi)

    # Set up an nan list to hold particle mesh regions
    mesh_elements = np.zeros(len(particle_data.points))
    mesh_elements[:] = np.nan

    # set mesh identifier counter
    counter = 0

    id_named_tuple = namedtuple("id_named_tuple",
                                ["id", "ang_lower_bound", "ang_upper_bound",
                                "rad_lower_bound", "rad_upper_bound",
                                "z_lower_bound", "z_upper_bound"])
    
    id_bounds = []

    # Loop through the lacey mesh elements
    for k in range(len(z_mesh_boundaries) - 1):

        # Particle above lower z boundary
        above_lower_z = particle_z >= z_mesh_boundaries[k]
        # Particle below upper z boundary
        below_upper_z = particle_z < z_mesh_boundaries[k+1]

        for i in range(len(radial_mesh_boundaries) - 1):

            # Particle above lower radial boundary
            above_lower_r = particle_radii >= radial_mesh_boundaries[i]

            # Particle below upper radial boundary
            below_upper_r = particle_radii < radial_mesh_boundaries[i+1]

            for j in range(len(angular_mesh_boundaries) - 1):

                # Particle above lower angular boundary
                above_lower_angle = resolved_angular_data >= angular_mesh_boundaries[j]
                # Particle below upper angular boundary
                below_upper_angle = resolved_angular_data < angular_mesh_boundaries[j+1]

                # pyvista_ndarray boolean mask outlining if a point lies within this
                # given mesh elemnent
                mesh_element = (
                    (above_lower_z & below_upper_z) &
                    (above_lower_r & below_upper_r) &
                    (above_lower_angle & below_upper_angle)
                )

                # Write mesh identifier to particles inside the mesh element
                mesh_elements[mesh_element] = counter

                id_bounds.append(id_named_tuple(counter, angular_mesh_boundaries[j],
                                                angular_mesh_boundaries[j+1],
                                                radial_mesh_boundaries[i], 
                                                radial_mesh_boundaries[i+1],
                                                z_mesh_boundaries[k], z_mesh_boundaries[k+1])
                )

                counter += 1

    mesh_column = "3D_cylinder_mesh"
    particle_data[mesh_column] = mesh_elements

    out_of_mesh_particles = sum(np.isnan(mesh_elements))
    in_mesh_particles = len(mesh_elements) - out_of_mesh_particles

    return (particle_data, cylinder_data, id_bounds, mesh_column, 
            in_mesh_particles, out_of_mesh_particles)