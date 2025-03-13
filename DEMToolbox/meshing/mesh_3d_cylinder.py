import numpy as np
import pandas as pd
import warnings

from ..classes.mesh_class import Mesh

def mesh_particles_3d_cylinder(particle_data, cylinder_data,
                               resolution, mesh_constant="volume", 
                               rotation=0, mesh_column="3D_mesh"):
    """Split the particles into a 3D mesh in cylindrical coordinates.

    Split the particles into a cylindrical mesh for a cylinder container 
    orientated in the z direction.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    cylinder_data : vtkPolyData
        The cylinder container vtk.
    resolution : list
        The resolution of the 3D mesh as [angular, radial, z].
    mesh_constant : str, optional
        The mesh constant to define the radial mesh spacing, 
        by default "volume".
    rotation : int or float, optional
        The rotation of the mesh in radians, by default 0.
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
        If resolution is not a 3 element list of integers.
    ValueError
        If resolution is not a list of integers.
    ValueError
        If mesh_constant is not "radius" or "volume".
    ValueError
        If rotation is not an integer or float.
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
        mesh_df = pd.DataFrame(columns=["mesh id", "radial_lower_bound",
                                        "radial_upper_bound",
                                        "angular_lower_bound",
                                        "angular_upper_bound",
                                        "z_lower_bound",
                                        "z_upper_bound",
                                        "n_particles"])
        mesh = Mesh(mesh_column, [], [], 0, 0, mesh_df)
        return (particle_data, mesh)
    
    if cylinder_data.n_points == 0:
        warnings.warn("cannot mesh empty container file", UserWarning)
        mesh_df = pd.DataFrame(columns=["mesh id", "radial_lower_bound",
                                        "radial_upper_bound",
                                        "angular_lower_bound",
                                        "angular_upper_bound",
                                        "z_lower_bound",
                                        "z_upper_bound",
                                        "n_particles"])
        mesh = Mesh(mesh_column, [], [], 0, 0, mesh_df)
        return (particle_data, mesh)
    
    if len(resolution) != 3:
        raise ValueError("resolution must be a list of 3 integers")
    
    if not all([isinstance(i, int) for i in resolution]):
        raise ValueError("resolution must be a list of 3 integers")
    
    if not isinstance(rotation, (int, float)):
        raise ValueError("rotation must be an integer or float")

    ang_mesh = resolution[0]
    rad_mesh = resolution[1]
    z_mesh = resolution[2]

    # Determine the radius of the cylinder mesh
    x_radii = abs(cylinder_data.bounds[1] - cylinder_data.bounds[0])/2
    y_radii = abs(cylinder_data.bounds[3] - cylinder_data.bounds[2])/2
    radii = max(x_radii, y_radii)

    # Define the mesh bounds linearly along the resolved container
    if mesh_constant == "radius":
        radial_bounds = np.linspace(0, radii, rad_mesh + 1)
    elif mesh_constant == "volume":
        radial_bounds = np.sqrt(np.linspace(0, radii**2, rad_mesh + 1))
    else:
        raise ValueError("Invalid mesh constant")

    z_bounds = np.linspace(cylinder_data.bounds[4], 
                           cylinder_data.bounds[5], 
                           z_mesh + 1)

    angular_bounds = np.linspace(0, 2*np.pi, ang_mesh + 1)

    x_center = cylinder_data.center[0]
    y_center = cylinder_data.center[1]

    particle_x_pos = particle_data.points[:,0]
    particle_y_pos = particle_data.points[:,1]
    particle_z_pos = particle_data.points[:,2]

    # Calculate the radial position of the particles
    particle_radial_pos = np.sqrt((particle_x_pos - x_center) ** 2
                                  + (particle_y_pos - y_center) ** 2)

    # Calculate the angular position of the particles
    resolved_angular_data = (np.arctan2(
                                        (particle_y_pos - y_center),
                                        (particle_x_pos - x_center)
                                        ) 
                            + np.pi + rotation) % (2*np.pi)

    # Create the empty mesh elements array
    mesh_elements = np.empty(particle_data.n_points)
    mesh_elements[:] = np.nan

    cells = []
    occupied_cells = []
    mesh_data = []    

    mesh_id = int(0)
    for k in range(len(z_bounds) - 1):

        above_lower_z = particle_z_pos >= z_bounds[k]
        below_upper_z = particle_z_pos < z_bounds[k+1]

        for i in range(len(radial_bounds) - 1):

            above_lower_r = particle_radial_pos >= radial_bounds[i]
            below_upper_r = particle_radial_pos < radial_bounds[i+1]

            for j in range(len(angular_bounds) - 1):

                above_lower_angle = resolved_angular_data >= angular_bounds[j]
                below_upper_angle = resolved_angular_data < angular_bounds[j+1]

                # Boolean array of particles in the mesh element
                mesh_element = (
                    (above_lower_z & below_upper_z) 
                    & (above_lower_r & below_upper_r) 
                    & (above_lower_angle & below_upper_angle)
                )

                # Write mesh identifier to particles inside the 
                # mesh element
                mesh_elements[mesh_element] = int(mesh_id)

                cells.append(mesh_id)
                if sum(mesh_element) > 0:
                    occupied_cells.append(mesh_id)

                # Store the mesh element id, bounds and number of 
                # particles
                mesh_data.append((mesh_id, radial_bounds[i],
                                  radial_bounds[i+1], angular_bounds[j],
                                  angular_bounds[j+1], z_bounds[k],
                                  z_bounds[k+1], sum(mesh_element)))
                
                mesh_id += int(1)
    
    mesh_df = pd.DataFrame(mesh_data, columns=["mesh id", "radial_lower_bound",
                                               "radial_upper_bound",
                                               "angular_lower_bound",
                                               "angular_upper_bound",
                                               "z_lower_bound",
                                               "z_upper_bound",
                                               "n_particles"])
    
    # Add the mesh elements to the particle data
    particle_data[mesh_column] = mesh_elements

    # Count the number of particles in and out of the mesh elements
    n_unmeshed_particles = sum(np.isnan(mesh_elements))
    n_meshed_particles = len(mesh_elements) - n_unmeshed_particles

    mesh = Mesh(mesh_column, cells, occupied_cells, n_meshed_particles,
                n_unmeshed_particles, mesh_df)

    return (particle_data, mesh)