import numpy as np
import pandas as pd
import warnings

from ..classes.particle_samples import ParticleSamples

def sample_3d_cylinder(particle_data, cylinder_data, resolution, 
                       sample_constant="volume", rotation=0, 
                       append_column="3D_cylinder_samples"):
    """Split the particles into samples split along a 3D cylinder.

    Split the particles into samples split along a 3D cylinder defined
    by a cylinder container vtk and a resolution in the angular, radial
    and z directions.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    cylinder_data : vtkPolyData
        The cylinder container vtk.
    resolution : list
        The resolution of the 3D sample space in the form [m, n, o].
    sample_constant : str, optional
        The constant to sample the radial direction, by default 
        "volume". If "volume" the radial direction will be sampled
        to generate equal volume sample elements. If "radius" the
        radial direction will be sampled to generate equal radius
        sample elements.
    rotation : int or float, optional
        The rotation of the cylinder in radians, by default 0.
    append_column : str, optional
        The name of the samples column to append to the particle data,
        by default "3D_cylinder_samples".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the samples column added.
    samples : ParticleSamples
        A ParticleSamples object containing the samples column name, a 
        list of sample elements, a list of occupied sample elements, 
        a list of the number of particles in the sample elements, the
        number of particles in the sample elements, the number of 
        sampled and unsampled particles and a dataframe containing the
        sample id, lower bound, upper bound and number of particles in
        the sample element.

    Raises
    ------
    ValueError
        If resolution is not a 3 element list of integers.
    ValueError
        If resolution is not a list of 3 integers.
    ValueError
        If rotation is not an integer or float.
    UserWarning
        If the particle data has no points return unedited particle
        data and an empty samples object.
    UserWarning
        If the container data has no points return unedited particle
        data and an empty samples object.
    """
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        samples = ParticleSamples(append_column, [], [], [], 0, 0)
        return (particle_data, samples)
    
    if cylinder_data.n_points == 0:
        warnings.warn("Cannot sample empty container file.", UserWarning)
        samples = ParticleSamples(append_column, [], [], [], 0, 0)
        return (particle_data, samples)
    
    if len(resolution) != 3:
        raise ValueError("Resolution must be a list of 3 integers.")
    
    if not all([isinstance(i, int) for i in resolution]):
        raise ValueError("Resolution must be a list of 3 integers.")
    
    if not isinstance(rotation, (int, float)):
        raise ValueError("Rotation must be an integer or float.")

    # Determine the radius of the cylinder container
    x_radii = abs(cylinder_data.bounds[1] - cylinder_data.bounds[0])/2
    y_radii = abs(cylinder_data.bounds[3] - cylinder_data.bounds[2])/2
    radii = max(x_radii, y_radii)

    # Define the sample bounds linearly along the resolved container
    if sample_constant == "radius":
        radial_bounds = np.linspace(0, radii, resolution[1] + 1)
    elif sample_constant == "volume":
        radial_bounds = np.sqrt(np.linspace(0, radii**2, resolution[1] + 1))
    else:
        raise ValueError("Invalid sample constant. "
                         "Must be 'radius' or 'volume'.")

    z_bounds = np.linspace(cylinder_data.bounds[4], 
                           cylinder_data.bounds[5], 
                           resolution[2] + 1)

    angular_bounds = np.linspace(0, 2*np.pi, resolution[0] + 1)

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

    # Create the empty sample elements array
    sample_elements = np.empty(particle_data.n_points)
    sample_elements[:] = np.nan

    cells = []
    occupied_cells = []
    sample_data = []
    cell_particles = []

    sample_id = int(0)
    for k in range(len(z_bounds) - 1):

        above_lower_z = particle_z_pos >= z_bounds[k]
        below_upper_z = particle_z_pos < z_bounds[k+1]

        for i in range(len(radial_bounds) - 1):

            above_lower_r = particle_radial_pos >= radial_bounds[i]
            below_upper_r = particle_radial_pos < radial_bounds[i+1]

            for j in range(len(angular_bounds) - 1):

                above_lower_angle = resolved_angular_data >= angular_bounds[j]
                below_upper_angle = resolved_angular_data < angular_bounds[j+1]

                # Boolean array of particles in the sample element
                sample_element = (
                    (above_lower_z & below_upper_z) 
                    & (above_lower_r & below_upper_r) 
                    & (above_lower_angle & below_upper_angle)
                )

                # Write sample identifier to particles inside the 
                # sample element
                sample_elements[sample_element] = int(sample_id)
                cells.append(sample_id)
                cell_particles.append(sum(sample_element))
                if sum(sample_element) > 0:
                    occupied_cells.append(sample_id)

                # Store the sample element id, bounds and number of 
                # particles
                sample_data.append((sample_id, radial_bounds[i],
                                  radial_bounds[i+1], angular_bounds[j],
                                  angular_bounds[j+1], z_bounds[k],
                                  z_bounds[k+1], sum(sample_element)))
                
                sample_id += int(1)
    
    sample_df = pd.DataFrame(sample_data, 
                             columns=["sample id", "radial_lower_bound",
                                      "radial_upper_bound", 
                                      "angular_lower_bound",
                                      "angular_upper_bound",
                                      "z_lower_bound",
                                      "z_upper_bound",
                                      "n_particles"],
                            )
    
    # Add the sample elements to the particle data
    particle_data[append_column] = sample_elements

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = sum(np.isnan(sample_elements))
    n_sampled_particles = len(sample_elements) - n_unsampled_particles

    samples = ParticleSamples(append_column, cells, occupied_cells, 
                              cell_particles, n_sampled_particles, 
                              n_unsampled_particles, sample_df)

    return (particle_data, samples)