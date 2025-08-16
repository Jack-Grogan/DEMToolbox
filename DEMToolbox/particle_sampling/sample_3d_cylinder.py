import numpy as np
import warnings

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def sample_3d_cylinder(particle_data, 
                       cylinder_data, 
                       resolution, 
                       sample_constant="volume", 
                       rotation=0, 
                       append_column="3D_cylinder_samples",
                       particle_id_column="id"):
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
    particle_id_column : str, optional
        The name of the particle id column in the particle data, by
        default "id".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the samples column added.
    samples : ParticleSamples
        A ParticleSamples object containing the samples column name,
        the particle ids and their corresponding sample ids stored in a
        ParticleAttribute object, a list of sample elements, a list of
        occupied sample elements, a list of the number of particles in
        the sample elements, the number of particles in the sample
        elements, the number of sampled and unsampled particles.

    Raises
    ------
    ValueError
        If resolution is not a 3 element list of integers.
    ValueError
        If resolution is not a list of 3 integers.
    ValueError
        If rotation is not an integer or float.
    ValueError
        If resolution is less than or equal to 0 in any dimension.
    UserWarning
        If the particle data has no points return unedited particle
        data and an empty samples object.
    UserWarning
        If the container data has no points return unedited particle
        data and an empty samples object.
    """
    if len(resolution) != 3:
        raise ValueError("Resolution must be a list of 3 integers.")
    
    if not all([isinstance(i, int) for i in resolution]):
        raise ValueError("Resolution must be a list of 3 integers.")
    
    if not isinstance(rotation, (int, float)):
        raise ValueError("Rotation must be an integer or float.")
    
    if resolution[0] <= 0 or resolution[1] <= 0 or resolution[2] <= 0:
        raise ValueError("Resolution must be greater than 0 in all "
                         "dimensions.")
    
    if particle_data.n_points == 0:
        warnings.warn("Cannot sample empty particles file.", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return (particle_data, samples)
    
    if cylinder_data.n_points == 0:
        warnings.warn("Cannot sample empty container file.", UserWarning)
        sample_attribute = ParticleAttribute(particle_id_column, 
                                        append_column,
                                        np.empty((0, 2)))
        samples = ParticleSamples(
            append_column, sample_attribute, [], [], [], 0, 0)
        return (particle_data, samples)

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


    i = np.digitize(particle_z_pos, z_bounds) - 1
    j = np.digitize(particle_radial_pos, radial_bounds) - 1
    k = np.digitize(resolved_angular_data, angular_bounds) - 1

    # Filter out particles outside the valid ranges
    mask = (
        (i >= 0) & (i < len(z_bounds) - 1) &
        (j >= 0) & (j < len(radial_bounds) - 1) &
        (k >= 0) & (k < len(angular_bounds) - 1)
    )
    i, j, k = i[mask], j[mask], k[mask] 


    sample_id = np.ravel_multi_index((i, j, k),
                                    (len(z_bounds)-1,
                                    len(radial_bounds)-1,
                                    len(angular_bounds)-1)).astype(int)
          
    cells = np.arange(resolution[0] * resolution[1] * resolution[2], dtype=int)
    occupied_cells, counts = np.unique(sample_id, return_counts=True)
    occupied_cells = occupied_cells[occupied_cells != -1]
    cell_particles = np.zeros_like(cells, dtype=int)

    np.add.at(cell_particles, occupied_cells, counts)

    sample_elements = np.full(particle_data.n_points, -1, dtype=int)
    sample_elements[mask] = sample_id
    
    # Add the sample elements to the particle data
    particle_data[append_column] = sample_elements

    # Create a DataFrame for the sample elements
    sample_data = np.array([particle_data["id"], sample_elements]).T
    sample_attribute = ParticleAttribute(particle_id_column, 
                                          append_column,
                                          sample_data)

    # Count the number of sampled and unsampled particles
    n_unsampled_particles = np.sum(sample_elements == -1)
    n_sampled_particles = len(sample_elements) - n_unsampled_particles

    samples = ParticleSamples(append_column,
                              sample_attribute,
                              cells, 
                              occupied_cells, 
                              cell_particles, 
                              n_sampled_particles, 
                              n_unsampled_particles,
                              )

    return (particle_data, samples)