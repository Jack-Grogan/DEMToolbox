import numpy as np
import warnings

def macro_scale_lacey_mixing(particle_data, split_column, mesh_column,
                             cell_conc_column=None, verbose=False):
    """Calculate the macro scale Lacey mixing index.
    
    The Lacey mixing index is a measure of the sample variance of a 
    target particle types concentration in a binary particle system.
    This is a macro scale version of the Lacey mixing index that 
    uses the mesh elements to calculate the variance. The contribution
    of each mesh element to the variance is weighted by the volume of
    particles in the mesh element in line with the work of
    Chandratilleke et al. [1]. The mesh element concentration is
    calculated as on a volume basis. The perfectly mixed variance is 
    calculated on the assumption that the sample of the cells in a 
    perfectly mixed system would have concentrations following a 
    binomial distribution [2]. The unmixed variance is calculated as 
    the variance of the bulk concentration of the target particle type.

    class_0 = target particle type
    class_1 = non-target particle type

    [1] Chandratilleke GR, Yu AB, Bridgwater J, Shinohara K. 
        A particle‐scale index in the quantification of mixing of 
        particles. AIChE journal. 2012 Apr;58(4):1099-118.

    [2] Fan LT, Too JR, Rubison RM, Lai FS. Studies on multicomponent 
        solids mixing and mixtures Part III. Mixing indices. Powder 
        Technology. 1979 Sep 1;24(1):73-89.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk must contain the split_column and mesh_column 
        and a radius column.
    split_column : str
        The name of the column in the particle data that defines the 
        particle type.
    mesh_column : str
        The name of the column in the particle data that defines the 
        mesh element that the particle resides in.
    cell_conc_column : str, optional
        The name of the column in the particle data to store the
        concentration of the target particle type in the mesh element.
    verbose : bool, optional
        Print the Lacey mixing index, by default False.

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the concentration column added.
    cell_conc_column : str
        The name of the column in the particle data that stores the
        concentration of the target particle type in the mesh element.
    lacey : float
        The Lacey mixing index.

    Raises
    ------
    Exception
        If the particle data contains more than 2 particle types.
    UserWarning
        If the particle data has no points return unedited particle
        data and a lacey of NaN.
    UserWarning
        If the split_column is not found in the particle data return
        unedited particle data and a lacey of NaN.
    UserWarning
        If the mesh_column is not found in the particle data return
        unedited particle data and a lacey of NaN.
    UserWarning
        If fewer than 2 non-empty lacey mesh for particle data return
        unedited particle data and a lacey of NaN.
    UserWarning
        If only one particle type present in the mesh at this timestep
        return unedited particle data and a lacey of NaN.
    """
    if particle_data.n_points == 0:
        warnings.warn(("Cannot calculate Lacey mixing index "
                      "for empty particle file"), UserWarning)
        return particle_data, np.nan, cell_conc_column
    
    if split_column not in particle_data.point_data.keys():
        warnings.warn((f"{split_column} not found in particle file, "
                      "returning NaN"), UserWarning)
        return particle_data, np.nan, cell_conc_column
    
    if mesh_column not in particle_data.point_data.keys():
        warnings.warn((f"{mesh_column} not found in particle file, "
                      "returning NaN"), UserWarning)
        return particle_data, np.nan, cell_conc_column

    if len(np.unique(particle_data[split_column])) != 2:
        raise Exception("Lacey can only support 2 particle types")

    # Boolean mask for class 0 particles
    class_0_split = (particle_data[split_column].astype(int)
                    ^ np.ones(len(particle_data[split_column])).astype(int))

    # Boolean mask for class 1 particles
    class_1_split = particle_data[split_column].astype(int)

    # Get the unique mesh ids so only the non-empty lacey mesh elements 
    # are considered
    mesh = particle_data[mesh_column]
    mesh_ids = np.unique(mesh)
    mesh_ids = mesh_ids[~np.isnan(mesh_ids)].astype(int)

    # Create a boolean mask for each lacey mesh element
    mesh_id_booleans = []
    for ids in mesh_ids:
        mesh_boolean_mask = mesh == ids
        mesh_id_booleans.append(mesh_boolean_mask)

    class_0_mesh_volume = np.zeros(len(mesh_ids))
    class_1_mesh_volume = np.zeros(len(mesh_ids))
    total_mesh_volume = np.zeros(len(mesh_ids))

    particles_concentration = np.empty(particle_data.n_points)
    particles_concentration[:] = np.nan

    for i, mesh_element in enumerate(mesh_id_booleans):

        # Boolean mask for particles of class 0 in the mesh element
        particles_class_0 = class_0_split & mesh_element

        # Boolean mask for particles of class 1 in the mesh element
        particles_class_1 = class_1_split & mesh_element

        # Calculate the volume of particles of class 0 mesh element
        class_0_radii = particle_data["radius"][particles_class_0.astype(bool)]
        class_0_volume = 4/3 * np.pi * class_0_radii**3

        # Calculate the volume of particles of class 1 mesh element
        class_1_radii = particle_data["radius"][particles_class_1.astype(bool)]
        class_1_volume = 4/3 * np.pi * class_1_radii**3

        # Total volume of particles in the mesh element 
        class_0_mesh_volume[i] = sum(class_0_volume)
        class_1_mesh_volume[i] = sum(class_1_volume)
        total_mesh_volume[i] = sum(class_0_volume) + sum(class_1_volume)

        # Assign the concentration value of the mesh element to all
        # particles that reside in the mesh element. Used for 
        # concentration visualisation
        particles_concentration[mesh_element] = (
            sum(class_0_volume)
                / (
                    sum(class_0_volume)
                    + sum(class_1_volume)
                )
            )

    # Append particle concentration and mesh elements to the particle_data
    if cell_conc_column is not None:
        particle_data[cell_conc_column] = particles_concentration
    else:
        particle_data[f"{split_column}_conc"] = particles_concentration
    
    if len(mesh_ids) < 2:
        warnings.warn(
            (f"Fewer than 2 non-empty lacey mesh for particle data."
            "Setting Lacey to NaN, consider refining lacey mesh"),
            UserWarning,
        )
        lacey = np.nan
    elif sum(class_0_mesh_volume) == 0 or sum(class_1_mesh_volume) == 0:
        warnings.warn(
            (f"Only one particle type present in the mesh at this timestep."
            " Setting Lacey to NaN."),
            UserWarning,
        )
        lacey = np.nan
    else:
        bulk_concentration = (
            np.sum(class_0_mesh_volume)
            / (
                np.sum(class_0_mesh_volume)
                + np.sum(class_1_mesh_volume)
            )
        )

        concentrations = (
            class_0_mesh_volume
            / (
                class_0_mesh_volume
                + class_1_mesh_volume
            )
        )

        variance = np.sum(total_mesh_volume
            / np.sum(total_mesh_volume)
            * (
                (concentrations - bulk_concentration) ** 2
            )
        )

        # Calculate the unmixed variance / segregated variance
        unmixed_variance = bulk_concentration * (1 - bulk_concentration)

        # mean radii of the particles
        mean_particle_radii = np.mean(particle_data["radius"])
        mean_paricle_volume = 4/3 * np.pi * mean_particle_radii**3

        # mean volume of the particles in the cell
        mean_cell_volume = np.mean(total_mesh_volume)

        mixed_variance = (unmixed_variance
                           / (mean_cell_volume / mean_paricle_volume))

        lacey = ((variance - unmixed_variance)
                  / (mixed_variance - unmixed_variance))

        if verbose:
            print(f"Lacey mixing index: {lacey}")
            
    return particle_data, cell_conc_column, lacey