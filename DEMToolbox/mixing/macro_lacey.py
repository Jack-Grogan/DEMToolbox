import numpy as np
import warnings

def macro_scale_lacey_mixing(particle_data, split_column, mesh_column, verbose=False):

    # Return NaN if the particles file is empty
    if split_column not in particle_data.point_data.keys():
        warnings.warn(f"{split_column} not found in particle file, returning NaN")
        return np.nan

    # Raise error if the file does not contain two particle types
    if len(np.unique(particle_data[split_column])) != 2:
        raise Exception("Lacey can only support 2 particle types")

    # Boolean mask for class 0 particles
    class_0_split = (particle_data[split_column].astype(int)
                    ^ np.ones(len(particle_data[split_column])).astype(int))

    # Boolean mask for class 1 particles
    class_1_split = particle_data[split_column].astype(int)

    # Get the unique mesh ids so that only the non-empty lacey mesh elements are considered
    mesh = particle_data[mesh_column]
    mesh_ids = np.unique(mesh)
    mesh_ids = mesh_ids[~np.isnan(mesh_ids)].astype(int)

    # Create a boolean mask for each lacey mesh element
    mesh_id_booleans = []
    for ids in mesh_ids:
        mesh_boolean_mask = mesh == ids
        mesh_id_booleans.append(mesh_boolean_mask)

    # Set up arrays to hold the cell wise volumes of particles of class 0 and 1 and the total
    basis_particle_class_0_meshed = np.zeros(len(mesh_ids))
    basis_particle_class_1_meshed = np.zeros(len(mesh_ids))
    total_basis_mesh_particle = np.zeros(len(mesh_ids))

    # Set up an nan list to hold particle concentrations
    particles_concentration = np.zeros(len(particle_data.points))
    particles_concentration[:] = np.nan

    # Loop through the lacey mesh elements
    for i, mesh_element in enumerate(mesh_id_booleans):

        # Boolean mask for particles of class 0 in the lacey mesh element
        mesh_particles_class_0 = class_0_split & mesh_element

        # Boolean mask for particles of class 1 in the lacey mesh element
        mesh_particles_class_1 = class_1_split & mesh_element

        # Calculate the volume of particles of class 0 lacey mesh element
        class_0_radii = particle_data["radius"][mesh_particles_class_0.astype(bool)]
        class_0_volume = 4/3 * np.pi * class_0_radii**3

        # Calculate the volume of particles of class 1 lacey mesh element
        class_1_radii = particle_data["radius"][mesh_particles_class_1.astype(bool)]
        class_1_volume = 4/3 * np.pi * class_1_radii**3

        # Calculate the total volume of particles in the lacey mesh element 
        basis_particle_class_0 = sum(class_0_volume)
        basis_particle_class_1 = sum(class_1_volume)

        # Write the total volume of particles of class 0 and 1 in the lacey mesh element
        # to arrays outside the loop
        basis_particle_class_0_meshed[i] = basis_particle_class_0
        basis_particle_class_1_meshed[i] = basis_particle_class_1
        total_basis_mesh_particle[i] = basis_particle_class_0 + basis_particle_class_1

        # Assign the concentration value of the mesh element to all particles that
        # reside in the mesh element. Used for concentration visualisation
        particles_concentration[mesh_element] = (
            basis_particle_class_1
                / (
                    basis_particle_class_0
                    + basis_particle_class_1
                )
            )

    # Append particle concentration and mesh elements to the particle_data
    particle_data[f"{split_column}_concentration"] = particles_concentration

    # Calculate lacey mixing index

    # Warn if there are fewer than 2 non-empty lacey mesh elements and return NaN
    if len(mesh_ids) < 2:
        warnings.warn(
            (f"Fewer than 2 non-empty lacey mesh for particle data."
            "Setting Lacey to NaN, consider refining lacey mesh"),
            UserWarning,
        )
        lacey = np.nan
    else:

        # Calculate the bulk concentration of particles of class 1
        bulk_concentration = (
            np.sum(basis_particle_class_1_meshed)
            / (
                np.sum(basis_particle_class_0_meshed)
                + np.sum(basis_particle_class_1_meshed)
            )
        )

        # Calculate the concentration of class 1 particles in each lacey mesh element
        concentrations = (
            basis_particle_class_1_meshed
            / (
                basis_particle_class_0_meshed
                + basis_particle_class_1_meshed
            )
        )

        # Calculate the variance of the concentration of class 1 particles
        variance = np.sum(total_basis_mesh_particle
            / np.sum(total_basis_mesh_particle)
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
        mean_cell_volume = np.mean(total_basis_mesh_particle)

        # Calculate the mixed variance
        mixed_variance = unmixed_variance / (mean_cell_volume / mean_paricle_volume)

        # If this is giving a runtime error it is likely due to a divide by zero
        # error. This is likely due to the particles being split in the z axis and
        # at the current timestep only one particle type is present in the mesh
        if unmixed_variance == 0:
            warnings.warn(
                (f"Unmixed variance is 0 for particle data."
                " Likely due to only one particle type being present"
                " in the mesh at this timestep. Setting Lacey to NaN."),
                UserWarning,
            )
            return np.nan


        lacey = (variance - unmixed_variance) / (mixed_variance - unmixed_variance)

        if verbose:
            print((f"lacey =  {variance} - {unmixed_variance}"
                f" / {mixed_variance} - {unmixed_variance} = {lacey}"))

    return particle_data, lacey