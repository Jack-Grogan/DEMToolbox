import numpy as np
import warnings

from ..classes.particle_samples import ParticleSamples
from ..classes.particle_attribute import ParticleAttribute

def macro_scale_lacey_mixing(particle_data, 
                             attribute:ParticleAttribute, 
                             samples:ParticleSamples,
                             append_column=None,
                             verbose=False):
    r"""Calculate the macro scale Lacey mixing index.
    
    The Lacey mixing index is a measure of the sample variance of a 
    target particle types concentration in a binary particle system.
    This is a macro scale version of the Lacey mixing index that uses 
    samples to calculate the variance. The contribution of each sample 
    to the variance is weighted by the volume of articles in the sample 
    in line with the work of Chandratilleke et al. [1]_ . The sample 
    concentration is calculated as on a volume basis. The perfectly 
    mixed variance is calculated on the assumption that samples in a
    perfectly mixed system would have concentrations following a 
    binomial distribution [2]_ . The unmixed variance is calculated as 
    the variance of the bulk concentration of the target particle type.

    The Lacey mixing index is calculated as:

    .. math::

        M = \frac{\sigma^2 - \sigma_0^2}{\sigma_r^2 - \sigma_0^2}

    .. math::

        \sigma_0^2 = P_0(1 - P_0)

    .. math::

        \sigma_r^2 = \frac{P_0(1 - P_0)}{\bar{n}_0 + \bar{n}_1}

    .. math::

        \sigma^2 = \sum_{i=1}^{N_s} \frac{v_i}{V} (p_{0,i} - \bar{p}_0)^2

    where :math:`\sigma^2` is the sample variance of the concentration
    of the target particle type in the samples, :math:`\sigma_0^2`
    is the unmixed variance, and :math:`\sigma_r^2` is the perfectly
    mixed variance.

    :math:`P_0`: 
        Bulk concentration of the target particle type

    :math:`\bar{n}_0`: 
        Mean number of particles of the target particle type in the 
        samples

    :math:`\bar{n}_1`:
        Mean number of particles of the non-target particle type in
        the samples

    :math:`v_i`:
        Volume of particles in sample :math:`i`

    :math:`V`:
        Total volume of particles in the samples

    :math:`p_{0,i}`:
        Volume fraction of the target particle type in sample :math:`i`

    :math:`\bar{p}_0`:
        Mean volume fraction of the target particle type in the samples

    :math:`N_s`:
        Number of samples

    References
    ----------

    [1] Chandratilleke GR, Yu AB, Bridgwater J, Shinohara K. 
        A particle‐scale index in the quantification of mixing of 
        particles. AIChE journal. 2012 Apr;58(4):1099-118.

    [2] Fan LT, Too JR, Rubison RM, Lai FS. Studies on multicomponent 
        solids mixing and mixtures Part III. Mixing indices. Powder 
        Technology. 1979 Sep 1;24(1):73-89.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    attribute : ParticleAttribute
        The attribute to calculate the Lacey mixing index for.
    samples : ParticleSamples
        The samples object containing the sample data.
    append_column : str, optional
        The name of the appended column, by default None.
    verbose : bool, optional
        Print the Lacey mixing index, by default False.

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the concentration column added.
    lacey : float
        The Lacey mixing index.

    Raises
    ------
    UserWarning
        If the particle data has no points return unedited particle
        data and NaN for Lacey.
    UserWarning
        If the attribute is not found in the particle data return
        unedited particle data and NaN for Lacey.
    UserWarning
        If the samples are not found in the particle data return
        unedited particle data and NaN for Lacey.
    UserWarning
        If the attribute contains only one particle type with a value
        of 0 or 1 return unedited particle data and NaN for Lacey.
    ValueError
        If the attribute contains particle types with values other 
        than 0 and 1 raise a ValueError.
    UserWarning
        If there are fewer than 2 non-empty samples in the particle
        data return unedited particle data and NaN for Lacey.
    UserWarning
        If only one particle type is present in the samples at this
        timestep return unedited particle data and NaN for Lacey.
    UserWarning
        If the mixed variance is equal to the unmixed variance,
        return NaN for Lacey.
    """
    if particle_data.n_points == 0:
        warnings.warn(("Cannot calculate Lacey mixing " 
                       "index for empty particle file."), UserWarning)
        return particle_data, np.nan
    
    if attribute.attribute not in particle_data.point_data.keys():
        warnings.warn((f"{attribute.attribute} not found in particle file, "
                      "returning NaN."), UserWarning)
        return particle_data, np.nan
    
    if samples.name not in particle_data.point_data.keys():
        warnings.warn((f"{samples.name} not found in particle file, "
                      "returning NaN."), UserWarning)
        return particle_data, np.nan

    ones_and_zeros = np.setdiff1d(particle_data[attribute.attribute], [1, 0])

    if len(ones_and_zeros) == 0:
        if len(np.unique(particle_data[attribute.attribute])) == 1:
            warnings.warn(("particle data contains only particle type "
                           f"{particle_data[attribute.attribute][0]}, "
                           "setting Lacey to NaN."), UserWarning)
            return particle_data, np.nan
    else:
        raise ValueError(("particle data contains particle types with values "
                          "other than 0 and 1, cannot calculate Lacey mixing "
                          f"index. Found particle types: {ones_and_zeros}"))

    # Calculate the volume of each particle and the mean particle volume
    particle_volumes = 4/3 * np.pi * particle_data["radius"] ** 3
    mean_particle_volume = np.mean(particle_volumes)

    # Array of sample ids for each particle, -1 for unsampled particles
    sample_ids = particle_data[samples.name].astype(int)

    # Create boolean masks for the two particle types
    class_1_mask = particle_data[attribute.attribute].astype(bool)
    class_0_mask = ~class_1_mask

    # valid is a boolean array indicating which particles have valid sample ids
    # sample_ids == -1 indicates unsampled particles, so valid is True for 
    # sampled particles
    valid = sample_ids != -1

    sample_ids_valid = sample_ids[valid]
    particle_volumes_valid = particle_volumes[valid]
    class_0_mask_valid = class_0_mask[valid]

    # Calculate the volume of each particle type in each sample using 
    # np.bincount
    class_0_volume_per_cell = np.bincount(
        sample_ids_valid[class_0_mask_valid],
        weights=particle_volumes_valid[class_0_mask_valid],
        minlength=samples.n_cells,
    )
    class_1_volume_per_cell = np.bincount(
        sample_ids_valid[~class_0_mask_valid],
        weights=particle_volumes_valid[~class_0_mask_valid],
        minlength=samples.n_cells,
    )

    # Discard empty samples from the volume arrays
    class_0_sample_volume = class_0_volume_per_cell[samples.occupied_cells]
    class_1_sample_volume = class_1_volume_per_cell[samples.occupied_cells]
    total_sample_volume = class_0_sample_volume + class_1_sample_volume

    # Calculate the concentration of the target particle type in each sample
    # that is occupied by at least one particle of either type. Unoccupied
    # samples will be ignored in the calculation of the Lacey mixing index.
    # The concentration is calculated as the volume fraction of the target
    # particle type in each sample.
    concentrations = class_0_sample_volume / total_sample_volume

    # Create an Nan array for the concentration of the target particle type 
    # in each sample including unoccupied samples. This will be used to assign 
    # the concentration of the target particle type to each particle based on 
    # its sample id.
    concentration_per_cell = np.full(samples.n_cells, np.nan)

    # Assign the concentration of the target particle type to each occupied sample
    concentration_per_cell[samples.occupied_cells] = concentrations

    # Create an array of particle concentrations based on their sample ids
    particles_concentration = np.full(particle_data.n_points, np.nan)

    # Assign the concentration of the target particle type to each particle 
    # based on its sample id. Concentration is only assigned to particles 
    # that are within the sample space (i.e., have a valid sample id (not -1),
    # -1 is used to indicate that a particle is not in any sample). Particles
    # that are not in any sample will have a concentration of NaN.
    particles_concentration[valid] = concentration_per_cell[sample_ids[valid]]

    # Append particle concentration and samples to the particle_data
    if append_column is not None:
        particle_data[append_column] = particles_concentration
    else:
        particle_data[f"{attribute.attribute}_conc"] = particles_concentration

    if samples.n_occupied_cells < 2:
        warnings.warn(
            ("Fewer than 2 non-empty samples in particle data. "
            "Setting Lacey to NaN, consider refining the sample "
            "resolution."),
            UserWarning,
        )
        lacey = np.nan
        variance = np.nan
        unmixed_variance = np.nan
        mixed_variance = np.nan

    else:
        bulk_concentration = (
            np.sum(class_0_sample_volume)
            / (
                np.sum(class_0_sample_volume)
                + np.sum(class_1_sample_volume)
            )
        )

        variance = np.sum(total_sample_volume
            / np.sum(total_sample_volume)
            * (
                (concentrations - bulk_concentration) ** 2
            )
        )

        # Calculate the unmixed variance / segregated variance
        unmixed_variance = bulk_concentration * (1 - bulk_concentration)
        
        # Calculate the perfectly mixed variance
        mixed_variance = (unmixed_variance
                           / (np.mean(total_sample_volume) 
                              / mean_particle_volume))
        
        if mixed_variance == unmixed_variance:
            warnings.warn(
                ("Mixed variance is equal to unmixed variance, "
                 "setting Lacey to NaN on account of division by zero. "
                 "This is likely due to the sample resolution being too "
                 "fine leading to each sample containing only one particle."
                 " Consider coarsening the sample resolution."), 
                UserWarning,
            )
            lacey = np.nan
        else:
            lacey = ((variance - unmixed_variance)
                    / (mixed_variance - unmixed_variance))

    if verbose:
        print(f"Lacey mixing index: {variance} - {unmixed_variance} " 
              f"/ {mixed_variance} - {unmixed_variance} = {lacey}")
            
    return particle_data, lacey