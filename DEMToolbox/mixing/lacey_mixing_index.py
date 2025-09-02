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
        If the attribute contains only one particle type with a value
        other than 0 or 1 raise a ValueError.
    ValueError
        If the attribute contains two particle types with values
        other than 0 or 1 raise a ValueError.
    ValueError
        If the attribute contains more than two particle types raise
        a ValueError.
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

    if len(np.setxor1d(particle_data[attribute.attribute], [1, 0])) != 0:
        if (len(np.unique(particle_data[attribute.attribute])) == 1 and
            len(np.setxor1d(particle_data[attribute.attribute]) == 1)):
            warnings.warn((f"particle data contains only particle type "
                           f"{particle_data[attribute.attribute][0]}, "
                           "setting Lacey to NaN."), UserWarning)
            return particle_data, np.nan
        elif len(np.unique(particle_data[attribute.attribute])) == 1:
            raise ValueError((f"particle data contains only one particle "
                             "type with value other than 0 and 1, cannot "
                             "calculate Lacey mixing index."))
        elif len(np.unique(particle_data[attribute.attribute])) == 2:
            raise ValueError((f"particle data contains two particle "
                             "types with values other than 0 and 1, cannot "
                             "calculate Lacey mixing index."))
        else:
            raise ValueError((f"particle data contains more than two "
                             "particle types, cannot calculate Lacey "
                             "mixing index."))
    
    # Boolean mask for class 0 particles
    class_0_split = (particle_data[attribute.attribute].astype(int)
                    ^ np.ones(
                        len(particle_data[attribute.attribute])).astype(int))

    # Boolean mask for class 1 particles
    class_1_split = particle_data[attribute.attribute].astype(int)

    # Calculate the mean volume of the particles
    particle_volumes = 4/3 * np.pi * particle_data["radius"] ** 3
    mean_paricle_volume = np.mean(particle_volumes)

    # Create a boolean mask for each Lacey sample
    sample_id_booleans = []
    for ids in samples.occupied_cells:
        sample_boolean_mask = particle_data[samples.name] == ids
        sample_id_booleans.append(sample_boolean_mask)

    class_0_sample_volume = np.zeros(samples.n_occupied_cells)
    class_1_sample_volume = np.zeros(samples.n_occupied_cells)
    total_sample_volume = np.zeros(samples.n_occupied_cells)

    particles_concentration = np.empty(particle_data.n_points)
    particles_concentration[:] = np.nan

    for i, sample_element in enumerate(sample_id_booleans):

        # Boolean mask for particles of class 0 in the sample
        particles_class_0 = class_0_split & sample_element

        # Boolean mask for particles of class 1 in the sample
        particles_class_1 = class_1_split & sample_element

        # Calculate the volume of particles of class 0 sample 
        class_0_radii = particle_data["radius"][particles_class_0.astype(bool)]
        class_0_volume = 4/3 * np.pi * class_0_radii ** 3
        
        # Calculate the volume of particles of class 1 sample
        class_1_radii = particle_data["radius"][particles_class_1.astype(bool)]
        class_1_volume = 4/3 * np.pi * class_1_radii ** 3

        # Total volume of particles in the sample 
        class_0_sample_volume[i] = sum(class_0_volume)
        class_1_sample_volume[i] = sum(class_1_volume)
        total_sample_volume[i] = sum(class_0_volume) + sum(class_1_volume)

        # Assign the concentration value of the sample element to all
        # particles that reside in the sample element. Used for 
        # concentration visualisation
        particles_concentration[sample_element] = (
            sum(class_0_volume)
                / (
                    sum(class_0_volume)
                    + sum(class_1_volume)
                )
            )

    # Append particle concentration and samples to the particle_data
    if append_column is not None:
        particle_data[append_column] = particles_concentration
    else:
        particle_data[f"{attribute.attribute}_conc"] = particles_concentration
    
    if samples.n_occupied_cells < 2:
        warnings.warn(
            (f"Fewer than 2 non-empty samples in particle data. "
            "Setting Lacey to NaN, consider refining the sample "
            "resolution."),
            UserWarning,
        )
        lacey = np.nan
        variance = np.nan
        unmixed_variance = np.nan
        mixed_variance = np.nan
    elif sum(class_0_sample_volume) == 0 or sum(class_1_sample_volume) == 0:
        warnings.warn(
            (f"Only one particle type present in the samples at this timestep."
            " Setting Lacey to NaN."),
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

        concentrations = (
            class_0_sample_volume
            / (
                class_0_sample_volume
                + class_1_sample_volume
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
                              / mean_paricle_volume))
        
        if mixed_variance == unmixed_variance:
            warnings.warn(
                ("Mixed variance is equal to unmixed variance, "
                 "setting Lacey to NaN."),
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