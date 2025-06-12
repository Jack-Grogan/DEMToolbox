import numpy as np
import warnings
import pandas as pd

from ..classes.particle_samples import ParticleSamples

def homogeneity_index(particle_data, 
                      attribute_column, 
                      samples:ParticleSamples, 
                      homogeniety_column=None, 
                      verbose=False):
    """ Calculate the homogeneity index of the particle data.

    The homogeneity index is a measure of the uniformity of a property
    across a sample of particles. It is calculated as the standard
    deviation of the sample mean of the property in each mesh element,
    with respect to the bulk mean of the property as defined in the 
    work of Windows-Yule et al. [1].

    References
    ----------

    [1] Windows-Yule K, Nicuşan L, Herald MT, Manger S, Parker D. 
        Positron emission particle tracking: A comprehensive guide. 
        IOP Publishing; 2022 Jun 1.

    Parameters
    ----------    
    
    particle_data : vtkPolyData
        The particle vtk.
    attribute_column : str
        The name of the attribute column whos deviation across the
        samples is to be calculated.
    samples : ParticleSamples
        The samples object containing the sample data to calculate the
        homogeneity index for.
    homogeniety_column : str, optional
        The name of the column to store the sample mean values.
        If None, the default is to use the attribute column name with
        "_homogeneity" appended to it.
    verbose : bool, optional
        Print the homogeneity index, bulk mean, sample means, and
        number of samples, by default False.

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the homogeneity column added.
    homogeneity_index : float
        The homogeneity index of the particle data.

    Raises
    ------
    UserWarning
        If the particle data has no points return unedited particle
        data and NaN for homogeneity index.
    UserWarning
        If the attribute column is not found in the particle data
        return unedited particle data and NaN for homogeneity index.
    UserWarning
        If the attribute column contains NaN values, the mean
        will be NaN and a warning will be raised.
    UserWarning
        If the sample column is not found in the particle data
        return unedited particle data and NaN for homogeneity index.
    ValueError
        If the attribute column is not numeric.
    ValueError
        If the sample column is not numeric.
    """
    if particle_data.n_points == 0:
        warnings.warn(("Cannot calculate homogeneity index for empty "
                       "particle file."), UserWarning)
        return particle_data, np.nan
    
    if attribute_column not in particle_data.point_data.keys():
        warnings.warn((f"{attribute_column} not found in particle file, "
                       "returning NaN."), UserWarning)
        return particle_data, np.nan
    
    if samples.name not in particle_data.point_data.keys():
        warnings.warn((f"{samples.name} not found in particle file, "
                       "returning NaN."), UserWarning)
        return particle_data, np.nan
    
    # Check if the attribute column is numeric
    if not np.issubdtype(particle_data[attribute_column].dtype, np.number):
        raise ValueError(f"Attribute column {attribute_column} is not numeric.")
    
    # Check if the sample column is numeric
    if not np.issubdtype(particle_data[samples.name].dtype, np.number):
        raise ValueError(f"Sample column {samples.name} is not numeric.")
    
    # Calculate the mean of the attribute column
    bulk_mean = np.mean(particle_data[attribute_column])
    if np.isnan(bulk_mean):
        warnings.warn("Bulk mean is NaN. Returning NaN for homogeneity index.")
        return particle_data, np.nan
    
    if homogeniety_column is None:
        homogeniety_column = attribute_column + "_homogeneity"

    particle_homogeneity_data = np.empty(particle_data.n_points)
    particle_homogeneity_data[:] = np.nan

    sample_means = np.zeros(samples.n_occupied_cells)
    for i, ids in enumerate(samples.occupied_cells):
        sample_boolean_mask = particle_data[samples.name] == ids

        # Calculate the mean of the attribute column for the mesh element
        mean = np.mean(
             particle_data[attribute_column][sample_boolean_mask.astype(bool)])
        sample_means[i] = mean
        
        # Update the particle homogeneity data with the mean value
        particle_homogeneity_data[sample_boolean_mask] = mean

    particle_data[homogeniety_column] = particle_homogeneity_data

    # Calculate the homogeneity index
    homogeneity_index = (sum((sample_means - bulk_mean) ** 2)
                         / len(sample_means)) ** 0.5
    
    if verbose:
        print("Homogeneity index: ", homogeneity_index)
        print("Bulk mean: ", bulk_mean)
        print("Sample means: ", sample_means)
        print("Number of samples: ", len(sample_means))

    return particle_data, homogeneity_index


        
        
   


