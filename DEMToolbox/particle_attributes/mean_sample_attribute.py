import numpy as np
import pandas as pd
import warnings

from ..classes.particle_attribute import ParticleAttribute
from ..classes.particle_samples import ParticleSamples

def mean_sample_attribute(particle_data,
                          attribute:ParticleAttribute, 
                          samples:ParticleSamples,
                          append_column=None):
    """Calculate the mean value of an attribute in each sample.

    Parameters
    ----------
    particle_data: vtkPolyData
        The particle vtk containing a field column.
    attribute: ParticleAttribute
        The attribute to calculate the mean of.
    samples: ParticleSamples
        The samples object containing the sample data to calculate the
        mean of the attribute for.
    append_column: str, optional
        The name of the appended column, by default None.

    Returns
    -------
    particle_data: vtkPolyData
        The particle vtk with the mean attribute column added.
    mean_attribute: ParticleAttribute
        A particle attribute object containing the field column, the
        attribute column and the mean array.

    Raises
    ------
    UserWarning
        If the particle data has no points return unedited particle data
        and an empty mean attribute object.
    UserWarning
        If the attribute is not found in the particle data return
        unedited particle data and an empty mean attribute object.
    UserWarning
        If the samples are not found in the particle data return
        unedited particle data and an empty mean attribute object.
    """
    if particle_data.n_points == 0:
        warnings.warn("no particles in particle_data vtk", UserWarning)
        mean_attribute = ParticleAttribute(attribute.field,
                                           append_column,
                                           [[None, None]])
        return particle_data, mean_attribute
    
    if attribute.attribute not in particle_data.point_data.keys():
        warnings.warn("attribute not in particle_data vtk", UserWarning)
        mean_attribute = ParticleAttribute(attribute.field,
                                           append_column,
                                           [[None, None]])
        return particle_data, mean_attribute
    
    if attribute.field not in particle_data.point_data.keys():
        warnings.warn("attribute field not in particle_data vtk", UserWarning)
        mean_attribute = ParticleAttribute(attribute.field,
                                           append_column,
                                           [[None, None]])
        return particle_data, mean_attribute
    
    if samples.name not in particle_data.point_data.keys():
        warnings.warn("samples not in particle_data vtk", UserWarning)
        mean_attribute = ParticleAttribute(attribute.field,
                                           append_column,
                                           [[None, None]])
        return particle_data, mean_attribute
    
    if append_column is None:
        append_column = attribute.attribute + "_mean"
    
    attribute_mean = np.empty(particle_data.n_points)
    attribute_mean[:] = np.nan
    for sample_id in samples.occupied_cells: 
        mesh_boolean_mask = particle_data[samples.name] == sample_id
        sample_mean = np.nanmean(
                        particle_data[attribute.attribute][mesh_boolean_mask])
        
        # Assign the sample mean to the particles in the sample
        attribute_mean[mesh_boolean_mask] = sample_mean

    # Add the sample mean to the particle data
    particle_data[append_column] = attribute_mean

    field = particle_data[attribute.field]

    attribute_data = [(field_i, attribute_i) 
                      for field_i, attribute_i in zip(field, attribute_mean)
                    ]

    mean_attribute = ParticleAttribute(attribute.field, 
                                       append_column, 
                                       attribute_data)

    return particle_data, mean_attribute