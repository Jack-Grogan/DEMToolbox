import numpy as np
import warnings

from ..classes.particle_attribute import ParticleAttribute

def append_attribute(particle_data, attribute:ParticleAttribute):
    """ Append the particle attribute to the particle data.

    Parameters
    ----------
    particle_data: vtkPolyData
        The particle vtk containing a field column.
    attribute: (ParticleAttribute)
        ParticleAttribute object containing the attribute data.

    Returns
    -------
    particle_data: vtkPolyData
        The particle vtk with the field attributes appended.

    Raises
    ------
    UserWarning
        If the particle data has no points return unedited particle data
        and raise a warning.
    UserWarning
        If the field is not found in the particle data return unedited
        particle data and raise a warning
    """
    field = attribute.field

    if (field in particle_data.point_data.keys() 
        and particle_data.n_points != 0):

        new_column = np.zeros(len(particle_data[field]))
        new_column[:] = np.nan

        for (field_i, attribute_i) in attribute.data:
            new_column[particle_data[field] == field_i] = attribute_i
        particle_data[attribute.attribute] = new_column

    else:
        warnings.warn(("No field column found in particles file or no "
                      "points in particles file therefore column not "
                      "appended"), 
                      UserWarning)
    
    return particle_data