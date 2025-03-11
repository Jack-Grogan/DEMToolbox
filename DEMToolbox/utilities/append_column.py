import numpy as np
import warnings

def append_on_id(particle_data, id_array, column_name):
    """Append a column to the particle data based on particle id"

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    id_array : list
        A list of tuples containing the particle id and the value to append.
    column_name : str
        The name of the column to append to the particle data.

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the column appended.

    Raises
    ------
    UserWarning
        If the particle data has no points return unedited particle data
        and raise a warning.
    """
    if ('id' in particle_data.point_data.keys() 
        and particle_data.n_points != 0):

        new_column = np.zeros(len(particle_data["id"]))
        new_column[:] = np.nan

        # Loop through the id array and assign the new column values
        for (particle_id, value) in id_array:
            new_column[particle_data["id"] == particle_id] = value

        particle_data[column_name] = new_column
    else:

        # If the particles file does not have an id column or has no points
        # raise a warning and return the object
        warnings.warn("No id column found in particles file or no points in "
                      "particles file therefore column not appended", 
                      UserWarning)

    return particle_data