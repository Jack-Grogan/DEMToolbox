import numpy as np
import warnings

from ..classes.particle_attribute import ParticleAttribute

def split_particles(particle_data, split_dimension, 
                    field_column="id", attribute_column=None):
    """Split the particles into two classes based on a dimension.

    Split the particles into two classes along a dimension. The 
    dimension can be x, y, z, r or radius. The particles are split into
    two classes 0 and 1 with an equal number of particles in each class.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    split_dimension : str
        The dimension to split the particles along. Can be x, y, z, r or
        radius.
    field_column : str, optional
        The name of the field column in the particle data, by default "id".
    attribute_column : str, optional
        The name of the attribute column in the particle data, by default
        None. If None, the column name will be "{split_dimension}_class".

    Returns
    -------
    particle_data : vtkPolyData
        The particle vtk with the attribute column added.
    split : ParticleAttribute
        A particle attribute object containing the field column, the 
        attribute column and the split array.

    Raises
    ------
    ValueError
        If split_dimension is not a recognised split dimension.
    UserWarning
        If the particle data has no points return nan for split array.
    """
    if particle_data.n_points == 0:
        warnings.warn("Cannot split empty particles file", UserWarning)
        split = ParticleAttribute(field_column, 
                                  attribute_column, 
                                  [[None, None]])
        return split

    if split_dimension == "x":
        split_class  = np.asarray(particle_data.points[:, 0]
                                  >= np.median(particle_data.points[:, 0])
                                  ).astype(int)
    elif split_dimension == "y":
        split_class  = np.asarray(particle_data.points[:, 1] 
                                  >= np.median(particle_data.points[:, 1])
                                  ).astype(int)
    elif split_dimension == "z":
        split_class  = np.asarray(particle_data.points[:, 2] 
                                  >= np.median(particle_data.points[:, 2])
                                  ).astype(int)
    elif split_dimension == "r":
        median_r2 = np.median(particle_data.points[:, 0] ** 2 
                              + particle_data.points[:, 1] ** 2)
        settled_r2 = (particle_data.points[:, 0] ** 2 
                      + particle_data.points[:, 1] ** 2)
        split_class  = np.asarray(settled_r2 >= median_r2).astype(int)

    elif split_dimension == "radius":
        radii = np.unique(particle_data["radius"])
        split_class = np.zeros(particle_data.n_points)
        split_class[:] = np.nan
        for i, radius in enumerate(radii):
            boolean_mask = [particle_data["radius"] == radius]
            split_class[boolean_mask] = i

    else:
        raise ValueError(
            f"{split_dimension} is not a recognised split dimension")
    
    if attribute_column is None:
        attribute_column = split_dimension + "_class"

    particle_data[attribute_column] = split_class

    split_array = np.asarray([[i,j] for i, j in 
                              zip(particle_data[field_column], 
                              split_class)])
    
    split = ParticleAttribute(field_column, attribute_column, split_array)

    return particle_data, split