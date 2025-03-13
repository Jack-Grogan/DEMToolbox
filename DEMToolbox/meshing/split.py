import numpy as np
import warnings

from ..classes.split_class import Split

def split_particles(particle_data, split_dimension):
    """Split the particles into two classes based on a dimension."

    Split the particles into two classes along a dimension. The 
    dimension can be x, y, z, r or radius. The particles are split into
    two classes 0 and 1 with an equal number of particles in each class.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    split_dimension : str
        The dimension to split the particles along.

    Returns
    -------
    split : Split
        A split object containing the split dimension and an array
        of particle ids and class.

    Raises
    ------
    ValueError
        If split_dimension is not x, y, z, r or radius.
    UserWarning
        If the particle data has no points return an empty array.
    """
    if particle_data.n_points == 0:
        warnings.warn("cannot split empty particles file", UserWarning)
        split = Split(np.asarray([[None, None]]), split_dimension)
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

    split_array = np.asarray([[i,j] for i, j in zip(particle_data["id"], 
                                                    split_class)])
    
    split = Split(split_array, split_dimension)

    return split