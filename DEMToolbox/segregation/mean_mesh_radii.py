import numpy as np
import pandas as pd
import warnings

def mean_mesh_element_radii(particle_data, mesh_column, mesh_df,
                            mean_radii_column="mean_radii"):
    """Calculate the mean particle radii for each mesh element.

    Calculate the mean particle radii for each mesh element in the
    particle data.

    Parameters
    ----------
    particle_data : vtkPolyData
        The particle vtk.
    mesh_column : str
        The name of the mesh column in the particle data.
    mesh_df : pd.DataFrame
        The mesh dataframe.
    mean_radii_column : str, optional
        The name of the mean radii column in the particle data,
        by default "mean_radii".

    Returns
    -------
    particle_data : vtkPolyData
        The particle data with the mean radii column added.
    mesh_mean_radii_df : pd.DataFrame
        A dataframe containing the mesh id and the mean particle radii.
    """
    if particle_data.n_points == 0:
        warnings.warn("no particles in particle_data vtk", UserWarning)
        return particle_data, pd.DataFrame(columns=["mesh id",
                                                    "mean particle radii"])
    
    if mesh_df.empty:
        warnings.warn("no mesh elements in mesh_df", UserWarning)
        return particle_data, pd.DataFrame(columns=["mesh id",
                                                    "mean particle radii"])
    
    if mesh_column not in particle_data.point_data.keys():
        warnings.warn("mesh column not in particle_data vtk", UserWarning)
        return particle_data, pd.DataFrame(columns=["mesh id",
                                            "mean particle radii"])

    mean_radii_list = []
    mesh_mean_radii = np.empty(particle_data.n_points)
    mesh_mean_radii[:] = np.nan

    mesh_ids = mesh_df["mesh id"].values
    for mesh_id in mesh_ids: 
        mesh_boolean_mask = particle_data[mesh_column] == mesh_id
        if not mesh_boolean_mask.any():
            continue

        mesh_radii = particle_data["radius"][mesh_boolean_mask.astype(bool)]
        mean_radii_list.append([mesh_id, np.mean(mesh_radii)])
        mesh_mean_radii[mesh_boolean_mask.astype(bool)] = np.mean(mesh_radii)

    particle_data[mean_radii_column] = mesh_mean_radii

    mesh_mean_radii_df = pd.DataFrame(mean_radii_list, 
                                      columns=["mesh id",
                                               "mean particle radii"])

    mesh_mean_radii_df = mesh_df.merge(mesh_mean_radii_df, on="mesh id")

    return particle_data, mesh_mean_radii_df