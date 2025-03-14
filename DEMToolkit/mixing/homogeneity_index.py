import numpy as np
import warnings
import pandas as pd

def homogeneity_index(particle_data, attribute_column, 
                      sample_column, homogeniety_column=None, verbose=False):
    
    """ Calculate the homogeneity index of the particle data.
    """

    if homogeniety_column is None:
        homogeniety_column = attribute_column + "_homogeneity"

    # Get the unique mesh ids
    mesh = particle_data[sample_column]
    mesh_ids = np.unique(mesh)
    mesh_ids = mesh_ids[~np.isnan(mesh_ids)].astype(int)

    mesh_data = []
    particle_homogeneity_data = np.empty(particle_data.n_points)
    particle_homogeneity_data[:] = np.nan
    for mesh_id in mesh_ids:
        mesh_boolean_mask = mesh == mesh_id
        if not mesh_boolean_mask.any():
            continue

        mean = np.mean(
             particle_data[attribute_column][mesh_boolean_mask.astype(bool)])
        
        particle_homogeneity_data[mesh_boolean_mask] = mean
        mesh_data.append([mesh_id, mean])

    particle_data[homogeniety_column] = particle_homogeneity_data
    mesh_homogeneity_df = pd.DataFrame(mesh_data, columns=["mesh id", "mean"])

    return particle_data, mesh_homogeneity_df


        
        
   


