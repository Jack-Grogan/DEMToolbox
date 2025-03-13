import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DEMToolbox.meshing import split_particles
from DEMToolbox.utilities import append_on_id
from DEMToolbox.meshing import mesh_particles_3d_cylinder
from DEMToolbox.mixing import macro_scale_lacey_mixing

from natsort import natsorted
import glob
import re
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

# Mesh parameters
cylinder_prefix = "mesh_"
split_dimensions = ["x", "y", "z", "r"]
mesh_resolution = [8,6,20]
mesh_constant = "volume"

# Simulation parameters
timestep = 1e-5
dumpstep = 0.1
settled_time = 2

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "mixing_analysis")
vtk_dir = os.path.join(save_dir, "post")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

# Get the particle files
glob_input = os.path.join(file_path, "post", "particles_*")
files = natsorted([f for f in glob.glob(glob_input) if "boundingBox" not in f])

# Split the particles at the settled time
settled_file = pv.read(files[round(settled_time/dumpstep)])
split_arrays = []
split_columns = []
for split_dimension in split_dimensions:
    split = split_particles(settled_file, split_dimension)
    split_arrays.append(split.split_array)
    split_columns.append(split.split_dimension + "_split")

x_split_array, y_split_array, z_split_array, r_split_array = split_arrays
x_split_column, y_split_column, z_split_column, r_split_column = split_columns

results = []
for particle_file in tqdm(files):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)
    cylinder_name = "mesh_" + file_name_id + '.vtk'
    time = float(file_name_id) * timestep
    cylinder_file = os.path.join(os.path.dirname(particle_file), cylinder_name)
    
    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)
    cylinder_data = pv.read(cylinder_file)

    # Mesh the particles
    particle_data, mesh = mesh_particles_3d_cylinder(particle_data, 
                                                            cylinder_data, 
                                                            mesh_resolution, 
                                                            mesh_constant)

    # Append the split columns to the particle data
    splits_n_class_0 = []
    splits_n_class_1 = []
    for split_array, split_column in zip(split_arrays, split_columns):
        particle_data = append_on_id(particle_data, split_array, split_column)
        if split_column in particle_data.point_data.keys():
            splits_n_class_0.append(np.sum(particle_data[split_column] == 0))
            splits_n_class_1.append(np.sum(particle_data[split_column] == 1))
        else:
            splits_n_class_0.append(np.nan)
            splits_n_class_1.append(np.nan)

    x_n_class_0, x_n_class_1 = splits_n_class_0[0], splits_n_class_1[0]
    y_n_class_0, y_n_class_1 = splits_n_class_0[1], splits_n_class_1[1]
    z_n_class_0, z_n_class_1 = splits_n_class_0[2], splits_n_class_1[2]
    r_n_class_0, r_n_class_1 = splits_n_class_0[3], splits_n_class_1[3]

    # Calculate the Lacey mixing index for each split dimension
    x_lacey_output = macro_scale_lacey_mixing(particle_data, x_split_column, 
                                              mesh.name, verbose=True)
    particle_data, _, x_lacey = x_lacey_output

    y_lacey_output = macro_scale_lacey_mixing(particle_data, y_split_column, 
                                              mesh.name, verbose=True)
    particle_data, _, y_lacey = y_lacey_output

    z_lacey_output = macro_scale_lacey_mixing(particle_data, z_split_column, 
                                              mesh.name, verbose=True)
    particle_data, _, z_lacey = z_lacey_output

    r_lacey_output = macro_scale_lacey_mixing(particle_data, r_split_column, 
                                              mesh.name, verbose=True)
    particle_data, _, r_lacey = r_lacey_output

    # Save the particle data
    vtk_file = os.path.join(vtk_dir,
                            ("lacey_particles_" + file_name_id + ".vtk"))
    particle_data.save(vtk_file)

    results.append([time, x_lacey, y_lacey, z_lacey, r_lacey,
                    mesh.n_meshed_particles, mesh.n_unmeshed_particles,
                    x_n_class_0, x_n_class_1, y_n_class_0, y_n_class_1,
                    z_n_class_0, z_n_class_1, r_n_class_0, r_n_class_1])

# Save the results to a csv file in the save directory
results_df = pd.DataFrame(results,
                          columns=["time", "x_lacey", "y_lacey", "z_lacey",
                                   "r_lacey", "n_meshed_particles",
                                   "out_of_mesh_particles_3d",
                                   "n_class_0_x", "n_class_1_x",
                                   "n_class_0_y", "n_class_1_y",
                                   "n_class_0_z", "n_class_1_z",
                                   "n_class_0_r", "n_class_1_r"])

results_df.to_csv(os.path.join(save_dir, "lacey_results.csv"), index=False)
