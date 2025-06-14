import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir)))

from DEMToolbox.particle_sampling import sample_3d_cylinder
from DEMToolbox.particle_sampling import sample_1d_volume
from DEMToolbox.classes.particle_attribute import ParticleAttribute
from DEMToolbox.utilities import append_attribute
from DEMToolbox.mixing import macro_scale_lacey_mixing

from natsort import natsorted
import glob
import re
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import numpy as np

# Sample parameters
cylinder_prefix = "mesh_"
split_dimensions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Split along x, y, z axes
split_columns = ["x_class", "y_class", "z_class"]
sample_resolution = [8,6,20]
split_resolution = 2
sample_constant = "volume"

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
splits = []
for split_dimension, split_column in zip(split_dimensions, split_columns):
    settled_file, split = sample_1d_volume(settled_file, 
                                split_dimension,
                                n_samples=split_resolution, 
                                append_column=split_column)
    splits.append(split.attribute)
    
results = []
for particle_file in tqdm(files):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)
    cylinder_name = cylinder_prefix + file_name_id + '.vtk'
    time = float(file_name_id) * timestep
    cylinder_file = os.path.join(os.path.dirname(particle_file), cylinder_name)
    
    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)
    cylinder_data = pv.read(cylinder_file)

    particle_data, samples = sample_3d_cylinder(particle_data,
                                                cylinder_data,
                                                sample_resolution,
                                                sample_constant)

    # Split the particles
    split_lacey = []
    for split in splits:
        particle_data = append_attribute(particle_data, split)
        particle_data, lacey = macro_scale_lacey_mixing(particle_data, 
                                                        split, 
                                                        samples,
                                                        verbose=True,
                                                        )
        split_lacey.append(lacey)
        
    # Save the particle data
    vtk_file = os.path.join(vtk_dir,
                            ("lacey_particles_" + file_name_id + ".vtk"))
    particle_data.save(vtk_file)

    results.append([time, *split_lacey, samples.n_sampled_particles,
                    samples.n_unsampled_particles])

# Save the results to a csv file in the save directory
results_df = pd.DataFrame(results,
                          columns=["time", 
                                   *[i + "_lacey" for i in split_columns],
                                   "n_sampled_particles",
                                   "n_unsampled_particles"]
                          )

results_df.to_csv(os.path.join(save_dir, "lacey_results.csv"), index=False)