import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir)))

from DEMToolbox.particle_sampling import sample_1d_volume_cylinder
from DEMToolbox.classes.particle_attribute import ParticleAttribute
from DEMToolbox.utilities import append_attribute

from natsort import natsorted
import glob
import re
import pyvista as pv
from tqdm import tqdm
import numpy as np

# Sample parameters
split_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Split along x, y, z axes
split_points = [[0, 0, 0], [0, 0.005, 0], [0, 0, 0]]  # Origin for each split
split_columns = ["cylinder_x_class", "cylinder_y_class", "cylinder_z_class"]
split_resolution = [5, 4, 3]  # Resolution for each split

# Simulation parameters
dumpstep = 0.1
settled_time = 2

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "equal_volume_sampling_cylinder_analysis")
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
iterator = zip(split_vectors, split_points, split_resolution, split_columns)
for vector, point, resolution, column in iterator:
    settled_file, split = sample_1d_volume_cylinder(
        settled_file, 
        point,
        vector,
        resolution=resolution, 
        append_column=column
    )
    
    splits.append(split.ParticleAttribute)
    
for particle_file in tqdm(files):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)

    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)

    # Split the particles
    for split in splits:
        particle_data = append_attribute(particle_data, split)

    # Save the particle data
    vtk_file = os.path.join(vtk_dir,
                            ("sampled_particles_" + file_name_id + ".vtk"))
    particle_data.save(vtk_file)