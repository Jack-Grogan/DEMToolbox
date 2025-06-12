import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir)))

from DEMToolbox.particle_sampling import sample_1d 
from DEMToolbox.particle_sampling import sample_2d
from DEMToolbox.particle_sampling import sample_2d_slice
from DEMToolbox.particle_sampling import sample_3d
from DEMToolbox.particle_sampling import sample_3d_cylinder

from natsort import natsorted
import glob
import re
import pyvista as pv
from tqdm import tqdm
import pandas as pd

# Mesh 1D
vector_1d = [0, 0, 1]
resolution_1d = 10
mesh_column_1d = "1D_sample"

# Mesh 2D
vector_1_2d = [0, 1, 0]
vector_2_2d = [0, 0, 1]
resolution_2d = [5, 10]
mesh_column_2d = "2D_sample"

# Mesh 2d Slice
point = [0, 0, 0]
vector_1_2d_slice = [0, 1, 0]
vector_2_2d_slice = [0, 0, 1]
plane_thickness = 0.006
resolution_2d_slice = [15, 15]
mesh_column_2d_slice = "2D_slice_sample"

# Mesh 3D
vector_1_3d = [1, 0, 0]
vector_2_3d = [0, 1, 0]
vector_3_3d = [0, 0, 1]
resolution_3d = [5, 10, 15]
mesh_column_3d = "3D_sample"

# Mesh 3D Cylinder
resolution_3d_cylinder = [5, 10, 15]
mesh_constant = "volume"
rotation = 0
mesh_column_3d_cylinder = "3D_cylinder_sample"

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "sample_analysis")
vtk_dir = os.path.join(save_dir, "post")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

# Get the particle files
glob_input = os.path.join(file_path, "post", "particles_*")
files = natsorted([f for f in glob.glob(glob_input) if "boundingBox" not in f])

samples_1d = []
samples_2d = []
samples_2d_slice = []
samples_3d = []
samples_3d_cylinder = []
for particle_file in tqdm(files):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)
    cylinder_name = "mesh_" + file_name_id + '.vtk'
    cylinder_file = os.path.join(os.path.dirname(particle_file), cylinder_name)
    
    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)
    cylinder_data = pv.read(cylinder_file)

    # Mesh the particles 1D
    particle_data, mesh_1d = sample_1d(particle_data, cylinder_data, vector_1d, 
                                       resolution_1d, mesh_column_1d)

    samples_1d.append([mesh_1d.n_sampled_particles, 
                           mesh_1d.n_unsampled_particles])

    # Mesh the particles 2D
    particle_data, mesh_2d = sample_2d(particle_data, cylinder_data, 
                                               vector_1_2d, vector_2_2d, 
                                               resolution_2d, mesh_column_2d)
    
    samples_2d.append([mesh_2d.n_sampled_particles,
                           mesh_2d.n_unsampled_particles])
    
    # Mesh the particles 2D Slice
    particle_data, mesh_2d_slice = sample_2d_slice(particle_data,
                                                   cylinder_data, 
                                                   point, 
                                                   vector_1_2d_slice,
                                                   vector_2_2d_slice,
                                                   plane_thickness, 
                                                   resolution_2d_slice, 
                                                   append_column = mesh_column_2d_slice
                                                   )
    
    samples_2d_slice.append([mesh_2d_slice.n_sampled_particles,
                                 mesh_2d_slice.n_unsampled_particles])

    # Mesh the particles 3D
    particle_data, mesh_3d = sample_3d(particle_data, cylinder_data, 
                                               vector_1_3d, vector_2_3d, 
                                               vector_3_3d, resolution_3d,
                                               mesh_column_3d)
    
    samples_3d.append([mesh_3d.n_sampled_particles,
                           mesh_3d.n_unsampled_particles])
    
    # Mesh the particles 3D Cylinder
    particle_data, mesh_3d_cylinder = sample_3d_cylinder(
                                                    particle_data, 
                                                    cylinder_data, 
                                                    resolution_3d_cylinder, 
                                                    mesh_constant, rotation, 
                                                    mesh_column_3d_cylinder)

    samples_3d_cylinder.append([mesh_3d_cylinder.n_sampled_particles,
                                    mesh_3d_cylinder.n_unsampled_particles])
    
    vtk_file = os.path.join(vtk_dir,
                            ("meshed_particles_" + file_name_id + '.vtk'))
    particle_data.save(vtk_file)