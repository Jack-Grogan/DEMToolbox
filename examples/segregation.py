import pyvista as pv
import glob
from natsort import natsorted
from tqdm import tqdm
import re
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DEMToolbox.particle_sampling import mesh_particles_1d
from DEMToolbox.mixing import mean_mesh_element_radii

# Mesh 1D
vector_1d = [0, 0, 1]
resolution_1d = 50
mesh_column_1d = "1D_mesh"

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "segregation_analysis")
vtk_dir = os.path.join(save_dir, "post")
csv_dir = os.path.join(save_dir, "csvs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Get the particle files
glob_input = os.path.join(file_path, "post", "particles_*")
files = natsorted([f for f in glob.glob(glob_input) if "boundingBox" not in f])

for i, particle_file in enumerate(tqdm(files)):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)
    cylinder_name = "mesh_" + file_name_id + '.vtk'
    cylinder_file = os.path.join(os.path.dirname(particle_file), cylinder_name)
    
    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)
    cylinder_data = pv.read(cylinder_file)

    # Mesh the particles
    particle_data, mesh = mesh_particles_1d(particle_data, 
                                                   cylinder_data, 
                                                   vector_1d, 
                                                   resolution_1d, 
                                                   mesh_column_1d)
    
    particle_data, mesh_mean_radii_df = mean_mesh_element_radii(particle_data, 
                                                                mesh.name, 
                                                                mesh.mesh_df)
    
    save_csv = os.path.join(csv_dir, f"mesh_{file_name_id}.csv")
    mesh_mean_radii_df.to_csv(save_csv, index=False)

    save_vtk = os.path.join(vtk_dir, os.path.basename(particle_file))
    particle_data.save(
        os.path.join(vtk_dir, 
                     ("mean_radii_" + os.path.basename(particle_file)))
        )