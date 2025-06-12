import pyvista as pv
import glob
from natsort import natsorted
from tqdm import tqdm
import re
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir)))

from DEMToolbox.particle_sampling import sample_1d
from DEMToolbox.mixing import homogeneity_index

# Mesh 1D
vector_1d = [0, 0, 1]
resolution_1d = 50
mesh_column_1d = "1D_mesh"

# Simulation parameters
timestep = 1e-5
dumpstep = 0.1
settled_time = 2

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "segregation_analysis")
vtk_dir = os.path.join(save_dir, "post")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

# Get the particle files
glob_input = os.path.join(file_path, "post", "particles_*")
files = natsorted([f for f in glob.glob(glob_input) if "boundingBox" not in f])

results = []
for i, particle_file in enumerate(tqdm(files)):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)
    cylinder_name = "mesh_" + file_name_id + '.vtk'
    time = float(file_name_id) * timestep
    cylinder_file = os.path.join(os.path.dirname(particle_file), cylinder_name)

    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)
    cylinder_data = pv.read(cylinder_file)

    # Mesh the particles
    particle_data, samples = sample_1d(particle_data, 
                                       cylinder_data, 
                                       vector_1d, 
                                       resolution_1d, 
                                       mesh_column_1d)
        
    particle_data, hom = homogeneity_index(particle_data,
                                           'radius',
                                           samples, 
                                           verbose=True)
    
    # Save the particle data
    save_vtk = os.path.join(vtk_dir, os.path.basename(particle_file))
    particle_data.save(
        os.path.join(vtk_dir, 
                     ("mean_radii_" + os.path.basename(particle_file)))
        )
    
    results.append([time, hom])

results_df = pd.DataFrame(results, columns=["time", "homogeneity"])
results_df.to_csv(os.path.join(save_dir, "segregation_results.csv"), index=False)