import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DEMToolbox.meshing import mesh_particles_1d 
from DEMToolbox.meshing import mesh_particles_2d
from DEMToolbox.meshing import mesh_particles_3d
from DEMToolbox.meshing import mesh_particles_3d_cylinder

from natsort import natsorted
import glob
import re
import pyvista as pv
from tqdm import tqdm
import pandas as pd

# Mesh 1D
vector_1d = [0, 0, 1]
resolution_1d = 10
mesh_column_1d = "1D_mesh"

# Mesh 2D
vector_1_2d = [0, 1, 0]
vector_2_2d = [0, 0, 1]
resolution_2d = [5, 10]
mesh_column_2d = "2D_mesh"

# Mesh 3D
vector_1_3d = [1, 0, 0]
vector_2_3d = [0, 1, 0]
vector_3_3d = [0, 0, 1]
resolution_3d = [5, 10, 15]
mesh_column_3d = "3D_mesh"

# Mesh 3D Cylinder
resolution_3d_cylinder = [5, 10, 15]
mesh_constant = "volume"
rotation = 0
mesh_column_3d_cylinder = "3D_cylinder_mesh"

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "meshing_analysis")
vtk_dir = os.path.join(save_dir, "post")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

# Get the particle files
glob_input = os.path.join(file_path, "post", "particles_*")
files = natsorted([f for f in glob.glob(glob_input) if "boundingBox" not in f])

in_out_mesh_1d = []
in_out_mesh_2d = []
in_out_mesh_3d = []
in_out_mesh_3d_cylinder = []
for particle_file in tqdm(files):

    # Retrieve corresponding cylinder file for each particle file
    file_name_id = re.search(r'particles_(.*).vtk', particle_file).group(1)
    cylinder_name = "mesh_" + file_name_id + '.vtk'
    cylinder_file = os.path.join(os.path.dirname(particle_file), cylinder_name)
    
    # Read the particle and cylinder files
    particle_data = pv.read(particle_file)
    cylinder_data = pv.read(cylinder_file)

    # Mesh the particles 1D
    particle_data, output_1d = mesh_particles_1d(particle_data, cylinder_data,
                                                 vector_1d, resolution_1d, 
                                                 mesh_column_1d)
    mesh_df_1d = output_1d[1]
    in_mesh_particles_1d = output_1d[2]
    out_of_mesh_particles_1d = output_1d[3]
    in_out_mesh_1d.append([in_mesh_particles_1d, out_of_mesh_particles_1d])

    # Mesh the particles 2D
    particle_data, output_2d = mesh_particles_2d(particle_data, cylinder_data, 
                                                 vector_1_2d, vector_2_2d, 
                                                 resolution_2d, mesh_column_2d)
    mesh_df_2d = output_2d[1]
    in_mesh_particles_2d = output_2d[2]
    out_of_mesh_particles_2d = output_2d[3]
    in_out_mesh_2d.append([in_mesh_particles_2d, out_of_mesh_particles_2d])

    # Mesh the particles 3D
    particle_data, output_3d = mesh_particles_3d(particle_data, cylinder_data, 
                                                 vector_1_3d, vector_2_3d, 
                                                 vector_3_3d, resolution_3d,
                                                 mesh_column_3d)
    mesh_df_3d = output_3d[1]
    in_mesh_particles_3d = output_3d[2]
    out_of_mesh_particles_3d = output_3d[3]
    in_out_mesh_3d.append([in_mesh_particles_3d, out_of_mesh_particles_3d])

    # Mesh the particles 3D Cylinder
    particle_data, output_3d_cylinder = mesh_particles_3d_cylinder(
                                                    particle_data, 
                                                    cylinder_data, 
                                                    resolution_3d_cylinder, 
                                                    mesh_constant, rotation, 
                                                    mesh_column_3d_cylinder)
    mesh_df_3d_cylinder = output_3d_cylinder[1]
    in_mesh_particles_3d_cylinder = output_3d_cylinder[2]
    out_of_mesh_particles_3d_cylinder = output_3d_cylinder[3]
    in_out_mesh_3d_cylinder.append([in_mesh_particles_3d_cylinder,
                                    out_of_mesh_particles_3d_cylinder])
    
    vtk_file = os.path.join(vtk_dir,
                            ("meshed_particles_" + file_name_id + '.vtk'))
    particle_data.save(vtk_file)

# Save the mesh dataframes
output_1d_df = pd.DataFrame(in_out_mesh_1d, columns=["in_mesh_particles",
                                                    "out_of_mesh_particles"])
output_2d_df = pd.DataFrame(in_out_mesh_2d, columns=["in_mesh_particles",
                                                    "out_of_mesh_particles"])
output_3d_df = pd.DataFrame(in_out_mesh_3d, columns=["in_mesh_particles",
                                                    "out_of_mesh_particles"])
output_3d_cylinder_df = pd.DataFrame(in_out_mesh_3d_cylinder,
                                    columns=["in_mesh_particles",
                                            "out_of_mesh_particles"])

# Save the dataframes of Lacey mixing indices 
output_1d_df.to_csv(os.path.join(save_dir, "in_out_mesh_1d.csv"), index=False)
output_2d_df.to_csv(os.path.join(save_dir, "in_out_mesh_2d.csv"), index=False)
output_3d_df.to_csv(os.path.join(save_dir, "in_out_mesh_3d.csv"), index=False)
output_3d_cylinder_df.to_csv(os.path.join(save_dir, 
                                          "in_out_mesh_3d_cylinder.csv"), 
                                          index=False)

# Save dataframes of mesh data at the end of the simulation
mesh_df_1d.to_csv(os.path.join(save_dir, "final_mesh_df_1d.csv"), index=False)
mesh_df_2d.to_csv(os.path.join(save_dir, "final_mesh_df_2d.csv"), index=False)
mesh_df_3d.to_csv(os.path.join(save_dir, "final_mesh_df_3d.csv"), index=False)
mesh_df_3d_cylinder.to_csv(os.path.join(save_dir, 
                                        "final_mesh_df_3d_cylinder.csv"), 
                                        index=False)