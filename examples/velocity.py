import pyvista as pv
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import re
import os
from DEMToolbox.velocity import velocity_vector_field 

# Mesh 2D
vector_1 = [0, 1, 0]
vector_2 = [0, 0, 1]
resolution = [15, 15]
point = [0, 0, 0]
plane_thickness = 0.006

# Cylinder dimesions
xmin = -0.03
xmax = 0.03
ymin = 0
ymax = 0.08

# Save directories
file_path = os.path.dirname(__file__)
save_dir = os.path.join(file_path, "velocity_analysis")
vtk_dir = os.path.join(save_dir, "post")
save_fig_dir = os.path.join(save_dir, "figures")
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(vtk_dir):
    os.makedirs(vtk_dir)

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

    particle_data, velocity = velocity_vector_field(particle_data, 
                                                    cylinder_data, 
                                                    point, 
                                                    vector_1, 
                                                    vector_2, 
                                                    plane_thickness,
                                                    resolution, 
                                                    )
    
    mags = velocity[0]
    vecs = velocity[1]
    
    save_vtk = os.path.join(
        vtk_dir,("velocity_" + os.path.basename(particle_file)))
    particle_data.save(save_vtk)

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=300)

    # cylinder top plate
    plt.plot([xmin, xmax], [ymax, ymax], 'k', lw=2)

    # cylinder bottom plate
    plt.plot([xmin, xmax], [ymin, ymin], 'k', lw=2)

    # cylinder left plate
    plt.plot([xmin, xmin], [ymin, ymax], 'k', lw=2)

    # cylinder right plate
    plt.plot([xmax, xmax], [ymin, ymax], 'k', lw=2)

    ax.set_xlabel("Dimension 1 (m)")
    ax.set_ylabel("Dimension 2 (m)")
    ax.set_ylim(-0.01, 0.09)
    ax.set_xlim(-0.035, 0.035)
    ax.set_title(f"Velocity Vector Field")
    ax.set_aspect('equal')

    if not np.isnan(mags).all(): 

        X = np.linspace(-0.03, 0.03, 2 * np.shape(vecs)[1] + 1)[1:-1:2]
        Y = np.linspace(0, 0.08, 2 * np.shape(vecs)[0] + 1)[1:-1:2]
        U = vecs[:,:,0]
        V = vecs[:,:,1]

        q = ax.quiver(X, Y, U, V, units='width', pivot='mid', angles='xy')

        xmin = -0.03
        xmax = 0.03
        ymin = 0
        ymax = 0.08

    im = ax.imshow(np.flipud(mags), cmap='viridis', 
                   extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=0.5)


    fig.colorbar(im, orientation='vertical', label='Velocity Magnitude (m/s)')

    save_fig_file = os.path.join(save_fig_dir, f"velocity_{file_name_id}.png")
    plt.savefig(save_fig_file)
    plt.close(fig)