import pyvista as pv
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import re
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DEMToolbox.meshing import mesh_particles_2d
from DEMToolbox.velocity import velocity_vector_field

# Mesh 2D
vector_1 = [0, 1, 0]
vector_2 = [0, 0, 1]
resolution = [5, 5]
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

    particle_data, data = velocity_vector_field(particle_data, cylinder_data, 
                                                point, vector_1, vector_2, 
                                                resolution, plane_thickness)
    
    save_vtk = os.path.join(
        vtk_dir,("velocity_" + os.path.basename(particle_file)))
    
    particle_data.save(save_vtk)

    velocity_vectors = data[3]
    velocity_magnitude = data[4]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300) # these are the default values



    X = np.linspace(-0.03, 0.03, 2 * np.shape(velocity_vectors)[1] + 1)[1:-1:2]
    Y = np.linspace(0, 0.08, 2 * np.shape(velocity_vectors)[0] + 1)[1:-1:2]
    U = velocity_vectors[:,:,0]
    V = velocity_vectors[:,:,1]

    q = ax.quiver(X, Y, U, V, units='width', pivot='mid', angles='xy')
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,label='Quiver key, length = 10', labelpos='E')


    xmin = -0.03
    xmax = 0.03
    ymin = 0
    ymax = 0.08

    im = ax.imshow(np.flipud(velocity_magnitude), cmap='viridis', 
                   extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=0.5)

    # for i in range(np.shape(velocity_vectors)[0]):
    #     for j in range(np.shape(velocity_vectors)[1]):

    #         # make vectors unit vectors
    #         velocity_vectors[i, j] = (velocity_vectors[i, j] / 
    #                                             np.linalg.norm(velocity_vectors[i, j]))

    #         x_cell_size = (xmax - xmin) / np.shape(velocity_vectors)[1]
    #         y_cell_size = (ymax - ymin) / np.shape(velocity_vectors)[0]

    #         x0 = x_cell_size * (j + 0.5) + xmin 
    #         y0 = y_cell_size * (np.shape(velocity_vectors)[0] - i - 0.5) + ymin

    #         if x_cell_size > y_cell_size:
    #             x_end = velocity_vectors[i, j][0] * y_cell_size * 0.5
    #             y_end = velocity_vectors[i, j][1] * y_cell_size * 0.5
    #             arrow_width = y_cell_size * 0.05
    #         else:
    #             x_end = velocity_vectors[i, j][0] * x_cell_size * 0.5
    #             y_end = velocity_vectors[i, j][1] * x_cell_size * 0.5
    #             arrow_width = y_cell_size * 0.05

    #         ax.arrow(x0, y0, x_end, y_end, width=arrow_width, length_includes_head=True,
    #                 head_width=arrow_width*4, head_length=arrow_width*4, fc='k', ec='white')
    
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

    fig.colorbar(im, orientation='vertical', label='Velocity Magnitude (m/s)')

    save_fig_file = os.path.join(save_fig_dir, f"velocity_{file_name_id}.png")
    plt.savefig(save_fig_file)
    fig.clf()
