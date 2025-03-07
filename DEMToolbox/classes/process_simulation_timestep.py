import pyvista as pv
import numpy as np
import warnings

import DEMToolbox

class ProcessSimulationTimestep():

    def __init__(self, particles_file, cylinder_file):

        self.particles_file = pv.read(particles_file)
        self.cylinder_file = pv.read(cylinder_file)
    
    def append_particle_column(self, id_array, column_name):

        # Check if the particles file has an id column and if it has points
        # If it does, append the new column to the particles file
        if 'id' in self.particles_file.point_data.keys() and self.particles_file.n_points != 0:

            new_column = np.zeros(len(self.particles_file["id"]))
            new_column[:] = np.nan

            # Loop through the id array and assign the new column values
            for (particle_id, value) in id_array:
                new_column[self.particles_file["id"] == particle_id] = value

            self.particles_file[column_name] = new_column

        else:

            # If the particles file does not have an id column or has no points
            # raise a warning and return the object
            warnings.warn("No id column found in particles file or no points in particles file"
                            " therefore column not appended")

        return self.particles_file
    
# --------------------------------------------------------------------------------------------------
#  Meshing functions
# --------------------------------------------------------------------------------------------------

    def mesh_particles_1d(self, mesh_vec, mesh_resolution_1d):

        mesh_results = DEMToolbox.meshing.mesh_particles_1d(self.particles_file, self.cylinder_file, 
                                                        mesh_vec, mesh_resolution_1d)
        
        self.particles_file = mesh_results[0]
        self.cylinder_file = mesh_results[1]
        id_bounds = mesh_results[2]
        mesh_column = mesh_results[3]
        in_mesh_particles = mesh_results[4]
        out_of_mesh_particles = mesh_results[5]

        return id_bounds, mesh_column, in_mesh_particles, out_of_mesh_particles


    def mesh_particles_2d(self, mesh_vec_x, mesh_vec_y, mesh_resolution_2d):

        mesh_results = DEMToolbox.meshing.mesh_particles_2d(self.particles_file, self.cylinder_file,
                                                        mesh_vec_x, mesh_vec_y, mesh_resolution_2d)
        
        self.particles_file = mesh_results[0]
        self.cylinder_file = mesh_results[1]
        id_bounds = mesh_results[2]
        mesh_column = mesh_results[3]
        in_mesh_particles = mesh_results[4]
        out_of_mesh_particles = mesh_results[5]

        return id_bounds, mesh_column, in_mesh_particles, out_of_mesh_particles

    def mesh_particles_3d(self, mesh_vec_x, mesh_vec_y, mesh_vec_z, mesh_resolution_3d):

        mesh_results = DEMToolbox.meshing.mesh_particles_3d(self.particles_file, self.cylinder_file,
                                                        mesh_vec_x, mesh_vec_y, mesh_vec_z, 
                                                        mesh_resolution_3d)
        
        self.particles_file = mesh_results[0]
        self.cylinder_file = mesh_results[1]
        id_bounds = mesh_results[2]
        mesh_column = mesh_results[3]
        in_mesh_particles = mesh_results[4]
        out_of_mesh_particles = mesh_results[5]

        return id_bounds, mesh_column, in_mesh_particles, out_of_mesh_particles
    

    def mesh_particles_3d_cylinder(self, mesh_resolution_3d, 
                                   mesh_constant="volume", start_rotation=0):
        
        mesh_results = DEMToolbox.meshing.mesh_particles_3d_cylinder(self.particles_file, 
                                                                 self.cylinder_file,
                                                                 mesh_resolution_3d, 
                                                                 mesh_constant, 
                                                                 start_rotation)
                
        self.particles_file = mesh_results[0]
        self.cylinder_file = mesh_results[1]
        id_bounds = mesh_results[2]
        mesh_column = mesh_results[3]
        in_mesh_particles = mesh_results[4]
        out_of_mesh_particles = mesh_results[5]

        return id_bounds, mesh_column, in_mesh_particles, out_of_mesh_particles

# --------------------------------------------------------------------------------------------------
#  Mixing Functions
# --------------------------------------------------------------------------------------------------

    def macro_scale_lacey_mixing(self, split_column, mesh_column, verbose=False):

        self.particles_file, lacey = DEMToolbox.mixing.macro_scale_lacey_mixing(self.particles_file, 
                                                                            split_column, 
                                                                            mesh_column, 
                                                                            verbose)
        return lacey
    
    def mean_mesh_element_radii(self, mesh_method, 
                                mesh_vec_1=None, mesh_vec_2=None, mesh_vec_3=None,
                                mesh_resolution=None, mesh_constant=None, start_rotation=None):
        
        results = DEMToolbox.mean_mesh_element_radii(self.particles_file, self.cylinder_file, 
                                                 mesh_method,mesh_vec_1, mesh_vec_2, mesh_vec_3,
                                                 mesh_resolution, mesh_constant, start_rotation)
        
        self.particles_file = results[0]
        self.cylinder_file = results[1]
        mean_particle_radii = results[2]

        return mean_particle_radii

# --------------------------------------------------------------------------------------------------
#  Velocity Vector Field Functions
# --------------------------------------------------------------------------------------------------

    def particle_slice(self, point, normal, plane_thickness):

        self.particles_file, slice_column = DEMToolbox.mixing.particle_slice(self.particles_file, 
                                                                  point, normal, plane_thickness)
        
        return slice_column


    def velocity_vector_field(self, point, mesh_vec_x, mesh_vec_y, 
                              mesh_resolution_2d, plane_thickness):
        
        results  = DEMToolbox.velocity.velocity_vector_field(self.particles_file, self.cylinder_file, 
                                                         point, mesh_vec_x, mesh_vec_y, 
                                                         mesh_resolution_2d, plane_thickness)
        
        self.particles_file = results[0]
        velocity_vectors = results[1]
        velocity_mag = results[2]

        return velocity_vectors, velocity_mag 
        

    def save_particles(self, save_file):
        self.particles_file.save(save_file)
        return
    
# --------------------------------------------------------------------------------------------------
#  Splitting Functions
# --------------------------------------------------------------------------------------------------

def split_particles(settled_file, split_dimension):

    settled_data = pv.read(settled_file)
    split_column = f"{split_dimension}_class"

    if split_dimension == "x":
        split_class  = np.asarray(
            settled_data.points[:, 0] >= np.median(settled_data.points[:, 0])).astype(int)

    elif split_dimension == "y":
        split_class  = np.asarray(
            settled_data.points[:, 1] >= np.median(settled_data.points[:, 1])).astype(int)

    elif split_dimension == "z":
        split_class  = np.asarray(
            settled_data.points[:, 2] >= np.median(settled_data.points[:, 2])).astype(int)

    elif split_dimension == "r":
        median_r2 = np.median(settled_data.points[:, 0]**2 + settled_data.points[:, 1]**2)
        settled_r2 = settled_data.points[:, 0]**2 + settled_data.points[:, 1]**2
        split_class  = np.asarray(settled_r2 >= median_r2).astype(int)

    elif split_dimension == "radius":

        radii = np.unique(settled_data["radius"])
        split_class = np.zeros(settled_data.n_points)
        split_class[:] = np.nan
        for i, radius in enumerate(radii):
            boolean_mask = [settled_data["radius"] == radius]
            split_class[boolean_mask] = i

    else:
        raise ValueError(f"{split_dimension} is not a recognised split dimension")

    split_array = np.asarray([[i,j] for i, j in zip(settled_data["id"], split_class)])

    return split_array, split_column