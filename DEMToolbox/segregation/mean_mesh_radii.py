import numpy as np
import warnings
from collections import namedtuple

from DEMToolbox import meshing

def mean_mesh_element_radii(particle_data, cylinder_data, mesh_method, 
                            mesh_vec_1=None, mesh_vec_2=None, mesh_vec_3=None,
                            mesh_resolution=None, mesh_constant=None, start_rotation=None):
    

    if mesh_method == "mesh_particles_1d":

        if mesh_vec_1 is None:
            raise ValueError("mesh_vec_1 must be provided for 1D meshing")
        
        if mesh_vec_2 is not None:
            raise ValueError("mesh_vec_2 must not be provided for 1D meshing")
        
        if mesh_vec_3 is not None:
            raise ValueError("mesh_vec_3 must not be provided for 1D meshing")

        if isinstance(mesh_resolution, int) == False:
            raise ValueError("mesh_resolution must be an integer")
        
        particle_data, cylinder_data, mesh_elements, mesh_column, _, _ = meshing.mesh_particles_1d(particle_data, cylinder_data, mesh_vec_1, mesh_resolution)
        
        radii_tuple = namedtuple("radii_tuple",["id", "mean_particle_radii", 
                                                "dim_1_lower_bound", "dim_1_upper_bound"])
        
        mesh = particle_data[mesh_column]
        mean_particle_radii = []

        for mesh_element in mesh_elements:
            mesh_boolean_mask = mesh == mesh_element.id
            mesh_radii = particle_data["radius"][mesh_boolean_mask.astype(bool)]
            mean_radii = np.mean(mesh_radii)

            mean_particle_radii.append(radii_tuple(mesh_element.id, mean_radii,
                                                mesh_element.dim_1_lower_bound, 
                                                mesh_element.dim_1_upper_bound))


    elif mesh_method == "mesh_particles_2d":

        if isinstance(mesh_resolution, list) == False:
            raise ValueError("mesh_resolution must be a list of 2 integers")
        
        elif len(mesh_resolution) != 2:
            raise ValueError("mesh_resolution must be a list of 2 integers")
        
        if mesh_vec_1 is None or mesh_vec_2 is None:
            raise ValueError("mesh_vec_1 and mesh_vec_2 must be provided for 2D meshing")
        
        if mesh_vec_3 is not None:
            raise ValueError("mesh_vec_3 must not be provided for 2D meshing")
        
        particle_data, cylinder_data, mesh_elements, mesh_column, _, _ = meshing.mesh_particles_2d(particle_data, cylinder_data, mesh_vec_1, mesh_vec_2, mesh_resolution)
        
        radii_tuple = namedtuple("radii_tuple",["id", "mean_particle_radii",
                                                "dim_1_lower_bound", "dim_1_upper_bound",
                                                "dim_2_lower_bound", "dim_2_upper_bound"])
        
        mesh = particle_data[mesh_column]
        mean_particle_radii = []

        for mesh_element in mesh_elements:
            mesh_boolean_mask = mesh == mesh_element.id
            mesh_radii = particle_data["radius"][mesh_boolean_mask.astype(bool)]
            mean_radii = np.mean(mesh_radii)

            mean_particle_radii.append(radii_tuple(mesh_element.id, mean_radii,
                                                mesh_element.dim_1_lower_bound, 
                                                mesh_element.dim_1_upper_bound,
                                                mesh_element.dim_2_lower_bound, 
                                                mesh_element.dim_2_upper_bound))


    elif mesh_method == "mesh_particles_3d":

        if mesh_vec_1 is None or mesh_vec_2 is None or mesh_vec_3 is None:
            raise ValueError("mesh_vec_1, mesh_vec_2 and mesh_vec_3 must be provided for 3D meshing")

        if isinstance(mesh_resolution, list) == False:
            raise ValueError("mesh_resolution must be a list of 3 integers")
        
        elif len(mesh_resolution) != 3:
            raise ValueError("mesh_resolution must be a list of 3 integers")
        
        particle_data, cylinder_data, mesh_elements, mesh_column, _, _ = meshing.mesh_particles_3d(particle_data, cylinder_data, mesh_vec_1, mesh_vec_2, mesh_vec_3, mesh_resolution)
        
        radii_tuple = namedtuple("radii_tuple",["id", "mean_particle_radii",
                                                "dim_1_lower_bound", "dim_1_upper_bound",
                                                "dim_2_lower_bound", "dim_2_upper_bound",
                                                "dim_3_lower_bound", "dim_3_upper_bound"])
        
        mesh = particle_data[mesh_column]
        mean_particle_radii = []

        for mesh_element in mesh_elements:
            mesh_boolean_mask = mesh == mesh_element.id
            mesh_radii = particle_data["radius"][mesh_boolean_mask.astype(bool)]
            mean_radii = np.mean(mesh_radii)

            mean_particle_radii.append(radii_tuple(mesh_element.id, mean_radii,
                                                mesh_element.dim_1_lower_bound, 
                                                mesh_element.dim_1_upper_bound,
                                                mesh_element.dim_2_lower_bound, 
                                                mesh_element.dim_2_upper_bound,
                                                mesh_element.dim_3_lower_bound, 
                                                mesh_element.dim_3_upper_bound))\
                                                

    elif mesh_method == "mesh_particles_3d_cylinder":

        if isinstance(mesh_resolution, list) == False:
            raise ValueError("mesh_resolution must be a list of 3 integers")
        
        elif len(mesh_resolution) != 3:
            raise ValueError("mesh_resolution must be a list of 3 integers")
        
        particle_data, cylinder_data, mesh_elements, mesh_column, _, _ = meshing.mesh_particles_3d_cylinder(mesh_resolution, mesh_constant,
                                                                start_rotation)
        
        mesh_column = "3D_cylinder_mesh"
        radii_tuple = namedtuple("radii_tuple",["id", "mean_particle_radii",
                                                "ang_lower_bound", "ang_upper_bound",
                                                "rad_lower_bound", "rad_upper_bound",
                                                "z_lower_bound", "z_upper_bound"])
        
        mesh = particle_data[mesh_column]
        mean_particle_radii = []

        for mesh_element in mesh_elements:
            mesh_boolean_mask = mesh == mesh_element.id
            mesh_radii = particle_data["radius"][mesh_boolean_mask.astype(bool)]
            mean_radii = np.mean(mesh_radii)

            mean_particle_radii.append(radii_tuple(mesh_element.id, mean_radii,
                                                mesh_element.ang_lower_bound, 
                                                mesh_element.ang_upper_bound,
                                                mesh_element.rad_lower_bound, 
                                                mesh_element.rad_upper_bound,
                                                mesh_element.z_lower_bound, 
                                                mesh_element.z_upper_bound))

    else:
        raise ValueError("Invalid mesh method")

    return particle_data, cylinder_data, mean_particle_radii