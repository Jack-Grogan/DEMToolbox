import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.velocity import velocity_vector_field 


def create_cylinder(radius=0.03, height=0.08, resolution=100):
    """Create a container_data mesh."""
    return pv.Cylinder(center=(0 ,0, height / 2),
                       direction=(0, 0, 1),
                       radius=radius, 
                       height=height, 
                       resolution=resolution
                       )


def create_particle_grid(positions, radii=None, velocities=None):
    """Create sample particle data."""

    particle_data = pv.PolyData(positions)

    if radii is not None:
        particle_data['radius'] = radii

    if velocities is not None:
        particle_data['v'] = velocities
    
    particle_data["id"] = np.arange(len(positions))

    return particle_data


class TestVectorFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

         # Create a grid of particles
        x_range = np.linspace(-0.03, 0.03, 31)[1::2] 
        y_range = np.linspace(-0.03, 0.03, 3)[1::2] 
        z_range = np.linspace(0, 0.08, 41)[1::2]
        x, y, z = np.meshgrid(x_range, y_range, z_range)

        positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        radii = [0.0005]*len(positions)
        velocity = [[0, 0.003, -0.003] for _ in range(len(positions))]

        particle_data = create_particle_grid(
            positions, radii=radii, velocities=velocity)
        container_data = create_cylinder(
            radius=0.03, height=0.08, resolution=100)

        # Simple test 1 particle in each sample defined by unit vectors
        point = [0, 0, 0]
        vector_1 = [1, 0, 0]
        vector_2 = [0, 0, 1]
        resolution = [15, 20]
        plane_thickness = 0.012

        # Call the velocity vector field function
        vector_field_results = velocity_vector_field(particle_data, 
                                                    container_data, 
                                                    point, 
                                                    vector_1, 
                                                    vector_2, 
                                                    plane_thickness,
                                                    resolution,
                                                    )
        
        # Store results
        cls.resolution = resolution
        cls.particle_data = vector_field_results[0]
        cls.velocity_vectors = vector_field_results[1]
        cls.samples = vector_field_results[2]
        cls.occupancy = cls.samples.particles.reshape(resolution[1], 
                                                      resolution[0])

    def test_vector_field_shape(self):
        assert (np.shape(self.velocity_vectors) 
                == (self.resolution[1], self.resolution[0], 2))
        
    def test_velocity_vectors_values(self):
        expected_value = [0, -0.003]
        for i in range(self.resolution[1]):
            for j in range(self.resolution[0]):
                assert np.all(self.velocity_vectors[i, j] == expected_value)
        
    def test_occupancy_shape(self):
        assert (np.shape(self.occupancy) 
                == (self.resolution[1], self.resolution[0]))
        
    def test_occupancy_values(self):
        assert np.all(self.occupancy == 1)
