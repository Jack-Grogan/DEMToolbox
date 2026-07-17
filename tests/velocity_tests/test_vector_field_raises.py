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


def set_up_simple_velocity_vector_field_test():

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
        
        return particle_data, container_data, 

def test_simple_velocity_vector_field_benchmark(benchmark):
    benchmark(set_up_simple_velocity_vector_field_test)


class TestSimpleVectorFieldRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, container_data = set_up_simple_velocity_vector_field_test()
        
        cls.particle_data = particle_data
        cls.container_data = container_data

    def test_velocity_vector_empty_particle_data(self):
        """Test velocity_vector_field with empty particle data."""
        empty_particle_data = pv.PolyData()
        point = [0, 0, 0]
        vector_1 = [1, 0, 0]
        vector_2 = [0, 0, 1]
        resolution = [15, 20]
        plane_thickness = 0.012

        with self.assertWarns(UserWarning) as context:
            results = velocity_vector_field(
                empty_particle_data, 
                self.container_data, 
                point, 
                vector_1, 
                vector_2, 
                plane_thickness,
                resolution,
            )
        
        self.assertEqual(str(context.warning),
                         "Cannot sample empty container file."
                         "Returning unedited particle data.")


        expected_occupancy = np.zeros((resolution[1], resolution[0]))

        assert results[0] == empty_particle_data
        assert np.all(np.isnan(results[1]))
        assert np.array_equal(results[2], expected_occupancy)
        assert results[3] is None
