import pyvista as pv
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d_volume_cylinder


def set_up_sample_1d_volume_cylinder_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                        os.pardir, "vtks")
        
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))

    return particle_data


class TestSample1DVolumeCylinderRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data = set_up_sample_1d_volume_cylinder_test()  
        
        cls.particle_data = particle_data

    def test_sample_1d_volume_cylinder_invalid_sample_vector(self):
        """Test sample_1d_volume_cylinder with invalid sample_vector."""

        point = [0, 0, 0]

        # Test with a vector that is not 3 elements
        sample_vector = [2, 1]

        with self.assertRaises(ValueError) as context:
            sample_1d_volume_cylinder(
                self.particle_data,
                point,
                sample_vector,
                resolution=10,
            )

        self.assertEqual(str(context.exception), 
                         "sample_vector must be a 3 element list.")

    def test_sample_1d_volume_cylinder_empty_particles(self):
        """Test sample_1d_volume_cylinder with empty particles."""

        point = [0, 0, 0]
        sample_vector = [2, 1, 3]

        # Test with empty particle data
        empty_particle_data = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_1d_volume_cylinder(
                empty_particle_data,
                point,
                sample_vector,
                resolution=10,
            )

        self.assertEqual(str(context.warning),
                         "Cannot sample with empty particle data. "
                         "Returning unedited particle data.")

    def test_sample_1d_volume_cylinder_invalid_resolution(self):
        """Test sample_1d_volume_cylinder with invalid resolution."""

        point = [0, 0, 0]
        sample_vector = [2, 1, 3]

        # Test with a non-integer resolution
        resolution = 10.5

        with self.assertRaises(ValueError) as context:
            sample_1d_volume_cylinder(
                self.particle_data,
                point,
                sample_vector,
                resolution=resolution,
            )

        self.assertEqual(str(context.exception), 
                         "resolution must be an integer.")

        # Test with a negative resolution
        resolution = -5

        with self.assertRaises(ValueError) as context:
            sample_1d_volume_cylinder(
                self.particle_data,
                point,
                sample_vector,
                resolution=resolution,
            )

        self.assertEqual(str(context.exception), 
                         "resolution must be greater than or equal to 2.")

        # Test with a resolution greater than the number of particles
        resolution = self.particle_data.n_points + 1

        with self.assertRaises(ValueError) as context:
            sample_1d_volume_cylinder(
                self.particle_data,
                point,
                sample_vector,
                resolution=resolution,
            )

        self.assertEqual(str(context.exception), 
                         "resolution must be less than or equal to the "
                         "number of particles in the particle data.")


    
