import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d_volume


def set_up_sample_1d_volume_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    
    return particle_data
                                            

class TestSample1DVolume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data = set_up_sample_1d_volume_test()

        cls.particle_data = particle_data

    def test_sample_1d_volume_invalid_vector(self):
        """Test sample_1d_volume with invalid vector."""

        # Test with a vector that is not 3 elements
        sample_vector = [8, 1]
        resolution = 21

        with self.assertRaises(ValueError) as context:
            sample_1d_volume(
                self.particle_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Sample_vector must be a 3 element list.")

    def test_sample_1d_volume_invalid_resolution(self):
        """Test sample_1d_volume with invalid resolution."""

        # Test with a non-integer resolution
        sample_vector = [0, 0, 1]
        resolution = 21.5

        with self.assertRaises(ValueError) as context:
            sample_1d_volume(
                self.particle_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be an integer.")

        # Test with a negative resolution
        resolution = -5

        with self.assertRaises(ValueError) as context:
            sample_1d_volume(
                self.particle_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be greater than or equal to 2.")

    def test_sample_1d_volume_empty_particle_data(self):
        """Test sample_1d_volume with empty particle data."""

        # Create an empty particle data object
        empty_particle_data = pv.PolyData()

        sample_vector = [0, 0, 1]
        resolution = 21

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, samples = sample_1d_volume(
                empty_particle_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.warning), 
                         "Cannot sample empty particles file.")

        assert samples.name == "sample_test"
        assert samples.ParticleAttribute.field == "id"
        assert samples.ParticleAttribute.attribute == "sample_test"
        assert np.array_equal(samples.ParticleAttribute.data, np.empty((0, 2)))
        assert len(samples.occupied_cells) == 0
        assert len(samples.cells) == 0
        assert samples.n_sampled_particles == 0
        assert samples.n_unsampled_particles == 0

        assert returned_particle_data == empty_particle_data

    def test_sample_1d_volume_more_samples_than_particles(self):
        """Test sample_1d_volume with more samples than particles."""

        sample_vector = [0, 0, 1]
        resolution = self.particle_data.n_points + 1

        with self.assertRaises(ValueError) as context:
            sample_1d_volume(
                self.particle_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be less than or equal to the "
                         "number of particles in the particle data.")

