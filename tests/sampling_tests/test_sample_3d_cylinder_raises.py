import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d_cylinder


def set_up_sample_3d_cylinder_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))

    return particle_data, cylinder_data


def test_sample_3d_cylinder_benchmark(benchmark):
    benchmark(set_up_sample_3d_cylinder_test)


class TestSample3DCylinderRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, cylinder_data = set_up_sample_3d_cylinder_test()
        
        cls.particle_data = particle_data
        cls.cylinder_data = cylinder_data

    def test_sample_3d_cylinder_invalid_resolution(self):
        """Test invalid resolution raises ValueError."""

        # Invalid resolution (only 2 dimensions)
        invalid_resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_3d_cylinder(self.particle_data,
                               self.cylinder_data,
                               invalid_resolution,
                               sample_constant="radius",
                               append_column="sample_test"
            )

        self.assertEqual(str(context.exception),
                            "Resolution must be a list of 3 integers.")

        # Invalid data type for resolution (string instead of list)
        invalid_resolution = ["3", "3", "3"]

        with self.assertRaises(ValueError) as context:
            sample_3d_cylinder(self.particle_data,
                               self.cylinder_data,
                               invalid_resolution,
                               sample_constant="radius",
                               append_column="sample_test"
            )

        self.assertEqual(str(context.exception),
                            "Resolution must be a list of 3 integers.")

        # Invalid resolution (contains zero)
        invalid_resolution = [3, 3, 0]

        with self.assertRaises(ValueError) as context:
            sample_3d_cylinder(self.particle_data,
                               self.cylinder_data,
                               invalid_resolution,
                               sample_constant="radius",
                               append_column="sample_test"
            )

        self.assertEqual(str(context.exception),
                         "Resolution must be greater than 0 in all "
                         "dimensions.",
        )

    def test_sample_3d_cylinder_invalid_rotation(self):
        """Test invalid rotation raises ValueError."""

        # Invalid rotation (only 2 dimensions)
        invalid_rotation = "1"

        with self.assertRaises(ValueError) as context:
            sample_3d_cylinder(self.particle_data,
                               self.cylinder_data,
                               resolution=[3, 3, 3],
                               rotation=invalid_rotation,
                               sample_constant="radius",
                               append_column="sample_test"
            )

        self.assertEqual(str(context.exception),
                            "Rotation must be an integer or float.")

    def test_sample_3d_cylinder_empty_particle_data(self):
        """Test empty particle data raises ValueError."""

        empty_particle_data = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_3d_cylinder(
                               empty_particle_data,
                               self.cylinder_data,
                               resolution=[3, 3, 3],
                               sample_constant="radius",
                               append_column="sample_test"
            )

        self.assertEqual(str(context.warning),
                         "Cannot sample with empty particle data. "
                         "Returning unedited particle data.")

        assert split.name == "sample_test"
        assert split.ParticleAttribute.attribute == "sample_test"
        assert split.ParticleAttribute.field == "id"
        assert np.array_equal(split.ParticleAttribute.data, np.empty((0, 2)))
        assert len(split.occupied_cells) == 0
        assert len(split.cells) == 0
        assert split.n_sampled_particles == 0
        assert split.n_unsampled_particles == 0

        assert returned_particle_data == empty_particle_data

    def test_sample_3d_cylinder_empty_cylinder_data(self):
        """Test empty cylinder data raises ValueError."""

        empty_cylinder_data = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_3d_cylinder(
                               self.particle_data,
                               empty_cylinder_data,
                               resolution=[3, 3, 3],
                               sample_constant="radius",
                               append_column="sample_test"
            )

        self.assertEqual(str(context.warning),
                         "Cannot sample empty container file."
                         "Returning unedited particle data.")

        assert split.name == "sample_test"
        assert split.ParticleAttribute.attribute == "sample_test"
        assert split.ParticleAttribute.field == "id"
        assert np.array_equal(split.ParticleAttribute.data, np.empty((0, 2)))
        assert len(split.occupied_cells) == 0
        assert len(split.cells) == 0
        assert split.n_sampled_particles == 0
        assert split.n_unsampled_particles == 0

        assert returned_particle_data == self.particle_data

    def test_sample_3d_cylinder_invalid_sample_constant(self):
        """Test invalid sample_constant raises ValueError."""

        # Invalid sample_constant (not a string)
        invalid_sample_constant = 123

        with self.assertRaises(ValueError) as context:
            sample_3d_cylinder(self.particle_data,
                               self.cylinder_data,
                               resolution=[3, 3, 3],
                               sample_constant=invalid_sample_constant,
                               append_column="sample_test"
            )

        self.assertEqual(str(context.exception),
                            "Invalid sample constant. "
                         "Must be 'radius' or 'volume'.")

    

        