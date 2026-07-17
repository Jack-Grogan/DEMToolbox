import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_2d


def set_up_sample_2d_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))

    return particle_data, cylinder_data


class TestSample2DRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, cylinder_data = set_up_sample_2d_test()

        cls.particle_data = particle_data
        cls.cylinder_data = cylinder_data

    def test_sample_2d_invalid_vectors(self):

        """Test sample_2d with invalid vectors."""

        # Test with a vector that is not 3 elements
        vector_1 = [1, 2]
        vector_2 = [4, -1, 2]
        resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Vectors must be 3 element lists.")

    def test_sample_2d_invalid_resolution(self):

        """Test sample_2d with invalid resolution."""

        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        resolution = [3]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be a 2 element list.")

        # Test with a non-integer resolution
        resolution = [2.5, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be a 2 element list of integers.")

        # Test with a negative resolution
        resolution = [1, -3]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Resolution must be a 2 element list of "
                         "integers greater than 0."
        )

    def test_sample_2d_empty_particles(self):
        """Test sample_2d with empty particles."""

        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        resolution = [3, 3]

        # Test with empty particle data
        empty_particle_data = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_2d(
                empty_particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
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

    def test_sample_2d_non_orthogonal_vectors(self):
        """Test sample_2d with non-orthogonal vectors."""

        vector_1 = [1, 0, 1]
        vector_2 = [1, 1, 0]

        resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Sample vectors must be orthogonal to each other.")

    def test_sample_2d_invalid_bounds(self):
        """Test sample_2d with invalid bounds."""

        vector_1 = [1, 0, 0]
        vector_2 = [0, 1, 0]

        resolution = [3, 3]

        # Invalid bounds (should be 6 elements)
        bounds = [1, 2, 3, 4, 5]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list of 6 elements: "
                         "[x_min, x_max, y_min, y_max, z_min, z_max].",
        )

        # Bounds with non-numeric values
        bounds = [1, 2, 3, 4, 5, "six"]

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list of integers or floats.",
        )

        # Bounds with min >= max
        bounds = np.array([1, 0, 3, 4, 5, 6])

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds are not valid. Ensure that "
                         "x_min < x_max, y_min < y_max and "
                         "z_min < z_max.",
        )

        # empty bounds (vtkPolyData with no points)
        empty_bounds = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_2d(
                self.particle_data,
                empty_bounds,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.warning),
                         "Cannot sample with empty bounds vtk file. "
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

        # Bounds neither a list nor a vtkPolyData
        invalid_bounds = "invalid_bounds"

        with self.assertRaises(ValueError) as context:
            sample_2d(
                self.particle_data,
                invalid_bounds,
                vector_1,
                vector_2,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list or array of 6 elements "
                         "or a vtkPolyData.",
        )

        