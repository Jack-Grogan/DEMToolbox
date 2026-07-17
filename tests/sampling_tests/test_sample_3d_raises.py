import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d


def set_up_sample_3d_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))

    return particle_data, cylinder_data


class TestSample3DRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, cylinder_data = set_up_sample_3d_test()
        
        cls.particle_data = particle_data
        cls.cylinder_data = cylinder_data

    def test_sample_3d_invalid_vectors(self):
        """Test sample_3d with invalid vectors."""

        # Test with a vector that is not 3 elements
        vector_1 = [1, 2]
        vector_2 = [4, -1, 2]
        vector_3 = [1, -2, -3]
        resolution = [3, 3, 3]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Vectors must be 3 element lists.")

    def test_sample_3d_invalid_cube_length(self):
        """Test sample_3d with invalid cube length."""

        # Test with no cube length provided 
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        vector_3 = [1, -2, -3]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "If resolution is None, "
                         "cube_length must be specified.",
        )

        # Test with a non-numeric cube length
        cube_length = "invalid"

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                cube_length=cube_length,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Cube length must be a positive integer or float.")

        # Test with a negative cube length
        cube_length = -5

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                cube_length=cube_length,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Cube length must be greater than 0.",
        )

    def test_sample_3d_invalid_resolution(self):

        # Test with cube also provided with valid resolution
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        vector_3 = [1, -2, -3]
        cube_length = 5
        resolution = [3, 3, 3]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                cube_length=cube_length,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "If resolution is specified, "
                         "cube_length must be None.",
        )

        # Unnecessary center_meshgrid argument set to True when 
        # resolution is specified

        center_mesh = True

        with self.assertWarns(UserWarning) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                center_meshgrid=center_mesh,
                append_column="sample_test",
            )

        self.assertEqual(str(context.warning),
                         "Centering the meshgrid is completed by "
                         "definition when resolution is specified.",
        )

        # Test with resolution that is not a 3 element list
        resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Resolution must be a 3 element list.")

        # Test with resolution that is not a 3 element list of integers
        resolution = [3, 3, 3.5]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Resolution must be a list of integers.")

        # Test with resolution that is not a 3 element list of 
        # integers greater than 0
        resolution = [3, 3, -3]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Resolution must be greater than 0 "
                         "in all dimensions.",
        )

    def test_sample_3d_empty_particles(self):
        """Test sample_3d with empty particles."""

        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        vector_3 = [1, -2, -3]
        resolution = [3, 3, 3]

        # Test with empty particle data
        empty_particle_data = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_3d(
                empty_particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
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

    def test_sample_3d_non_orthogonal_vectors(self):
        """Test sample_3d with non-orthogonal vectors."""

        vector_1 = [1, 0, 1]
        vector_2 = [1, 1, 0]
        vector_3 = [0, 1, 1]
        resolution = [3, 3, 3]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                self.cylinder_data,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Sample vectors must be orthogonal to each other.")

    def test_sample_3d_invalid_bounds(self):

        # Test with bounds that are not a 6 element list
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        vector_3 = [1, -2, -3]
        resolution = [3, 3, 3]

        bounds = [0, 1, 2, 3, 4]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list of 6 elements: "
                         "[x_min, x_max, y_min, y_max, z_min, z_max].",
        )


        # Test with bounds that are not a 6 element list of integers or floats
        bounds = [0, 1, 2, 3, 4, "invalid"]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list of integers or floats.",
        )

        # Test with bounds with impossible values (min > max)
        bounds = [1, 0, 3, 2, 5, 4]

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds are not valid. Ensure that for each "
                         "dimension min < max.",
        )

        # bounds provided with empty polydata
        bounds = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_3d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
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

        bounds = "invalid"

        with self.assertRaises(ValueError) as context:
            sample_3d(
                self.particle_data,
                bounds,
                vector_1,
                vector_2,
                vector_3,
                resolution=resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list of 6 elements or a "
                         "vtkPolyData.",
        )

