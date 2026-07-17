import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_2d_slice


def set_up_sample_2d_slice_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))

    return particle_data, cylinder_data


class TestSample2DSliceRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, cylinder_data = set_up_sample_2d_slice_test()

        cls.particle_data = particle_data
        cls.cylinder_data = cylinder_data

    def test_sample_2d_slice_invalid_vectors(self):
        """Test sample_2d_slice with invalid vectors."""

        point = [0, 0, 0.04]

        # Test with a vector that is not 3 elements
        vector_1 = [1, 2]
        vector_2 = [4, -1, 2]
        plane_thickness = 0.01
        resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Vectors must be 3 element lists.")

    def test_sample_2d_slice_invalid_point(self):
        """Test sample_2d_slice with invalid point."""

        # Test with a point that is not 3 elements
        point = [0, 0]
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        plane_thickness = 0.01
        resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Point must be a 3 element list.")

    def test_sample_2d_slice_invalid_resolution(self):
        """Test sample_2d_slice with invalid resolution."""

        point = [0, 0, 0.04]
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        plane_thickness = 0.01

        # Test with a resolution that is not 2 elements
        resolution = [3]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be a 2 element list.")

        # Test with a non-integer resolution
        resolution = [2.5, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be a 2 element list of integers.")

        # Test with a resolution that has a zero or negative value
        resolution = [3, 0]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be a 2 element list of "
                         "integers greater than 0.",
        )

    def test_sample_2d_slice_invalid_plane_thickness(self):
        """Test sample_2d_slice with invalid plane thickness."""

        point = [0, 0, 0.04]
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        resolution = [3, 3]

        # Test with a non-numeric plane thickness
        plane_thickness = "invalid"

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "plane_thickness must be an integer or float.")

        # Test with a negative plane thickness
        plane_thickness = -0.01

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "plane_thickness must be greater than 0.")

    def test_sample_2d_slice_non_orthogonal_vectors(self):
        """Test sample_2d_slice with non-orthogonal vectors."""

        point = [0, 0, 0.04]
        vector_1 = [1, 2, -1]
        vector_2 = [2, 4, -2]  # This vector is not orthogonal to vector_1
        plane_thickness = 0.01
        resolution = [3, 3]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Sample vectors must be orthogonal to each other.")

    def test_sample_2d_slice_empty_particle_data(self):
        """Test sample_2d_slice with empty particle data."""

        empty_particle_data = pv.PolyData()
        point = [0, 0, 0.04]
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        plane_thickness = 0.01
        resolution = [3, 3]

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_2d_slice(
                empty_particle_data,
                self.cylinder_data,
                point,
                vector_1,
                vector_2,
                plane_thickness,
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

    def test_sample_2d_slice_invalid_bounds(self):
        """Test sample_2d_slice with invalid bounds."""

        point = [0, 0, 0.04]
        vector_1 = [1, 2, -1]
        vector_2 = [4, -1, 2]
        plane_thickness = 0.01
        resolution = [3, 3]

        # Test with a bounds that is not 6 elements
        bounds = [0, 1, 0, 1]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                bounds,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds must be a list of 6 elements: "
                         "[x_min, x_max, y_min, y_max, z_min, z_max].")

        # Test with a bounds that has a non-numeric value
        bounds = [0, 1, 0, 1, 0, "invalid"]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                bounds,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds must be a list of integers or floats.")

        # Test with a bounds that has a min value greater than the max value
        bounds = [1, 0, 0, 1, 0, 1]

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                bounds,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds are not valid. Ensure that "
                         "x_min < x_max, y_min < y_max and "
                         "z_min < z_max.",
        )

        # Test empty bounds vtk file
        empty_bounds = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_2d_slice(
                self.particle_data,
                empty_bounds,
                point,
                vector_1,
                vector_2,
                plane_thickness,
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

        # Test with a bounds that is not a list or numpy array
        bounds = "invalid"

        with self.assertRaises(ValueError) as context:
            sample_2d_slice(
                self.particle_data,
                bounds,
                point,
                vector_1,
                vector_2,
                plane_thickness,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception),
                         "Bounds must be a list of 6 elements or a "
                         "vtkPolyData.",
        )

        
        


        