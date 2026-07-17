import pyvista as pv
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d
from DEMToolbox.classes import ParticleSamples, ParticleAttribute


def set_up_sample_1d_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                        os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))
    
    return particle_data, cylinder_data



class TestSample1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, cylinder_data = set_up_sample_1d_test()        

        cls.particle_data = particle_data
        cls.cylinder_data = cylinder_data


    def test_sample_1d_invalid_vector(self):
        """Test sample_1d with invalid vector."""

        sample_vector = [8, 1]
        resolution = 21

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                self.cylinder_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Vector must be a 3 element list.")

 

    def test_sample_1d_invalid_resolution(self):
        """Test sample_1d with invalid resolution."""

        sample_vector = [8, 1, 4]
        resolution = "invalid"

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                self.cylinder_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be an integer.")


    def test_sample_1d_invalid_resolution_value(self):
        """Test sample_1d with invalid resolution value."""

        sample_vector = [8, 1, 4]
        resolution = -5

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                self.cylinder_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Resolution must be greater than 0.")

    def test_sample_1d_empty_particle_data(self):
        """Test sample_1d with empty particle data."""

        sample_vector = [8, 1, 4]
        resolution = 21

        empty_particle_data = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_1d(
                empty_particle_data,
                self.cylinder_data,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.warning), 
                         "Cannot sample empty particles file.")

        expected_split = ParticleSamples(
            name="sample_test",
            sample_attribute=ParticleAttribute(
                "id", "sample_test", np.empty((0, 2))
                ),
            cells=[],
            occupied_cells=[],
            particles=[],
            n_sampled_particles=0,
            n_unsampled_particles=0,
        )

        assert split.name == expected_split.name

        assert (split.ParticleAttribute.field 
                == expected_split.ParticleAttribute.field)
        
        assert (split.ParticleAttribute.attribute 
                == expected_split.ParticleAttribute.attribute)
        
        assert np.array_equal(split.ParticleAttribute.data, 
                              expected_split.ParticleAttribute.data)
        
        assert len(split.cells) == len(expected_split.cells)

        assert (len(split.occupied_cells) 
                == len(expected_split.occupied_cells))
        
        assert len(split.particles) == len(expected_split.particles)

        assert (split.n_sampled_particles 
                == expected_split.n_sampled_particles)
        
        assert (split.n_unsampled_particles 
                == expected_split.n_unsampled_particles)

        self.assertEqual(returned_particle_data, empty_particle_data)

    def test_sample_1d_invalid_bounds(self):

        """Test sample_1d with invalid bounds."""

        sample_vector = [8, 1, 4]
        resolution = 21

        # test with invalid bounds length
        bounds = [0, 1, 2, 3, 4] 

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                bounds,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds must be a list of 6 elements: "
                         "[x_min, x_max, y_min, y_max, z_min, z_max].")

    def test_sample_1d_invalid_bounds_type(self):

        """Test sample_1d with invalid bounds type."""

        sample_vector = [8, 1, 4]
        resolution = 21

        # test with invalid bounds type
        bounds = [0, 1, 2, 3, 4, "invalid"] 

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                bounds,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds must be a list of integers or floats.")

    def test_sample_1d_impossible_bounds(self):

        """Test sample_1d with impossible bounds."""

        sample_vector = [8, 1, 4]
        resolution = 21

        # test with impossible bounds
        bounds = [1, 0, 3, 2, 5, 4] 

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                bounds,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds are not valid. Ensure that for each "
                         "dimension min < max.")

    def test_sample_1d_empty_polydata_bounds(self):
        """Test sample_1d with empty vtkPolyData bounds."""

        sample_vector = [8, 1, 4]
        resolution = 21

        # test with empty vtkPolyData bounds
        empty_polydata = pv.PolyData()

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, split = sample_1d(
                self.particle_data,
                empty_polydata,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.warning), 
                        "Cannot sample with empty bounds vtk file. "
                        "Returning unedited particle data.")

        expected_split = ParticleSamples(
            name="sample_test",
            sample_attribute=ParticleAttribute(
                "id", "sample_test", np.empty((0, 2))
                ),
            cells=[],
            occupied_cells=[],
            particles=[],
            n_sampled_particles=0,
            n_unsampled_particles=0,
        )

        assert split.name == expected_split.name

        assert (split.ParticleAttribute.field 
                == expected_split.ParticleAttribute.field)
        
        assert (split.ParticleAttribute.attribute 
                == expected_split.ParticleAttribute.attribute)
        
        assert np.array_equal(split.ParticleAttribute.data, 
                              expected_split.ParticleAttribute.data)
        
        assert len(split.cells) == len(expected_split.cells)

        assert (len(split.occupied_cells) 
                == len(expected_split.occupied_cells))
        
        assert len(split.particles) == len(expected_split.particles)

        assert (split.n_sampled_particles 
                == expected_split.n_sampled_particles)
        
        assert (split.n_unsampled_particles 
                == expected_split.n_unsampled_particles)

        self.assertEqual(returned_particle_data, self.particle_data)

    def test_sample_1d_invalid_bounds_neither_list_nor_polydata(self):
        """Test sample_1d with invalid bounds type."""

        sample_vector = [8, 1, 4]
        resolution = 21

        # test with invalid bounds type
        bounds = "invalid" 

        with self.assertRaises(ValueError) as context:
            sample_1d(
                self.particle_data,
                bounds,
                sample_vector,
                resolution,
                append_column="sample_test",
            )

        self.assertEqual(str(context.exception), 
                         "Bounds must be a list or array of 6 elements "
                         "or a vtkPolyData.")