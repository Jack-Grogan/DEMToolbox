import pyvista as pv
import numpy as np
import os
import sys
import unittest
import io
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.mixing import macro_scale_lacey_mixing 
from DEMToolbox.particle_sampling import sample_1d_volume
from DEMToolbox.particle_sampling import sample_3d_cylinder
from DEMToolbox.particle_sampling import sample_3d
from DEMToolbox.classes import ParticleAttribute
from DEMToolbox.utilities import append_attribute

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


def set_up_lacey_mixing_index_part_mixed_test():
    
    # Create a grid of particles
    x_range = np.linspace(-0.03, 0.03, 13)[1::2] 
    y_range = np.linspace(-0.03, 0.03, 13)[1::2] 
    z_range = np.linspace(0, 0.08, 61)[1::2]
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    radii = [0.0005]*len(positions)

    particle_data = create_particle_grid(positions, radii=radii)
    cylinder_data = create_cylinder(
        radius=0.03, height=0.08, resolution=100)
    
    
    # test with non normalised vector
    particle_data, split = sample_1d_volume(particle_data, 
                                            [1, 1, 3],
                                            resolution=2,
                                            append_column="z_split",
                                            )


    sample_resolution = [1,1,3]
    sample_constant = "volume"
    particle_data, samples = sample_3d_cylinder(particle_data,
                                                cylinder_data,
                                                sample_resolution,
                                                sample_constant)
    
    return (particle_data, samples, sample_resolution, split, cylinder_data)


def test_lacey_mixing_index_part_mixed_benchmark(benchmark):
    benchmark(set_up_lacey_mixing_index_part_mixed_test)


class TestLaceyMixingIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        result = set_up_lacey_mixing_index_part_mixed_test()

        # Unpack results
        (particle_data, samples, sample_resolution, 
         split, cylinder_data) = result        
        
        # Store results
        cls.particle_data = particle_data
        cls.samples = samples
        cls.resolution = sample_resolution
        cls.split = split
        cls.cylinder_data = cylinder_data


    def test_lacey_mixing_index_no_particles(self):
        """Test error handling for empty particle data."""

        empty_particle_data = pv.PolyData()


        with self.assertWarns(UserWarning) as context:
            particle_data, lacey = macro_scale_lacey_mixing(
                empty_particle_data, 
                self.split.ParticleAttribute, 
                self.samples,
            )

        self.assertEqual(
            str(context.warning),
            "Cannot calculate Lacey mixing index for empty particle file."
        )

        self.assertTrue(np.isnan(lacey))
        self.assertEqual(particle_data, empty_particle_data)


    def test_lacey_mixing_index_missing_attribute_column(self):
        """Test error handling for missing attribute column."""

        particle_data = self.particle_data.copy()
        missing_attribute_data = np.zeros((particle_data.n_points, 2))

        missing_attribute_data[:, 0] = particle_data["id"]
        missing_attribute_data[:, 1][::2] = [1] * (particle_data.n_points // 2)

        missing_attribute = ParticleAttribute(
            "id",
            "missing_attribute",
            missing_attribute_data
        )
            
        with self.assertWarns(UserWarning) as context:
            particle_data, lacey = macro_scale_lacey_mixing(
                particle_data, 
                missing_attribute,
                self.samples,
            )

        self.assertEqual(
            str(context.warning),
            "missing_attribute not found in particle file, returning NaN."
        )

        self.assertTrue(np.isnan(lacey))
        self.assertEqual(particle_data, self.particle_data)


    def test_lacey_mixing_index_missing_samples_column(self):
        """Test error handling for missing samples column."""

        unmodified_particle_data = self.particle_data.copy()
        _, samples = sample_3d_cylinder(self.particle_data,
                                        self.cylinder_data,
                                        self.resolution,
                                        "volume",
                                        append_column="missing_samples"
                                        )

        with self.assertWarns(UserWarning) as context:
            particle_data, lacey = macro_scale_lacey_mixing(
                unmodified_particle_data, 
                self.split.ParticleAttribute,
                samples,    
            )

        self.assertEqual(
            str(context.warning),
            "missing_samples not found in particle file, returning NaN."
        )

        self.assertTrue(np.isnan(lacey))
        self.assertEqual(particle_data, unmodified_particle_data)


    def test_lacey_mixing_index_only_type_1(self):
        """Test error handling for only one particle type present."""

        particle_data = self.particle_data.copy()
        attribute_data = np.zeros((particle_data.n_points, 2))
       
        attribute_data[:, 0] = particle_data["id"]
        attribute_data[:, 1] = [1] * particle_data.n_points

        attribute = ParticleAttribute(
            "id",
            "1D_samples",
            attribute_data
        )

        particle_data = append_attribute(particle_data, attribute)

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, lacey = macro_scale_lacey_mixing(
                particle_data, 
                attribute,
                self.samples,
            )

        self.assertEqual(
            str(context.warning),
            "particle data contains only particle type 1.0, "
            "setting Lacey to NaN."
        )

        self.assertTrue(np.isnan(lacey))
        self.assertEqual(returned_particle_data, particle_data)


    def test_lacey_mixing_index_only_type_0(self):
        """Test error handling for only one particle type present."""

        particle_data = self.particle_data.copy()
        attribute_data = np.zeros((particle_data.n_points, 2))
       
        attribute_data[:, 0] = particle_data["id"]
        attribute_data[:, 1] = [0] * particle_data.n_points

        attribute = ParticleAttribute(
            "id",
            "1D_samples",
            attribute_data
        )

        particle_data = append_attribute(particle_data, attribute)

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, lacey = macro_scale_lacey_mixing(
                particle_data, 
                attribute,
                self.samples,
            )

        self.assertEqual(
            str(context.warning),
            "particle data contains only particle type 0.0, "
            "setting Lacey to NaN."
        )

        self.assertTrue(np.isnan(lacey))
        self.assertEqual(returned_particle_data, particle_data)


    def test_lacey_mixing_index_only_type_2(self):
        """Test error handling for only one invalid particle type present."""

        particle_data = self.particle_data.copy()
        attribute_data = np.zeros((particle_data.n_points, 2))
       
        attribute_data[:, 0] = particle_data["id"]
        attribute_data[:, 1] = [2] * particle_data.n_points

        attribute = ParticleAttribute(
            "id",
            "1D_samples",
            attribute_data
        )

        particle_data = append_attribute(particle_data, attribute)

        with self.assertRaises(ValueError) as context:
            macro_scale_lacey_mixing(
                particle_data, 
                attribute,
                self.samples,
            )

        self.assertEqual(
            str(context.exception),
            "particle data contains particle types with values "
            "other than 0 and 1, cannot calculate Lacey mixing "
            "index. Found particle types: [2.]"
        )


    def test_lacey_mixing_index_type_02(self):
        """Test error handling for one invalid particle type present."""

        particle_data = self.particle_data.copy()
        attribute_data = np.zeros((particle_data.n_points, 2))
       
        attribute_data[:, 0] = particle_data["id"]
        attribute_data[:, 1][1::2] = [2] * (particle_data.n_points // 2)

        attribute = ParticleAttribute(
            "id",
            "1D_samples",
            attribute_data
        )

        particle_data = append_attribute(particle_data, attribute)

        with self.assertRaises(ValueError) as context:
            macro_scale_lacey_mixing(
                particle_data, 
                attribute,
                self.samples,
            )

        self.assertEqual(
            str(context.exception),
            "particle data contains particle types with values "
            "other than 0 and 1, cannot calculate Lacey mixing "
            "index. Found particle types: [2.]"
        )

    
    def test_lacey_mixing_index_one_occupied_sample(self):
        """Test error handling for only one occupied sample present."""

        particle_data = self.particle_data.copy()
        cylinder_data = self.cylinder_data.copy()

        sample_resolution = [1, 1, 1]

        particle_data, samples = sample_3d(
            particle_data,
            cylinder_data,
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            resolution=sample_resolution,
            append_column="3D_samples",
        )

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, lacey = macro_scale_lacey_mixing(
                particle_data, 
                self.split.ParticleAttribute, 
                samples,
        )

        self.assertEqual(
            str(context.warning),
            "Fewer than 2 non-empty samples in particle data. "
            "Setting Lacey to NaN, consider refining the sample "
            "resolution."
        )

    def test_lacey_mixing_index_one_particle_per_sample(self):
        """Test error handling for one particle per sample."""

        particle_data = self.particle_data.copy()
        cylinder_data = self.cylinder_data.copy()

        sample_resolution = [6, 6, 30]

        particle_data, samples = sample_3d(
            particle_data,
            cylinder_data,
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            resolution=sample_resolution,
            append_column="3D_samples",
        )

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, lacey = macro_scale_lacey_mixing(
                particle_data, 
                self.split.ParticleAttribute, 
                samples,
        )

        self.assertEqual(
            str(context.warning),
            "Mixed variance is equal to unmixed variance, "
            "setting Lacey to NaN on account of division by zero. "
            "This is likely due to the sample resolution being too "
            "fine leading to each sample containing only one particle."
            " Consider coarsening the sample resolution."
        )

        self.assertTrue(np.isnan(lacey))
        self.assertEqual(returned_particle_data, particle_data)

    def test_lacey_mixing_index_verbose(self):
        """Test verbose output of the Lacey mixing index."""

        captured_output = io.StringIO()

        with patch("sys.stdout", new=captured_output):
            macro_scale_lacey_mixing(
                self.particle_data.copy(), 
                self.split.ParticleAttribute, 
                self.samples,
                verbose=True
            )

        output = captured_output.getvalue()

        self.assertEqual(
            "Lacey mixing index: 0.1666666666666667 - 0.25 / "
            "0.0007812499999999983 - 0.25 = 0.334378265412748\n",
            output
        )