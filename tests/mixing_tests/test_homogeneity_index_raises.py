import pyvista as pv
import numpy as np
import os
import sys
import unittest
import io
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.mixing import homogeneity_index 
from DEMToolbox.particle_sampling import sample_1d

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


def set_up_homogeneity_no_particles():
    """Set up the test class."""

    # Create a grid of particles
    x_range = np.linspace(-0.03, 0.03, 13)[1::2] 
    y_range = np.linspace(-0.03, 0.03, 13)[1::2] 
    z_range = np.linspace(0, 0.08, 21)[1::2]
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    radii = np.digitize(positions[:, 2], np.linspace(0, 0.08, 11)) / 4000

    particle_data = create_particle_grid(positions, radii=radii)
    cylinder_data = create_cylinder(
        radius=0.03, height=0.08, resolution=100)

    particle_data, samples = sample_1d(particle_data,
                                        cylinder_data,
                                        [0, 0, 1],
                                        resolution=10)
    
    return particle_data, samples, cylinder_data


def test_homogeneity_index_non_zero_benchmark(benchmark):
     benchmark(set_up_homogeneity_no_particles)


class TestHomogeneityIndexRaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        result = set_up_homogeneity_no_particles()
        particle_data, samples, cylinder_data = result

        # Store results
        cls.particle_data = particle_data
        cls.samples = samples
        cls.cylinder_data = cylinder_data


        return
    

    def test_homogeneity_index_no_particles(self):

        particle_data = pv.PolyData([])

        with self.assertWarns(UserWarning) as context:
            result_particle_data, result_hi = homogeneity_index(
                particle_data, "radius", self.samples
            )

        self.assertEqual(
            str(context.warning),
            "Cannot calculate homogeneity index for empty particle file."
        )

        self.assertTrue(np.isnan(result_hi))
        self.assertIs(result_particle_data, particle_data)

    def test_homogeneity_index_missing_column(self):

        with self.assertWarns(UserWarning) as context:
            result_particle_data, result_hi = homogeneity_index(
                self.particle_data, "missing_column", self.samples
            )

        self.assertEqual(
            str(context.warning),
            "missing_column not found in particle file, returning NaN."
        )

        self.assertTrue(np.isnan(result_hi))
        self.assertIs(result_particle_data, self.particle_data)

    def test_homogeneity_index_no_samples(self):

        unmodified_particle_data = self.particle_data.copy()
        _, samples = sample_1d(self.particle_data,
                               self.cylinder_data,
                               [1, 0, 0],
                               resolution=3,
                               append_column="missing_samples"
        )

        print(unmodified_particle_data.point_data.keys())
        with self.assertWarns(UserWarning) as context:
            result_particle_data, result_hi = homogeneity_index(
                unmodified_particle_data, "radius", samples
            )

        self.assertEqual(
            str(context.warning),
            "missing_samples not found in particle file, returning NaN."
        )

        self.assertTrue(np.isnan(result_hi))
        self.assertIs(result_particle_data, unmodified_particle_data)

    def test_homogeneity_index_non_numeric_attribute(self):

        unmodified_particle_data = self.particle_data.copy()
        unmodified_particle_data["non_numeric"] = ["a"] * unmodified_particle_data.n_points

        with self.assertRaises(ValueError) as context:
            result_particle_data, result_hi = homogeneity_index(
                unmodified_particle_data, "non_numeric", self.samples
            )

        self.assertEqual(
            str(context.exception),
            "Attribute column non_numeric is not numeric."
        )

    def test_homogeneity_index_non_numeric_samples(self):

        unmodified_particle_data = self.particle_data.copy()
        unmodified_particle_data, samples = sample_1d(
                               unmodified_particle_data,
                               self.cylinder_data,
                               [1, 0, 0],
                               resolution=3,
                               append_column="non_numeric_samples"
        )

        unmodified_particle_data["non_numeric_samples"] = ["a"] * unmodified_particle_data.n_points

        with self.assertRaises(ValueError) as context:
            result_particle_data, result_hi = homogeneity_index(
                unmodified_particle_data, "radius", samples
            )

        self.assertEqual(
            str(context.exception),
            "Sample column non_numeric_samples is not numeric."
        )

    def test_homogeneity_index_nan_mean(self):

        unmodified_particle_data = self.particle_data.copy()
        unmodified_particle_data["radius"] = [np.nan] * unmodified_particle_data.n_points

        with self.assertWarns(UserWarning) as context:
            result_particle_data, result_hi = homogeneity_index(
                unmodified_particle_data, "radius", self.samples
            )

        self.assertEqual(
            str(context.warning),
            "Bulk mean is NaN. Returning NaN for homogeneity index."
        )

        self.assertTrue(np.isnan(result_hi))
        self.assertIs(result_particle_data, unmodified_particle_data)

    def test_homogeneity_index_verbose(self):

        captured_output = io.StringIO()

        with patch("sys.stdout", new=captured_output):
            homogeneity_index(
                self.particle_data,
                "radius",
                self.samples,
                verbose=True,
            )

        output = captured_output.getvalue()

        self.assertIn("Homogeneity index:  0.0007180703308172537\nBulk mean:  "
                      "0.001375\nSample means:  [0.00025 0.0005  0.00075 0.001 "
                      "  0.00125 0.0015  0.00175 0.002   0.00225\n 0.0025 "
                      "]\nNumber of samples:  10\n", 
                      output
        )
