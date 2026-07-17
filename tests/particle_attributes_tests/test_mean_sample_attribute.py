import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_attributes import mean_sample_attribute
from DEMToolbox.classes import ParticleAttribute
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


def set_up_mean_sample_attribute_test():
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

    ids = particle_data.point_data["id"]
    radii = particle_data.point_data["radius"]
    attribute_data = np.column_stack((ids, radii))

    attribute = ParticleAttribute(
        "id",
        "radius",
        attribute_data
    )

    particle_data, mean = mean_sample_attribute(
        particle_data,
        attribute,
        samples,
    )
        
    return particle_data, attribute, samples, mean, cylinder_data


def test_mean_sample_attribute_benchmark(benchmark):
     benchmark(set_up_mean_sample_attribute_test)


class TestMeanSampleAttribute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        result = set_up_mean_sample_attribute_test()
        particle_data, attribute, samples, mean, cylinder_data = result

        # Store results
        cls.particle_data = particle_data
        cls.attribute = attribute
        cls.samples = samples
        cls.mean = mean
        cls.cylinder_data = cylinder_data

    def test_mean_sample_attribute_1(self):

        # Check that the mean attribute is calculated correctly
        mean_values = self.particle_data.point_data['radius_mean']
        radii_values = self.particle_data.point_data['radius']

        print("Mean values:", mean_values - radii_values)
        assert np.allclose(mean_values, radii_values)

    def test_mean_sample_attribute_2(self):

        particle_data = self.particle_data.copy()

        partcle_data, samples = sample_1d(
            particle_data,
            self.cylinder_data,
            [1, 0, 0],
            resolution=6,
        )

        particle_data, mean = mean_sample_attribute(
            particle_data,
            self.attribute,
            samples,
            append_column="radius_mean_2"
        )

        assert mean.attribute == "radius_mean_2"
        assert mean.field == "id"
        assert mean.data.shape == (particle_data.n_points, 2)
        assert mean.data[:, 0].tolist() == particle_data.point_data["id"].tolist()
        assert mean.data[:, 1].tolist() == particle_data.point_data["radius_mean_2"].tolist()

        expected_mean = np.mean(np.arange(1, 11) / 4000)
        expected_mean = np.array([expected_mean] * particle_data.n_points)

        self.assertTrue(np.allclose(particle_data.point_data["radius_mean_2"], expected_mean))
        self.assertTrue(np.allclose(mean.data[:, 1], expected_mean))

        assert "radius_mean_2" in particle_data.point_data.keys()


    def test_mean_sample_attribute_no_particles(self):

        """Test the mean_sample_attribute function with no particles."""
        particle_data = pv.PolyData([])

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, mean = mean_sample_attribute(
                particle_data,
                self.attribute,
                self.samples,
                append_column="radius_mean_3"
            )

        self.assertEqual(
            str(context.warning),
            "No particles in particle_data vtk. " 
            "Returning unedited particle_data and "
            "empty mean_attribute."
        )

        assert mean.attribute == "radius_mean_3"
        assert mean.field == "id"
        assert mean.data.shape == (1, 2)
        assert mean.data[0, 0] is None
        assert mean.data[0, 1] is None
        assert returned_particle_data == particle_data


    def test_mean_sample_attribute_missing_attribute_column(self):
        
        with self.assertWarns(UserWarning) as context:
            returned_particle_data, mean = mean_sample_attribute(
                self.particle_data,
                ParticleAttribute("id", "missing_column", None),
                self.samples,
                append_column="radius_mean_4"
            )

        self.assertEqual(
            str(context.warning),
            "Attribute column not in particle_data vtk. "
            "Returning unedited particle_data and "
            "empty mean_attribute."
        )

        assert mean.attribute == "radius_mean_4"
        assert mean.field == "id"
        assert mean.data.shape == (1, 2)
        assert mean.data[0, 0] is None
        assert mean.data[0, 1] is None
        assert returned_particle_data == self.particle_data


    def test_mean_sample_attribute_missing_samples_column(self):

        unmodified_particle_data = self.particle_data.copy()
        _, samples = sample_1d(self.particle_data,
                               self.cylinder_data,
                               [1, 0, 0],
                               resolution=3,
                               append_column="missing_samples"
        )

        with self.assertWarns(UserWarning) as context:
            returned_particle_data, mean = mean_sample_attribute(
                unmodified_particle_data,
                self.attribute,
                samples,
                append_column="radius_mean_5"
            )

        self.assertEqual(
            str(context.warning),
            "Samples column not in particle_data vtk. "
            "Returning unedited particle_data and "
            "empty mean_attribute."
        )

        assert mean.attribute == "radius_mean_5"
        assert mean.field == "id"
        assert mean.data.shape == (1, 2)
        assert mean.data[0, 0] is None
        assert mean.data[0, 1] is None
        assert returned_particle_data == unmodified_particle_data



