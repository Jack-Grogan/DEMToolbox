import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.mixing import macro_scale_lacey_mixing 
from DEMToolbox.particle_sampling import sample_1d_volume
from DEMToolbox.particle_sampling import sample_3d_cylinder

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


def set_up_lacey_mixing_index_segregated_test():
    # Create a grid of particles
    x_range = np.linspace(-0.03, 0.03, 13)[1::2] 
    y_range = np.linspace(-0.03, 0.03, 13)[1::2] 
    z_range = np.linspace(0, 0.08, 41)[1::2]
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    radii = [0.0005]*len(positions)

    particle_data = create_particle_grid(positions, radii=radii)
    cylinder_data = create_cylinder(
        radius=0.03, height=0.08, resolution=100)
    
    
    # test with non normalised vector
    particle_data, split = sample_1d_volume(particle_data, 
                                            [0, 0, 3],
                                            resolution=2,
                                            append_column="z_split",
                                            )

    sample_resolution = [2,2,8]
    sample_constant = "volume"

    particle_data, samples = sample_3d_cylinder(particle_data,
                                                cylinder_data,
                                                sample_resolution,
                                                sample_constant)
    
    particle_data, lacey = macro_scale_lacey_mixing(particle_data, 
                                                    split.ParticleAttribute,
                                                    samples,
                                                    )

    return (particle_data, samples, sample_resolution, 
            lacey, split, cylinder_data)


def test_lacey_mixing_index_segregated_benchmark(benchmark):
    benchmark(set_up_lacey_mixing_index_segregated_test)


class TestLaceyMixingIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        result = set_up_lacey_mixing_index_segregated_test()
        
        # Unpack results
        (particle_data, samples, sample_resolution, 
         lacey, split, cylinder_data) = result

        # Store results
        cls.particle_data = particle_data
        cls.samples = samples
        cls.lacey = lacey
        cls.resolution = sample_resolution
        cls.split = split
        cls.cylinder_data = cylinder_data

    def test_lacey_mixing_index_segregated(self):
        """Test value of the Lacey mixing index."""

        expected_value = 0
        assert np.isclose(self.lacey, expected_value, atol=1e-10)
