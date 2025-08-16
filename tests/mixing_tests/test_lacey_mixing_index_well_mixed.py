import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.mixing import macro_scale_lacey_mixing 
from DEMToolbox.classes import ParticleAttribute
from DEMToolbox.utilities import append_attribute
from DEMToolbox.particle_sampling import sample_3d

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


def set_up_lacey_mixing_index_well_mixed_test():
    # Create a grid of particles
    x_range = np.linspace(-0.03, 0.03, 11)[1::2] 
    y_range = np.linspace(-0.03, 0.03, 11)[1::2] 
    z_range = np.linspace(0, 0.08, 21)[1::2]
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    radii = [0.0005]*len(positions)

    particle_data = create_particle_grid(positions, radii=radii)
    cylinder_data = create_cylinder(
        radius=0.03, height=0.08, resolution=100)
    
    ids = np.asarray(particle_data["id"])
    data = np.column_stack((ids, ids % 2))

    split = ParticleAttribute("id", "mixed", data)
    particle_data = append_attribute(particle_data, split)

    sample_resolution = [5, 5, 5]
    particle_data, samples = sample_3d(particle_data,
                                        cylinder_data,
                                        vector_1=[1, 0, 0],
                                        vector_2=[0, 1, 0],
                                        vector_3=[0, 0, 1],
                                        resolution=sample_resolution,
                                        )
    
    particle_data, lacey = macro_scale_lacey_mixing(particle_data, 
                                                    split,
                                                    samples,
                                                    )
    
    return (particle_data, samples, sample_resolution, 
            lacey, split, cylinder_data)


def test_lacey_mixing_index_well_mixed_benchmark(benchmark):
    benchmark(set_up_lacey_mixing_index_well_mixed_test)


class TestLaceyMixingIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        result = set_up_lacey_mixing_index_well_mixed_test()

        # Unpack the result
        (particle_data, samples, sample_resolution, 
         lacey, split, cylinder_data) = result
        
        particle_data, lacey = macro_scale_lacey_mixing(particle_data, 
                                                        split,
                                                        samples,
                                                        )

        # Store results
        cls.particle_data = particle_data
        cls.samples = samples
        cls.lacey = lacey
        cls.resolution = sample_resolution
        cls.split = split
        cls.cylinder_data = cylinder_data

    def test_lacey_mixing_index_well_mixed(self):
        """Test value of the Lacey mixing index."""

        # 50% of particles in the bulk are of target particle type
        unmixed_variance = 0.5 * (1 - 0.5)
        # 2 particles in each of the 125 samples
        mixed_variance = unmixed_variance / 2
        # Equal volume sampling all samples perfectly mixed
        variance = 0

        expected_value = ((variance - unmixed_variance)
                          / (mixed_variance - unmixed_variance))

        assert np.isclose(self.lacey, expected_value, atol=1e-10)
