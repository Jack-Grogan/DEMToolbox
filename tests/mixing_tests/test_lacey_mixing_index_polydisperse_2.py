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


class TestLaceyMixingIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

         # Create a grid of particles
        x_range = np.linspace(-0.03, 0.03, 11)[1::2] 
        y_range = np.linspace(-0.03, 0.03, 11)[1::2] 
        z_range = np.linspace(0, 0.08, 21)[1::2]
        x, y, z = np.meshgrid(x_range, y_range, z_range)

        positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        radii = np.digitize(positions[:, 2], np.linspace(0, 0.08, 11)) / 4000

        # Add a particle to the grid at cell boundary
        positions = np.vstack([positions, [-0.018001, -0.018001, 0.015999]])
        radii = np.append(radii, 0.005)

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
        
        # Store results
        cls.particle_data = particle_data
        cls.samples = samples
        cls.lacey = lacey
        cls.resolution = sample_resolution
        cls.split = split
        cls.cylinder_data = cylinder_data

    def test_lacey_mixing_index_polydisperse_2(self):
        """Test value of the Lacey mixing index."""

        # Number of particles in a row
        n_particles = 25
        r_1 = 0.00025
        r_insert = 0.005
        r_2 = 0.0005

        row_1_volume_1_1 = [(np.pi * r_1**3 * 4/3 
                             + np.pi * r_insert**3 * 4/3),
                            np.pi * r_2**3 * 4/3]
        row_1_volume = [(n_particles - 1) * np.pi * r_1**3 * 4/3, 
                        (n_particles - 1) * np.pi * r_2**3 * 4/3]
        
        r_3 = 0.00075
        r_4 = 0.001
        row_2_volume = [n_particles * np.pi * r_3**3 * 4/3, 
                        n_particles * np.pi * r_4**3 * 4/3]
        
        r_5 = 0.00125
        r_6 = 0.0015
        row_3_volume = [n_particles * np.pi * r_5**3 * 4/3, 
                        n_particles * np.pi * r_6**3 * 4/3]
        
        r_7 = 0.00175
        r_8 = 0.002
        row_4_volume = [n_particles * np.pi * r_7**3 * 4/3, 
                        n_particles * np.pi * r_8**3 * 4/3]
        
        r_9 = 0.00225
        r_10 = 0.0025
        row_5_volume = [n_particles * np.pi * r_9**3 * 4/3, 
                        n_particles * np.pi * r_10**3 * 4/3]
        
        type_0_volume = (row_1_volume_1_1[0]
                         + row_1_volume[0]
                         + row_2_volume[0]
                         + row_3_volume[0]
                         + row_4_volume[0]
                         + row_5_volume[0])
        
        type_1_volume = (row_1_volume_1_1[1]
                         + row_1_volume[1]
                         + row_2_volume[1]
                         + row_3_volume[1]
                         + row_4_volume[1]
                         + row_5_volume[1])
        
        total_volume = type_0_volume + type_1_volume

        bulk_conc = type_0_volume / total_volume
        unmixed_variance = bulk_conc * (1 - bulk_conc)

        mean_particle_volume = total_volume / 251
        mean_samples_volume = total_volume / 125
        particles_per_sample = mean_samples_volume / mean_particle_volume

        mixed_variance = unmixed_variance / particles_per_sample

        variance = 0
        for row in [row_1_volume_1_1, row_1_volume,
                    row_2_volume, row_3_volume,
                    row_4_volume, row_5_volume]:
            row_sum = sum(row)
            variance += (row_sum / total_volume 
                         * (row[0] / row_sum - bulk_conc) ** 2)
            
        expected_value = ((variance - unmixed_variance)
                          / (mixed_variance - unmixed_variance))

        assert np.isclose(self.lacey, expected_value, atol=1e-10)