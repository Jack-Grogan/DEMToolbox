import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.velocity import mean_velocity_vector_field 


class TestVectorFields(unittest.TestCase):
    def test_mean_velocity_vector_field_1(self):
        """Test the mean velocity vector field calculation."""


        # Bottom left corner of paticle grid
        velocity_vectors_1 = np.zeros((10, 10, 2))
        velocity_vectors_1[:5, :5, :] = [0., -0.003]
        occupancy_1 = np.zeros((10, 10))
        occupancy_1[:5, :5] = 1

        # Bottom right corner of particle grid
        velocity_vectors_2 = np.zeros((10, 10, 2))
        velocity_vectors_2[:5, 5:, :] = [0., -0.003]
        occupancy_2 = np.zeros((10, 10))
        occupancy_2[:5, 5:] = 1

        # Top left corner of particle grid
        velocity_vectors_3 = np.zeros((10, 10, 2))
        velocity_vectors_3[5:, :5, :] = [0., -0.003]
        occupancy_3 = np.zeros((10, 10))
        occupancy_3[5:, :5] = 1 

        # Top right corner of particle grid
        velocity_vectors_4 = np.zeros((10, 10, 2))
        velocity_vectors_4[5:, 5:, :] = [0., -0.003]
        occupancy_4 = np.zeros((10, 10))
        occupancy_4[5:, 5:] = 1

        # Combine all corners
        velocity_vectors = [velocity_vectors_1, velocity_vectors_2,
                            velocity_vectors_3, velocity_vectors_4]
        
        occupancies = [occupancy_1, occupancy_2, 
                       occupancy_3, occupancy_4]
        
        mean_velocity_vectors = mean_velocity_vector_field(
                                velocity_vectors, occupancies)

        expected_velocity = np.zeros((10, 10, 2))
        expected_velocity[:, :, :] = [0., -0.003]

        assert np.all(mean_velocity_vectors == expected_velocity)

    def test_mean_velocity_vector_field_2(self):
        """Test the mean velocity vector field calculation with different occupancy."""
        
        # Frame 1 vectors and occupancy
        velocity_vectors_1 = np.zeros((10, 10, 2))
        velocity_vectors_1[:, :, :] = [0.003, -0.003]
        occupancy_1 = np.zeros((10, 10))
        occupancy_1[:, :] = 2

        # Frame 2 vectors and occupancy
        velocity_vectors_2 = np.zeros((10, 10, 2))
        velocity_vectors_2[:, :, :] = [0.006, 0.]
        occupancy_2 = np.zeros((10, 10))
        occupancy_2[:, :] = 1

        # Combine all corners
        velocity_vectors = [velocity_vectors_1, velocity_vectors_2]
        occupancies = [occupancy_1, occupancy_2]

        mean_velocity_vectors = mean_velocity_vector_field(
                                velocity_vectors, occupancies)

        expected_velocity = np.zeros((10, 10, 2))
        expected_velocity[:, :, :] = [0.004, -0.002]

        assert np.all(mean_velocity_vectors == expected_velocity)

if __name__ == "__main__":
    unittest.main()
