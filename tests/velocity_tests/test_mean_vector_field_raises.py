import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.velocity import mean_velocity_vector_field 


def test_mean_velocity_vector_field_invalid_velocity_vectors():
    """Test mean_velocity_vector_field with invalid velocity_vectors."""

    # Bottom left corner of particle grid
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
    velocity_vectors = [[velocity_vectors_1, velocity_vectors_2,
                        velocity_vectors_3, velocity_vectors_4]]
    
    occupancies = [occupancy_1, occupancy_2, 
                    occupancy_3, occupancy_4]

    with np.testing.assert_raises(ValueError) as context:
        mean_velocity_vector_field(
            velocity_vectors, occupancies)

    assert (str(context.exception) == 
        "velocity_vectors must be a 4D array with shape "
        "(n_frames, resolution[1], resolution[0], 2)."
    )

def test_mean_velocity_vector_field_invalid_occupancies():
    """Test mean_velocity_vector_field with invalid occupancies."""

    # Bottom left corner of particle grid
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
    
    occupancies = [[occupancy_1, occupancy_2, 
                    occupancy_3, occupancy_4]]

    with np.testing.assert_raises(ValueError) as context:
        mean_velocity_vector_field(
            velocity_vectors, occupancies)

    assert (str(context.exception) == 
        "occupancies must be a 3D array with shape "
        "(n_frames, resolution[1], resolution[0])."
    )

def test_mean_velocity_vector_field_mismatched_shapes():
    """Test mean_velocity_vector_field with missmatched shapes"""

    # Bottom left corner of particle grid
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
                    occupancy_3, occupancy_4, occupancy_4]

    with np.testing.assert_raises(ValueError) as context:
        mean_velocity_vector_field(
            velocity_vectors, occupancies)

    assert (str(context.exception) == 
        "velocity_vectors and occupancies must have the same "
        "shape for the first three dimensions."
    )

    