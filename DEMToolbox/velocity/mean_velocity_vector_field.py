import numpy as np

def mean_velocity_vector_field(velocity_vectors, occupancies):
    """
    Calculate the mean velocity vector field from the velocity vectors.
    mean velocity is calculated from the mean particle velocities in a 
    given sample averaged over the frames weighted by the samples 
    particle occupancies in each frame.

    Parameters
    ----------
    velocity_vectors : np.ndarray
        An array of the velocity vectors for particles in the sample 
        space. Shape should be (n_frames, resolution[1], 
        resolution[0], 2).
    occupancies : np.ndarray
        An array of the occupancy values for each point in the sample 
        space. Shape should be (n_frames, resolution[1], resolution[0]).

    Returns
    -------
    np.ndarray
        An array of the mean velocity vectors for each point in the 
        sample space. Shape will be (resolution[1], resolution[0], 2).
    """
    if velocity_vectors.ndim != 4 or velocity_vectors.shape[-1] != 2:
        raise ValueError("velocity_vectors must be a 4D array with shape "
                         "(n_frames, resolution[1], resolution[0], 2).")
    
    if occupancies.ndim != 3 or occupancies.shape[-1] != 1:
        raise ValueError("occupancies must be a 3D array with shape "
                         "(n_frames, resolution[1], resolution[0]).")
    
    if velocity_vectors.shape[:3] != occupancies.shape:
        raise ValueError("velocity_vectors and occupancies must have the same "
                         "shape for the first three dimensions.")

    velocity_vectors = np.asarray(velocity_vectors)
    occupancies = np.asarray(occupancies)

    mean_velocity = np.zeros_like(velocity_vectors[0])
    mean_velocity[:] = np.nan
    for i in range(mean_velocity.shape[0]):
        for j in range(mean_velocity.shape[1]):
            if np.sum(occupancies[:, i, j]) > 0:
                mean_velocity[i, j] = np.nansum(
                    *[velocity_vectors[:, i, j] 
                     * occupancies[:, i, j][:, np.newaxis]
                    ], axis=0) / np.sum(occupancies[:, i, j])

    return mean_velocity
