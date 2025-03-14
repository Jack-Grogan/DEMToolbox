import numpy as np
class ParticleSamples():
    """Class to store particle sample information.

    Attributes
    ----------
    name: (str)
        Name of the samples column
    cells: (np.ndarray)
        Array of possible cell ids
    occupied_cells: (np.ndarray)
        Array of cell ids that contain particles
    particles: (np.ndarray)
        Array of number of particles in each cell
    n_sampled_particles: (int)
        Number of particles in the sampled cells
    n_unsampled_particles: (int)
        Number of particles not in the sampled cells
    sample_df: (pd.DataFrame) optional
        DataFrame containing the sample information. Default is None.
    """
    def __init__(self, name, cells, occupied_cells,
                 particles, n_sampled_particles, n_unsampled_particles, 
                 sample_df = None):
        self.name = name
        self.cells = cells
        self.occupied_cells = occupied_cells
        self.particles = particles
        self.n_cells = np.size(cells)
        self.n_occupied_cells = np.size(occupied_cells)
        self.n_sampled_particles = n_sampled_particles
        self.n_unsampled_particles = n_unsampled_particles
        self.sample_df = sample_df
