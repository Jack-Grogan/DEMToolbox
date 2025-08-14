import numpy as np
from .particle_attribute import ParticleAttribute

class ParticleSamples():
    """Class to store particle sample information.

    Attributes
    ----------
    name: (str)
        Name of the samples column
    sample_attribute: (ParticleAttribute)
        ParticleAttribute object containing the particle ids and their
        corresponding sample ids
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
    data: (np.ndarray)
        Array of particle ids and their corresponding sample ids
    """
    def __init__(self, 
                 name, 
                 sample_attribute:ParticleAttribute, 
                 cells, 
                 occupied_cells,
                 particles, 
                 n_sampled_particles, 
                 n_unsampled_particles):
        
        self.name = name
        self.ParticleAttribute = sample_attribute
        self.cells = np.asarray(cells)
        self.occupied_cells = np.asarray(occupied_cells)
        self.particles = np.asarray(particles)
        self.n_cells = np.size(cells)
        self.n_occupied_cells = np.size(occupied_cells)
        self.n_sampled_particles = n_sampled_particles
        self.n_unsampled_particles = n_unsampled_particles
