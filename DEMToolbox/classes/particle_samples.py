import numpy as np
from .particle_attribute import ParticleAttribute

class ParticleSamples():
    """Class to store particle sample information.

    Attributes
    ----------
    name: (str)
        Name of the samples column
    ParticleAttribute: (ParticleAttribute)
        ParticleAttribute object containing the particle ids and their
        corresponding sample ids
    cells: (np.ndarray)
        Array of possible cell ids
    occupied_cells: (np.ndarray)
        Array of cell ids that contain particles
    particles: (np.ndarray)
        Array of number of particles in each cell
    n_cells: (int)
        Number of possible cells
    n_occupied_cells: (int)
        Number of cells that contain particles
    n_sampled_particles: (int)
        Number of particles in the sampled cells
    n_unsampled_particles: (int)
        Number of particles not in the sampled cells
    vector_1_centers: (np.ndarray)
        1D array of the vector 1 cell centers in the sample space
    vector_1_bounds: (np.ndarray)
        Array of the vector 1 bounds in the sample space
    vector_2_centers: (np.ndarray)
        1D array of the vector 2 cell centers in the sample space
    vector_2_bounds: (np.ndarray)
        Array of the vector 2 bounds in the sample space
    vector_3_centers: (np.ndarray)
        1D array of the vector 3 cell centers in the sample space
    vector_3_bounds: (np.ndarray)
        Array of the vector 3 bounds in the sample space
    """
    def __init__(self, 
                 name, 
                 sample_attribute:ParticleAttribute, 
                 cells, 
                 occupied_cells,
                 particles, 
                 n_sampled_particles, 
                 n_unsampled_particles,
                 vector_1_centers=None,
                 vector_1_bounds=None,
                 vector_2_centers=None,
                 vector_2_bounds=None,
                 vector_3_centers=None,
                 vector_3_bounds=None,
                 ):
        
        self.name = name
        self.ParticleAttribute = sample_attribute
        self.cells = np.asarray(cells)
        self.occupied_cells = np.asarray(occupied_cells)
        self.particles = np.asarray(particles)
        self.n_cells = np.size(cells)
        self.n_occupied_cells = np.size(occupied_cells)
        self.n_sampled_particles = n_sampled_particles
        self.n_unsampled_particles = n_unsampled_particles
        self.vector_1_centers = vector_1_centers
        self.vector_1_bounds = vector_1_bounds
        self.vector_2_centers = vector_2_centers
        self.vector_2_bounds = vector_2_bounds
        self.vector_3_centers = vector_3_centers
        self.vector_3_bounds = vector_3_bounds