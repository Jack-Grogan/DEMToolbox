import numpy as np

class Mesh():
    """Class to store mesh information.

    Attributes
    ----------
    name: (str)
        Name of the mesh.
    cells: (np.ndarray)
        Array of cell coordinates.
    occupied_cells: (np.ndarray)
        Array of occupied cell coordinates.
    n_cells: (int)
        Number of cells in the mesh.
    n_occupied_cells: (int)
        Number of occupied cells in the mesh.
    n_meshed_particles: (int)
        Number of particles that are within the mesh container
    n_unmeshed_particles: (int)
        Number of particles that are outside the mesh container
    mesh_df: (pd.DataFrame)
        DataFrame containing the mesh information.
    """
    def __init__(self, name, cells, occupied_cells, n_meshed_particles,
                 n_unmeshed_particles, mesh_df):
        self.name = name
        self.cells = cells
        self.occupied_cells = occupied_cells
        self.n_cells = np.size(cells)
        self.n_occupied_cells = np.size(occupied_cells)
        self.n_meshed_particles = n_meshed_particles
        self.n_unmeshed_particles = n_unmeshed_particles
        self.mesh_df = mesh_df
