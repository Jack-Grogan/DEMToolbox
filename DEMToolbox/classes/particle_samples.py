import numpy as np
import pyvista as pv
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

    Methods
    -------
    save(filename=None):
        Save the sampled cells as a StructuredGrid in 3D space. If filename 
        is provided, the StructuredGrid will be saved as a .vtk file with the 
        specified name. If filename is not provided, the StructuredGrid will be 
        saved as a .vtk file with the name of the samples column.
    to_vtm(filename=None):
        Render the sampled cells as cubes in 3D space. If filename 
        is provided, the rendered cubes will be saved as a .vtm file 
        with the specified name. If filename is not provided, 
        the rendered cubes will be saved as a .vtm file with the 
        name of the samples column. vtp files are saved for each 
        individual cube in the sample space, in a folder named after 
        the samples column.
    """
    def __init__(self, 
                 name, 
                 sample_attribute:ParticleAttribute, 
                 cells, 
                 occupied_cells,
                 particles, 
                 n_sampled_particles, 
                 n_unsampled_particles,
                 vector_1=None,
                 vector_1_centers=None,
                 vector_1_bounds=None,
                 vector_2=None,
                 vector_2_centers=None,
                 vector_2_bounds=None,
                 vector_3=None,
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
        self.vector_1 = vector_1
        self.vector_1_centers = vector_1_centers
        self.vector_1_bounds = vector_1_bounds
        self.vector_2 = vector_2
        self.vector_2_centers = vector_2_centers
        self.vector_2_bounds = vector_2_bounds
        self.vector_3 = vector_3
        self.vector_3_centers = vector_3_centers
        self.vector_3_bounds = vector_3_bounds


    def save(self, filename=None):
        """Save the sampled cells as a StructuredGrid in 3D space. If filename 
        is provided, the StructuredGrid will be saved as a .vtk file with the 
        specified name. If filename is not provided, the StructuredGrid will be 
        saved as a .vtk file with the name of the samples column.
        """
        z, y, x = np.meshgrid(self.vector_3_bounds,
                              self.vector_2_bounds,
                              self.vector_1_bounds,
                              indexing='ij'
        )

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = np.array([self.vector_1, 
                                             self.vector_2, 
                                             self.vector_3]).T

        meshgrid = pv.StructuredGrid(x, y, z)
        meshgrid.transform(rotation_matrix, inplace=True)

        # Save the meshgrid to a .vtm file if filename is provided, 
        # otherwise save with the name of the samples column
        if filename is not None:
            meshgrid.save(filename)
        else:
            meshgrid.save(f"{self.name}.vtk") # pragma: no cover
        return
    

    def to_vtm(self, filename=None):

        z, y, x = np.meshgrid(self.vector_3_centers,
                              self.vector_2_centers,
                              self.vector_1_centers,
                              indexing='ij'
        )

        cube_centers = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = np.array([self.vector_1, 
                                             self.vector_2, 
                                             self.vector_3]).T
        data = []
        for i, (x, y, z) in enumerate(cube_centers):
            cube = pv.Cube(
                center=(x, y, z),
                x_length=self.vector_1_bounds[1] - self.vector_1_bounds[0],
                y_length=self.vector_2_bounds[1] - self.vector_2_bounds[0],
                z_length=self.vector_3_bounds[1] - self.vector_3_bounds[0],
            )

            # Rotate the blocks to align with the original vectors
            cube.transform(rotation_matrix, inplace=True)
            cube["id"] = np.full(cube.n_cells, i)

            data.append(cube)

        blocks = pv.MultiBlock(data)

        # Save the blocks to a .vtm file if filename is provided, 
        # otherwise save with the name of the samples column
        if filename is not None:
            blocks.save(filename)
        else:
            blocks.save(f"{self.name}.vtm") # pragma: no cover
        return