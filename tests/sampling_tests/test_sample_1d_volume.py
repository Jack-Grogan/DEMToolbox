import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d_volume


class TestSample1DVolume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        vtk_file_path = os.path.join(os.path.dirname(__file__),
                                     os.pardir, "vtks",)
        
        particle_data = pv.read(os.path.join(vtk_file_path,
                                           "particles.vtk"))
        
        sample_vector = [8, 1, 4]
        # test with non normalised vector
        particle_data, split = sample_1d_volume(particle_data,
                                                sample_vector,
                                                5)
                                                

        cls.particle_data = particle_data
        cls.split = split


    def test_sample_1d_volume_name(self):
        """Test column name in particle data."""

        normalised_vector = np.linalg.norm([8, 1, 4])
        sample_str = (f"{8/normalised_vector}_{1/normalised_vector}_"
                      f"{4/normalised_vector}_volume_sample")

        # check the particle data has the new column
        assert self.split.name == sample_str


    def test_sample_1d_volume_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == self.split.name
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_1d_volume_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 5

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4])
                  )
        

    def test_sample_1d_volume_occupied_cells(self):
        """Test occupied cells in split data."""

        assert self.split.occupied_cells.shape[0] == 5 

    
    def test_sample_1d_volume_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [17879, 17842, 17835, 17855, 17839])
                   )
        

    def test_sample_1d_volume_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data[self.split.name][:10],
                    [0, 0, 1, 2, 0, 1, 0, 0, 2, 1])
                   )


    def test_sample_1d_volume_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 5


    def test_sample_1d_volume_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 5


    def test_sample_1d_volume_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_1d_volume_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0