import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d


class TestSample1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        vtk_file_path = os.path.join(os.path.dirname(__file__),
                                     os.pardir, "vtks",)
        
        particle_data = pv.read(os.path.join(vtk_file_path,
                                           "particles.vtk"))
        cylinder_data = pv.read(os.path.join(vtk_file_path,
                                           "mesh.vtk"))
        
        sample_vector = [8, 1, 4]
        resolution = 21

        # test with non normalised vector
        particle_data, split = sample_1d(particle_data,
                                         cylinder_data,
                                         sample_vector,
                                         resolution,
                                         append_column="sample_test")

        cls.particle_data = particle_data
        cls.split = split


    def test_sample_1d_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_1d_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_1d_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 21

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6])
                  )
        

    def test_sample_1d_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 
                                           10, 11, 12, 13, 14, 15, 16, 
                                           17, 18, 19])
                   )
        

    def test_sample_1d_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [0, 78, 796, 1912, 3070, 4803, 
                                           6572, 7662, 8337, 8901, 9095, 
                                           9110, 8685, 7520, 5730, 3733,
                                           2185, 889, 169, 3, 0])
                   )
        

    def test_sample_1d_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                    [6, 3, 8, 9, 1, 9, 2, 2, 9, 9])
                   )


    def test_sample_1d_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 21


    def test_sample_1d_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 19


    def test_sample_1d_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_1d_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0