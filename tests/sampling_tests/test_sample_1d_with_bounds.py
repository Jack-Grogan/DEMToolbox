import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d


def set_up_sample_1d_bounds_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                        os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    
    bounds = [-0.03, 0.03, -0.03, 0.03, -0.00388999, 0.07611]
    
    sample_vector = [8, 1, 4]
    resolution = 21

    # test with non normalised vector
    particle_data, split = sample_1d(particle_data,
                                     bounds,
                                     sample_vector,
                                     resolution,
                                     append_column="sample_test")

    return particle_data, split


def test_sample_1d_bounds_benchmark(benchmark):
    benchmark(set_up_sample_1d_bounds_test)


class TestSample1DBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_1d_bounds_test()        

        cls.particle_data = particle_data
        cls.split = split


    def test_sample_1d_bounds_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_1d_bounds_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_1d_bounds_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 21

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6])
                  )
        

    def test_sample_1d_bounds_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 
                                           10, 11, 12, 13, 14, 15, 16, 
                                           17, 18, 19])
                   )
                

    def test_sample_1d_bounds_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [0, 4, 307, 1468, 2722, 4425, 
                                           6528, 8038, 8780, 9496, 9790,
                                           9721, 9153, 7713, 5492, 3389, 
                                           1700, 492, 32, 0,0])
                   )
        

    def test_sample_1d_bounds_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                    [6, 3, 9, 10, 1, 9, 3, 2, 9, 9])
                   )


    def test_sample_1d_bounds_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 21


    def test_sample_1d_bounds_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 18


    def test_sample_1d_bounds_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_1d_bounds_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0


    def test_sample_1d_bounds_vector_1_centers(self):
        """Test vector 1 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_centers,
                     [-0.02945375, -0.02490349, -0.02035322,
                      -0.01580296, -0.01125269, -0.00670243,
                      -0.00215217, 0.0023981, 0.00694836,
                      0.01149863, 0.01604889, 0.02059916,
                      0.02514942, 0.02969968, 0.03424995, 
                      0.03880021, 0.04335048, 0.04790074,
                      0.05245101, 0.05700127, 0.06155153])
                   )
        
        assert len(self.split.vector_1_centers) == 21


    def test_sample_1d_bounds_vector_1_bounds(self):
        """Test vector 1 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_bounds,
                     [-0.03172888, -0.02717862, -0.02262836,
                      -0.01807809, -0.01352783, -0.00897756,
                      -0.0044273, 0.00012297, 0.00467323,
                      0.00922349, 0.01377376, 0.01832402,
                      0.02287429, 0.02742455, 0.03197482, 
                      0.03652508, 0.04107534, 0.04562561,
                      0.05017587, 0.05472614, 0.0592764, 0.06382667])
                   )
        
        assert len(self.split.vector_1_bounds) == 22
        

    def test_sample_1d_bounds_vector_2_centers(self):
        """Test vector 2 cell centers in split data."""

        assert self.split.vector_2_centers is None


    def test_sample_1d_bounds_vector_2_bounds(self):
        """Test vector 2 bounds in split data."""

        assert self.split.vector_2_bounds is None


    def test_sample_1d_bounds_vector_3_centers(self):
        """Test vector 3 cell centers in split data."""

        assert self.split.vector_3_centers is None


    def test_sample_1d_bounds_vector_3_bounds(self):
        """Test vector 3 bounds in split data."""

        assert self.split.vector_3_bounds is None