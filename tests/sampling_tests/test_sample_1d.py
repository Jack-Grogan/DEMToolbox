import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d


def set_up_sample_1d_test():

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
    
    return particle_data, split


def test_sample_1d_benchmark(benchmark):
    benchmark(set_up_sample_1d_test)


class TestSample1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_1d_test()        

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


    def test_sample_1d_vector_1_centers(self):
        """Test vector 1 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_centers,
                     [-0.02645526, -0.02220484, -0.01795443, 
                      -0.01370401, -0.0094536, -0.00520318,
                      -0.00095277, 0.00329765, 0.00754806, 
                      0.01179848, 0.01604889, 0.02029931,
                      0.02454972, 0.02880014, 0.03305055, 
                      0.03730097, 0.04155138, 0.0458018,
                      0.05005221, 0.05430263, 0.05855304])
                   )
        
        assert len(self.split.vector_1_centers) == 21
        

    def test_sample_1d_vector_1_bounds(self):
        """Test vector 1 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_bounds,
                     [-0.02858047, -0.02433005, -0.02007964,
                      -0.01582922, -0.01157881, -0.00732839,
                      -0.00307798, 0.00117244, 0.00542285, 
                      0.00967327, 0.01392368, 0.0181741,
                      0.02242451, 0.02667493, 0.03092534, 
                      0.03517576, 0.03942617, 0.04367659,
                      0.047927, 0.05217742, 0.05642783, 0.06067825])
                   )
        
        assert len(self.split.vector_1_bounds) == 22

    def test_sample_1d_vector_2_centers(self):
        """Test vector 2 cell centers in split data."""

        assert self.split.vector_2_centers is None


    def test_sample_1d_vector_2_bounds(self):
        """Test vector 2 bounds in split data."""

        assert self.split.vector_2_bounds is None


    def test_sample_1d_vector_3_centers(self):
        """Test vector 3 cell centers in split data."""

        assert self.split.vector_3_centers is None


    def test_sample_1d_vector_3_bounds(self):
        """Test vector 3 bounds in split data."""

        assert self.split.vector_3_bounds is None