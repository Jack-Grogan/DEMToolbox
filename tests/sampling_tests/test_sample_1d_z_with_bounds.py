import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d


def set_up_sample_1d_z_bounds_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                        os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    
    bounds = [-0.03, 0.03, -0.03, 0.03, -0.02, 0.08]
    
    sample_vector = [0, 0, 1]
    resolution = 10

    # test with non normalised vector
    particle_data, split = sample_1d(particle_data,
                                     bounds,
                                     sample_vector,
                                     resolution,
                                     append_column="sample_test")

    return particle_data, split


def test_sample_1d_z_bounds_benchmark(benchmark):
    benchmark(set_up_sample_1d_z_bounds_test)


class TestSample1DZBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_1d_z_bounds_test()        

        cls.particle_data = particle_data
        cls.split = split


    def test_sample_1d_z_bounds_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_1d_z_bounds_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_1d_z_bounds_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 10

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                  )
        

    def test_sample_1d_z_bounds_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [1, 2, 3, 4, 5, 6, 7, 8, 9])
                   )
        

    def test_sample_1d_z_bounds_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [0, 1, 1766, 16487, 18522, 
                                           19200, 18877, 13689, 704, 4])
                   )
        

    def test_sample_1d_z_bounds_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
                   )


    def test_sample_1d_z_bounds_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 10


    def test_sample_1d_z_bounds_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 9


    def test_sample_1d_z_bounds_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_1d_z_bounds_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0


    def test_sample_1d_z_bounds_vector_1_centers(self):
        """Test vector 1 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_centers,
                    [-0.015, -0.005, 0.005, 0.015, 0.025, 
                     0.035, 0.045, 0.055, 0.065, 0.075])
                   )
        
        assert len(self.split.vector_1_centers) == 10


    def test_sample_1d_z_bounds_vector_1_bounds(self):
        """Test vector 1 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_bounds,
                     [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03,
                      0.04, 0.05, 0.06, 0.07, 0.08])
                   )
        
        assert len(self.split.vector_1_bounds) == 11
        

    def test_sample_1d_z_bounds_vector_2_centers(self):
        """Test vector 2 cell centers in split data."""

        assert self.split.vector_2_centers is None


    def test_sample_1d_z_bounds_vector_2_bounds(self):
        """Test vector 2 bounds in split data."""

        assert self.split.vector_2_bounds is None


    def test_sample_1d_z_bounds_vector_3_centers(self):
        """Test vector 3 cell centers in split data."""

        assert self.split.vector_3_centers is None


    def test_sample_1d_z_bounds_vector_3_bounds(self):
        """Test vector 3 bounds in split data."""

        assert self.split.vector_3_bounds is None