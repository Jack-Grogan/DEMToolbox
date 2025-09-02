import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d


def set_up_sample_3d_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))
    
    vector_1 = [1, 2, -1]
    vector_2 = [4, -1, 2]
    vector_3 = [1, -2, -3]

    resolution = [3, 3, 3]

    # test with non normalised vector
    particle_data, split = sample_3d(particle_data,
                                     list(cylinder_data.bounds),
                                     vector_1,
                                     vector_2,
                                     vector_3,
                                     resolution,
                                     append_column="sample_test"
    )

    return particle_data, split


def test_sample_3d_benchmark(benchmark):
    benchmark(set_up_sample_3d_test)


class TestSample3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_3d_test()
        
        cls.particle_data = particle_data
        cls.split = split
        

    def test_sample_3d_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_3d_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_3d_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 27

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26])
                  )
        

    def test_sample_3d_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [0, 1, 3, 4, 5, 7, 8, 9, 10, 
                                           11, 12, 13, 14, 15, 16, 17,
                                           18, 19, 21, 22, 23, 25, 26])
                   )
        

    def test_sample_3d_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [15, 523, 0, 130, 4395, 
                                           269, 0, 238, 11, 366, 
                                           11590, 1369, 7933, 33717,
                                           9934, 901, 7627, 401, 22, 
                                           775, 0, 262, 7241, 1228, 
                                           0, 282, 21])
                   )
        

    def test_sample_3d_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                        [22, 19, 22, 22, 10, 23, 10, 10, 23, 23])
                   )
        

    def test_sample_3d_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 27


    def test_sample_3d_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 23


    def test_sample_3d_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_3d_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0