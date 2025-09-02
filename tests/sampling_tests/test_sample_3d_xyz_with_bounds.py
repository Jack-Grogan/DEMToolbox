import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d


def set_up_sample_2d_xyz_bounds_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))
    
    vector_1 = [1, 0, 0]
    vector_2 = [0, 1, 0]
    vector_3 = [0, 0, 1]

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


def test_sample_2d_xyz_bounds_benchmark(benchmark):
    benchmark(set_up_sample_2d_xyz_bounds_test)


class TestSample3DXYZBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_2d_xyz_bounds_test()
        
        cls.particle_data = particle_data
        cls.split = split
        

    def test_sample_2d_xyz_bounds_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_2d_xyz_bounds_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_2d_xyz_bounds_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 27

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26])
                  )
        

    def test_sample_2d_xyz_bounds_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26])
                   )
        

    def test_sample_2d_xyz_bounds_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [1640, 2545, 1452, 3713, 
                                           3237, 3739, 2082, 3084, 
                                           1887, 3440, 6525, 2534, 
                                           7619, 8319, 6320, 3995, 
                                           7696, 4020, 1268, 2639, 
                                           1187, 1923, 2767, 2317,
                                           558, 1671, 1073])
                   )
        

    def test_sample_2d_xyz_bounds_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                        [1, 0, 2, 2, 3, 5, 3, 3, 4, 4])
                   )
        

    def test_sample_2d_xyz_bounds_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 27


    def test_sample_2d_xyz_bounds_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 27


    def test_sample_2d_xyz_bounds_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_2d_xyz_bounds_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0