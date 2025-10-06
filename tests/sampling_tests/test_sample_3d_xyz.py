import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d


def set_up_sample_2d_xyz_test():

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
                                     cylinder_data,
                                     vector_1,
                                     vector_2,
                                     vector_3,
                                     resolution,
                                     append_column="sample_test"
    )

    return particle_data, split


def test_sample_2d_xyz_benchmark(benchmark):
    benchmark(set_up_sample_2d_xyz_test)


class TestSample3DXYZ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_2d_xyz_test()
        
        cls.particle_data = particle_data
        cls.split = split
        

    def test_sample_2d_xyz_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_2d_xyz_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_2d_xyz_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 27

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26])
                  )
        

    def test_sample_2d_xyz_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26])
                   )
        

    def test_sample_2d_xyz_particles(self):
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
        

    def test_sample_2d_xyz_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                        [1, 0, 2, 2, 3, 5, 3, 3, 4, 4])
                   )
        

    def test_sample_2d_xyz_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 27


    def test_sample_2d_xyz_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 27


    def test_sample_2d_xyz_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_2d_xyz_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0


    def test_sample_2d_xyz_vector_1_centers(self):
        """Test vector 1 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_centers,
                       [-0.02, 0.00, 0.02])
                    )
        assert len(self.split.vector_1_centers) == 3


    def test_sample_2d_xyz_vector_1_bounds(self):
        """Test vector 1 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_bounds,
                        [-0.03, -0.01, 0.01, 0.03])
                   )
        
        assert len(self.split.vector_1_bounds) == 4


    def test_sample_2d_xyz_vector_2_centers(self):
        """Test vector 2 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_2_centers,
                        [-0.02, 0.00, 0.02])
                   )
        
        assert len(self.split.vector_2_centers) == 3


    def test_sample_2d_xyz_vector_2_bounds(self):
        """Test vector 2 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_2_bounds,
                        [-0.03, -0.01, 0.01, 0.03])
                   )
        
        assert len(self.split.vector_2_bounds) == 4


    def test_sample_2d_xyz_vector_3_centers(self):
        """Test vector 3 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_3_centers,
                     [0.00944334, 0.03611, 0.06277667])
                   )
        
        assert len(self.split.vector_3_centers) == 3


    def test_sample_2d_xyz_vector_3_bounds(self):
        """Test vector 3 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_3_bounds,
                     [-0.00388999, 0.02277667, 0.04944334, 0.07611])
                   )
        
        assert len(self.split.vector_3_bounds) == 4