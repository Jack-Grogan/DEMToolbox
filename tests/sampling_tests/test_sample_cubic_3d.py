import pyvista as pv
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d

def set_up_sample_cubic_3d_test():

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
                                     cube_length=0.02,
                                     append_column="sample_test"
    )

    return particle_data, split


def test_sample_3d_benchmark(benchmark):
    benchmark(set_up_sample_cubic_3d_test)


class TestSample3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_cubic_3d_test()
        
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

        assert self.split.cells.shape[0] == 36

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26, 27, 28, 29, 
                                           30, 31, 32, 33, 34, 35])
                  )
        

    def test_sample_3d_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                           9, 10, 11, 12, 13, 14, 15, 
                                           16, 17, 18, 19, 20, 21, 22, 
                                           23, 24, 25, 26, 27, 28, 29, 
                                           30, 31, 32, 33, 34, 35])
                   )
        

    def test_sample_3d_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [789, 1145, 833, 1874, 1434, 
                                           1965, 1066, 1163, 879, 2388, 
                                           4523, 2010, 5525, 5857, 5043, 
                                           3068, 5861, 2988, 2902, 5174, 
                                           1735, 5668, 6291, 4563, 2411, 
                                           5027, 2867, 269, 867, 595, 
                                           188, 742, 805, 90, 399, 246])
                   )
        

    def test_sample_3d_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                        [1, 0, 2, 2, 3, 5, 3, 3, 4, 4])
                   )
        

    def test_sample_3d_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 36


    def test_sample_3d_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 36


    def test_sample_3d_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_3d_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0


    def test_sample_3d_vector_1_centers(self):
        """Test vector 1 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_centers,
                       [-0.02, 0.00, 0.02])
                    )
        
        assert len(self.split.vector_1_centers) == 3


    def test_sample_3d_vector_1_bounds(self):
        """Test vector 1 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_bounds,
                        [-0.03, -0.01, 0.01, 0.03])
                   )
        
        assert len(self.split.vector_1_bounds) == 4


    def test_sample_3d_vector_2_centers(self):
        """Test vector 2 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_2_centers,
                       [-0.02, 0.00, 0.02])
                    )
        
        assert len(self.split.vector_2_centers) == 3


    def test_sample_3d_vector_2_bounds(self):
        """Test vector 2 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_2_bounds,
                        [-0.03, -0.01, 0.01, 0.03])
                   )
        
        assert len(self.split.vector_2_bounds) == 4


    def test_sample_3d_vector_3_centers(self):
        """Test vector 3 cell centers in split data."""

        print(self.split.vector_3_centers)
        assert all(round(a, 5) == b for a, b 
                   in zip(self.split.vector_3_centers,
                       [0.00611, 0.02611, 0.04611, 0.06611])
                    )

        assert len(self.split.vector_3_centers) == 4


    def test_sample_3d_vector_3_bounds(self):
        """Test vector 3 bounds in split data."""

        print(self.split.vector_3_bounds)
        assert all(round(a, 5) == b for a, b 
                   in zip(self.split.vector_3_bounds,
                        [-0.00389, 0.01611, 0.03611, 0.05611, 0.07611])
                   )
        assert len(self.split.vector_3_bounds) == 5