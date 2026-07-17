import pyvista as pv
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d

def set_up_sample_3d_cubic_with_bounds_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))

    cylinder_data = list(cylinder_data.bounds)
    
    vector_1 = [1, 0, 0]
    vector_2 = [0, 1, 0]
    vector_3 = [0, 0, 1]

    # test with non normalised vector
    particle_data, split = sample_3d(particle_data,
                                     cylinder_data,
                                     vector_1,
                                     vector_2,
                                     vector_3,
                                     cube_length=0.05,
                                     center_meshgrid=True,
                                     append_column="sample_test"
    )

    return particle_data, split


def test_sample_3d_benchmark(benchmark):
    benchmark(set_up_sample_3d_cubic_with_bounds_test)


class TestSample3DCubic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_3d_cubic_with_bounds_test()
        
        cls.particle_data = particle_data
        cls.split = split
        

    def test_sample_3d_cubic_with_bounds_name(self):
        """Test column name in particle data."""

        assert self.split.name == "sample_test"


    def test_sample_3d_cubic_with_bounds_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_3d_cubic_with_bounds_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 8

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7])
                  )
        

    def test_sample_3d_cubic_with_bounds_occupied_cells(self):
        """Test occupied cells in split data."""

        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7])
                   )
        

    def test_sample_3d_cubic_with_bounds_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [11049, 10676, 13477, 13209, 
                                           11019, 9779, 9604, 10437])
                   )
        

    def test_sample_3d_cubic_with_bounds_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                        [0, 0, 1, 1, 2, 3, 2, 2, 3, 3])
                   )
        

    def test_sample_3d_cubic_with_bounds_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 8


    def test_sample_3d_cubic_with_bounds_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 8


    def test_sample_3d_cubic_with_bounds_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_3d_cubic_with_bounds_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0


    def test_sample_3d_cubic_with_bounds_vector_1_centers(self):
        """Test vector 1 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_centers,
                       [-0.025, 0.025])
                    )
        
        assert len(self.split.vector_1_centers) == 2


    def test_sample_3d_cubic_with_bounds_vector_1_bounds(self):
        """Test vector 1 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_1_bounds,
                        [-0.05, 0.0, 0.05])
                   )
        
        assert len(self.split.vector_1_bounds) == 3


    def test_sample_3d_cubic_with_bounds_vector_2_centers(self):
        """Test vector 2 cell centers in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_2_centers,
                       [-0.025, 0.025])
                    )
        
        assert len(self.split.vector_2_centers) == 2


    def test_sample_3d_cubic_with_bounds_vector_2_bounds(self):
        """Test vector 2 bounds in split data."""

        assert all(round(a, 8) == b for a, b 
                   in zip(self.split.vector_2_bounds,
                        [-0.05, 0.0, 0.05])
                   )
        
        assert len(self.split.vector_2_bounds) == 3


    def test_sample_3d_cubic_with_bounds_vector_3_centers(self):
        """Test vector 3 cell centers in split data."""

        assert all(round(a, 5) == b for a, b 
                   in zip(self.split.vector_3_centers,
                       [0.01111, 0.06111])
                    )

        assert len(self.split.vector_3_centers) == 2


    def test_sample_3d_cubic_with_bounds_vector_3_bounds(self):
        """Test vector 3 bounds in split data."""

        assert all(round(a, 5) == b for a, b 
                   in zip(self.split.vector_3_bounds,
                        [-0.01389, 0.03611, 0.08611])
                   )
        assert len(self.split.vector_3_bounds) == 3