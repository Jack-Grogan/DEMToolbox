import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_2d_slice


def set_up_sample_2d_slice_bounds_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))
    
    point = [0, 0, 0.04]
    vector_1 = [2, 3, 4]
    vector_2 = [1, -2, 1]
    plane_thickness = 0.01

    resolution = [4, 5]

    # test with non normalised vector
    particle_data, split = sample_2d_slice(particle_data,
                                            list(cylinder_data.bounds),
                                            point,
                                            vector_1,
                                            vector_2,
                                            plane_thickness,
                                            resolution,
                                            append_column="sample_test"
    )

    return particle_data, split


def test_sample_2d_slice_bounds_benchmark(benchmark):
    benchmark(set_up_sample_2d_slice_bounds_test)


class TestSample2DSliceBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_2d_slice_bounds_test()

        cls.particle_data = particle_data
        cls.split = split


    def test_sample_2d_slice_bounds_name(self):
        """Test column name in particle data."""

        # check the particle data has the new column
        assert self.split.name == "sample_test"


    def test_sample_2d_slice_bounds_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        # check the particle attribute data
        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == "sample_test"
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_2d_slice_bounds_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 20

        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                                           10, 11, 12, 13, 14, 15, 16, 
                                           17, 18, 19])
                  )
        

    def test_sample_2d_slice_bounds_occupied_cells(self):
        """Test occupied cells in split data."""
        
        assert all(a == b for a, b in zip(self.split.occupied_cells, 
                                          [1, 2, 4, 5, 6, 7, 8, 9, 10, 
                                           11, 12, 13, 14, 15, 17, 
                                           18])
                   )
        

    def test_sample_2d_slice_bounds_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [0, 313, 1, 0, 
                                           116, 4169, 2944, 19, 
                                           776, 4526, 4386, 216, 
                                           53, 2333, 2989, 3, 
                                           0, 3, 196, 0])
                   )
        

    def test_sample_2d_slice_bounds_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data["sample_test"][:10],
                        [-1, 8, -1, -1, 4, -1, 5, 5, -1, -1])
                   )


    def test_sample_2d_slice_bounds_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 20


    def test_sample_2d_slice_bounds_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 16


    def test_sample_2d_slice_bounds_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 23043


    def test_sample_2d_slice_bounds_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 66207