import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_1d_volume_cylinder


class TestSample1DVolumeCylinder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        vtk_file_path = os.path.join(os.path.dirname(__file__),
                                     os.pardir, "vtks",)
        
        particle_data = pv.read(os.path.join(vtk_file_path,
                                           "particles.vtk"))
        
        point = [0, 0, 0]
        sample_vector = [2, 1, 3]
        # test with non normalised vector
        particle_data, split = sample_1d_volume_cylinder(particle_data,
                                                         point,
                                                         sample_vector,
                                                         resolution=10)
        
        cls.particle_data = particle_data
        cls.split = split

    def test_sample_1d_volume_cylinder_name(self):
        """Test column name in particle data."""

        normalised_vector = np.linalg.norm([2, 1, 3])
        sample_str = f"{2/normalised_vector}_{1/normalised_vector}_" \
                      f"{3/normalised_vector}_volume_sample"

        # check the particle data has the new column
        assert self.split.name == sample_str

    def test_sample_1d_volume_cylinder_ParticleAttribute(self):
        """Test ParticleAttribute in split data."""

        assert self.split.ParticleAttribute.field == "id"
        assert self.split.ParticleAttribute.attribute == self.split.name
        assert (self.split.ParticleAttribute.data.shape[0] 
                == len(self.particle_data.points))
        assert self.split.ParticleAttribute.data.shape[1] == 2


    def test_sample_1d_volume_cylinder_cells(self):
        """Test cells in split data."""

        assert self.split.cells.shape[0] == 10

        # check all cells are labelled correctly
        assert all(a == b for a, b in zip(self.split.cells, 
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                  )
        

    def test_sample_1d_volume_cylinder_occupied_cells(self):
        """Test occupied cells in split data."""

        assert self.split.occupied_cells.shape[0] == 10 


    def test_sample_1d_volume_cylinder_particles(self):
        """Test particles in each cell."""

        assert all(a == b for a, b in zip(self.split.particles,
                                          [8914, 8913, 8928, 8917, 8919, 
                                           8942, 8922, 8952, 8925, 8918])
                   )
        

    def test_sample_1d_volume_cylinder_sample_ids(self):
        """Test sample ids in particle data."""

        assert all(a == b for a, b in 
                   zip(self.particle_data[self.split.name][:10],
                                          [5, 5, 3, 3, 5, 0, 3, 4, 0, 0])
                   )


    def test_sample_1d_volume_cylinder_n_cells(self):
        """Test the number of cells in split data."""

        assert self.split.n_cells == 10


    def test_sample_1d_volume_cylinder_n_occupied_cells(self):
        """Test the number of occupied cells in split data."""

        assert self.split.n_occupied_cells == 10

    def test_sample_1d_volume_cylinder_n_sampled_particles(self):
        """Test the number of sampled particles in split data."""

        assert self.split.n_sampled_particles == 89250


    def test_sample_1d_volume_cylinder_n_unsampled_particles(self):
        """Test the number of unsampled particles in split data."""

        assert self.split.n_unsampled_particles == 0