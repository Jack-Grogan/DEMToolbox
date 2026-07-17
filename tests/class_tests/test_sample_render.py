import pyvista as pv
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.particle_sampling import sample_3d


def set_up_sample_3d_bounds_test():

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


def test_sample_3d_bounds_benchmark(benchmark):
    benchmark(set_up_sample_3d_bounds_test)


class TestSample3DBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, split = set_up_sample_3d_bounds_test()
        
        cls.particle_data = particle_data
        cls.split = split

        
    def test_sample_3d_to_vtm(self):
        """Test the to_vtm method of the Sample3D class."""

        output_dir = os.path.join(os.path.dirname(__file__), "generated_test_files")
        os.makedirs(output_dir, exist_ok=True)

        vtm_file = os.path.join(output_dir, "samples.vtm")

        self.split.to_vtm(filename=vtm_file)
        self.particle_data.save(os.path.join(output_dir, "particles.vtp"))

        assert os.path.exists(vtm_file)
        assert os.path.getsize(vtm_file) > 0


    def test_sample_3d_save(self):
        """Test the save method of the Sample3D class."""

        output_dir = os.path.join(os.path.dirname(__file__), "generated_test_files")
        os.makedirs(output_dir, exist_ok=True)

        vtk_file = os.path.join(output_dir, "meshgrid.vtk")

        self.split.save(filename=vtk_file)

        assert os.path.exists(vtk_file)
        assert os.path.getsize(vtk_file) > 0

    # Commented out as generating files in the root directory is not ideal for testing purposes.

    # def test_sample_3d_to_vtm_autoname(self):
    #     """Test the to_vtm method of the Sample3D class."""

    #     self.split.to_vtm()

    #     assert os.path.exists("sample_test.vtm")
    #     assert os.path.getsize("sample_test.vtm") > 0


    # def test_sample_3d_save_autoname(self):
    #     """Test the save method of the Sample3D class."""

    #     self.split.save()

    #     assert os.path.exists("sample_test.vtk")
    #     assert os.path.getsize("sample_test.vtk") > 0
     