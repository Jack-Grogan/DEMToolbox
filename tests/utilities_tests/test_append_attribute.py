import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.utilities import append_attribute
from DEMToolbox.classes import ParticleAttribute


def set_up_append_attribute_test():

    vtk_file_path = os.path.join(os.path.dirname(__file__),
                                    os.pardir, "vtks",)
    
    particle_data = pv.read(os.path.join(vtk_file_path,
                                        "particles.vtk"))
    cylinder_data = pv.read(os.path.join(vtk_file_path,
                                        "mesh.vtk"))

    return particle_data, cylinder_data


class TestAppendAttribute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""

        particle_data, cylinder_data = set_up_append_attribute_test()
        
        cls.particle_data = particle_data
        cls.cylinder_data = cylinder_data

    def test_append_attribute(self):
        """Test append_attribute function."""

        particle_data = self.particle_data.copy()
        field = particle_data["id"]
        attribute = np.random.rand(len(field))
        data = np.column_stack((field, attribute))

        # Create a ParticleAttributes object
        attribute = ParticleAttribute(
            field="id",
            attribute="test_attribute",
            data=data,
        )

        # Append the attribute to the particle data
        updated_particle_data = append_attribute(
            particle_data, attribute
        )

        # Check if the attribute was appended correctly
        assert "test_attribute" in updated_particle_data.point_data
        assert np.array_equal(updated_particle_data["test_attribute"], 
                              attribute.data[:, 1])

    def test_append_attribute_invalid_field(self):
        """Test append_attribute with invalid field."""

        particle_data = self.particle_data.copy()
        field = particle_data["id"]
        attribute = np.random.rand(len(field))
        data = np.column_stack((field, attribute))

        # Create a ParticleAttributes object with an invalid field
        attribute = ParticleAttribute(
            field="invalid_field",
            attribute="test_attribute",
            data=data,
        )

        # Attempt to append the attribute to the particle data
        with self.assertWarns(UserWarning) as context:
            updated_particle_data = append_attribute(
                particle_data, attribute
            )

        self.assertEqual(str(context.warning),
                         "No field column found in particles file or no "
                         "points in particles file therefore column not "
                         "appended",
        )

        assert "test_attribute" not in updated_particle_data.point_data.keys()
        