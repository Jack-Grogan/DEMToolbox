import pyvista as pv
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                os.pardir, os.pardir)))
from DEMToolbox.mixing import lacey_mixing_curve_fit 
from DEMToolbox.mixing import lacey_mixing_curve

class TestLaceyMixingIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        
        # Store times for testing
        cls.times =  np.linspace(0, 10, 101)


    def test_lacey_mixing_curve_fit_zero(self):
        """Test the lacey_mixing_curve_fit function with zero input."""

        lacey_0 = np.zeros_like(self.times)

        # Fit the curve
        results = lacey_mixing_curve_fit(self.times, lacey_0)

        k = results[0][0]
        tau = results[0][1]

        # Check that the parameters are close to zero
        self.assertAlmostEqual(k, 0.0, places=5)
        self.assertAlmostEqual(tau, 0.0, places=5)

        self.assertEqual(results[1][0, 0], 1.0363836214613491e-22)
        self.assertEqual(results[1][0, 1], 9.111118648114095e-14)
        self.assertEqual(results[1][1, 0], 9.111118648114094e-14)
        self.assertEqual(results[1][1, 1], 0.0031584677190661177)

        assert np.all(results[2] == self.times)
        assert np.all(results[3] == lacey_0)
        assert np.all(results[4] == lacey_mixing_curve(self.times, k, tau, lacey_0[0]))


    def test_lacey_mixing_curve_fit_one(self):
        """Test the lacey_mixing_curve_fit function with one input."""

        lacey_1 = np.ones_like(self.times)
        
        # Fit the curve
        results = lacey_mixing_curve_fit(self.times, lacey_1)

        k = results[0][0]
        tau = results[0][1]

        # Check that the parameters are close to zero
        self.assertAlmostEqual(k, 0.0, places=5)
        self.assertAlmostEqual(tau, 0.0, places=5)

        self.assertEqual(results[1][0, 0], 0.0)
        self.assertEqual(results[1][0, 1], 0.0)
        self.assertEqual(results[1][1, 0], 0.0)
        self.assertEqual(results[1][1, 1], 0.0)

        assert np.all(results[2] == self.times)
        assert np.all(results[3] == lacey_1)
        assert np.all(results[4] == lacey_mixing_curve(self.times, k, tau, lacey_1[0]))

    def test_lacey_mixing_curve_fit_zero_to_one(self):
        """Test the lacey_mixing_curve_fit for 0 to 1 in 0.1 seconds."""
        
        # Fit the curve
        lacey_values = np.ones_like(self.times)
        lacey_values[0] = 0.0  # Set values to 0 for times < 0.1 seconds

        results = lacey_mixing_curve_fit(self.times, lacey_values)

        k = results[0][0]
        tau = results[0][1]

        # Check that the parameters are close to zero
        self.assertEqual(k, 84.06014038940896)
        self.assertEqual(tau, 5.15602967908695e-12)

        self.assertEqual(results[1][0, 0], 20052087.00180876)
        self.assertEqual(results[1][0, 1], 23854.41667977134)
        self.assertEqual(results[1][1, 0], 23854.416679771337)
        self.assertEqual(results[1][1, 1], 28.377755579520105)

        assert np.all(results[2] == self.times)
        assert np.all(results[3] == lacey_values)
        assert np.all(results[4] == lacey_mixing_curve(self.times, k, tau, lacey_values[0]))

    def test_lacey_mixing_curve_fit_perfect_fit(self):
        """Test the lacey_mixing_curve_fit for a perfect fit."""

        # Create a perfect fit
        k = 0.5
        tau = 1.0
        m0 = 0.2

        lacey_values = lacey_mixing_curve(self.times, k, tau, m0)
        results = lacey_mixing_curve_fit(self.times, np.array(lacey_values))

        k_fit = results[0][0]
        tau_fit = results[0][1]

        # Check that the parameters are close to the original values
        self.assertAlmostEqual(k, k_fit)
        self.assertAlmostEqual(tau, tau_fit)

        self.assertEqual(results[1][0, 0], 6.187977008560435e-21)
        self.assertEqual(results[1][0, 1], 1.2991315443332372e-20)
        self.assertEqual(results[1][1, 0], 1.2991315443332372e-20)
        self.assertEqual(results[1][1, 1], 5.175835209191755e-20)

        assert np.all(results[2] == self.times)
        assert np.all(results[3] == lacey_values)
        assert np.all(results[4] == lacey_mixing_curve(self.times, k_fit, tau_fit, lacey_values[0]))

    def test_lacey_mixing_curve_raises(self):
        """Test the lacey_mixing_curve_fit function raises errors for invalid inputs."""

        k = 0.5
        tau = 1.0
        m0 = 0.2

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve(list(self.times), k, tau, m0)

        self.assertEqual(
            str(context.exception),
            "time must be an array-like object"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve(self.times, "invalid input type", tau, m0)

        self.assertEqual(
            str(context.exception),
            "k must be an integer or float"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve(self.times, k, "invalid input type", m0)

        self.assertEqual(
            str(context.exception),
            "tau must be an integer or float"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve(self.times, k, tau, "invalid input type")    

        self.assertEqual(
            str(context.exception),
            "m0 must be an integer or float"
        )

    def test_lacey_mixing_curve_fit_raises(self):
        """Test the lacey_mixing_curve_fit function raises errors for invalid inputs."""

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve_fit(list(self.times), np.ones_like(self.times))

        self.assertEqual(
            str(context.exception),
            "time must be an array-like object"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve_fit(self.times, list(np.ones_like(self.times)))

        self.assertEqual(
            str(context.exception),
            "m must be an array-like object"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve_fit(self.times, np.ones_like(self.times), t0="invalid input type")

        self.assertEqual(
            str(context.exception),
            "t0 must be an integer or float"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve_fit(self.times, np.ones_like(self.times), tend="invalid input type")

        self.assertEqual(
            str(context.exception),
            "tend must be an integer or float"
        )

        with self.assertRaises(ValueError) as context:
            lacey_mixing_curve_fit(self.times, np.ones(len(self.times)+1))

        self.assertEqual(
            str(context.exception),
            "time and m must be the same length"
        )