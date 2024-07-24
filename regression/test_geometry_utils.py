import unittest
import numpy as np
from plasma_math_tools.geometry_utils import get_phi_tor_from_two_points

class TestGeomtryUtils(unittest.TestCase):

    def test_get_phi_tor_from_two_points(self):
        for x1_vec, x2_vec, phi_check in zip([np.array([8, 0.5, 0.01]),
                                   np.array([-0.9807852804032304, -0.1950903220161284, 0.006681805010885])],
                                  [np.array([4, 0.8, 0.02]),
                                   np.array([-2.3048453154126185, -0.45846223813263876, 0.04097180441021919])],
                                   [7.8654877038163695, -180.0]):
            phi = get_phi_tor_from_two_points(x1_vec, x2_vec)
            self.assertTrue(np.allclose(phi, phi_check, rtol=1.e-5, atol=1.e-5))

if __name__ == "__main__":
    test = TestGeomtryUtils()
    test.test_get_phi_tor_from_two_points()