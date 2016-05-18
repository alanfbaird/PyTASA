#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the rotation routines 

These test cases are taken from MSAT
"""
import unittest
import numpy as np

import pytasa.rotate

olivine_cij_voigt = np.array([[320.5, 68.1, 71.6, 0.0, 0.0, 0.0],
                              [68.1, 196.5, 76.8, 0.0, 0.0, 0.0],
                              [71.6, 76.8, 233.5, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 78.7]])

class PytasaRotTestCase(unittest.TestCase):
    """
    Test case for rot functions
    """
    def test_identity(self):
        """Check that roation by the identity matrix does not rotate
           
           We should end up with what we started with. This is from MSAT
           test_MS_rotR_simple.
        """
        np.testing.assert_almost_equal(pytasa.rotate.rot_c(
             olivine_cij_voigt, np.eye(3)), olivine_cij_voigt, decimal=6)
    
    def test_a_90(self):
        """Check that roation around X axis works
           
           Result can be calculated by hand by permuting values. 
           This is from MSAT test_MS_rotR_simple.
        """
        c_ol_rot_a_90 = np.array([[320.5, 71.6, 68.1, 0.0, 0.0, 0.0],
                                  [71.6, 233.5, 76.8, 0.0, 0.0, 0.0],
                                  [68.1, 76.8, 196.5, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 78.7, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 77.0]])
        g = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        np.testing.assert_almost_equal(pytasa.rotate.rot_c(
             olivine_cij_voigt, g), c_ol_rot_a_90, decimal=6)
    
    def test_b_90(self):
        """Check that roation around Y axis works
           
           Result can be calculated by hand by permuting values. 
           This based on MSAT test_MS_rotR_simple.
        """
        c_ol_rot_b_90 = np.array([[233.5, 76.8, 71.6, 0.0, 0.0, 0.0],
                                  [76.8, 196.5, 68.1, 0.0, 0.0, 0.0],
                                  [71.6, 68.1, 320.5, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 78.7, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 64.0]])
        g = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        print(pytasa.rotate.rot_c(olivine_cij_voigt, g))
        print(c_ol_rot_b_90)
        np.testing.assert_almost_equal(pytasa.rotate.rot_c(
             olivine_cij_voigt, g), c_ol_rot_b_90, decimal=6)
    
    def test_c_90(self):
        """Check that roation around Z axis works
           
           Result can be calculated by hand by permuting values. 
           This based on MSAT test_MS_rotR_simple.
        """
        c_ol_rot_c_90 = np.array([[196.5, 68.1, 76.8, 0.0, 0.0, 0.0],
                                  [68.1, 320.5, 71.6, 0.0, 0.0, 0.0],
                                  [76.8, 71.6, 233.5, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 77.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 64.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 78.7]])
        g = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        print(pytasa.rotate.rot_c(olivine_cij_voigt, g))
        print(c_ol_rot_c_90)
        np.testing.assert_almost_equal(pytasa.rotate.rot_c(
             olivine_cij_voigt, g), c_ol_rot_c_90, decimal=6)
    
def suite():
    return unittest.makeSuite(PytasaRotTestCase, 'test')
    
    
if __name__ == '__main__':
    unittest.main(defaultTest='suite')
        
