#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the IO routines - based on the MSAT test cases
"""
import os
import inspect # To work out where we are...
import unittest

import numpy as np

import pytasa.io

# We need to know where to get the data from
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data", "io_test_data")

olivine_cij_voigt = np.array([[320.5, 68.1, 71.6, 0.0, 0.0, 0.0],
                              [68.1, 196.5, 76.8, 0.0, 0.0, 0.0],
                              [71.6, 76.8, 233.5, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 78.7]])
olivine_rho = 3355.0

class PytasaIOTestCase(unittest.TestCase):
    """
    Test case for low level reading from files in various formats
    """

    def test_load_ematrix(self):
       C_input = pytasa.io.load_ematrix(os.path.join(DATA,
                                        "test_MS_load_ematrix.txt"))
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)

    def test_load_ematrix_gz(self):
       C_input = pytasa.io.load_ematrix(os.path.join(DATA,
                                        "test_MS_load_ematrix.txt.gz"))
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)

    def test_load_ematrix_fh(self):
       with open(os.path.join(DATA, "test_MS_load_ematrix.txt")) as f:
           C_input = pytasa.io.load_ematrix(f)
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)

    def test_load_msat_simple_file(self):
       C_input, rho = pytasa.io.load_mast_simple(os.path.join(DATA, 
                                            "test_MS_load_default.txt"))
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)
       np.testing.assert_almost_equal(olivine_rho, rho)

    def test_load_msat_simple_fh(self):
       with open(os.path.join(DATA, "test_MS_load_default.txt")) as f:
           C_input, rho = pytasa.io.load_mast_simple(f)
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)
       np.testing.assert_almost_equal(olivine_rho, rho)

    def test_load_msat_simple_file_gz(self):
       C_input, rho = pytasa.io.load_mast_simple(os.path.join(DATA, 
                                            "test_MS_load_default.txt.gz"))
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)
       np.testing.assert_almost_equal(olivine_rho, rho)

    def test_load_msat_Aij_file(self):
       C_input, rho = pytasa.io.load_mast_simple(os.path.join(DATA, 
                                            "test_MS_load_Aij.txt"), dnorm=True,
                                            eunit="Pa")
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input, decimal=3)
       np.testing.assert_almost_equal(olivine_rho, rho)

def suite():
    return unittest.makeSuite(PytasaIOTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')



