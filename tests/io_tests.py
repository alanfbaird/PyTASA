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

    def test_expand_isotropic(self):
        """Text for expand isotropic, taken from MSAT test_MS_expand"""
        res_valid = np.array([[166.6667, 66.6667, 66.6667, 0.0, 0.0, 0.0],
                              [66.6667, 166.6667, 66.6667, 0.0, 0.0, 0.0],
                              [66.6667, 66.6667, 166.6667, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
        res_test = pytasa.io.expand_isotropic(166.6667, 66.6667)

    def test_expand_cubic(self):
        """Text for expand cubic, taken from MSAT test_MS_expand"""
        res_valid = np.array([[166.6667, 66.6667, 66.6667, 0.0, 0.0, 0.0],
                              [66.6667, 166.6667, 66.6667, 0.0, 0.0, 0.0],
                              [66.6667, 66.6667, 166.6667, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])
        res_test = pytasa.io.expand_cubic(166.6667, 50.0, 66.6667)
        np.testing.assert_almost_equal(res_valid, res_test, decimal=4)

    def test_expand_hexagonal(self):
        """Text for expand hexagonal, taken from MSAT test_MS_expand"""
        res_valid = np.array([[153.6, 76.8, 76.0444, 0.0, 0.0, 0.0],
                              [76.8, 153.6, 76.0444, 0.0, 0.0, 0.0],
                              [76.0444, 76.0444, 128.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 32.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0,  32.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0,  38.4]])
        res_test = pytasa.io.expand_hexagonal(153.6, 128.0, 32.0, 76.8, 76.0444)
        np.testing.assert_almost_equal(res_valid, res_test, decimal=4)

    def test_expand_hexagonal(self):
        """Text for expand orthorhombc, taken from MSAT test_MS_expand"""
        res_valid = np.array([[153.6, 76.82, 76.0444, 0.0, 0.0, 0.0],
                              [76.82, 155.6, 76.1444, 0.0, 0.0, 0.0],
                              [76.0444, 76.1444, 128.1, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 32.1, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 32.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 38.4]])
        res_test = pytasa.io.expand_orthorhombic(153.6, 155.6, 128.1, 32.1,
                                                 32.0, 38.4, 76.82, 76.0444,
                                                 76.1444)
        np.testing.assert_almost_equal(res_valid, res_test, decimal=4)

    def test_build_isotropic(self):
        """Test build_isotropic for all combinations of input. However, this
        is rather circular."""

        C, lam, mu, K, E, nu, M = pytasa.io.build_iso(K=300.8, mu=123.7)

        np.testing.assert_almost_equal(C, pytasa.io.build_iso(lam=C[0,1], 
                                                              mu=C[3,3])[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(mu=C[3,3], 
                                                              lam=C[0,1])[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(M=C[0,0], 
                                                              mu=C[3,3])[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(mu=C[3,3], 
                                                              M=C[0,0])[0])

        np.testing.assert_almost_equal(C, pytasa.io.build_iso(E=E, mu=mu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(mu=mu, E=E)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(E=E, mu=mu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(K=K, lam=lam)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(lam=lam, K=K)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(K=K, mu=mu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(mu=mu, K=K)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(nu=nu,lam=lam)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(lam=lam,nu=nu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(nu=nu, mu=mu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(mu=mu, nu=nu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(E=E, nu=nu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(nu=nu, E=E)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(K=K, nu=nu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(nu=nu, K=K)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(K=K, E=E)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(E=E, K=K)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(M=M, mu=mu)[0])
        np.testing.assert_almost_equal(C, pytasa.io.build_iso(mu=mu, M=M)[0])


def suite():
    return unittest.makeSuite(PytasaIOTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')



