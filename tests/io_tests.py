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

class PytasaIOTestCase(unittest.TestCase):
    """
    Test case for low level reading from files in various formats
    """

    def test_load_Aij(self):
       C_input = pytasa.io.load_ematrix(os.path.join(DATA,
                                        "test_MS_load_ematrix.txt"))
       np.testing.assert_almost_equal(olivine_cij_voigt, C_input)

def suite():
    return unittest.makeSuite(PytasaIOTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')



