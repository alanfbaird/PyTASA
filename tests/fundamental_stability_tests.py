#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the IO routines - based on the MSAT test cases
"""

# help run this code without installation
import sys
sys.path.append("..")

import unittest

import numpy as np

import pytasa.fundamental


class TestCijStability(unittest.TestCase):

    def test_stability_false(self):
        """This is the non-positivedefinate test from MSAT"""
        bad_cij = -1.0 * np.ones((6,6))
        assert(not pytasa.fundamental.cij_stability(bad_cij))

    def test_stability_true(self):
        """Olivine should be stable"""
        olivine_cij_voigt = np.array([[320.5, 68.1, 71.6, 0.0, 0.0, 0.0],
                              [68.1, 196.5, 76.8, 0.0, 0.0, 0.0],
                              [71.6, 76.8, 233.5, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 78.7]])
        assert(pytasa.fundamental.cij_stability(olivine_cij_voigt))

def suite():
    return unittest.makeSuite(TestCijStability, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

