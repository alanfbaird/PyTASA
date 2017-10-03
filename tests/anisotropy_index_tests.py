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

import pytasa.anisotropy_index


class TestAnisotropyIndex(unittest.TestCase):

    def setUp(self):
        """Some useful matricies for testing"""
        self.olivine = np.array([[320.5, 68.1, 71.6, 0.0, 0.0, 0.0],
                            [68.1, 196.5, 76.8, 0.0, 0.0, 0.0],
                            [71.6, 76.8, 233.5, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 64.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 77.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 78.7]])
        self.isotropic = np.array([[166.6667, 66.6667, 66.6667, 0.0, 0.0, 0.0],
                             [66.6667, 166.6667, 66.6667, 0.0, 0.0, 0.0],
                             [66.6667, 66.6667, 166.6667, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 50.0]])

    def test_isotropic_zenner(self):
        """Test from MSAT - are isotropic results isotropic"""
        np.testing.assert_almost_equal(pytasa.anisotropy_index.zenerAniso(
            self.isotropic), [1.0, 0.0])

    def test_isotropic_universal(self):
        """Test from MSAT - are isotropic results isotropic"""
        np.testing.assert_almost_equal(pytasa.anisotropy_index.uAniso(
            self.isotropic), [0.0, 0.0])

def suite():
    return unittest.makeSuite(TestCijStability, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

