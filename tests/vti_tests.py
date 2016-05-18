#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the vti routines - based on results from my MATLAB implementation 
"""
import unittest
import numpy as np

import pytasa.vti

class PytasaVTITestCase(unittest.TestCase):
    """
    Test case for vti functions
    """
    def test_example_point(self):
        """Check we get the same results as those given in old files

           This test is for lat=-60, lon=85, r=3555 with cij from:
           TX2008.V2.T9.6.topo.P001.geog_cij.dat and Xi from:
           TX2008.V2.T9.6.topo.P001.Xi.dat
        """
        ln_xi_percent_file = 24.319 
        cij_file = np.array([[1091.91560284,  458.538622806, 456.496942875,
                             -1.04865250962, 1.84037540811, -38.9969923147],
                             [458.538622806, 1101.84460008, 452.074983665,
                             2.75366574376, 0.5841747463, 24.6917896573],
                             [456.496942875, 452.074983665, 1224.90181259,
                             -4.09920876836, -5.74073877119, 6.43661390972],
                             [-1.04865250962, 2.75366574376, -4.09920876836,
                             262.58650102, 1.908492257, 2.66401429285],
                             [1.84037540811, 0.5841747463, -5.74073877119,
                             1.908492257, 264.1072491, 0.433616876376],
                             [-38.9969923147, 24.6917896573, 6.43661390972,
                             2.66401429285, 0.433616876376, 356.605598817]])
        xi, phi, vti_cij = pytasa.vti.planar_ave(cij_file)

        np.testing.assert_almost_equal(np.log(xi)*100.0, ln_xi_percent_file, decimal=3)
    
    
def suite():
    return unittest.makeSuite(PytasaVTITestCase, 'test')
    
    
if __name__ == '__main__':
    unittest.main(defaultTest='suite')
        
