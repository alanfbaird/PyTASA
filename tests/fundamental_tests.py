#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the fundamental routines - based on the MSAT test cases
"""
import unittest
import numpy as np
import pytasa.fundamental

stishovite_cij = np.array([[453.0, 211.0, 203.0,   0.0,   0.0,   0.0],
                           [211.0, 453.0, 203.0,   0.0,   0.0,   0.0],
                           [203.0, 203.0, 776.0,   0.0,   0.0,   0.0],
                           [  0.0,   0.0,   0.0, 252.0,   0.0,   0.0],
                           [  0.0,   0.0,   0.0,   0.0, 252.0,   0.0],
                           [  0.0,   0.0,   0.0,   0.0,   0.0, 302.0]])

stishovite_rho = 4290.0



class PytasaFundamentalTestCase(unittest.TestCase):
    """
    Test case for fundamental functions
    """
    def test_phasevels_stishovite_graph(self):
        """Check we get the same results as those given in Mainprices review (Figure 3)."""
        
        # [001], symmetry axis
        pol,avs,vs1,vs2,vp,S1P,S2P = pytasa.fundamental.phasevels(C, rh, 90, 0);
        
        np.testing.assert_almost_equal(vs1,vs2)
        np.testing.assert_almost_equal(avs,0.0)
        np.testing.assert_almost_equal(vp-13.5,0.0)
        np.testing.assert_almost_equal(vs1-7.7,0.0)
        
        
    
    
    
def suite():
    return unittest.makeSuite(PytasaFundamentalTestCase, 'test')
    
    
if __name__ == '__main__':
    unittest.main(defaultTest='suite')
        