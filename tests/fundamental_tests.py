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
        pol,avs,vs1,vs2,vp,S1P,S2P = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, 90, 0)

        np.testing.assert_almost_equal(vs1,vs2)
        np.testing.assert_almost_equal(avs,0.0)
        assert (vp-13.5)**2 < 0.1**2
        assert (vs1-7.7)**2 < 0.1**2
        assert np.isnan(pol)
        
        # [100]
        pol,avs,vs1,vs2,vp,S1P,S2P = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, 0, 0)

        assert (vp-10.2)**2 < 0.1**2
        assert (vs1-8.4)**2 < 0.1**2
        assert (vs2-7.7)**2 < 0.1**2
        
        # [010]
        pol,avs,vs1,vs2,vp,S1P,S2P = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, 0, 90)

        assert (vp-10.2)**2 < 0.1**2
        assert (vs1-8.4)**2 < 0.1**2
        assert (vs2-7.7)**2 < 0.1**2
        
    
    def test_phasevels_stishovite_list(self):
        """docstring for test_phasevels_stishovite_list"""
        pol,avs,vs1,vs2,vp,S1P,S2P = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, [90,90,90], [0,0,0])
        
        np.testing.assert_array_almost_equal(vs1,vs2)
        np.testing.assert_array_almost_equal(avs,[0.0,0.0,0.0])
        np.testing.assert_allclose(vp,[13.5,13.5,13.5],atol=0.5)
        np.testing.assert_allclose(vs1,[7.7,7.7,7.7],atol=0.5)
        
        
        
        
    pass

        
        
    
    
    
def suite():
    return unittest.makeSuite(PytasaFundamentalTestCase, 'test')
    
    
if __name__ == '__main__':
    unittest.main(defaultTest='suite')
        