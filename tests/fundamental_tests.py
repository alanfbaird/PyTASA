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
        pol,avs,vs1,vs2,vp = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, 90, 0)

        np.testing.assert_almost_equal(vs1,vs2)
        np.testing.assert_almost_equal(avs,0.0)
        assert (vp-13.5)**2 < 0.1**2
        assert (vs1-7.7)**2 < 0.1**2
        assert np.isnan(pol)
        
        # [100]
        pol,avs,vs1,vs2,vp = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, 0, 0)

        assert (vp-10.2)**2 < 0.1**2
        assert (vs1-8.4)**2 < 0.1**2
        assert (vs2-7.7)**2 < 0.1**2
        
        # [010]
        pol,avs,vs1,vs2,vp = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, 0, 90)

        assert (vp-10.2)**2 < 0.1**2
        assert (vs1-8.4)**2 < 0.1**2
        assert (vs2-7.7)**2 < 0.1**2
        
    
    def test_phasevels_stishovite_list(self):
        
        pol,avs,vs1,vs2,vp = pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, [90,90,90], [0,0,0])
        
        np.testing.assert_array_almost_equal(vs1,vs2)
        np.testing.assert_array_almost_equal(avs,[0.0,0.0,0.0])
        np.testing.assert_allclose(vp,[13.5,13.5,13.5],atol=0.5)
        np.testing.assert_allclose(vs1,[7.7,7.7,7.7],atol=0.5)
        
    
    def test_phasevels_stishovite_errors(self):
        
        with self.assertRaises(ValueError):
            pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, [90,90,90], [0,0])
            pytasa.fundamental.phasevels(stishovite_cij, stishovite_rho, [90,90], [0,0,0])
            
    # Not implemented   
    #def test_phasevels_Cinvalid(self):
    #    
    #    C = stishovite_cij.copy()
    #    C[2,5]=-675.0
    #    self.assertRaises(ValueError,pytasa.fundamental.phasevels(C, stishovite_rho, [90,90], [0,0]))
    #
    
    def test_phasevels_stishovite_min_max(self):
        """
        Test that max and min phase velocity, group velocity and slowness 
        matches figure 5 of Mainprice (2007)
        """
        
        azi=np.arange(-180.,181.,1)
        inc=np.zeros_like(azi)
        VGP, VGS1, VGS2, PE, S1E, S2E, SNP, SNS1, SNS2, VPP, VPS1, VPS2 = pytasa.fundamental.groupvels(stishovite_cij,stishovite_rho,inc,azi,slowout=True)
        
        # resort S velocities and slownesses into SV and SH
        VGSV=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[VGS2.T,VGS1.T]).T
        VGSH=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[VGS1.T,VGS2.T]).T

        SVE=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[S2E.T,S1E.T]).T
        SHE=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[S1E.T,S2E.T]).T

        SNSV=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[SNS2.T,SNS1.T]).T
        SNSH=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[SNS1.T,SNS2.T]).T

        VPSV=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[VPS2.T,VPS1.T]).T
        VPSH=np.choose((S1E[:,2]>S2E[:,2]).astype(int),[VPS1.T,VPS2.T]).T
        
        
        # minmax phase velocities
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in VPP]),10.28,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in VPP]),12.16,decimal=2)        
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in VPSH]),5.31,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in VPSH]),8.39,decimal=2)        
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in VPSV]),7.66,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in VPSV]),7.66,decimal=2)
        
        # minmax slownesses
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in SNP]),0.08,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in SNP]),0.10,decimal=2)        
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in SNSH]),0.12,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in SNSH]),0.19,decimal=2)        
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in SNSV]),0.13,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in SNSV]),0.13,decimal=2)
        
        # minmax group velocities
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in VGP]),10.28,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in VGP]),12.16,decimal=2)        
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in VGSH]),5.31,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in VGSH]),9.40,decimal=2)        
        np.testing.assert_almost_equal(min([np.linalg.norm(i) for i in VGSV]),7.66,decimal=2)
        np.testing.assert_almost_equal(max([np.linalg.norm(i) for i in VGSV]),7.66,decimal=2)

        
        
    
    
    
def suite():
    return unittest.makeSuite(PytasaFundamentalTestCase, 'test')
    
    
if __name__ == '__main__':
    unittest.main(defaultTest='suite')
        