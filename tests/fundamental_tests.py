#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the fundamental routines - based on the MSAT test cases
"""

# help run this code without installation
import sys
sys.path.append("..")

import unittest
import numpy as np
import numpy.testing as npt
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

        
        
    
    
    
        

class TestInvertCijFunctions(unittest.TestCase):

    def setUp(self):
        self.inmatrix = np.matrix([[0.700, 0.200],[0.400, 0.600]])
        self.inerrors = np.matrix([[0.007, 0.002],[0.004, 0.006]])
        self.true_inv = np.matrix([[1.765, -0.588],[-1.177, 2.059]])
        self.true_err = np.sqrt(np.matrix([[5.269E-4, 1.603E-4],
                                           [6.413E-4, 7.172E-4]]))
        self.true_cov = np.array([[[[5.269E-4, -2.245E-4], 
                                    [-4.490E-4, 2.514E-4]],
                                   [[-2.245E-4, 1.603E-4],
                                    [2.514E-4, -2.619E-4]]],
                                  [[[-4.490E-4, 2.514E-4],
                                    [6.413E-4, -5.238E-4]],
                                   [[2.514E-4, -2.619E-4],
                                    [-5.238E-4,7.172E-4]]]])
        self.calc_inv, self.calc_err, self.calc_cov = pytasa.fundamental. \
                                      invert_cij(self.inmatrix, self.inerrors)

    def test_inverse(self):
        npt.assert_array_almost_equal(self.calc_inv, self.true_inv, 3,
           'Calculated inverse of test matrix is wrong')

    def test_inverseErrors(self):
        npt.assert_array_almost_equal(self.calc_err, self.true_err, 5,
           'Calculated propogated std. errors of test matrix are wrong')

    def test_inverseCovar(self):
        npt.assert_array_almost_equal(self.calc_cov, self.true_cov,7,
           'Calculated propogated var-covar matrix of test matrix is wrong')


def suite():
    asuit = unittest.makeSuite(TestInvertCijFunctions, 'test')
    asuit.addTest(suite_1())
    return asuit


def suite_1():
    return unittest.makeSuite(PytasaFundamentalTestCase, 'test')
    
    
if __name__ == '__main__':
    unittest.main(defaultTest='suite')
