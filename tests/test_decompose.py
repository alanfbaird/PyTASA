#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the routines in pytasa.decompose

Most of these tests confirm that our functions return the values given in the
paper by Browaeys and Chevrot (2004) and match test cases found in MSAT
(Walker and Wookey 2012).

References:

    Browaeys, J. T. and S. Chevrot (2004) Decomposition of the elastic tensor
             and geophysical applications. Geophysical Journal international,
             159:667-678.
    Walker, A. M. and Wookey, J. (2012) MSAT -- a new toolkit for the analysis
             of elastic and seismic anisotropy. Computers and Geosciences,
             49:81-90.
"""
import numpy as np
import pytasa.decompose

# Define input and output from Browaeys and Chevrot (olivine, page 671)
ol_c_ref = np.array([[ 192.0,  66.0,  60.0,   0.0,   0.0,   0.0 ],
                     [  66.0, 160.0,  56.0,   0.0,   0.0,   0.0 ],
                     [  60.0,  56.0, 272.0,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,  60.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,  62.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,  49.0 ]])
ol_c_iso = np.array([[ 194.7,  67.3,  67.3,   0.0,   0.0,   0.0 ],
                     [  67.3, 194.7,  67.3,   0.0,   0.0,   0.0 ],
                     [  67.3,  67.3, 194.7,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,  63.7,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,  63.7,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,  63.7 ]])
ol_norm_iso = 0.793
ol_c_hex = np.array([[ -21.7,   1.7,  -9.3,   0.0,   0.0,   0.0 ],
                     [   1.7, -21.7,  -9.3,   0.0,   0.0,   0.0 ],
                     [  -9.3,  -9.3,  77.3,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,  -2.7,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,  -2.7,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0, -11.7 ]])
ol_norm_hex = 0.152 
ol_c_tet = np.array([[   3.0,  -3.0,   0.0,   0.0,   0.0,   0.0 ],
                     [  -3.0,   3.0,   0.0,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,  -3.0 ]])
# NB: we do not know tet and ort from the paper (they are reported together)
ol_c_ort = np.array([[  16.0,   0.0,   2.0,   0.0,   0.0,   0.0 ],
                     [   0.0, -16.0,  -2.0,   0.0,   0.0,   0.0 ],
                     [   2.0,  -2.0,   0.0,   0.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,  -1.0,   0.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   1.0,   0.0 ],
                     [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ]])
ol_norm_tet_and_ort = 0.055
ol_c_mon = np.zeros((6,6))
ol_norm_mon = 0.0
ol_c_tri = np.zeros((6,6))
ol_norm_tri = 0.0

def test_olivine_norms():
    """Check that we get the correct norms if we feed in the decomposed
       values for olivine"""
    norms = pytasa.decompose.norms(ol_c_ref, ol_c_iso, ol_c_hex, ol_c_tet,
                                   ol_c_ort, ol_c_mon, ol_c_tri)
    np.testing.assert_almost_equal(norms.isotropic, ol_norm_iso, decimal=3)
    np.testing.assert_almost_equal(norms.hexagonal, ol_norm_hex, decimal=3)
    np.testing.assert_almost_equal(norms.orthorhombic + norms.tetragonal,
                                   ol_norm_tet_and_ort, decimal=3)
    np.testing.assert_almost_equal(norms.monoclinic, ol_norm_mon, decimal=3)
    np.testing.assert_almost_equal(norms.triclinic, ol_norm_tri, decimal=3)
 
