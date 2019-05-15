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
import pytasa.rotate

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

albite_c = np.array([[ 69.9,  34.0,  30.8,   5.1,  -2.4,  -0.9],
                     [ 34.0, 183.5,   5.5,  -3.9,  -7.7,  -5.8],
                     [ 30.8,   5.5, 179.5,  -8.7,   7.1,  -9.8],
                     [  5.1,  -3.9,  -8.7,  24.9,  -2.4,  -7.2],
                     [ -2.4,  -7.7,   7.1,  -2.4,  26.8,   0.5],
                     [ -0.9,  -5.8,  -9.8,  -7.2,   0.5,  33.5]])

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

def test_olivine_decompose():
    """Check that we get the correct decomposed elasticity for olivine"""
    decomp = pytasa.decompose.decompose(ol_c_ref) 
    np.testing.assert_allclose(decomp[0], ol_c_iso, atol=0.1)
    np.testing.assert_allclose(decomp[1], ol_c_hex, atol=0.1)
    np.testing.assert_allclose(decomp[2], ol_c_tet, atol=0.1)
    np.testing.assert_allclose(decomp[3], ol_c_ort, atol=0.1)
    np.testing.assert_allclose(decomp[4], ol_c_mon, atol=0.1)
    np.testing.assert_allclose(decomp[5], ol_c_tri, atol=0.1)

def test_olivine_axes():
    """Check that we get the correct rotation for olivine

    This checks that the olivine example from page 671 od B&C always comes
    out with the correct axes however we rotate the input"""

    # Rotate the correct result on to the x3>x2>x1 reference frame
    # this won't matter for future decomposition but we need it for the test
    # cases.
    c_ref = pytasa.rotate.rot_3(ol_c_ref, 0, 0, 90)

    for a, b, g in zip([0.0, 0.0, 45.0, 33.234, 123.0, 0.0, 0.0, 77.5, 266.0], 
                       [0.0, 2.0, 2.0, 44.7, 266.0, 33.3, 0.0, 77.5, 0.0],
                       [0.0, 0.0, 0.0, 345.0, 0.0, 10.0, 10.0, 10.0, 10.0]):
        c_test = pytasa.rotate.rot_3(c_ref, a, b, g)
        np.testing.assert_allclose(pytasa.decompose.axes(c_test)[0], c_ref, 
                                   atol=0.000001)
        
def test_triclinic_axes():
    """Check that different rotations for albite always give us the same axes"""

    c1, _ = pytasa.decompose.axes(albite_c)

    for a, b, g in zip([0.0, 0.0, 45.0, 33.234, 123.0, 0.0, 0.0, 77.5, 266.0],
                       [0.0, 2.0, 2.0, 44.7, 266.0, 33.3, 0.0, 77.5, 0.0],
                       [0.0, 0.0, 0.0, 345.0, 0.0, 10.0, 10.0, 10.0, 10.0]):
        c2 = pytasa.rotate.rot_3(albite_c, a, b, g)
        c2, _ = pytasa.decompose.axes(c2)

        np.testing.assert_allclose(c1, c2)

def test_isortho():
    "Check that out orthogonal check routine works"
    # These should be true
    assert pytasa.decompose._isortho([1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert pytasa.decompose._isortho([0, 1, 0], [1, 0, 0], [0, 0, 1])
    assert pytasa.decompose._isortho([1, 0, 0], [0, 0, 1], [0, 1, 0])
    assert pytasa.decompose._isortho([0, 0, 1], [0, 1, 0], [1, 0, 0])
    assert pytasa.decompose._isortho([1, 0, 0], [0, 0, 1], [0, 1, 0])
    assert pytasa.decompose._isortho([0, 1, 0], [0, 0, 1], [1, 0, 0])
    assert pytasa.decompose._isortho([-1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert pytasa.decompose._isortho([0, -1, 0], [1, 0, 0], [0, 0, 1])
    assert pytasa.decompose._isortho([-1, 0, 0], [0, 0, 1], [0, 1, 0])
    assert pytasa.decompose._isortho([0, 0, -1], [0, 1, 0], [1, 0, 0])
    assert pytasa.decompose._isortho([-1, 0, 0], [0, 0, 1], [0, 1, 0])
    assert pytasa.decompose._isortho([0, -1, 0], [0, 0, 1], [1, 0, 0])
    # These should be false
    assert not(pytasa.decompose._isortho([1, 0, 0], [0, 1, 0.01], [0, 0, 1]))
    assert not(pytasa.decompose._isortho([1, 0, 0], [0.01, 1, 0], [0, 0, 1]))
    assert not(pytasa.decompose._isortho([1, 0, 0], [0, 1, 0], [0.01, 0, 1]))
