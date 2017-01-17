# -*- coding: utf-8 -*-
"""
pytasa.decompose - decompose tensors following Browaeys & Chevrot (2004)

This module provides functions that implement the decomposition of elasticity
tensors according to their symmetry using the approach described by Browaeys
and Chevrot (2004). The code is more-or-less a direct translation from the 
Matlab implementation in MSAT (Walker and Wookey, 2012)

References:

    Browaeys, J. T. and S. Chevrot (2004) Decomposition of the elastic tensor 
             and geophysical applications. Geophysical Journal international,
             159:667-678
    Walker, A. M. and Wookey, J. (2012) MSAT -- a new toolkit for the analysis
             of elastic and seismic anisotropy. Computers and Geosciences,
             49:81-90.

"""
from __future__ import division
import collections

import numpy as np

ElasticNorms = collections.namedtuple('ElasticNorms', ['isotropic', 'hexagonal',
                    'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'])

def norms(c_ref, c_iso, c_hex, c_tet, c_ort, c_mon, c_tri):

    x_ref = _c2x(c_ref)
    n = np.sqrt(np.dot(x_ref, x_ref))

    c_tot = np.zeros((6,6))
    result = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i, c in enumerate([c_iso, c_hex, c_tet, c_ort, c_mon, c_tri]):
        c_tot = c_tot + c
        x_tot = _c2x(c_tot)
        x_d = x_ref - x_tot
        result[i] = 1 - (np.sqrt(np.dot(x_d, x_d)) / n)

    # Rework the fractions
    result[1:6] = result[1:6] - result[0:5]

    return ElasticNorms(result[0], result[1], result[2], result[3], result[4],
                        result[5])

def _c2x(c):
    """Convert elastic constants tensor (Voigt notation) into vector 'x'
    """
    x = np.zeros(21)
	
    x[0]  = c[0,0]
    x[1]  = c[1,1]
    x[2]  = c[2,2]
    x[3]  = np.sqrt(2.0) * c[1,2]
    x[4]  = np.sqrt(2.0) * c[0,2]
    x[5]  = np.sqrt(2.0) * c[0,1]
    x[6]  = 2.0 * c[3,3]
    x[7]  = 2.0 * c[4,4]
    x[8]  = 2.0 * c[5,5]
    x[9]  = 2.0 * c[0,3]
    x[10] = 2.0 * c[1,4]
    x[11] = 2.0 * c[2,5]
    x[12] = 2.0 * c[2,3]
    x[13] = 2.0 * c[0,4]
    x[14] = 2.0 * c[1,5]
    x[15] = 2.0 * c[1,3]
    x[16] = 2.0 * c[2,4]
    x[17] = 2.0 * c[0,5]
    x[18] = 2.0 * np.sqrt(2.0) * c[4,5]
    x[19] = 2.0 * np.sqrt(2.0) * c[3,5] ;
    x[20] = 2.0 * np.sqrt(2.0) * c[3,4] ;

    return x
