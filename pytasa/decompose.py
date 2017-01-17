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


def decompose(c):

    c_work = np.copy(c) # so we don't stuff up input
    out = []
    for i in range(5):
        x = _c2x(c_work)
        m = _get_projector(i)
        xh = np.dot(m, x)
        ch = _x2c(xh)
        out.append(ch)
        c_work = c_work - ch
    out.append(c_work) # triclinic
    return out


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


def _x2c(x):
    """Convert vector 'x' into elastic constants tensor (Voigt notation)
    """
    c = np.zeros((6,6))

    c[0,0] = x[0]
    c[1,1] = x[1]
    c[2,2] = x[2]
    c[1,2] = 1.0 / np.sqrt(2.0) * x[3]
    c[0,2] = 1.0 / np.sqrt(2.0) * x[4]
    c[0,1] = 1.0 / np.sqrt(2.0) * x[5]
    c[3,3] = 1.0 / 2.0 * x[6]
    c[4,4] = 1.0 / 2.0 * x[7]
    c[5,5] = 1.0 / 2.0 * x[8]
    c[0,3] = 1.0 / 2.0 * x[9]
    c[1,4] = 1.0 / 2.0 * x[10]
    c[2,5] = 1.0 / 2.0 * x[11]
    c[2,3] = 1.0 / 2.0 * x[12]
    c[0,4] = 1.0 / 2.0 * x[13]
    c[1,5] = 1.0 / 2.0 * x[14]
    c[1,3] = 1.0 / 2.0 * x[15]
    c[2,4] = 1.0 / 2.0 * x[16]
    c[0,5] = 1.0 / 2.0 * x[17]
    c[4,5] = 1.0 / 2.0 * np.sqrt(2.0) * x[18]
    c[3,5] = 1.0 / 2.0 * np.sqrt(2.0) * x[19]
    c[3,4] = 1.0 / 2.0 * np.sqrt(2.0) * x[20]

    for i in range(6):
        for j in range(6):
		c[j,i] = c[i,j]

    return c


def _get_projector(order):
    """Return the projector, M, for a given symmetry.

    Projector is a 21 by 21 numpy array for a given symmetry given by the
    order argument thus:
        order = 0 -> isotropic
        order = 1 -> hexagonal
        order = 2 -> tetragonal
        order = 3 -> orthorhombic
        order = 4 -> monoclinic
    there is no projector for triclinic (it is just whatever is left once all
    other components have been removed).

    NB: real division (like python 3) is used below, hence the __future__
    import at the top of the file.
    """
   
    srt2 = np.sqrt(2.0)
    
    # Isotropic
    if order == 0:
        m = np.zeros((21,21))
        m[0:9,0:9] = [
           [3/15, 3/15, 3/15, srt2/15, srt2/15, srt2/15, 2/15, 2/15, 2/15],
           [3/15, 3/15, 3/15, srt2/15, srt2/15, srt2/15, 2/15, 2/15, 2/15],
           [3/15, 3/15, 3/15, srt2/15, srt2/15, srt2/15, 2/15, 2/15, 2/15],
           [srt2/15, srt2/15, srt2/15, 4/15, 4/15, 4/15, -srt2/15, -srt2/15,
                 -srt2/15],
           [srt2/15, srt2/15, srt2/15, 4/15, 4/15, 4/15, -srt2/15, -srt2/15,
                 -srt2/15],
           [srt2/15, srt2/15, srt2/15, 4/15, 4/15, 4/15, -srt2/15, -srt2/15,
                 -srt2/15],
           [2/15, 2/15, 2/15, -srt2/15, -srt2/15, -srt2/15, 1/5, 1/5, 1/5],
           [2/15, 2/15, 2/15, -srt2/15, -srt2/15, -srt2/15, 1/5, 1/5, 1/5],
           [2/15, 2/15, 2/15, -srt2/15, -srt2/15, -srt2/15, 1/5, 1/5, 1/5]]

    # hexagonal
    elif order == 1:
        m = np.zeros((21,21))
        m[0:9,0:9] = [[3/8, 3/8, 0, 0, 0, 1/(4*srt2), 0, 0, 1/4],
                      [3/8, 3/8, 0, 0, 0, 1/(4*srt2), 0, 0, 1/4],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],
                      [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],
                      [1/(4*srt2), 1/(4*srt2), 0, 0, 0, 3/4, 0, 0, -1/(2*srt2)],
                      [0, 0, 0, 0, 0, 0, 1/2, 1/2, 0],
                      [0, 0, 0, 0, 0, 0, 1/2, 1/2, 0],
                      [1/4, 1/4, 0, 0, 0, -1/(2*srt2), 0, 0, 1/2]]

    # tetragonal
    elif order == 2:
        m = np.zeros((21,21))
        m[0:9,0:9] = [[1/2, 1/2, 0, 0, 0, 0, 0, 0, 0],
                      [1/2, 1/2, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],
                      [0, 0, 0, 1/2, 1/2, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1/2, 1/2, 0],
                      [0, 0, 0, 0, 0, 0, 1/2, 1/2, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]]

    # orthorhombic
    elif order == 3:
        m = np.zeros((21,21))
        for jj in range(9):
            m[jj,jj] = 1

    # monoclinic
    elif order == 4:
        m = np.eye(21)
        for jj in [9, 10, 12, 13, 15, 16, 18, 19]:
            m[jj,jj] = 0

    else:
        raise ValueError("Order must be 0, 1, 2, 3 or 4")

    return m

