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

from . import rotate
from . import fundamental

ElasticNorms = collections.namedtuple('ElasticNorms', ['isotropic', 'hexagonal',
                    'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic'])


def axes(c, x3_stiff=False):

    det_thresh = 100.0 * np.sqrt(np.spacing(1))

    # Dilational stiffness tensor
    dst = np.array([[c[0,0] + c[0,1] + c[0,2], c[0,5] + c[1,5] + c[2,5],
                       c[0,4] + c[1,4] + c[2,4]],
                    [c[0,5] + c[1,5] + c[2,5], c[0,1] + c[1,1] + c[2,1],
                       c[0,3] + c[1,3] + c[2,3]],
                    [c[0,4] + c[1,4] + c[2,4], c[0,3] + c[1,3] + c[2,3],
                       c[0,2] + c[1,2] + c[2,2]]])

    # Voigt stiffness tensor
    vst = np.array([[c[0,0] + c[5,5] + c[4,4], c[0,5] + c[1,5] + c[3,4],  
                       c[0,4] + c[2,4] + c[3,5]],
                    [c[0,5] + c[1,5] + c[3,4], c[5,5] + c[1,1] + c[3,3],
                       c[1,3] + c[2,3] + c[4,5]],
                    [c[0,4] + c[2,4] + c[3,5], c[1,3] + c[2,3] + c[4,5],
                       c[4,4] + c[3,3] + c[2,2]]])

    # Eigenvectors of these symmetric arrays
    # these are dst_evecs[:,0], dst_evecs[:,1] etc.
    dst_vals, dst_evecs = np.linalg.eigh(dst)
    vst_vals, vst_evecs = np.linalg.eigh(vst)

    assert _isortho(dst_evecs[:,0], dst_evecs[:,1], dst_evecs[:,2]),\
        'Eigenvectors of dilational stiffness tensor not orthogonal'
    assert _isortho(vst_evecs[:,0], vst_evecs[:,1], vst_evecs[:,2]),\
        'Eigenvectors of Voigt stiffness tensor not orthogonal'

    # Number and order of distinct vectors
    dst_ndist, dst_ind = _ndistinct(dst_vals, thresh=np.linalg.norm(dst_vals)
                                                        / 1000.0)
    vst_ndist, vst_ind = _ndistinct(vst_vals, thresh=np.linalg.norm(vst_vals)
                                                        / 1000.0)
    # FIXME: dd a unittest for _ndistinct

    do_rot = False
    mono_or_tri = False
    # The symmetry depends on the number of distinct eigenvalues. These
    # should be the same for the two tensors for high symmetry cases.
    if dst_ndist == 1:
        # Isotropic case
        x1 = np.array([1, 0, 0])
        x2 = np.array([0, 1, 0])
        x3 = np.array([0, 0, 1])

    elif dst_ndist == 2:
        # Hexagonal or tetragonal
        if x3_stiff:
            # x3 should be the stiff direction, treat like ortho
            inds = np.argsort(dst_vals)
            if not(inds[0] == 0 and inds[1] == 1 and inds[2] == 2):
                do_rot = True
            x1 = dst_evecs[:, inds[0]]
            x2 = dst_evecs[:, inds[1]]
            x3 = dst_evecs[:, inds[2]]

        else:
            # x3 should be the distinct direction
            x3 = dst_evecs[:, dst_ind]
            if not((x3[0] == 0) and (x3[1] == 0)):
                do_rot = True
                # NB: this fails when x3 = [0 1 0] - check MSAT!
                x2 = np.cross(x3, np.array([x3[0], x3[1], 0]))
                x2 = x2/np.linalg.norm(x2)
                x1 = np.cross(x3,x2)
                x1 = x1/np.linalg.norm(x1)
            else:
                # we won't need to rotate, but we do need x1 and x2
                x1 = np.array([1, 0, 0])
                x2 = np.array([0, 1, 0])

    elif dst_ndist == 3:
        # orthorhombic or lower symmetry
        # If the two tensors are alligned we are ortho, otherwise lower sym
        neq = 0
        for i in range(3):
            for j in range(3):
                neq = neq + _veceq(dst_evecs[:,i], vst_evecs[:,j], 0.01)
                # FIXME: add a unittest for _veceq

        if neq == 3:
            # Ortho significant axes are the three eigenvectors
            do_rot = True
            x1 = dst_evecs[:, 0]
            x2 = dst_evecs[:, 1]
            x3 = dst_evecs[:, 2]

        else:        
            if x3_stiff:
                # x3 should be the stiff direction, should use d
                inds = np.argsort(dst_vals)
                if not(inds[0] == 0 and inds[1] == 1 and inds[2] == 2):
                    do_rot = True
                x1 = dst_evecs[:, inds[0]]
                x2 = dst_evecs[:, inds[1]]
                x3 = dst_evecs[:, inds[2]]

            else:
                # monoclinic or triclinic. Here we have to make a 'best-guess'.
                # Following Browraeys and Chevrot we use the bisectrix of each
                # of the eigenvectors of d and its closest match in v.
                x = _estimate_basis(dst_evecs, vst_evecs)
                x1 = x[:,0]
                x2 = x[:,1]
                x3 = x[:,2]
                do_rot = True
                mono_or_tri = True

    else:
        # This should be impossible
        raise ValueError("Number of distinct eigenvalues is impossible")

    # Now apply the necessary rotation. The three new axes define the rotation
    # matrix which turns the 3x3 unit matrix (original axes) into the best
    # projection. So the rotation matrix we need to apply to the elastic
    # tensor is the inverse of this.
    r1 = np.stack((x1, x2, x3), axis=1)
    rr = r1.T

    # We don't want to flip the axes (and make a LH system). If this
    # is what we would do avoid it by reflecting X1.
    if (np.linalg.det(rr) + 1 < det_thresh):
        print("FIXME: need to issue a warning")
        r1 = np.stack((-x1, x2, x3), axis=1)
        rr = r1.T

    cr = rotate.rot_c(c, rr)

    if mono_or_tri:
        # if crystal is monoclinic or triclinic, we have some more work to do.
        # We need to make sure that differently rotated crystals return the 
        # same result. So they are consistent we try 180 degree rotations 
        # around all the principle axes. For each flip, we choose the result
        # that gives the fastest P-wave velocity in the [1 1 1] direction
        cr, rr = _tryrotations(cr, rr)

    return cr, rr


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


def _isortho(vec1, vec2, vec3, thresh=10.0*np.sqrt(np.spacing(1))):
    "Confirms three vectors are mutually orthogonal, returns True or False"
    dot_prods = np.absolute([np.dot(vec1, vec2), np.dot(vec1, vec3),
                             np.dot(vec2, vec3)])
    return not(np.any(dot_prods > thresh))


def _ndistinct(vec, thresh=10.0*np.sqrt(np.spacing(1))):
    """Returns the number and indicies of distinct values in a length-3 array

       Ignores differences of thresh.
    """
    if ((np.abs(vec[0] - vec[1]) < thresh) and
        (np.abs(vec[0] - vec[2]) < thresh)):
        # All are the same
        nd = 1
        i = None
    elif ((np.abs(vec[0] - vec[1]) > thresh) and
          (np.abs(vec[0] - vec[2]) > thresh) and
          (np.abs(vec[1] - vec[2]) > thresh)):
        # All are the different
        nd = 3
        i = None
    else:
        # One is different
        nd = 2
        if np.abs(vec[0] - vec[1]) < thresh:
            i = 2
        elif np.abs(vec[0] - vec[2]) < thresh:
            i = 1
        elif np.abs(vec[1] - vec[2]) < thresh:
            i - 0
        else:
            assert False, "Logic error in ndistinct!"
    return nd, i


def _veceq(v1, v2, thresh):
    """Return 1 if two 3D eigen-vectors are equal, otherwise 0

       Ignores differences smaller than thresh and switch
       sign of the vector if needed.
    """
    # eignvector sign is arbitary, so, for each vector compare
    # the sign of the largest element, and switch such that it is
    # positive
    i = np.absolute(v1).argmax()
    if v1[i] < 0:
        v1 = v1 * -1
    if v2[i] < 0:
        v2 = v2 * -1
                
    if (np.any((v1 - v2) > thresh)):
        return 0
    else:
        return 1


def _estimate_basis(a, b):
    """Estimate the best basis vectors for decomposition.

    This is the bisectrices of vectors a and b."""
    c = np.zeros((3,3))
    for j in [0, 1, 2]:
        i = np.argmax(np.absolute([np.dot(a[:,j], b[:,0]),
                                   np.dot(a[:,j], b[:,1]),
                                   np.dot(a[:,j], b[:,2])]))
        if np.dot(a[:,j], b[:,i]) < 0.0:
            c[:,j] = _bisectrix(a[:,j], -b[:,i])
        else:
            c[:,j] = _bisectrix(a[:,j], b[:,i]) 
   
    # Enforce orthoganality, and renormalise
    c[:,2] = np.cross(c[:,0], c[:,1])
    c[:,1] = np.cross(c[:,0], c[:,2])
    c[:,0] = c[:,0] / np.linalg.norm(c[:,0])
    c[:,1] = c[:,1] / np.linalg.norm(c[:,1])
    c[:,2] = c[:,2] / np.linalg.norm(c[:,2])

    return c


def _bisectrix(a,b):
    """return the unit length bisectrix of 3-vectors A and B"""
    c = a + b
    c = c / np.linalg.norm(c)
    return c


def _tryrotations(cr, rr):
    """try all combinations of 180 rotations around principle axes
       select the one that gives the highest Vp in the [1 1 1] direction
    """
    x1r = [000, 180, 000, 000, 180, 000]
    x2r = [000, 000, 180, 000, 180, 180]
    x3r = [000, 000, 000, 180, 000, 180]
    vp = []
    for r1, r2, r3 in zip(x1r, x2r, x3r):
        _, _, _, _, p = fundamental.phasevels(rotate.rot_3(cr, r1, r2, r3),
                                              2000, 45, 45)
        vp.append(p[0])
    inds = np.argsort(vp)

    # Rotate to fastest direction
    a = x1r[inds[-1]]
    b = x2r[inds[-1]]
    g = x3r[inds[-1]]
    crr = rotate.rot_3(cr, a, b, g)

    # Add rotation to rr rotation matrix
    a = a * np.pi/180.0
    b = b * np.pi/180.0
    g = g * np.pi/180.0
    r1 = np.array([[1.0, 0.0, 0.0], 
                   [0.0, np.cos(a), np.sin(a)], 
                   [0.0, -np.sin(a), np.cos(a)]])
    r2 = np.array([[np.cos(b), 0.0, -np.sin(b)],
                   [0.0, 1.0, 0.0],
                   [np.sin(b), 0.0, np.cos(b)]])
    r3 = np.array([[np.cos(g), np.sin(g), 0.0],
                   [-np.sin(g), np.cos(g), 0.0], 
                   [0.0, 0.0, 1.0]])
    rrr = np.dot(r3, np.dot(r2, np.dot(r1, rr)))
    return crr, rrr
