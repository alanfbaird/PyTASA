# -*- coding: utf-8 -*-
"""
pytasa.fundamental - basic operations with elastic constant matrices

This module provides functions that operate on single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""

import numpy as np

# FIXME: this is a good place to put things like phase velocity 
#        calculation or tensor rotation. Basic rule should be that
#        the functions take a 6x6 np. array (or matrix?) as input
#        and return something. We can then build an OO interface on top

def mat2tens(cij_mat, compl=False):
    """Convert from Voigt to full tensor notation 

       Convert from the 6*6 elastic constants matrix to 
       the 3*3*3*3 tensor representation. Recoded from 
       the Fortran implementation in DRex. Use the optional 
       argument "compl" for the elastic compliance (not 
       stiffness) tensor to deal with the multiplication 
       of elements needed to keep the Voigt and full 
       notation consistant.

    """
    cij_tens = np.zeros((3,3,3,3))
    m2t = np.array([[0,5,4],[5,1,3],[4,3,2]])
    if compl:
        cij_mat = cij_mat / np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                      [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                      [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                      [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                      [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                      [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    cij_tens[i,j,k,l] = cij_mat[m2t[i,j],m2t[k,l]]

    return cij_tensA

def tens2mat(cij_tens, compl=False):
    """Convert from full tensor to Voigt notation

       Convert from the 3*3*3*3 elastic constants tensor to 
       to 6*6 matrix representation. Recoded from the Fortran
       implementation in DRex. Use the optional 
       argument "compl" for the elastic compliance (not 
       stiffness) tensor to deal with the multiplication 
       of elements needed to keep the Voigt and full 
       notation consistant.

    """
    t2m = np.array([[0,1,2,1,2,0],[0,1,2,2,0,1]])
    cij_mat = np.zeros((6,6))
    # Convert back to matrix form
    for i in range(6):
        for j in range(6):
            cij_mat[i,j] = cij_tens[t2m[0,i],t2m[1,i],t2m[0,j],t2m[1,j]]
    
    if compl:
        cij_mat = cij_mat * np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                      [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                      [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                      [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                      [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                      [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]])

    return cij_mat


def rotT(T, g):
    """Rotate a rank 4 tensor, T, using a rotation matrix, g
       
       Tensor rotation involves a summation over all combinations
       of products of elements of the unrotated tensor and the 
       rotation matrix. Like this for a rank 3 tensor:

           T'(ijk) -> Sum g(i,p)*g(j,q)*g(k,r)*T(pqr)
       
       with the summation over p, q and r. The obvious implementation
       involves (2*rank) length 3 loops building up the summation in the
       inner set of loops. This optimized implementation >100 times faster 
       than that obvious implementaton using 8 nested loops. Returns a 
       3*3*3*3 numpy array representing the rotated tensor, Tprime. 

       NB: For Voigt notation we could use Bond transform, as used in MSAT,
       which is probably quicker.

    """
    gg = np.outer(g, g) # Flatterns input and returns 9*9 array
                        # of all possible products
    gggg = np.outer(gg, gg).reshape(4 * g.shape)
                        # 81*81 array of double products reshaped
                        # to 3*3*3*3*3*3*3*3 array...
    axes = ((0, 2, 4, 6), (0, 1, 2, 3)) # We only need a subset 
                                        # of gggg in tensordot...
    return np.tensordot(gggg, T, axes)

