# -*- coding: utf-8 -*-
"""
pytasa.polycrystal - functions to deal with polycrystals

"""

import numpy as np

from .fundamental import invert_cij

def isotropic_limits(Cij, eCij=np.zeros((6,6))):
    """Returns voight-reuss-hill average of elastic constants tensor
    and propogated error given the 6*6 matrix of elastic constants
    and the 6*6 matrix of errors. The errors are optional. Assumes
    no co-varance between the errors on the elastic constants but
    does include the co-varenance on the (calculated) compliance
    matrix."""

    # Need compliances too:
    (sij, eSij, covSij) = invert_cij(Cij, eCij)

    # These equations are valid for all crystal systems (only 9 of
    # the 21 elastic constants ever have to be used, e.g. see Anderson
    # theory of the Earth, pg. 122 or the introduction to Hill, 1952).

    voigtB = (1.0/9)*(Cij[0,0] + Cij[1,1] + Cij[2,2] ) \
           + (2.0/9)*(Cij[0,1] + Cij[0,2] + Cij[1,2])

    evB = np.sqrt( (1.0/81)*(eCij[0,0]**2 + eCij[1,1]**2 + eCij[2,2]**2) \
                  +(2.0/81)*(eCij[0,1]**2 + eCij[0,2]**2 + eCij[1,2]**2) )

    reussB = 1.0/((sij[0,0]+sij[1,1]+sij[2,2]) + 2*(sij[0,1]+sij[0,2]+sij[1,2]))

    # Note that COV(X+Z,Y) = COV(X,Y)+COV(Z,Y) and
    # COV(SUM(Xi),SUM(Yj)) = SUM(SUM(COV(Xi,Yj)
    # c.f. http://mathworld.wolfram.com/Covariance.html
    erB = (np.sqrt(eSij[0,0]**2 + eSij[1,1]**2 + eSij[2,2]**2  \
                   + 4*eSij[0,1]**2 + 4*eSij[0,2]**2 + 4*eSij[1,2]**2  \
                   + 2*covSij[0,0,1,1] + 2*covSij[0,0,2,2] + 2*covSij[1,1,2,2] \
                   + 4*covSij[0,0,0,1] + 4*covSij[0,0,0,2] + 4*covSij[0,0,1,2] \
                   + 4*covSij[1,1,0,1] + 4*covSij[1,1,0,2] + 4*covSij[1,1,1,2] \
                   + 4*covSij[2,2,0,1] + 4*covSij[2,2,0,2] + 4*covSij[2,2,1,2] \
                   + 8*covSij[0,1,0,2] + 8*covSij[0,1,1,2] + 8*covSij[0,2,1,2] )) \
        * reussB**2

    voigtG = (1.0/15)*(Cij[0,0] + Cij[1,1] + Cij[2,2] - \
                       Cij[0,1] - Cij[0,2] - Cij[1,2]) + \
             (1.0/5)*(Cij[3,3] + Cij[4,4] + Cij[5,5])

    evG = np.sqrt( (1.0/225)*(eCij[0,0]**2 + eCij[1,1]**2 + \
                              eCij[2,2]**2 + eCij[0,1]**2 + \
                              eCij[0,2]**2 + eCij[1,2]**2) + \
                    (1.0/25)*(eCij[3,3]**2 + eCij[4,4]**2 + eCij[5,5]**2) )

    reussG = 15.0/(4*(sij[0,0]+sij[1,1]+sij[2,2]) - \
                   4*(sij[0,1]+sij[0,2]+sij[1,2]) + 3*(sij[3,3]+sij[4,4]+sij[5,5]))

    erG = np.sqrt( \
                  16*(eSij[0,0]**2 + eSij[1,1]**2 + eSij[2,2]**2) \
                + 16*(eSij[0,1]**2 + eSij[0,2]**2 + eSij[1,2]**2) \
                +  9*(eSij[3,3]**2 + eSij[4,4]**2 + eSij[5,5]**2) \
                + 32*covSij[0,0,1,1] + 32*covSij[0,0,2,2] + 32*covSij[1,1,2,2] \
                + 32*covSij[0,0,0,1] + 32*covSij[0,0,0,2] + 32*covSij[0,0,1,2] \
                + 32*covSij[1,1,0,1] + 32*covSij[1,1,0,2] + 32*covSij[1,1,1,2] \
                + 32*covSij[2,2,0,1] + 32*covSij[2,2,0,2] + 32*covSij[2,2,1,2] \
                + 32*covSij[0,1,0,2] + 32*covSij[0,1,1,2] + 32*covSij[0,2,1,2] \
                + 24*covSij[0,0,3,3] + 24*covSij[0,0,4,4] + 24*covSij[0,0,5,5] \
                + 24*covSij[1,1,3,3] + 24*covSij[1,1,4,4] + 24*covSij[1,1,5,5] \
                + 24*covSij[2,2,3,3] + 24*covSij[2,2,4,4] + 24*covSij[2,2,5,5] \
                + 24*covSij[0,1,3,3] + 24*covSij[0,1,4,4] + 24*covSij[0,1,5,5] \
                + 24*covSij[0,2,3,3] + 24*covSij[0,2,4,4] + 24*covSij[0,2,5,5] \
                + 24*covSij[1,2,3,3] + 24*covSij[1,2,4,4] + 24*covSij[1,2,5,5] \
                + 18*covSij[3,3,4,4] + 18*covSij[3,3,5,5] + 18*covSij[4,4,5,5] \
                ) * (reussG**2 / 15)

    return (voigtB, reussB, voigtG, reussG, ((voigtB+reussB)/2.0), ((voigtG+reussG)/2.0),
               evB, erB, evG, erG, ((evB+erB)/2), ((evG+erG)/2))

