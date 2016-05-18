# -*- coding: utf-8 -*-
"""
pytasa.rotate - rotations and related transforms for elasticity 

"""

import numpy as np

def rot_c(c, g):
    """Rotate a elastic constants matrix given a rotation matrix

       This implementation makes use of the Bond transform, which avoids
       expanding the 6x6 matrix to a 3x3x3x3 tensor.

       See Bowers 'Applied Mechanics of Solids', Chapter 3
    """
    k = np.empty((6,6))
    # k1
    k[0:3,0:3] = g[0:3,0:3]**2

    # k2
    k[0:3,3] = g[0:3,1]*g[0:3,2]*2.0
    k[0:3,4] = g[0:3,2]*g[0:3,0]*2.0
    k[0:3,5] = g[0:3,0]*g[0:3,1]*2.0

    # k3
    k[3,0:3] = g[1,0:3]*g[2,0:3]
    k[4,0:3] = g[2,0:3]*g[0,0:3]
    k[5,0:3] = g[0,0:3]*g[1,0:3]

    # k4
    k[3,3] = g[1,1]*g[2,2] + g[1,2]*g[2,1]
    k[3,4] = g[1,2]*g[2,0] + g[1,0]*g[2,2]
    k[3,5] = g[1,0]*g[2,1] + g[1,1]*g[2,0]
    k[4,3] = g[2,1]*g[0,2] + g[2,2]*g[0,1]
    k[4,4] = g[2,2]*g[0,0] + g[2,0]*g[0,2]
    k[4,5] = g[2,0]*g[0,1] + g[2,1]*g[0,0]
    k[5,3] = g[0,1]*g[1,2] + g[0,1]*g[1,1]
    k[5,4] = g[0,2]*g[1,0] + g[0,0]*g[1,2]
    k[5,5] = g[0,0]*g[1,1] + g[0,1]*g[1,0]

    cr = np.dot(np.dot(k, c), k.T)

    return cr
