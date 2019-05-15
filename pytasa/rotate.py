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
    k[5,3] = g[0,1]*g[1,2] + g[0,2]*g[1,1]
    k[5,4] = g[0,2]*g[1,0] + g[0,0]*g[1,2]
    k[5,5] = g[0,0]*g[1,1] + g[0,1]*g[1,0]

    cr = np.dot(np.dot(k, c), k.T)

    return cr


def rot_3(c, alpha, beta, gamma, order=None):
    """Rotate a elastic constants matrix around the three axes.

       Rotate a 6x6 numpy array representing an elastic constants matrix
       (c) by alpha degrees about the 1-axis, beta degrees about the 2-axis,
       and gamma degress about the 3-axies. All rotations are clockwise when
       looking at the origin from the positive axis. The order of the rotations
       matter and the default is to rotate about the 1-axis first, the 2-axis
       second and the 3-axis third. Passing a three-tuple of integers to the
       optional "order" argument can be used to change this. For example:

          rot_3(c, 45, 90, 30, order=(2, 3, 1)

       will result in a rotation of 90 degrees about the 2-axis, 30 degrees
       about the 3-axis then 45 degrees about the 1-axis.
    """
    if order is None:
        order = (1, 2, 3)

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Three rotation matrices - in a list so we can order them
    # given "order"
    r_list = [np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(alpha), np.sin(alpha)],
                        [0.0, -1.0*np.sin(alpha), np.cos(alpha)]]),
              np.array([[np.cos(beta), 0.0, -1.0*np.sin(beta)],
                        [0.0, 1.0, 0.0],
                        [np.sin(beta), 0.0, np.cos(beta)]]),
              np.array([[np.cos(gamma), np.sin(gamma), 0.0],
                        [-1.0*np.sin(gamma), np.cos(gamma), 0.0],
                        [0.0, 0.0, 1.0]])]

    rot_matrix = np.dot(np.dot(r_list[order[2]-1], r_list[order[1]-1]), 
                        r_list[order[0]-1])

    return rot_c(c, rot_matrix)
