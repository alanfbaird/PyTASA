# -*- coding: utf-8 -*-
"""
pytasa.vti - tools for working with VTI symmetry elastic constants

"""

import numpy as np

from .validate import cij_stability
from .rotate import rot_c

def planar_ave(this_cij):

    # Find VTI elastic constants
    if (np.amax(this_cij) <= 0.0) and (np.amin(this_cij) >= 0.0):
        # no data at this point. Fill with NaN
        xi = float('nan')
        phi = float('nan')

    else:
        # Find VRH mean by rotating around X3 axis
        voigt_cij = np.zeros((6,6))
        reuss_sij = np.zeros((6,6))
        num_rots = 0.0
        for theta in range(-180, 180, 5):
            theta = np.radians(theta)
            g = np.array([[np.cos(theta), np.sin(theta), 0.0],
                          [-np.sin(theta), np.cos(theta), 0.0],
                          [0.0, 0.0, 1.0]])
            rot_cij = rot_c(this_cij, g)
            assert cij_stability(rot_cij)
            voigt_cij = voigt_cij + rot_cij
            reuss_sij = reuss_sij + np.linalg.inv(rot_cij)
            num_rots = num_rots + 1.0
        voigt_cij = voigt_cij / num_rots
        reuss_cij = np.linalg.inv(reuss_sij / num_rots)
        vrh_cij = (voigt_cij + reuss_cij) / 2.0

        # Now build new matrix enforcing VTI symmetry
        vti_cij = np.zeros((6,6))
        vti_cij[0,0] = (vrh_cij[0,0] + vrh_cij[1,1])/2.0
        vti_cij[1,1] = vti_cij[0,0]
        vti_cij[2,2] = vrh_cij[2,2]
        vti_cij[3,3] = (vrh_cij[3,3] + vrh_cij[4,4])/2.0
        vti_cij[4,4] = vti_cij[3,3]
        vti_cij[5,5] = vrh_cij[5,5]
        vti_cij[0,1] = vti_cij[0,0] - 2.0*vti_cij[5,5]
        vti_cij[0,2] = (vrh_cij[0,2] + vrh_cij[1,2])/2.0
        vti_cij[1,2] = vti_cij[0,2]
        # Lower half
        vti_cij[1,0] = vti_cij[0,1]
        vti_cij[2,1] = vti_cij[1,2]
        vti_cij[2,0] = vti_cij[0,2]

        assert cij_stability(vti_cij)

        # Calculate velocities - for xi and phi the density does not matter
        vph, vpv, vsh, vsv = vti_velocities(vti_cij, 20000.0)
        # Calculate anisotropy measures - note that v and h swap for s and p
        phi = (vpv**2/vph**2)
        xi = (vsh**2/vsv**2)

        return xi, phi, vti_cij


def vti_velocities(c, r):

    c = c * r

    A = (3.0/8.0)*(c[0,0]+c[1,1])+0.25*c[0,1]+0.5*c[5,5]
    C = c[2,2]
    F = 0.5*(c[0,2]+c[1,2]) 
    N = (1.0/8.0)*(c[0,0]+c[1,1])-0.25*c[0,1]+0.5*c[5,5] 
    L = 0.5*(c[3,3]+c[4,4])

    vph = np.sqrt(A/r)
    vpv = np.sqrt(C/r)

    vsh = np.sqrt(N/r)
    vsv = np.sqrt(L/r)

    return vph, vpv, vsh, vsv
