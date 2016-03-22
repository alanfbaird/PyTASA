# -*- coding: utf-8 -*-
"""
pytasa.anisotropy_index - measures of elastic anisotropy

This module provides functions that report on the magnitude of elastic
anisotropy.
"""

import numpy as np


def zenerAniso(Cij,eCij=np.zeros((6,6))):
    """Returns Zener anisotropy index, A, defined as
    2C44/(C11-C12). This is unity for an isotropic crystal
    and, for a cubic crystal C44 and 1/2(C11-C12) are shear
    strains accross the (100) and (110) planes, respectivly.
    See Zener, Elasticity and Anelasticity of Metals, 1948
    or doi:10.1103/PhysRevLett.101.055504 (c.f. uAniso).
    Also returns the error on the anisotriopy index.
    Note that we don't check that the crystal is cubic!"""
    zA = (Cij[3,3]*2)/(Cij[0,0]-Cij[0,1])
    ezA = np.sqrt(((eCij[0,0]/Cij[0,0])**2 + (eCij[0,1]/Cij[0,1])**2) +\
           (2*(eCij[3,3]/Cij[3,3])**2)) * zA
    return (zA, ezA)


def uAniso(Cij,eCij):
    """Returns the Universal elastic anisotropy index defined
    by Ranganathan and Ostoja-Starzewski (PRL 101, 05504; 2008
    doi:10.1103/PhysRevLett.101.055504 ). Valid for all systems."""
    (voigtB, reussB, voigtG, reussG, hillB, hillG,
                       evB, erB, evG, erG, ehB, ehG) = polyCij(Cij,eCij)
    uA = (5*(voigtG/reussG))+(voigtB/reussB)-6
    euA = np.sqrt((np.sqrt((evG/voigtG)**2 +\
                  (erG/reussG)**2)*(voigtG/reussG))**2 + \
                  (np.sqrt((evB/voigtB)**2 + \
                  (erB/reussB)**2)*(voigtB/reussB))**2)
    return (uA, euA)

