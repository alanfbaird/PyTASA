# -*- coding: utf-8 -*-
"""
pytasa.validate - tools for checking elaatic constants

"""

import numpy as np

def cij_stability(cij):
    """Check that the elastic constants matrix is positive
    definite - i.e. that the structure is stable to small
    strains. This is done by finding the eigenvalues by
    diagonalization and checking that they are all positive.
    See Born & Huang, "Dynamical Theory of Crystal Lattices"
    (1954) page 141."""

    stable = False
    try:
        L = np.linalg.cholesky(cij)
        stable = True

    except np.linalg.LinAlgError:
        # Swallow this exception and return False
        print("Crystal not stable to small strains")
        print("(Cij not positive definite)")
        print("Matrix: " + str(cij))

    return stable
