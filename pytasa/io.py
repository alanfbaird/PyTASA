# -*- coding: utf-8 -*-
"""
pytasa.io - read and write elastic constants data

This module provides functions that read or write single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""

import numpy as np

def load_ematrix(filename):

    Cout = np.zeros((6,6))

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # Skip the header...
            if i > 2:
                cijstr = line.split()
                for j in range(6):
                    Cout[i-3, j] = float(cijstr[j])

    # Units are Mbar - convert to GPa!
    Cout = Cout * 100.0

    return Cout
            
