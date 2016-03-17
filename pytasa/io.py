# -*- coding: utf-8 -*-
"""
pytasa.io - read and write elastic constants data

This module provides functions that read or write single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""
import os
import gzip
import numpy as np


def openfile(read_function):
    """Decorator to read data from files or file like instances.

       Adding the @openfile decorator to a function designed to read from a
       open file handle permits filenames to be given as arguments. If a string
       argument is given it is treated as a file name and opened prior to the 
       main read function being called. If the file name ends in .gz, the file
       is also uncompressed on the fly.
    """
    def wrapped_function(f):
        if isinstance(f, str):
            if os.path.splitext(f)[1] == '.gz':
                with gzip.open(f, 'rb') as f:
                    return read_function(f)
            else:
                with open(f, 'r') as f:
                    return read_function(f)
        else:
            return read_function(f)

    return wrapped_function


@openfile
def load_ematrix(fh):
    """Load an 'ematrix' file"""
    Cout = np.zeros((6,6))

    for i, line in enumerate(fh):
        # Skip the header...
        if i > 2:
            # We really should do some sanity checking...
            cijstr = line.split()
            for j in range(6):
                Cout[i-3, j] = float(cijstr[j])

    # Units are Mbar - convert to GPa!
    Cout = Cout * 100.0

    return Cout
            
