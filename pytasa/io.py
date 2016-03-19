# -*- coding: utf-8 -*-
"""
pytasa.io - read and write elastic constants data

This module provides functions that read or write single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""
import os
import gzip
import numpy as np


class PytasaIOError(Exception):
    pass

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


def convert_to_gpa(C, units):
    """Convert the elastic constants in C to GPa from specified units"""
    units = units.lower()
    if units == "gpa":
        pass
    elif units == "pa":
        C = C / 1.0E9
    elif units == "mbar":
        C = C * 100.0
    elif units == "bar":
        C = C / 10.0E3
    else:
        raise ValueError("Elasticity unit not recognised")
    return C


def convert_to_kgm3(rho, units):
    """Convert the density to Kg/m^3 from specified units"""
    units = units.lower()
    if units == "kgm3":
        pass
    elif units == "gcc":
        rho = rho * 1.0E3
    else:
        raise ValueError("Density unit not recognised")
    return rho


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
    Cout = convert_to_gpa(Cout, "Mbar")

    return Cout


@openfile
def load_mast_simple(fh):
    """Load a MSAT 'simple' file"""
    c_seen = np.zeros((6,6),dtype=bool)
    c_out = np.zeros((6,6))
    rho_seen = False
    rho = None

    for i, line in enumerate(fh):
        # Strip comments and empty lines
        line = line.split('%')[0].strip()
        vals = line.split()
        if len(vals)!=0:
            if len(vals)!=3:
                raise PytasaIOError("Invalid format on line {}".format(i+1))
            try:
                ii = int(vals[0])
                jj = int(vals[1])
                cc = float(vals[2])
            except ValueError:
                raise PytasaIOError("Value not parsed on line {}".format(i+1))
            if (ii >= 1 and ii <= 6) and (jj >= 1 and jj <= 6):
                # A Cij value
                if (not c_seen[ii-1,jj-1]) and (not c_seen[jj-1,ii-1]):
                    c_out[ii-1, jj-1] = cc
                    c_seen[ii-1,jj-1] = True
                    c_out[jj-1, ii-1] = cc
                    c_seen[jj-1,ii-1] = True
                else:
                    raise PytasaIOError("Double specified value on line {}".format(i+1))
            else:
                if not rho_seen:
                    rho = cc
                    rho_seen = True
                else:
                    raise PytasaIOError("Double specified value on line {}".format(i))

    return c_out
             
