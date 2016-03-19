# -*- coding: utf-8 -*-
"""
pytasa.io - read and write elastic constants data

This module provides functions that read or write single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""
import os
import gzip
from functools import wraps

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
    @wraps(read_function)
    def wrapped_function(f, **kwargs):
        if isinstance(f, str):
            if os.path.splitext(f)[1] == '.gz':
                with gzip.open(f, 'rt') as f:
                    return read_function(f, **kwargs)
            else:
                with open(f, 'r') as f:
                    return read_function(f, **kwargs)
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


def unnormalise_density(c, rho):
    """Reverses any desnity normalisation

    As a general rule this should be done using the units read in from the file,
    with unit conversion done afterwards"""

    return c * rho


@openfile
def load_ematrix(fh, eunit="Mbar"):
    """Load an 'ematrix' file

    This file format does not support storage or density, or density normalisation. The
    default 'ematrix' unit of elasticity is Mbar but files with other units exist. Returns
    a 6x6 numpy array containing the elastity in GPa and in Voigt notation.

    Note that the header information and line of rotations are not used."""
    Cout = np.zeros((6,6))

    for i, line in enumerate(fh):
        # Skip the header...
        if i > 2:
            # We really should do some sanity checking...
            cijstr = line.split()
            for j in range(6):
                Cout[i-3, j] = float(cijstr[j])

    # Units are normally Mbar - convert to GPa!
    Cout = convert_to_gpa(Cout, eunit)

    return Cout


@openfile
def load_mast_simple(fh, eunit="GPa", dunit="Kgm3", dnorm=False):
    """Load a MSAT 'simple' file

    This file format consists of a serise of lines, each with two integers and a float,
    the integers represent the indicies of the elastic constant (in Voigt notation) and
    the float is it's value. Major and minor symmetry is assumed and only one of each 
    pair of off diagonal elements can be provided. Elements outside the range i=(1,6),
    j=(1,6) are assumed to be the density. Only one desnity can be present. Empty
    lines are ignored and the characters after the comment symbol '%' are removed 
    prior to reading. Any other characters result in an error. 

    The file format does not carry unit information, so this needs to be supplied
    by the user. The default is GPa for elasticity and Kgm^-1 ('kgm3') for the density.
    Other units can be provided via the eunit and dunit keyword arguments. 

    Sometimes elasticity is provided in density normalised form (i.e. units of velocity
    squared). This normalisation is removed if the dnorm optional argument is provided.
    
    The function returns a 6x6 numpy array representing the elastic constants in GPa and
    Voigt notation, and a float representing the density in Kgm^-3."""
    c_seen = np.zeros((6,6),dtype=bool)
    c_out = np.zeros((6,6))
    rho_seen = False
    rho = None

    for i, line in enumerate(fh):
        # Strip comments and empty lines and chop up
        vals = line.split('%')[0].strip().split()
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

    if dnorm:
        c_out = unnormalise_density(c_out, rho)
    c_out = convert_to_gpa(c_out, eunit)
    rho = convert_to_kgm3(rho, dunit) 

    return c_out, rho
             
