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
def load_msat_simple(fh, eunit="GPa", dunit="Kgm3", dnorm=False, symmetry=None):
    """Load a MSAT 'simple' file

    This file format consists of a series of lines, each with two integers and a float,
    the integers represent the indicies of the elastic constant (in Voigt notation) and
    the float is its value. Major and minor symmetry is assumed and only one of each 
    pair of off diagonal elements can be provided. Elements outside the range i=(1,6),
    j=(1,6) are assumed to be the density. Only one density can be present. Empty
    lines are ignored and the characters after the comment symbol '%' are removed 
    prior to reading. Any other characters result in an error. 

    The file format does not carry unit information, so this needs to be supplied
    by the user. The default is GPa for elasticity and Kgm^-1 ('kgm3') for the density.
    Other units can be provided via the eunit and dunit keyword arguments. 

    Sometimes elasticity is provided in density normalised form (i.e. units of velocity
    squared). This normalisation is removed if the dnorm optional argument is provided.
    
    The function returns a 6x6 numpy array representing the elastic constants in GPa and
    Voigt notation, and a float representing the density in Kgm^-3.

    Setting the symmetry keyword argument to something other than `None` will 
    fill out the elastic tensor based on symmetry, defined by the string mode. 
    This can take the following values:
         None   - nothing attempted, unspecified Cijs are zero (default)
        'auto'  - assume symmetry based on number of Cijs specified 
        'iso'   - isotropic (nCij=2) ; C33 and C66 must be specified.
        'hex'   - hexagonal (nCij=5) ; C11, C33, C44, C66 and C13 must be
                  specified, x3 is symmetry axis
        'vti'   - synonym for hexagonal
        'cubic' - cubic (nCij=3) ; C33, C66 and C12 must be specified
    """
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
                    if symmetry is None:
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

    if symmetry is not None:
        c_out = expand(c_out, symmetry)

    if dnorm:
        c_out = unnormalise_density(c_out, rho)
    c_out = convert_to_gpa(c_out, eunit)
    rho = convert_to_kgm3(rho, dunit) 

    return c_out, rho


def expand(c_in, mode='auto'):
    """Expand a minimal set of elastic constants based on a specifed symmetry to a full Cij tensor. 

    Fill out elastic tensor C based on symmetry, defined by mode. This can take the following 
    values:
        'auto'  - assume symmetry based on number of Cijs specified 
        'iso'   - isotropic (nec=2) ; C33 and C66 must be specified.
        'hex'   - hexagonal (nec=5) ; C33, C44, C11, C66 and C13 must be specified, x3 is symmetry
                  axis
        'vti'   - synonym for hexagonal
        'cubic' - cubic (nec=3) ; C33, C66 and C12 must be specified
        'ortho' - orthorhombic (nec=9); All six diagonal (C11-C66), C12,
                  C13 and C23 must be specified

    Cijs *not* specified in the appropriate symmetry should be zero in the input matrix. 
    """

    if mode == 'auto':
        nelc = np.count_nonzero(c_in)
        if nelc == 2:
            mode = 'iso'
        elif nelc == 3:
            mode = 'cubic'
        elif nelc == 5:
            mode = 'hex'
        elif nelc == 9:
            mode = 'ortho'
        else:
            raise PytasaIOError("Auto expansion not supported for {} constants".format(nelc))
    elif mode == 'vti':
        mode = 'hex'
        
    if mode == 'iso':
        c_out = expand_isotropic(c_in[2,2], c_in[5,5])
    elif mode == 'cubic':
        c_out = expand_cubic(c_in[2,2], c_in[5,5], c_in[0,1])
    elif mode == 'hex':
        c_out = expand_hexagonal(c_in[0,0], c_in[2,2], c_in[3,3], c_in[0,0]-2.0*c_in[5,5], c_in[0,2])
    elif mode == 'ortho':
        c_out = expand_orthorhombic(c_in[0,0], c_in[1,1], c_in[2,2], c_in[3,3], c_in[4,4],
                                    c_in[5,5], c_in[0,1], c_in[0,2], c_in[1,2])
    else:
        raise PytasaIOError("Symmetry expansion not supported for mode {}".format(mode))

    return c_out


def expand_isotropic(c11, c44):
    """Return an isotropic  Cij tensor given two elastic constants"""
    # Note: c11 is M and c44 is mu, so use build_iso
    return build_iso(M=c11, mu=c44)[0]


def expand_cubic(c11,c44,c12):
    """Return a Cij tensor of cubic symmetry based on a list of elastic constants"""
    
    C = np.array([[ c11, c12, c12, 0.0, 0.0, 0.0], 
                  [ c12, c11, c12, 0.0, 0.0, 0.0], 
                  [ c12, c12, c11, 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, c44, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, c44, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, c44]])
    
    return C
    
def expand_hexagonal(c11,c33,c44,c12,c13):
    """Return a Cij tensor of hexagonal symmetry based on a list of elastic constants"""
    
    c66=(c11-c12)/2.
    
    C = np.array([[ c11, c12, c13, 0.0, 0.0, 0.0], 
                  [ c12, c11, c13, 0.0, 0.0, 0.0], 
                  [ c13, c13, c33, 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, c44, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, c44, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, c66]])
    
    return C

def expand_trigonal(c11,c33,c44,c12,c13,c14,c15=0.0):
    """Return a Cij tensor of trigonal symmetry based on a list of elastic constants"""
    
    c66=(c11-c12)/2.
    
    C = np.array([[ c11, c12, c13, c14, c15, 0.0], 
                  [ c12, c11, c13,-c14,-c15, 0.0], 
                  [ c13, c13, c33, 0.0, 0.0, 0.0],
                  [ c14,-c14, 0.0, c44, 0.0,-c15],
                  [ c15,-c15, 0.0, 0.0, c44, c14],
                  [ 0.0, 0.0, 0.0,-c15, c14, c66]])
    return C

def expand_orthorhombic(c11,c22,c33,c44,c55,c66,c12,c13,c23):
    """Return a Cij tensor of orthorhombic symmetry based on a list of elastic constants"""
    C = np.array([[ c11, c12, c13, 0.0, 0.0, 0.0], 
                  [ c12, c22, c23, 0.0, 0.0, 0.0], 
                  [ c13, c23, c33, 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, c44, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, c55, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, c66]])
    return C

def expand_tetragonal(c11,c33,c44,c66,c12,c13,c16=0.0):
    """Return a Cij tensor of tetragonal symmetry based on a list of elastic constants"""
    C = np.array([[ c11, c12, c13, 0.0, 0.0, c16], 
                  [ c12, c11, c13, 0.0, 0.0,-c16], 
                  [ c13, c13, c33, 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, c44, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, c44, 0.0],
                  [ c16,-c16, 0.0, 0.0, 0.0, c66]])
    return C

def expand_monoclinic(c11,c22,c33,c44,c55,c66,c12,c13,c23,c15,c25,c35,c46):
    """Return a Cij tensor of monoclinic symmetry based on a list of elastic constants"""
    C = np.array([[ c11, c12, c13, 0.0, c15, 0.0], 
                  [ c12, c22, c23, 0.0, c25, 0.0], 
                  [ c13, c23, c33, 0.0, c35, 0.0],
                  [ 0.0, 0.0, 0.0, c44, 0.0, c46],
                  [ c15, c25, c35, 0.0, c55, 0.0],
                  [ 0.0, 0.0, 0.0, c46, 0.0, c66]])
    return C

    
def build_iso(**kwargs):
    """Given two isotropic moduli create an elasticity matrix for an isotropic material
    and also return all other moduli.
    
    Two elastic moduli are required and should be given as keyword arguments.
    Permitted moduli are:
        lam - first Lame parameter, in GPa    
        mu - shear modulus (second Lame parameter), in GPa
        K - bulk modulus, in GPa.
        E - Young's modulus, in GPa.
        nu - Poisson's ratio, dimensionless.
        M - 'P-wave modulus', in GPa.
        
    The function returns a 6x6 numpy array representing the elastic constants in GPa in
    Voigt notation, and floats representing lam, mu, K, E, nu, and M.
    """
    
    if len(kwargs) != 2:
        raise ValueError('Two (and only two) elastic constants are required')
    
    for key in kwargs:
        if key not in ['lam','mu','K','E','nu','M']:
            raise ValueError('keyword '+key+' not recognized')
        
    if 'lam' in kwargs:
        lam = kwargs['lam']
        if 'mu' in kwargs:
            mu = kwargs['mu']
        elif 'K' in kwargs:
            mu = 3.0*(kwargs['K']-lam)/2.0
        elif 'E' in kwargs:
            E=kwargs['E']
            R = np.sqrt(E**2+9*lam**2+2*E*lam)
            mu = (E-3.0*lam+R)/4.0
        elif 'nu' in kwargs:
            mu = lam*(1.0-2.0*kwargs['nu'])/(2.0*kwargs['nu'])
        elif 'M' in kwargs:
            mu = (kwargs['M']-lam)/2.0
    elif 'mu' in kwargs:
        mu = kwargs['mu']
        if 'K' in kwargs:
            lam = kwargs['K']-2.0*mu/3.0
        elif 'E' in kwargs:
            lam = mu*(kwargs['E']-2.0*mu)/(3.0*mu-kwargs['E'])
        elif 'nu' in kwargs:
            lam = 2.0*mu*kwargs['nu']/(1.0-2.0*kwargs['nu'])
        elif 'M' in kwargs:
            lam = kwargs['M']-2.0*mu
    elif 'K' in kwargs:
        if 'E' in kwargs:
            mu = 3.0*kwargs['K']*kwargs['E']/(9.0*kwargs['K']-kwargs['E'])
            lam = mu*(kwargs['E']-2.0*mu)/(3.0*mu-kwargs['E'])
        elif 'nu' in kwargs:
            lam = 3.0*kwargs['K']*kwargs['nu']/(1+kwargs['nu'])
            mu = lam*(1.0-2.0*kwargs['nu'])/(2.0*kwargs['nu'])
        elif 'M' in kwargs:
            lam = (3.0*kwrgs['K']-kwargs['M'])/2.0
            mu = (kwargs['M']-lam)/2.0
    elif 'E' in kwargs:
        if 'nu' in kwargs:
            mu = kwargs['E']/(2*(1+kwargs['nu']))
            lam = 2.0*mu*kwargs['nu']/(1.0-2.0*kwargs['nu'])
        elif 'M' in kwargs:
            E=kwargs['E']
            M=kwargs['M']
            S=np.sqrt(E**2 + 9*M**2 - 10*E*M)
            lam = (M-E+S)/4.0
            mu = (M-lam)/2.0
    elif 'nu' in kwargs:
        if 'M' in kwargs:
            lam = kwargs['M']*kwargs['nu']/(1-kwargs['nu'])
            mu = (kwargs['M']-lam)/2.0
            
    
    M = 2*mu+lam
    E = (mu*((3*lam)+(2*mu)))/(lam+mu)
    K = lam + (2.0/3.0)*mu
    nu = lam/(2*(lam+mu))
    

    C = np.array([[  M, lam, lam, 0.0, 0.0, 0.0],
                  [lam,   M, lam, 0.0, 0.0, 0.0],
                  [lam, lam,   M, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0,  mu, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0,  mu, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0,  mu]])
    
    
    return C, lam,mu,K,E,nu,M


             
def latexCij(Cij, outputfile, eCij=np.zeros((6,6)),  nt=False):
    """Write out elastic constants and derived properties in a format
    that can be processed as a LaTeX table."""

    nt_string = "& {0:5.1f}$\pm${1:3.1f}  \n"
    wt_string = "c$_{{{0}{1}}}$ & {2:5.1f}$\pm${3:3.1f} \\\\ \n"

    with open(outputfile,"w") as f:
        for i in range(6):
            for j in range(i,6):
                if ((Cij[i,j] != 0.0) and (eCij[i,j] != 0.0)):
                    if (nt):
                        f.write(nt_string.format(Cij[i,j],eCij[i,j]))
                    else:
                        f.write(wt_string.format(i+1,j+1,Cij[i,j],eCij[i,j]))


def txtCij(Cij, filename):
    """Add elastic constants to a text file as a single line
    (e.g. for bulk plotting). Order of elastic constants is:
    C11 C12 C13 ... C16 C22 ... C26 C33 ... C55 C56 C66"""

    with open(filename, "a") as f:
        for i in range(6):
            for j in range(i,6):
                f.write("{0:5.1f} ".format(Cij[i,j]))
            f.write("\n")

