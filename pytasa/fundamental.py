# -*- coding: utf-8 -*-
"""
pytasa.fundamental - basic operations with elastic constant matrices

This module provides functions that operate on single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""
from __future__ import print_function

import numpy as np


def _velo(X,rh,C):
    """PHASE-VELOCITY SURFACES IN AN ANISOTROPIC MEDIUM
       revised April 1991
           X(3) - DIRECTION OF INTEREST
           RHO - DENSITY
           V - PHASE VELOCITIES 0,1,2= P,S,SS)
           EIGVEC[3,3] - eigenvectors stored by columns
       Translated to MATLAB by James Wookey   
       Translated to Python by Alan Baird      
    """
    gamma = np.array([[ X[0],  0.0,  0.0,  0.0, X[2], X[1]],
                      [  0.0, X[1],  0.0, X[2],  0.0, X[0]],
                      [  0.0,  0.0, X[2], X[1], X[0],  0.0]])
    T = np.dot(np.dot(gamma,C),np.transpose(gamma))
    
    # determine the eigenvalues of symmetric tij
    EIVAL,EIVEC = np.linalg.eig(T)
    
    # calculate velocities and sort
    V_RAW = (np.sqrt(EIVAL/rh))*10.
    IND=np.argsort(V_RAW) 
    #np.argsort() assumes ascending sort, reverse indices for descending IND[::-1]
    V=V_RAW[IND[::-1]]
    EIGVEC = EIVEC
    EIGVEC=EIVEC[:,IND[::-1]]

    return V, EIGVEC


def _is_isotropic(C,tol):
    """Are we isotropic - assume matrix is symmetrical at this point."""
    l = (abs(C[0,0]-C[1,1]) < tol) and (abs(C[0,0]-C[2,2]) < tol) and \
        (abs(C[0,1]-C[0,2]) < tol) and (abs(C[0,1]-C[1,2]) < tol) and \
        (abs(C[3,3]-C[4,4]) < tol) and (abs(C[3,3]-C[5,5]) < tol) and \
        (abs(C[0,3]) < tol) and (abs(C[0,4]) < tol) and (abs(C[0,5]) < tol) and \
        (abs(C[1,3]) < tol) and (abs(C[1,4]) < tol) and (abs(C[1,5]) < tol) and \
        (abs(C[2,3]) < tol) and (abs(C[2,4]) < tol) and (abs(C[2,5]) < tol) and \
        (abs(C[3,4]) < tol) and (abs(C[3,5]) < tol) and (abs(C[4,5]) < tol) and \
        (((C[0,0]-C[0,1])/2.0)-C[3,3] < tol)
    return l


def _cart2(inc, azm):
    """  convert from spherical to cartesian co-ordinates
         north x=100  west y=010 up z=001
         irev=+1 positive vector x
         irev=-1 negative vector x
         NB: pre-converting azm and inc to radians and using
             cos and sin directly (instead to cosd and sind) 
             is ~10x faster making the function ~4x faster.
    """
    azmr = azm*(np.pi/180.0)
    incr = inc*(np.pi/180.0)
    caz=np.cos(azmr)  
    saz=np.sin(azmr)  
    cinc=np.cos(incr) 
    sinc=np.sin(incr) 
    X=[caz*cinc,-saz*cinc,sinc] 
    # normalise to direction cosines
    r=np.sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2])
    X = X/r 
    return X


def _V_rot_gam(V,gam):
    """docstring for V_rot_gam"""
    #  Make rotation matrix
    g = gam * np.pi/180.
    RR = [ [np.cos(g), np.sin(g), 0] , [-np.sin(g), np.cos(g), 0] , [0, 0, 1] ] 
    VR = np.dot(V, RR) 
    return VR


def _V_rot_bet(V,bet):
    """docstring for V_rot_bet"""
    #  Make rotation matrix
    b = bet * np.pi/180.
    RR = [ [np.cos(b), 0, -np.sin(b)] , [0, 1, 0], [np.sin(b), 0, np.cos(b)]] 
    VR = np.dot(V,RR) 
    return VR


def _rayvel(Cin,SN,rho):
    """
    Calculate the ray-velocity vector corresponding to a slowness vector.
    Based on original fortran code by Mike Kendall and Sean Guest as part of ATRAK.
    Converted to Python by Alan Baird
    
    C: Stiffness tensor in Voigt Notation (6X6).
    SN: Slowness vector (3).
    rho: Density
    
    returns VG: Group velocity vector (3)    
    """
    C = Cin/rho

    ijkl = np.array([[0,5,4],
                     [5,1,3],
                     [4,3,2]])

    gamma = np.array([[ SN[0],   0.0,   0.0,   0.0, SN[2], SN[1]],
                      [   0.0, SN[1],   0.0, SN[2],   0.0, SN[0]],
                      [   0.0,   0.0, SN[2], SN[1], SN[0],  0.0]])

    cp = np.dot(np.dot(gamma,C),np.transpose(gamma))

    dcp = np.zeros((3,3,3))
    for r in range(3):
        for i in range(3):
            for l in range(3):
                x = 0.0
                y = 0.0
                for j in range(3):
                    m=ijkl[i,j]
                    n=ijkl[r,l]
                    x = x+C[m,n]*SN[j]
                for k in range(3):
                    m=ijkl[i,r]
                    n=ijkl[k,l]
                    y = y+C[m,n]*SN[k]
                dcp[r,i,l]=x+y

    det=np.zeros(3)            

    for r in range(3):
        for m in range(3):
            a = cp - np.identity(3)
            for i in range(3):
                a[i,m] = dcp[r,i,m]
            det[r]= det[r]+np.linalg.det(a)

    den = 0.0
    VG = np.zeros(3)
    for i in range(3):
        den = den + SN[i]*det[i]

    for i in range(3):
        VG[i] = det[i]/den    

    return VG


def phasevels(Cin,rh,incl,azim,vecout=False):
    """Calculate the group velocity details for an elsticity matrix. 

        Usage: 
            VGP, VGS1, VGS2, PE, S1E, S2E = groupvels( Cin,rh,incl,azim )                    
                Calculate group velocity vectors from elasticity matrix C (in GPa) and
                density rh (in kg/m^3) corresponding to a phase angle defined by
                an inclination and azimuth (both in degrees). Additionally P, S1 and
                S2-wave polarisations are output in vector form.
        
            VGP, VGS1, VGS2, PE, S1E, S2E, SNP, SNS1, SNS2, VPP, VPS1, VPS2 = 
                                             groupvels( Cin,rh,incl,azim,slowout=True )
                Additionally output P, S1 and S2-wave slownesses (SNP, ...) and 
                phase velocities (VPP, ...) in vector form, as calculated by phasevels.
        
        
        Notes:
            Based on original fortran code by Mike Kendall and Sean Guest as part of ATRAK.
            Converted to Python by Alan Baird.
    """
    # Copy C to avoid mutating input
    C=Cin.copy()

    # Convert inc and azi to arrays of at least 1 dimension otherwise can't iterate over scalars
    # rh=rho
    inc=np.atleast_1d(incl)
    azi=np.atleast_1d(azim)

    if (np.size(inc)!=np.size(azi)):
        raise ValueError("AZI and INC must be scalars or vectors of the same dimension")

    isotol = np.sqrt(np.spacing(1)); # Mbars

    #Check that C is valid (if check not suppressed)
    #MS_checkC(C);

    #  ** convert GPa to MB file units (Mbars), density to g/cc
    C[:] = C[:] * 0.01

    rh = rh / 1e3 

    avs = np.zeros(np.size(azi))
    vp  = np.zeros(np.size(azi))
    vs1 = np.zeros(np.size(azi))
    vs2 = np.zeros(np.size(azi))
    pol = np.zeros(np.size(azi))
    S1P = np.zeros((np.size(azi),3)) 
    S2P = np.zeros((np.size(azi),3)) 

    # Eigenvectors
    PE  = np.zeros((np.size(azi),3)) 
    S1E = np.zeros((np.size(azi),3)) 
    S2E = np.zeros((np.size(azi),3)) 

    # Cartesion propagation vectors
    XIS = np.zeros((np.size(azi),3)) 

    #start looping
    for ipair in range(np.size(inc)):
        cazi = azi[ipair]
        cinc = inc[ipair]

        # create the cartesian vector
        XI = _cart2(cinc,cazi)
        XIS[ipair,:] = XI

        # Compute phase velocities      
        V,EIGVEC=_velo(XI,rh,C)
        #print 'V',V

        # pull out the eigenvectors
        P  = EIGVEC[:,0]
        S1 = EIGVEC[:,1]
        S2 = EIGVEC[:,2]

        PE[ipair,:]  = P
        S1E[ipair,:] = S1
        S2E[ipair,:] = S2

        # calculate projection onto propagation plane      

        S1N = np.cross(XI,S1)
        S1P[ipair,:] = np.cross(XI,S1N)
        S2N = np.cross(XI,S2)
        S2P[ipair,:] = np.cross(XI,S2N)

        #rotate into y-z plane to calculate angles
        #     (use functions optimised for the two needed 
        #      rotations, see below).

        S1PR  = _V_rot_gam(S1P[ipair,:],cazi) 
        S1PRR = _V_rot_bet(S1PR,cinc) 

        ph = np.arctan2(S1PRR[1],S1PRR[2]) * 180/np.pi 

        #  transform angle to between -90 and 90
        if (ph < -90.): ph = ph + 180.
        if (ph >  90.): ph = ph - 180.

        #   ** calculate some useful values
        dVS =  (V[1]-V[2]) 
        VSmean = (V[1]+V[2])/2.0 

        avs[ipair] = 100.0*(dVS/VSmean) 
        vp[ipair] =  V[0]
        vs1[ipair] = V[1]
        vs2[ipair] = V[2]
        pol[ipair] = ph

    # If any directions have zero avs (within machine accuracy)
    # set pol to np.nan - array wise:
    # make sure invalid error warnings are suppressed.
    with np.errstate(invalid='ignore'):
        isiso = (avs > np.sqrt(np.spacing(1))).astype(float) # list of 1.0 and 0.0.
        pol = pol * np.divide(isiso,isiso) # times by 1.0 or np.nan. 

        S1P[:,0] = S1P[:,0] * np.divide(isiso,isiso)
        S1P[:,1] = S1P[:,1] * np.divide(isiso,isiso)
        S1P[:,2] = S1P[:,2] * np.divide(isiso,isiso)

    # if polout:
    #     return pol,avs,vs1,vs2,vp,S1P,S2P,PE,S1E,S2E,XIS
    if vecout:
        VPP  = (XIS.T * vp).T
        VPS1 = (XIS.T * vs1).T
        VPS2 = (XIS.T * vs2).T
        SNP  = (XIS.T * 1/vp).T
        SNS1 = (XIS.T * 1/vs1).T
        SNS2 = (XIS.T * 1/vs2).T
        return pol,avs,vs1,vs2,vp,PE,S1E,S2E,VPP,VPS1,VPS2,SNP,SNS1,SNS2
    else:
        return pol,avs,vs1,vs2,vp


def groupvels(Cin,rh,incl,azim,slowout=False):
    """Calculate the group velocity details for an elsticity matrix."""

    # copy C to avoid mutating input
    C=Cin.copy()

    #Convert inc and azi to arrays of at least 1 dimension otherwise can't iterate over scalars
    inc=np.atleast_1d(incl)
    azi=np.atleast_1d(azim)

    if (np.size(inc)!=np.size(azi)):
        raise ValueError("AZI and INC must be scalars or vectors of the same dimension")

    isotol = np.sqrt(np.spacing(1)); # Mbars

    # pol,avs,vs1,vs2,vp,S1P,S2P,PE,S1E,S2E,XIS = phasevels(Cin,rh,inc,azi,polout=True)
    pol,avs,vs1,vs2,vp,PE,S1E,S2E,VPP,VPS1,VPS2,SNP,SNS1,SNS2 = phasevels(Cin,rh,inc,azi,vecout=True)

    #  ** convert density to g/cc

    rh = rh / 1e3

    # Group velocity vectors
    VGP  = np.zeros((np.size(azi),3)) 
    VGS1 = np.zeros((np.size(azi),3)) 
    VGS2 = np.zeros((np.size(azi),3))

    #start looping
    for ipair in range(np.size(inc)):
        
        # Group velocity vectors
        VGP[ipair,:]  = _rayvel(C,SNP[ipair,:],rh)
        VGS1[ipair,:] = _rayvel(C,SNS1[ipair,:],rh)
        VGS2[ipair,:] = _rayvel(C,SNS2[ipair,:],rh)

    if slowout:
        return VGP, VGS1, VGS2, PE, S1E, S2E, SNP, SNS1, SNS2, VPP, VPS1, VPS2
    else:
        return VGP, VGS1, VGS2, PE, S1E, S2E


def cij_stability(cij):
    """Check that the elastic constants matrix is positive definite 

    That is,  check that the structure is stable to small strains. This
    is done by finding the eigenvalues of the Voigt elastic stiffness matrix
    by diagonalization and checking that they are all positive.

    See Born & Huang, "Dynamical Theory of Crystal Lattices" (1954) page 141.
    """
    stable = False
    (eigenvalues, eigenvectors) = np.linalg.eig(cij)
    if (np.amin(eigenvalues) > 0.0):
        stable = True
    else:
        print("Crystal not stable to small strains")
        print("(Cij not positive definite)")
        print("Eigenvalues: " + str(eigenvalues))

    return stable


def invert_cij(cij, ecij):
    """Given a square matrix and a square matrix of the errors
    on each element, return the inverse of the matrix and the
    propogated errors on the inverse.

    We use numpy for the inversion and eq.10 of Lefebvre,
    Keeler, Sobie and White ('Propagation of errors for
    matrix inversion' Nuclear Instruments and Methods in
    Physics Research A 451 pp.520-528; 2000) to calculate
    the errors. The errors can be reported directly as the
    errors on the inverse matrix but to do useful further
    propogation we need to report the covar matrix too.
    This is calculated from eq.9 and we then extract the
    diagonal elements to get the errors (rather than implementing
    eq.10 too).

    Tested with the matrix:
            0.700(7) 0.200(2)
            0.400(4) 0.600(6)
    which gives back the inverse and squared errors reported
    in Table 1 of the above reference.

    This is coded up for an elastic constants matrix (cij) and
    its inverse (the elastic compliance matrix, cij), but should
    work for any rank 2 square matrix.
    """
    if (np.ndim(cij) != 2):
        raise ValueError("Matrix must be rank 2")
    if (np.shape(cij)[0] != np.shape(cij)[1]):
        raise ValueError("Matrix must be square")
    if (np.shape(cij) != np.shape(ecij)):
        raise ValueError("Matrix and error matrix must have same shape")

    # Calculate the inverse using numpy
    sij = np.linalg.inv(cij)

    # Set up output arrays (init as zeros)
    esij = np.zeros_like(ecij)
    array_size = esij[0].size
    vcovsij = np.zeros((array_size,array_size,array_size,array_size),
                       dtype=type(esij))

    # Build covariance arrays (i.e COV(C^-1[a,b],S^-1[b,c] - a 4d array).
    # This is an implementation of eq.9 of Lefebvre et al.
    for a in range (array_size):
        for b in range (array_size):
            for c in range (array_size):
                for d in range (array_size):
                    for i in range (array_size):
                        for j in range (array_size):
                            vcovsij[a,b,c,d] = vcovsij[a,b,c,d] + \
                             ((sij[a,i]*sij[c,i]) * (ecij[i,j]**2) * \
                              (sij[j,b]*sij[j,d]))

    # Extrct the "diagonal" terms, which are
    # the errors on the elements of the inverse
    # and could also be calculated using eq.10
    for a in range (array_size):
        for b in range (array_size):
            esij[a,b] = np.sqrt(vcovsij[a,b,a,b])

    return (sij, esij, vcovsij)

