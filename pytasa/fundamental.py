# -*- coding: utf-8 -*-
"""
pytasa.fundamental - basic operations with elastic constant matrices

This module provides functions that operate on single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""

import numpy as np

# FIXME: this is a good place to put things like phase velocity 
#        calculation or tensor rotation. Basic rule should be that
#        the functions take a 6x6 np. array (or matrix?) as input
#        and return something. We can then build an OO interface on top


def velo(X,rh,C):
    """PHASE-VELOCITY SURFACES IN AN ANISOTROPIC MEDIUM
       revised April 1991
           X(3) - DIRECTION OF INTEREST
           RHO - DENSITY
           V - PHASE VELOCITIES 0,1,2= P,S,SS)
           EIGVEC[3,3] - eigenvectors stored by columns
       Translated to MATLAB by James Wookey   
       Translated to Python by Alan Baird      
    """
    ijkl = np.array([[0,5,4],
                     [5,1,3],
                     [4,3,2]])
                     
    # Form symmetric matrix Tik=Cijkl*Xj*Xl (summation convention)
    # note that this mostly-unrolled approach is ~5x faster than the
    # direct (four-looping) construction and allows us to use the 
    # symmetry of T to reduce the cost. I suspect there is a better
    # way, involving kron(X,X') and a single line for the summation, 
    # but I cannot see it.

    #T = np.zeros((3,3))
    #
    #for j in range(0,3):
    #    T[0,0]=T[0,0] + sum(C[ijkl[0,j],ijkl[0,:]]*X[j]*X[:])
    #    T[0,1]=T[0,1] + sum(C[ijkl[0,j],ijkl[1,:]]*X[j]*X[:])
    #    T[0,2]=T[0,2] + sum(C[ijkl[0,j],ijkl[2,:]]*X[j]*X[:])
    #    T[1,1]=T[1,1] + sum(C[ijkl[1,j],ijkl[1,:]]*X[j]*X[:])
    #    T[1,2]=T[1,2] + sum(C[ijkl[1,j],ijkl[2,:]]*X[j]*X[:])
    #    T[2,2]=T[2,2] + sum(C[ijkl[2,j],ijkl[2,:]]*X[j]*X[:])
    #    
    ## Impose the symmetry (build the lower-left corner).
    #T[1,0] = T[0,1]
    #T[2,0] = T[0,2]
    #T[2,1] = T[1,2]
    
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


    
def isIsotropic(C,tol):
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


def cart2(inc, azm):
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


    
def V_rot_gam(V,gam):
    """docstring for V_rot_gam"""
    #  Make rotation matrix
    g = gam * np.pi/180.
    RR = [ [np.cos(g), np.sin(g), 0] , [-np.sin(g), np.cos(g), 0] , [0, 0, 1] ] 
    VR = np.dot(V, RR) 
    return VR


def V_rot_bet(V,bet):
    """docstring for V_rot_bet"""
    #  Make rotation matrix
    b = bet * np.pi/180.
    RR = [ [np.cos(b), 0, -np.sin(b)] , [0, 1, 0], [np.sin(b), 0, np.cos(b)]] 
    VR = np.dot(V,RR) 
    return VR


def phasevels(C,rh,incl,azim):
    """docstring for PS_phasevels"""
    
    #Convert inc and azi to arrays of at least 1 dimension otherwise can't iterate over scalars
    #C=Cin
    #rh=rho
    inc=np.atleast_1d(incl)
    azi=np.atleast_1d(azim)
    
    if (np.size(inc)!=np.size(azi)):
        print 'AZI and INC must be scalars or vectors of the same dimension'
        return
        

    isotol = np.sqrt(np.spacing(1)); # Mbars
    

    #Check that C is valid (if check not suppressed)
    #MS_checkC(C);
      
    #  ** convert GPa to MB file units (Mbars), density to g/cc
    

    #C[:] = C[:] * 0.01
    C[:] = C[:] * 0.00001
    rh = rh / 1e3 
    
    avs = np.zeros(np.size(azi))
    vp  = np.zeros(np.size(azi))
    vs1 = np.zeros(np.size(azi))
    vs2 = np.zeros(np.size(azi))
    pol = np.zeros(np.size(azi))
    S1  = np.zeros((np.size(azi),3)) 
    S1P = np.zeros((np.size(azi),3)) 
    S2P = np.zeros((np.size(azi),3)) 
    
    # ** Handle isotropic case quickly
    #if isIsotropic(C, isotol):
    #    vp[:] = np.sqrt(( ((1.0/3.0)*(C[0,0]+2*C[0,1]))+((4.0/3.0)*C[3,3]) )/rh)*10.0
    #    vs1[:] = np.sqrt(C[3,3]/rh)*10.0  # Factor of 10 converts from
    #    vs2 = vs1                      # Mbar to Pa.
    #    avs[:] = 0.0
    #    pol[:] = np.nan # Both waves have same velocity... meaningless.
    #    S1P[:] = np.nan
    #    S2P[:] = np.nan
    #    return pol,avs,vs1,vs2,vp,S1P,S2P

    #start looping
    for ipair in range(np.size(inc)):
        cazi = azi[ipair]
        cinc = inc[ipair]

        # create the cartesian vector
    	XI = cart2(cinc,cazi)

        # Compute phase velocities		
    	V,EIGVEC=velo(XI,rh,C)
        #print 'V',V
		
        # pull out the eigenvectors
        P  = EIGVEC[:,0]
        S1 = EIGVEC[:,1]
        S2 = EIGVEC[:,2]

          
        # calculate projection onto propagation plane      
        
        S1N = np.cross(XI,S1)
        S1P[ipair,:] = np.cross(XI,S1N)
        S2N = np.cross(XI,S2)
        S2P[ipair,:] = np.cross(XI,S2N)

        #rotate into y-z plane to calculate angles
        #     (use functions optimised for the two needed 
        #      rotations, see below).
        
        S1PR  = V_rot_gam(S1P[ipair,:],cazi) 
    	S1PRR = V_rot_bet(S1PR,cinc) 

        ph = np.arctan2(S1PRR[1],S1PRR[2]) * 180/np.pi 

        #  transform angle to between -90 and 90
        if (ph < -90.): ph = ph + 180.
        if (ph >  90.): ph = ph - 180.

        #	** calculate some useful values
        dVS =  (V[1]-V[2]) 
        VSmean = (V[1]+V[2])/2.0 

        avs[ipair] = 100.0*(dVS/VSmean) 
        vp[ipair] =  V[0]
        vs1[ipair] = V[1]
        vs2[ipair] = V[2]
        pol[ipair] = ph


    ## If any directions have zero avs (within machine accuracy)
    ## set pol to np.nan - array wise:
    #isiso = float(avs > np.sqrt(np.spacing(1))) # list of 1.0 and 0.0.
    #pol = pol * (isiso/isiso) # times by 1.0 or np.nan. 
    #
    #S1P[:,0] = S1P[:,0] * (isiso/isiso);
    #S1P[:,1] = S1P[:,1] * (isiso/isiso);
    #S1P[:,2] = S1P[:,2] * (isiso/isiso);
    
    return pol,avs,vs1,vs2,vp,S1P,S2P