# -*- coding: utf-8 -*-
"""
pytasa.fundamental - basic operations with elastic constant matrices

This module provides functions that operate on single elastic constant
matrices represented as 6x6 numpy arrays (Voigt notation). 
"""

import numpy as np

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
    
    
def rayvel(C,SN,rho):
    """
    TO CALCULATE THE RAY-VELOCITY VECTOR CORRESPONDING TO A SLOWNESS VECTOR.
    Original fortran by David Mainprice as part of the EMATRIX code.
    Converted to Python by Alan Baird
    
    C: Stiffness tensor in Voigt Notation (6X6).
    SN: Slowness vector (3).
    rho: Density
    
    returns VG: Group velocity vector (3)
    
    
    """
    
    ijkl = np.array([[0,5,4],
                     [5,1,3],
                     [4,3,2]])
    
    
    # F[I,K]: COMPONENTS OF DETERMINANT IN TERMS OF COMPONENTS OF THE SLOWNESS VECTOR

    # F = np.zeros((3,3))
    # for i in range(3):
    #     for k in range(3):
    #         F[i,k]=0.0
    #         for j in range(3):
    #             for l in range(3):
    #                 m=ijkl[i,j]
    #                 n=ijkl[k,l]
    #                 F[i,k]=F[i,k]+C[m,n]*SN[j]*SN[l]
    #         if i==k:
    #             F[i,k]=F[i,k]-rho
    
    gamma = np.array([[ SN[0],   0.0,   0.0,   0.0, SN[2], SN[1]],
                      [   0.0, SN[1],   0.0, SN[2],   0.0, SN[0]],
                      [   0.0,   0.0, SN[2], SN[1], SN[0],  0.0]])
    F = np.dot(np.dot(gamma,C),np.transpose(gamma))-np.identity(3)*rho
    
    
    
    # Signed cofactors of F[i,k]
    CF = np.zeros((3,3))
    
    CF[0,0]=F[1,1]*F[2,2]-F[1,2]**2
    CF[1,1]=F[0,0]*F[2,2]-F[0,2]**2
    CF[2,2]=F[0,0]*F[1,1]-F[0,1]**2
    CF[0,1]=F[1,2]*F[2,0]-F[1,0]*F[2,2]
    CF[1,0]=CF[0,1]
    CF[1,2]=F[2,0]*F[0,1]-F[2,1]*F[0,0]
    CF[2,1]=CF[1,2]
    CF[2,0]=F[0,1]*F[1,2]-F[0,2]*F[1,1]
    CF[0,2]=CF[2,0]
    
    # Derivatives of determinant elements
    DF = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                DF[i,j,k]=0.0
                for l in range(3):
                    
                    DF[i,j,k] = DF[i,j,k] + (C[ijkl[i,j],ijkl[k,l]] + C[ijkl[k,j],ijkl[i,l]]) * SN[l]
                    
    # Components of gradient
    DFD = np.zeros(3)
    for k in range(3):
        DFD[k]=0.0
        for i in range(3):
            for j in range(3):
                DFD[k]=DFD[k]+DF[i,j,k]*CF[i,j]
                
    # Normalize to obtain group velocity
    denom = 0.0
    VG = np.zeros(3)
    for i in range(3):
        denom=denom+SN[i]*DFD[i]
    for i in range(3):
        VG[i]=DFD[i]/denom
                
            
    return VG
    


def phasevels(Cin,rh,incl,azim,polout=False):
    """docstring for phasevels"""
    
    #copy C to avoid mutating input
    C=Cin.copy()
    
    #Convert inc and azi to arrays of at least 1 dimension otherwise can't iterate over scalars
    #rh=rho
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
    	XI = cart2(cinc,cazi)
        XIS[ipair,:] = XI

        # Compute phase velocities		
    	V,EIGVEC=velo(XI,rh,C)
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


    # If any directions have zero avs (within machine accuracy)
    # set pol to np.nan - array wise:
    # make sure invalid error warnings are suppressed.
    with np.errstate(invalid='ignore'):
        isiso = (avs > np.sqrt(np.spacing(1))).astype(float) # list of 1.0 and 0.0.
        pol = pol * np.divide(isiso,isiso) # times by 1.0 or np.nan. 
        
        S1P[:,0] = S1P[:,0] * np.divide(isiso,isiso)
        S1P[:,1] = S1P[:,1] * np.divide(isiso,isiso)
        S1P[:,2] = S1P[:,2] * np.divide(isiso,isiso)
    
    
    if polout:
        return pol,avs,vs1,vs2,vp,S1P,S2P,PE,S1E,S2E,XIS        
    else:
        return pol,avs,vs1,vs2,vp,S1P,S2P


def groupvels(Cin,rh,incl,azim,slowout=False):
    """Calculate the group velocity details for an elsticity matrix."""
    
    #copy C to avoid mutating input
    C=Cin.copy()
    
    
    #Convert inc and azi to arrays of at least 1 dimension otherwise can't iterate over scalars
    inc=np.atleast_1d(incl)
    azi=np.atleast_1d(azim)
    
    if (np.size(inc)!=np.size(azi)):
        raise ValueError("AZI and INC must be scalars or vectors of the same dimension")
        

    isotol = np.sqrt(np.spacing(1)); # Mbars
    
    pol,avs,vs1,vs2,vp,S1P,S2P,PE,S1E,S2E,XIS = phasevels(Cin,rh,inc,azi,polout=True)
    
    
    # Slowness vectors
    SNP  = np.zeros((np.size(azi),3)) 
    SNS1 = np.zeros((np.size(azi),3)) 
    SNS2 = np.zeros((np.size(azi),3))

    # Group velocity vectors
    VGP  = np.zeros((np.size(azi),3)) 
    VGS1 = np.zeros((np.size(azi),3)) 
    VGS2 = np.zeros((np.size(azi),3))
    

    #start looping
    for ipair in range(np.size(inc)):

        
        # Slowness vectors
        SNP[ipair,:] = XIS[ipair,:]/vp[ipair]
        SNS1[ipair,:] = XIS[ipair,:]/vs1[ipair]
        SNS2[ipair,:] = XIS[ipair,:]/vs2[ipair]
        
        # Group velocity vectors (need to convert density back first)
        VGP[ipair,:]  = rayvel(C,SNP[ipair,:],rh/1e3)
        VGS1[ipair,:] = rayvel(C,SNS1[ipair,:],rh/1e3)
        VGS2[ipair,:] = rayvel(C,SNS2[ipair,:],rh/1e3)
        

    if slowout:
        return VGP, VGS1, VGS2, PE, S1E, S2E, SNP, SNS1, SNS2
    else:
        return VGP, VGS1, VGS2, PE, S1E, S2E,
    
