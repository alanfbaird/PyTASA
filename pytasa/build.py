# -*- coding: utf-8 -*-
"""
pytasa.build - funtions to build elastic constant matrices

This module provides functions that take input parameters and return 
elastic constant matrices represented as 6x6 numpy arrays (Voigt notation). 
"""

import numpy as np






def cvtifrac_znzt(vp_v,vs_v,rho,znzt,xi,alp,eps,gam,delt):
    """
    Generates a stiffness tensor for one vertical fracture set and a vti fabric
    using an input zn and zt 
    
    Input parameters are:
       vp = vertical P-wave velocity
       vs = vertical S-wave velocity
       eps = Thomsen's epsilon
       gam = Thomsen's gamma
       del = Thomsen's delta
       xi = Fracture density of the fracture set
       alp = strike of the fracture set
    
    
    Original Fortran by J.P. Verdon
    Converted to Python by A.F. Baird
    """
    zth = xi
    znh = znzt*zth
    
    alpha=90.+alp
    #alpha=90.+alp
    


    # Find Gueguen h values
    c=cvti(vp,vs,rho,0.0,0.0,0.0)
    s,ae,anu,amu,h = get_s_nus_mus_and_es(c)
    zn=znh/h[0]
    zt=zth/h[0]
    c=cvti(vp,vs,rho,eps,gam,delt)   # Construct the VTI background



    # Add compliance to background
    s=np.linalg.inv(c)
    s[0,0]=s[0,0]+zn
    s[4,4]=s[4,4]+zt
    s[5,5]=s[5,5]+zt
    # Invert for stiffness matrix:    
    c=np.linalg.inv(s)  ###########################
    
    # Construct the 3x3 rotation matrix 
    rot=np.zeros((3,3))
    rot[2,2]=1.0
    rot[0,0]=np.cos(alpha*np.pi/180)
    rot[1,1]=np.cos(alpha*np.pi/180)
    rot[0,1]=0.0+np.sin(alpha*np.pi/180)
    rot[1,0]=0.0-rot[0,1]
    


    cout=rotR(c,rot)

    
    #rot2=np.zeros((3,3))
    #rot2[2,2]=1.0
    #rot2[0,0]=np.cos(-alpha*np.pi/180)
    #rot2[1,1]=np.cos(-alpha*np.pi/180)
    #rot2[0,1]=0.0-np.sin(-alpha*np.pi/180)
    #rot2[1,0]=0.0-rot[0,1]
    #
    #print rot2
    #
    #cback=rotR(cout,rot2)
    #
    #
    ##cout=rotCij(c,0,alpha*np.pi/180.)
    ##cout=rotCij(c,0,-alpha*np.pi/180.)
    
    return cout
    









def cvti(vp,vs,rho,eps,gam,delt):
    """
    Generates a vti stiffness tensor based on input P and S velocities and
    Thomsen's parameters
   
    Input parameters are:
       vp = P-wave velocity
       vs = S-wave velocity
       eps = Thomsen's epsilon
       gam = Thomsen's gamma
       delt = Thomsen's delta
    """
    c=np.zeros((6,6))
    c[2,2]=vp*vp*rho
    c[3,3]=vs*vs*rho
    c[0,0]=c[2,2]*(2*eps+1)
    c[5,5]=c[3,3]*(2*gam+1)     
    c[1,1]=c[0,0]
    c[4,4]=c[3,3]
    c[0,1]=c[0,0]-2*c[5,5] 
    c[0,2]=np.sqrt(2*delt*c[2,2]*(c[2,2]-c[3,3]) + (c[2,2]-c[3,3])*(c[2,2]-c[3,3])) - c[3,3]     
    if ((2*delt*c[2,2]*(c[2,2]-c[3,3]) + (c[2,2]-c[3,3])*(c[2,2]-c[3,3]) )< 0):
        raise NameError('HiThere')



    c[1,2]=c[0,2]
    
    # Impose the symmetry (build the lower-left corner).
    for i in range(5):
        for j in range(i,6):
            c[j,i] = c[i,j]
            
    return c

  









def get_s_nus_mus_and_es(c):
    """docstring for get_s_nus_mus_and_es"""
    
    s=np.linalg.inv(c) #invert for compliance
    ae=np.zeros(3)
    for i in range(3):
        ae[i]=1.0/s[i,i] #Young's modulus
        
    anu=np.zeros(3)
    anu[0]=ae[0]*(-s[0,1]-s[0,2])/2. # Poisson's ratio
    anu[1]=ae[1]*(-s[1,0]-s[1,2])/2.
    anu[2]=ae[2]*(-s[2,0]-s[2,1])/2.
    amu=np.zeros(3)
    amu[0]=(c[5,5]+c[4,4])/2.    # Average shear modulus
    amu[1]=(c[5,5]+c[3,3])/2.
    amu[2]=(c[4,4]+c[3,3])/2.
    h=np.zeros(3)
    for i in range(3):
        h[i]=3.*ae[i]*(2.-anu[i])/(32.*(1.-anu[i]*anu[i]))
    
    
    return s,ae,anu,amu,h


def rotR(C,R):
    """docstring for rotR"""
    # form the K matrix (based on Bowers 'Applied Mechanics of Solids', Chapter 3)
    K1 = np.array([[ R[0,0]**2, R[0,1]**2, R[0,2]**2 ],
                   [ R[1,0]**2, R[1,1]**2, R[1,2]**2 ],
                   [ R[2,0]**2, R[2,1]**2, R[2,2]**2 ]]) 

    K2 = np.array([[ R[0,1]*R[0,2], R[0,2]*R[0,0], R[0,0]*R[0,1]],
                   [ R[1,1]*R[1,2], R[1,2]*R[1,0], R[1,0]*R[1,1]],
                   [ R[2,1]*R[2,2], R[2,2]*R[2,0], R[2,0]*R[2,1]]])

    K3 = np.array([[ R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2] ],
                   [ R[2,0]*R[0,0], R[2,1]*R[0,1], R[2,2]*R[0,2] ],
                   [ R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2] ]])

    K4 = np.array([[ R[1,1]*R[2,2]+R[1,2]*R[2,1],    R[1,2]*R[2,0]+R[1,0]*R[2,2],    R[1,0]*R[2,1]+R[1,1]*R[2,0]],
                   [ R[2,1]*R[0,2]+R[2,2]*R[0,1],    R[2,2]*R[0,0]+R[2,0]*R[0,2],    R[2,0]*R[0,1]+R[2,1]*R[0,0]],     
                   [ R[0,1]*R[1,2]+R[0,2]*R[1,1],    R[0,2]*R[1,0]+R[0,0]*R[1,2],    R[0,0]*R[1,1]+R[0,1]*R[1,0]]])

    K = np.vstack((np.hstack((K1,2*K2)),np.hstack((K3,K4))))

    #print K
    CR = np.dot(np.dot(K,C),np.transpose(K))
    
    return CR


    
    
    