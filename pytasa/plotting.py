# -*- coding: utf-8 -*-
"""
pytasa.plotting - functions for plotting
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from .fundamental import phasevels

def _maskevery(length,num):
    """
    Function for masking arrays except for every numth entry
    Useful for plotting tick marks less frequently
    """
    mask = np.zeros(length)
    mask[::num]=1
    return np.expand_dims(mask,axis=1)
    

def plot_hemi(c,rho,maxsws=None,title=None,scale=200,width=0.004):
    
    # set up inclinations and azimuths 
    inc,az=np.meshgrid(np.arange(90,-1,-6),np.arange(0,361,6))

    # Calculate the polarizations and velocities
    #print "c: ",c
    pol,avs,vs1,vs2,vp=phasevels(c,rho,inc.flatten(),az.flatten())
    #print "c ",c

    # convert inclinations so that 0=vertical
    inc = 90. - inc

    # switch azimuths to radians for plotting
    az = az*np.pi/180.
    
    # shape anisotropy into a grid for plotting
    avs=avs.reshape(61,16)

    # Set up polar figure
    fig=plt.figure(figsize=(7,7))
    ax=plt.subplot(111,projection = 'polar')
    ax.set_theta_direction(-1) # make theta clockwise positive 
    ax.set_theta_zero_location("N") # make 0 degrees point North
    ax.set_ylim((0,90))

    # contour anisotropy
    contours=ax.contourf(az,inc,avs,100,vmin=0,vmax=maxsws,cmap=plt.cm.jet_r)
    if maxsws != None:
        contours.set_clim(0,maxsws)
    cbar=fig.colorbar(contours,shrink=0.8)
    
    cbar.draw_all()
    #cbar=fig.colorbar(0,maxsws)
    #cbar.set_clim(0,7)
    #cbar.set_ticks([0,maxsws])
    
    cbar.set_label('Anisotropy(%)')


    # set up mask for plotting tick marks at less frequent intervals
    azmask=np.hstack(((_maskevery(61,20)), (_maskevery(61,20)), (_maskevery(61,15)),
                      (_maskevery(61,15)), (_maskevery(61,12)), (_maskevery(61,12)),
                      (_maskevery(61,10)), (_maskevery(61,10)), (_maskevery(61,6)),
                      (_maskevery(61,6)),(_maskevery(61,5)),(_maskevery(61,5)),
                      (_maskevery(61,4)),(_maskevery(61,4)),(_maskevery(61,3)),
                      (_maskevery(61,3))))

    incmask = np.ones(16)
    incmask[1::2] = 0
    incmask=np.tile(incmask,61).reshape(61,16)

    mask = 1-azmask*incmask


    # Mask the arrays

    azm = np.ma.masked_array(az,mask=mask)
    incm = np.ma.masked_array(inc,mask=mask)
    avsm = np.ma.masked_array(avs,mask=mask)
    polm = np.ma.masked_array(pol,mask=mask)

    avsm = avsm.flatten()
    polm = polm.flatten()

    # set up quiver plots 
    X = azm.flatten()
    Y = incm.flatten()
    U=avsm*-np.sin(polm*np.pi/180)
    V=avsm*-np.cos(polm*np.pi/180)

    points=ax.quiver(X,Y,V*np.sin(X) +U*np.cos(X),V*np.cos(X)-U*np.sin(X),pivot='mid',headwidth=0,headlength=0,headaxislength=0,scale=scale,scale_units=None,linewidths=(0.5,),edgecolors=('k'),width=width)
    
    if title != None:
        ax.set_title(title)
    
    return fig
