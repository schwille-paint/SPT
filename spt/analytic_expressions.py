import numpy as np

#%%
def msd_free(tau,a,b=0):
    '''
    Simple brownian diffusion taking loclization precision into account: msd=a*tau+b.
    According to: Xavier Michalet, Physical Review E, 82, 2010
    '''
    msd=b*tau+b
    return msd

#%%
def msd_anomal(tau,a,b):
    '''
    Anomalous diffusion: msd=a*tau**b
    According to: Carlo Manzo, Report on Progress in Physics, 78, 2015
    '''
    msd=a*tau**b
    return msd

#%%
def exp_tracks_per_frame(x,a,b,c):
    '''
    Exponential like decay to fit number of tracks per frame: NoTracks=a*np.exp(-x/b)+c
    '''
    NoTracks=a*np.exp(-x/b)+c
    return NoTracks