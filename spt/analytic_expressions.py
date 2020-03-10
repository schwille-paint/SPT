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

#%%
def ecdf(x):
    """
    Calculate experimental continuous distribution function (ECDF) of random variable x
    so that counts(value)=probability(x>=value). I.e last value of counts=1.
    
    Equivalent to :
        matplotlib.pyplot.hist(tau_dist,bins=numpy.unique(tau_dist),normed=True,cumulative=True)
    
    Parameters
    ---------
    x : numpy.ndarray
        1 dimensional array of random variable  
    Returns
    -------
    values : numpy.ndarray
         Bins of ECDF corresponding to unique values of x.
    counts : numpy.ndarray
        counts(value)=probability(x<=value).
    """
    x=x[x!=np.nan]
    values,counts=np.unique(x,return_counts=True) # Give unique values and counts in x
    counts_leq=np.cumsum(counts) # Empirical cfd counts(X<=x)
    counts_l=np.concatenate([[0],counts_leq[0:-1]]) # Empirical cfd counts(X<x)
    counts_l = counts_l/counts_leq[-1] # normalize that sum(counts) = 1, i.e. now P(X<x)
    counts_l_inv=1-counts_l # Empirical cfd inverse, i.e. P(X>=x)
    
    return [values,counts_l_inv]

#%%
def gauss_1D(x,x0,sigma,A):
    '''
    

    Args:
        x (TYPE): DESCRIPTION.
        A (TYPE): DESCRIPTION.
        x0 (TYPE): DESCRIPTION.
        sigma (TYPE): DESCRIPTION.

    Returns:
        y (TYPE): DESCRIPTION.

    '''
    y=np.absolute(A)*np.exp(-(x-x0)**2/sigma**2)
    
    return y


#%%
def gauss_Ncomb(x,p,N):
    
    y=0
    for i in range(N):
        ### Version with least degrees of freedom
        # y+=gauss_1D(x,(i+1)*p[0],np.sqrt(i+1)*p[1],p[2+i])
        ### Allow x0 to be different to dx
        # y+=gauss_1D(x,p[0]+i*p[1],np.sqrt(i+1)*p[2],p[3+i])
        ### Allow power law spacing
        y+=gauss_1D(x,p[0]*(i+1)**p[1],np.sqrt(i+1)*p[2],p[3+i])
    return y
    
    