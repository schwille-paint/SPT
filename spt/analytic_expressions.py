'''
.. _michalet:
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.041914
.. _manzo:
    https://iopscience.iop.org/article/10.1088/0034-4885/78/12/124601
'''
import numpy as np

#%%
def msd_free(tau,a,b=0):
    '''
    MSD fitting eq. for simple brownian diffusion taking localization precision into account: ``msd=a*tau+b``
    According to: Xavier Michalet, Physical Review E, 82, 2010 (michalet_)
    '''
    msd=a*tau+b
    return msd

#%%
def msd_anomal(tau,a,b):
    '''
    MSD fitting eq. for anomalous diffusion: ``msd=a*tau**b``
    According to: Carlo Manzo, Report on Progress in Physics, 78, 2015 (manzo_)
    '''
    msd=a*tau**b
    return msd

#%%
def exp_tracks_per_frame(x,a,b):
    '''
    Exponential like decay to fit number of tracks per frame: ``NoTracks=a*np.exp(-x/b)+c``
    '''
    NoTracks=a*np.exp(-x/b)
    return NoTracks

#%%
def ecdf(x):
    """
    Calculate experimental continuous distribution function (ECDF), i.e. no binning, of random variable x
    so that ECDF(value)=probability(x>=value). I.e first value of counts=1.
    
    Equivalent to inverse of:
        ``matplotlib.pyplot.hist(tau_dist,bins=numpy.unique(tau_dist),normed=True,cumulative=True)``
        but with non-equidistant binning, but bins are chosen according to unique values in x.
    
    Args:
        x(numpy.array):1 dimensional array of random variable  
    Returns:
        list:
        - [0](numpy.array): Bins of ECDF corresponding to unique values of x.
        - [1](numpy.array): ECDF(value)=probability(x>=value).
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
    Simple 1D non-normalized Gaussian function: ``y=np.absolute(A)*np.exp(-(x-x0)**2/sigma**2)``
    '''
    y=np.absolute(A)*np.exp(-(x-x0)**2/sigma**2)
    
    return y


#%%
def gauss_Ncomb(x,p,N):
    '''
    Sum of N 1D Gaussian functions, see gauss_1D(x,x0,sigma,A).
    The nth Gaussian function with ``n in [0,N[`` is:
        
        - centered at multiples of first Gaussian center ``(n+1)*x0``
        - has a width of ``sqrt(n+1)*sigma`` assuming Poissonian broadening
        - but decoupled Amplitudes ``An``
    
    Args:
        x(np.array):  Values at which function is evaluated
        p(list):      ``[x0,sigma,A0,A1,...,AN]`` input parameters for sum of Gaussians (len=N+2)
        N(integer):   Number of Gaussian functions that are summed up
    Returns:
        np.array: Evaluation of function at ``x``, ``p``, ``N``
    '''
    y=0
    for i in range(N):
        ### Version with least degrees of freedom
        y+=gauss_1D(x,(i+1)*p[0],np.sqrt(i+1)*p[1],p[2+i]) 

    return y
    
    