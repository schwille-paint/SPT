import numpy as np


#%%
def msd_abs_std(msd,N):
    '''
    Get relative standard deviation of points in msd.
    According to: Qian et al., Biophsical Journal, 60, 1991
    '''
    # n=np.arange(0,len(msd),1)
    # n=np.ones(len(msd))*len(msd)
    # var_rel=(2*n**2+1)/((3*n)*(N-n+1)) # Relative variance
    # var_rel=(2*n)/(3*(N-n+1)) # Relative variance
    # std_rel=np.sqrt(var_rel) # Relative standard deviation
    std_rel=np.ones(len(msd))*0.2
    std_abs=msd*std_rel # Absolute value of standard deviation
    
    return std_abs

#%%
def msd_free(tau,a,b=0):
    '''
    Simple brownian diffusion taking loclization precision into account: msd=a+b*tau.
    According to: Xavier Michalet, Physical Review E, 82, 2010
    '''
    msd=a*tau+b
    return msd

#%%
def msd_confined(tau,a,b):
    '''
    Confined diffusion: msd=(a/3)+(1-np.exp(-tau/b))
    According to: Carlo Manzo, Report on Progress in Physics, 78, 2015
    '''
    msd=(a/3)*(1-np.exp(-tau/b))
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
def jdist_cdf_free(rsquare,sigma):
    cdf=1-np.exp(-rsquare/sigma)
    return cdf
