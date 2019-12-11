import numpy as np

#%%
def cdf_freemotion(rsquare,sigma):
    cdf=1-np.exp(-rsquare/sigma)
    return cdf


#%%
# #### Absolute error of logarithmic msd
# def abs_err_logmsd(msd):
#     #### Relative error of msd
#     def rel_err_msd(N,n):
#         sr_msd=(2*n**2+1)/((3*n)*(N-n+1))
#         sr_msd=np.sqrt(sr_msd)
#         return sr_msd
    
#     sa_logmsd=np.log(msd)+np.log(rel_err_msd(len(msd),msd.index.astype(float)))
#     #### Set last value to finite number
#     sa_logmsd.iloc[-1]=sa_logmsd.iloc[-2]
#     return sa_logmsd
    
# #%%
# def logmsd_diffusive(tau,A,n):
#     '''
    
#     '''
#     logmsd=np.log(A)+n*np.log(tau)
    
#     return logmsd