import numpy as np
import numba


@numba.jit(nopython=True,nogil=True,cache=True)
def mean_displacement_moments(t,x,y):
    '''

    '''
    N=t[-1]-t[0] # Trajectory length
    max_lag=int(np.floor(0.2*N)) # Set maximum lagtime to 0.2*trajectory length
    
    ### Create nan arrays of length N to distribute x,y and fill gaps with nans
    x_gap=np.ones(N)*np.nan
    y_gap=np.ones(N)*np.nan
    idx=t-t[0] # Indices where we find finite coordinates
    ### Fill in values
    x_gap[idx]=x
    y_gap[idx]=y
    
    moments=np.ones((max_lag,3),dtype=np.float32)
    for l in range(max_lag):
        ### One dimensional jumps for lag l
        dx=x_gap[l:]-x_gap[:N-l]
        dy=y_gap[l:]-y_gap[:N-l]
      
        moments[l,0]=l
        moments[l,1]=np.nanmean(dx**2+dy**2)
        moments[l,2]=np.nanmean(dx**4+dy**4)
    
    moments=moments[1:,:]
    return moments