import numpy as np
import numba


@numba.jit(nopython=True,nogil=True,cache=True)
def displacement_moments(t,x,y):
    '''

    '''
    N=t[-1]-t[0]+1 # Trajectory length
    max_lag=int(np.floor(0.25*N)) # Set maximum lagtime to 0.2*trajectory length
    
    ### Create nan arrays of length N to distribute x,y and fill gaps with nans
    x_gap=np.ones(N)*np.nan
    y_gap=np.ones(N)*np.nan
    idx=t-t[0] # Indices where we find finite coordinates
    ### Fill in values
    x_gap[idx]=x
    y_gap[idx]=y
    
    moments=np.ones((max_lag,5),dtype=np.float32)
    r2max_leql=np.zeros(N+1,dtype=np.float64) #Init max distance traveled
    r4max_leql=np.zeros(N+1,dtype=np.float64)
    for l in range(max_lag):
        ### One dimensional jumps for lag l
        dx=x_gap[l:]-x_gap[:N-l]
        dy=y_gap[l:]-y_gap[:N-l]
        
        ### Two dimensional jumps to the power of two and four
        r2_l=dx**2+dy**2
        r4_l=dx**4+dy**4
        
        ### Assign mean moments
        moments[l,0]=l
        moments[l,1]=np.nanmean(r2_l)
        moments[l,2]=np.nanmean(r4_l)
        
        ### Update rXmax_leql to maximum of past steps and current
        r2max_leql=np.maximum(r2max_leql[:-1],r2_l)
        r4max_leql=np.maximum(r4max_leql[:-1],r4_l)
        
        ### Assign max moments
        moments[l,3]=np.nanmean(r2max_leql)
        moments[l,4]=np.nanmean(r4max_leql)

    
    moments=moments[1:,:]
    return moments

#%%
@numba.jit(nopython=True,nogil=True,cache=True)
def msd_ratio(moments):
    '''
    MSD ratio (classifier H.) according to P. Kowalek, Physical Review E, 100, 2019.
    '''
    moments=moments[moments[:,1]!=0,:]
    kappa=np.nanmean(moments[:-1,1]/moments[1:,1]-moments[:-1,0]/moments[1:,0])

    return kappa

#%%
@numba.jit(nopython=True,nogil=True,cache=True)
def straightness(x,y):
    '''
    Straightness (classifier I.) according to P. Kowalek, Physical Review E, 100, 2019.
    '''
    ### One dimensional jumps
    dx=x[1:]-x[:-1]
    dy=y[1:]-x[:-1]
    
    ### Sum of steps
    stepsum=np.sum(np.sqrt(dx**2+dy**2))
    ### Total distance travelled
    distance=np.sqrt((x[-1]-x[0])*2+(y[-1]-y[0])*2)
    ### Straighness
    if stepsum!=0:
        straightness=distance/stepsum
    else:
        straightness=np.nan
    
    return straightness