'''
.. _tejedor:
    https://www.cell.com/biophysj/fulltext/S0006-3495(09)06097-4
.. _kowalek:
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.032410
'''
import numpy as np
import numba


@numba.jit(nopython=True,nogil=True,cache=True)
def displacement_moments(t,x,y):
    '''
    Numba optimized calulation of trajectory ``(t,x,y)`` moments. Calulation accounts for short gaps, i.e. missed localizations 
    recovered by allowed ``memory`` values. Moments are only calulated up to maximum lag time of ``l_max = 0.25*N`` with ``N=len(t)``.
    Calulated moments are:
        
        - Mean square displacement (MSD)
        - Mean displacement moment of 4th order (MSD corresponds to 2nd order)
        - Mean maximal excursion of 2nd order (MME)
        - Mean maximal excursion of 4th order
    
    MME is calculated according to: Vincent Tejedor, Biophysical Journal, 98, 7, 2010 (tejedor_)
    
    Args:
        t(np.array): time
        x(np.array): x-position
        y(np.array): y-position
    Returns:
        np.array of size ``(l_max,5)``:
            
            - ``[:,0]``: lag time
            - ``[:,1]``: MSD
            - ``[:,2]``: Mean displacement moment of 4th order
            - ``[:,3]``: MME
            - ``[:,4]``: Mean maximal excursion of 4th order
    '''
    N=t[-1]-t[0]+1 # Trajectory length
    max_lag=int(np.floor(0.25*N)) # Set maximum lagtime to 0.25*trajectory length
    
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
        ### rXmax_leql will be always shortened 
        ### to the size of rX_l while going through the loop!
        r2max_leql=np.maximum(r2max_leql[:-1],r2_l)
        r4max_leql=np.maximum(r4max_leql[:-1],r4_l)
        
        ### Assign max moments
        moments[l,3]=np.nanmean(r2max_leql)
        moments[l,4]=np.nanmean(r4max_leql)

    ### Remove first entry
    moments=moments[1:,:]
    ### Remove NaNs due to gaps in trace
    moments=moments[np.isfinite(moments[:,1])]
    
    return moments

#%%
@numba.jit(nopython=True,nogil=True,cache=True)
def msd_ratio(moments):
    '''
    MSD ratio (classifier H.) according to P. Kowalek, Physical Review E, 100, 2019 (kowalek_).
    
    Args:
        moments(np.array): Return of displacement_moments(t,x,y)
    Returns:
        float: MSD ratio
    '''
    moments=moments[moments[:,1]!=0,:]
    kappa=np.nanmean(moments[:-1,1]/moments[1:,1]-moments[:-1,0]/moments[1:,0])

    return kappa

#%%
@numba.jit(nopython=True,nogil=True,cache=True)
def straightness(x,y):
    '''
    Straightness (classifier I.) according to P. Kowalek, Physical Review E, 100, 2019 (kowalek_).
    
    Args:
        x(np.array): x-positions of trajectory
        y(np.array): y-positions of trajectory
    Returns:
        float: Straightness
    '''
    ### One dimensional jumps
    dx=x[1:]-x[:-1]
    dy=y[1:]-x[:-1]
    
    ### Sum of steps
    stepsum=np.sum(np.sqrt(dx**2+dy**2))
    ### Total distance travelled
    distance=np.sqrt((x[-1]-x[0])**2+(y[-1]-y[0])**2)
    ### Straightness
    if stepsum!=0:
        straightness=distance/stepsum
    else:
        straightness=np.nan
    
    return straightness