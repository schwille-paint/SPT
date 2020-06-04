'''
.. _michalet:
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.041914
.. _spt:
    https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1
.. _picasso.localize:
    https://picassosr.readthedocs.io/en/latest/localize.html
'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import importlib
import dask.dataframe as dd
import multiprocessing as mp
import time

import picasso_addon.io as addon_io

import spt.immobile_props as improps
import spt.motion_metrics as metrics
import spt.analytic_expressions as express

importlib.reload(metrics)
importlib.reload(express)

#%%
def fit_msd_free(lagtimes,msd,offset=False):
    '''
    Unweighted least square fit of invidual msd by linear model ``msd=a*lagtimes+b``, see analytic_expressions.msd_free(),
    i.e. assuming free Browninan motion. If there was less then two data-points or fit was not succesfull 
    NaNs are returned as optimum parameters.
    
    Args:
        lagtimes(np.array): Array of msd lagtimes
        msd(np.array):      Mean square displacement (msd) at lagtimes
        offset(bool=False): If True offset is used in linear fit model, if False
    Returns:
        pandas.Series: Column ``a`` corresponds to slope, ``b`` corresponds to offset of linear function applied. 
        
    '''
    x=lagtimes
    y=msd
    N=len(y)

    if N>=2: # Try ftting if more than one point in msd
        ### Init fit parameters
        p0=[(y[-1]-y[0])/N,y[0]] # Start values
       
        try:
            if offset==True:
                popt,pcov=curve_fit(express.msd_free,x,y,p0=p0)
            else:
                popt,pcov=curve_fit(express.msd_free,x,y,p0=p0[0])
        except:
            popt=np.full(2,np.nan)

    else:
        popt=np.full(2,np.nan)
    
    ### Assign to output
    if offset==True:
        s_out=pd.Series({'a':popt[0],'b':popt[1]})
    else:
        s_out=pd.Series({'a':popt[0],'b':0})
        
    return s_out

#%%
def fit_msd_anomal(lagtimes,msd):
    '''
    Unweighted least square fit of invidual msd by anomalous model ``msd=a*lagtimes**b``, 
    see analytic_expressions.msd_anomal(). If there was less then two data-points or fit was not succesfull 
    NaNs are returned as optimum parameters.
    
    Args:
        lagtimes(np.array): Array of msd lagtimes
        msd(np.array):      Mean square displacement (msd) at lagtimes
    Returns:
        pandas.Series: Column ``a`` corresponds to slope, ``b`` corresponds to diffusion mode. 
        
    '''
    x=lagtimes
    y=msd
    N=len(y)
    
    if N>=2: # Try ftting if more than one point in msd
        ### Init fit parameters
        p0=[(y[-1]-y[0])/N,1.0] # Start values
        try:
            popt,pcov=curve_fit(express.msd_anomal,x,y,p0=p0)
        except:
            popt=np.full(2,np.nan)
    else:
        popt=np.full(2,np.nan)
    
    ### Assign to output
    s_out=pd.Series({'a':popt[0],'b':popt[1]})
    
    return s_out

#%%
def fit_msd_free_iterative(lagtimes,msd,max_it=5):
    '''
    Unweighted least square fit of invidual msd by linear model ``msd=a*lagtimes+b`` in **iterative manner**
    to find optimum fitting range of msd according to: Xavier Michalet, Physical Review E, 82, 2010 (michalet_).
    In first iteration msd is fitted up to a maximum lagtime of ``lag_max=0.5*Nmsd`` with ``Nmsd`` being the full msd length.
    Notice that motion_metrics.displacement_moments() calculates msd only up to ``Nmsd=0.25*N`` hence ``lag_max=0.125*N``
    with Nbeing the full lenght of the trajectory. Then fitting range is updated according to rule 
    ``lag_max=int(np.round(2+2.3*(b/a)**0.52))``. For a detailed illustration please see SI of spt_.
    
    Args:
        lagtimes(np.array): Array of msd lagtimes
        msd(np.array):      Mean square displacement (msd) at lagtimes
        max_it(int=5):      Maximum number of iterations
    Returns:
        pandas.Series:
            
            - ``a`` slope  of linear function applied.
            - ``b`` offset of linear function applied
            - ``p`` maximum lagtime up to which msd was fitted
            - ``max_it`` resulting number of iterations until convergence was achieved
    '''
    ### Set inital track length that will be fitted to half the msd, 
    ### which is already set to only 0.25 of full track length, hence only 12.5% are used!
    p=[int(np.floor(0.5*len(msd)))]
    
    i=0
    while i<max_it:
        ### Truncate msd up to p for optimal fitting result
        t=lagtimes[:p[-1]]
        y=msd[:p[-1]]    

        ### Fit truncated msd
        s_out=fit_msd_free(t,y,offset=True)
        
        ### Update x 
        # x=np.abs((4*lp)/s_out['a'])
        x=np.abs(s_out['b']/(s_out['a']))
        
        ### Assign iteration and fitted track length
        s_out['p']=p[-1]
        s_out['max_it']=i
        
        ### Update optimal track length to be fitted
        try:
            p_update=int(np.round(2+2.3*x**0.52))
            if p_update<=2:
                p_update=2
            p=p+[p_update]
        except:
            break
            
        if np.abs(p[-1]-p[-2])<1:
            break
        i+=1
    
    
    return s_out

#%%
def getfit_moments(df):
    '''
    Calculate msd of single trajectory using metrics.displacement_moments() and apply both linear iterative fitting
    according to fit_msd_free_iterative() and anomalous diffsuion model fitting using fit_msd_anomal() to msd.
    
    Args:
        df(pandas.DataFrame): Trajectories (_pickedxxxx.hdf5) as obtained by linklocs.main()
    Returns:
        pandas.Series: 
            Concatenated output of fit_msd_free_iterative() and fit_msd_anomal().
            
            - ``a_iter`` slope  of iterative linear fit
            - ``b_iter`` offset of iterative linear fit
            - ``p_iter`` maximum lagtime up to which msd was fitted for iterative linear fit
            - ``max_iter`` resulting number of iterations until convergence was achieved for iterative linear fit
            - ``a`` slope of anomalous fit 
            - ``b`` diffusion mode of anomalous fit
    '''

    ### Get displacement moments
    moments=metrics.displacement_moments(df.frame.values,
                                         df.x.values,
                                         df.y.values) 
    
    ########################## MSD fitting
    x=moments[:,0] # Define lagtimes, x values for fit
    y=moments[:,1] # Define MSD, y values for fit
    
    ### Anomalous diffusion (0.25 length)
    s_anom=fit_msd_anomal(x,y).rename({'a':'a_anom','b':'b_anom'})
    
    ### Iterative fit
    s_iter=fit_msd_free_iterative(x,y).rename({'a':'a_iter','b':'b_iter','p':'p_iter','max_it':'max_iter'})
    

    ### Asign output series
    s_out=pd.concat([s_anom,
                     s_iter,
                     ])
    
    return s_out

#%%
def get_props(df):
    """ 
    Combination of immobile_props.get_var(df) and getfit_moments(df).
    
    Args:
        df(pandas.DataFrame): Trajectories (_pickedxxxx.hdf5) as obtained by linklocs.main()
    Returns:
        pandas.Series:       Concatenated output of immobile_props.get_var(df) and getfit_moments(df).
    """
    
    # Call individual functions
    
    s_var=improps.get_var(df)
    s_msd=getfit_moments(df)
        
    # Combine output
    s_out=pd.concat([s_var,s_msd])
    
    return s_out

#%%
def apply_props(df):
    """ 
    Group trajectories list (_pickedxxxx.hdf5) as obtained by linklocs.main() by groups (i.e. trajectories) and 
    apply get_props() to each group to get mobile properties. See also `spt`_.
    
    Args:
        df(pandas.DataFrame): Trajectories list (_pickedxxxx.hdf5) as obtained by linklocs.main()
    Returns:
        pandas.DataFrame:     Output of get_props() for each group in ``df`` (groupby-apply approach).     
    """
    tqdm.pandas() # For progressbar under apply
    df_props = df.groupby('group').progress_apply(get_props)
    
    return df_props

#%%
def apply_props_dask(df): 
    """
    Same as apply_props() but in parallelized version using DASK by partitioning df. 
    Local DASK cluster has to be started manually for efficient computation, see cluster_setup_howto().
    
    Args:
        df(pandas.DataFrame): Trajectories list (_pickedxxxx.hdf5) as obtained by linklocs.main()
    Returns:
        pandas.DataFrame:     Output of get_props() for each group in ``df`` (groupby-apply approach).
    """
    
    ### Define groupby.apply function for dask which will be applied to different partitions of df
    def apply_props_2part(df): return df.groupby('group').apply(get_props)
    
    t0=time.time() # Timing
    ### Set up DataFrame for dask
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!! 
    NoPartitions=max(1,int(0.8 * mp.cpu_count()))
    df=dd.from_pandas(df,npartitions=NoPartitions)                
        
    ### Compute using running dask cluster, if no cluster is running dask will start one with default settings (maybe slow since not optimized for computation!)
    df_props=df.map_partitions(apply_props_2part).compute()
    dt=time.time()-t0
    print('... Computation time %.1f s'%(dt)) 
    return df_props

#%%
def main(locs,info,path,**params):
    '''
    Get mobile properties for each group in trajectories list (_pickedxxxx.hdf5) file as obtained by linklocs.main().
    
    
    Args:
        locs(pandas.DataFrame):    Trajectories list (_pickedxxxx.hdf5) as obtained by linklocs.main()
        info(list):                Info _pickedxxxx.yaml to _pickedxxxx.hdf5 trajectories as list of dictionaries.
        path(str):                 Path to _pickedxxxx.hdf5 file.
        
    Keyword Args:
        parallel(bool=True):       Apply parallel computing using DASK? Local cluster should be started before according to cluster_setup_howto()
    
    Returns:
        list:
            
        - [0](dict):             Dict of keyword arguments passed to function.
        - [1](pandas.DataFrame): Mobile properties of each group in ``locs`` as calulated by apply_props()
    '''
    ##################################### Params and file handling
    
    ### Path of file that is processed and number of frames
    path=os.path.splitext(path)[0]
    
    ### Define standard 
    standard_params={'parallel':True,
                     }
    ### Set standard if not contained in params
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    ### Remove keys in params that are not needed
    delete_key=[]
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
      
    ### Processing marks
    params['generatedby']='spt.mob_props.main()'
    
    
    ##################################### Calculate kinetic properties
    print('Calculating kinetic information ...')
    if params['parallel']==True:
        print('... in parallel')
        locs_props=apply_props_dask(locs)
    else:
        locs_props=apply_props(locs)

    print('Saving _tmobprops ...')
    locs_props=locs_props.assign(group=locs_props.index.values) # Write group index into separate column
    info_props=info.copy()+[params]
    addon_io.save_locs(path+'_tmobprops.hdf5',
                       locs_props,
                       info_props,
                       mode='picasso_compatible')

    return [params,locs_props]