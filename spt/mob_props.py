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
def fit_msd_free_iterative(lagtimes,msd,lp,max_it=5):
    '''
    
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
        x=np.abs((4*lp)/s_out['a'])
        # x=np.abs(s_out['b']/(s_out['a']))
        
        ### Assign iteration and fitted track length
        s_out['p']=p[-1]
        s_out['max_it']=i
        
        ### Update optimal track length to be fitted
        try:
            p_update=int(np.ceil(2+2.7*x**0.5))
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
    Calculate msd of single trjecory using metrics.mean_displacement_moments() and apply all fit methods.
    '''

    ### Get displacement moments
    moments=metrics.displacement_moments(df.frame.values,
                                         df.x.values,
                                         df.y.values) 
    
    
    ### Get some metrics
    meanmoment_ratio=np.median(moments[:,2]/(moments[:,1]**2))
    maxmoment_ratio=np.median(moments[:,4]/(moments[:,3]**2))
    msd_ratio=metrics.msd_ratio(moments)
    straight=metrics.straightness(df.x.values,
                                  df.y.values)
    
    s_other=pd.Series({'meanmoment_ratio':meanmoment_ratio,
                       'maxmoment_ratio':maxmoment_ratio,
                       'msd_ratio':msd_ratio,
                       'straight':straight,
                       })
    
    ########################## MSD fitting
    x=moments[:,0] # Define lagtimes, x values for fit
    y=moments[:,1] # Define MSD, y values for fit
    
    ### Anomalous diffusion (0.25 length)
    s_anom_msd=fit_msd_anomal(x,y).rename({'a':'a_anom','b':'b_anom'})
    
    ### Iterative fit
    lp=np.mean(df.lpx**2+df.lpy**2) # Get localization precision as input for iterative fit
    s_iter=fit_msd_free_iterative(x,y,lp).rename({'a':'a_iter','b':'b_iter','p':'p_iter','max_it':'max_iter'})
    
    ########################## MME fitting
    x=moments[:,0] # Define lagtimes, x values for fit
    y=moments[:,3] # Define MME, y values for fit
    
    ## Anomalous diffusion (0.25 length)
    s_anom_mme=fit_msd_anomal(x,y).rename({'a':'a_mme_anom','b':'b_mme_anom'})
    
    ### Asign output series
    s_out=pd.concat([s_other,
                     s_anom_msd,
                     s_iter,
                     s_anom_mme,
                     ])
    
    return s_out

#%%
def get_props(df):
    """ 
    Wrapper function to combine:
    
    Parameters
    ---------
    df : pandas.DataFrame
        'locs' of locs_picked.hdf5 as given by Picasso

    Returns
    -------
    s : pandas.DataFrame
        Columns as defined in individual functions. Index corresponds to 'group'.
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
          
    """
    tqdm.pandas() # For progressbar under apply
    df_props = df.groupby('group').progress_apply(get_props)
    
    return df_props

#%%
def apply_props_dask(df): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
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
    Cluster detection (pick) in localization list by thresholding in number of localizations per cluster.
    Cluster centers are determined by creating images of localization list with set oversampling.
    
    
    args:
        locs(pd.Dataframe):        Picked localizations as created by picasso render
        info(list(dict)):          Info to picked localizations
        path(str):                 Path to _picked.hdf5 file.
        
    **kwargs: If not explicitly specified set to default, also when specified as None
        parallel(bool=True):       Apply parallel computing? (better speed, but a few lost groups)
    
    return:
        list[0](dict):             Dict of **kwargs passed to function.
        list[1](pandas.DataFrame): Kinetic properties of all groups.
                                   Will be saved with extension '_picked_tprops.hdf5' for usage in picasso.filter
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