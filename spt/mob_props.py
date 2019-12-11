import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from scipy.optimize import curve_fit
# import warnings
# warnings.filterwarnings("ignore")

import picasso_addon.io as addon_io
import trackpy.motion as motion
import spt.analyze as analyze
import spt.immobile_props as improps


#%%
def get_msd(df,detail=False):
    '''
    Return mean-square-displacement of one group using trackpy implementation in units of px^2/frame.
    
    args:
        df(pd.DataFrame):     Localizations of single trajectory (see trackpy.motion.msd())
        max_lagtime(int=200): Maximum lagtime up to which msd is computed.
        detail(bool=False):   Detail as defined in trackpy.motion.msd().

    '''
    max_lagtime=int(np.floor(0.2*len(df))) # Only compute msd up to a quarter of trajectory length
    msd=motion.msd(df,
                   mpp=1,
                   fps=1,
                   max_lagtime=max_lagtime,
                   detail=detail,
                   pos_columns=None,
                   )
    
    msd=msd.msd # Get only mean square displacement, since aslo <x>, <x^2>, ... are computed
    
    return msd

#%%
def revert_time(df):
    '''
    Revert time of trace to check for multiple diffusion modes within trace.
    '''
    df_revert=df.copy()
    df_revert.x=df.x.values[::-1]
    df_revert.y=df.y.values[::-1]
    
    return df_revert

#%%
def get_jdist(df,interval):
    '''
    Return jump distances r^2 within time interval (frames).
    '''
    ### Get coordinates ad timestamp
    x=df.x.values
    y=df.y.values
    t=df.frame.values
    ### Get jumps
    dx=x[interval:]-x[:-interval]
    dy=y[interval:]-y[:-interval]
    dt=t[interval:]-t[:-interval]
    ### Get squared radial jump distance
    dr=dx**2+dy**2
    ### Remove jump distances where time stamp difference is greater than interval
    ### du to missin frames!
    dr=dr[dt==interval]
    
    return dr

#%%
def get_jdist_ecdf(jdist): 
   '''
   Get empirical cumulative distribution function of jump distance distribution.
   '''
   ecdf=analyze.get_ecdf(jdist)
   ecdf[1]=1-ecdf[1]
   
   return ecdf

#%%
# def fit_jdist_ecdf(jdist):
#     ecdf=get_jdist_ecdf(jdist)
    
#     try:
        
    
    
#%%
def get_props(df,ignore):
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
    s_msd=pd.Series([])
    
        
    # Combine output
    s_out=pd.concat([s_msd,s_var])
    
    return s_out

#%%
def apply_props(df,
                ignore=1):
    """ 
          
    """
    tqdm.pandas() # For progressbar under apply
    df_props = df.groupby('group').progress_apply(lambda df: get_props(df,ignore))

    df_props.dropna(inplace=True)
    
    return df_props

#%%
def apply_props_dask(df,
                     ignore=1,
                     NoPartitions=30): 
    """
    Applies mob_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
    """
    ### Load packages
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    ### Prepare dask.DataFrame
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!!
    df=dd.from_pandas(df,npartitions=NoPartitions) 
    
    ### Define apply_props for dask which will be applied to different partitions of df
    def apply_props_2part(df,ignore): return df.groupby('group').apply(lambda df: get_props(df,ignore))
    
    ### Map apply_props_2part to every partition of df for parallelized computing    
    with ProgressBar():
        df_props=df.map_partitions(apply_props_2part,ignore).compute(scheduler='processes')
    return df_props

#%%
def main(locs,info,**params):
    '''
    Cluster detection (pick) in localization list by thresholding in number of localizations per cluster.
    Cluster centers are determined by creating images of localization list with set oversampling.
    
    
    args:
        locs(pd.Dataframe):        Picked localizations as created by picasso render
        info(list(dict)):          Info to picked localizations
    
    **kwargs: If not explicitly specified set to default, also when specified as None
        ignore(int=1):             Ignore value for bright frame
        parallel(bool=True):       Apply parallel computing? (better speed, but a few lost groups)
        NoPartitions(int=30):      Number of partitions in case of parallel computing
        filter(string='paint'):    Which filter to use, either None, 'paint' or 'fix'
    
    return:
        list[0](dict):             Dict of **kwargs passed to function.
        list[1](pandas.DataFrame): Kinetic properties of all groups.
                                   Will be saved with extension '_picked_tprops.hdf5' for usage in picasso.filter
    '''
    
    ### Define standard 
    standard_params={'ignore':1,
                     'parallel':True,
                     'NoPartitions':30,
                     #'filter':'paint'
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
    
    ### Override ignore if memory in info
    try:
        params['ignore']=info[-1]['memory']
    except:
        pass
    
    ### Processing marks: extension&generatedby
    try: extension=info[-1]['extension']+'_tmobprops'
    except: extension='_locs_pickedxxxx_tmobprops'
    params['extension']=extension
    params['generatedby']='spt.mob_props.main()'
    
    ### Get path of and number of frames
    path=info[0]['File']
    path=os.path.splitext(path)[0]
    
    ### Calculate kinetic properties
    print('Calculating kinetic information ...')
    if params['parallel']==True:
        print('... in parallel')
        locs_props=apply_props_dask(locs,
                                    ignore=params['ignore'],
                                    NoPartitions=params['NoPartitions'],
                                    )
    else:
        locs_props=apply_props(locs,
                                    ignore=params['ignore'],
                                    )
    # ### Filtering
    # print('Filtering ..(%s)'%(params['filter']))
    # params['NoGroups_nofilter']=len(locs_props) # Number of groups before filter
    # locs_props=filter_(locs_props,NoFrames,params['filter']) # Apply filter
    # params['NoGroups_filter']=len(locs_props) # Number of groups after filter


    print('Saving _tprops ...')
    info_props=info.copy()+[params]
    addon_io.save_locs(path+extension+'.hdf5',
                        locs_props,
                        info_props,
                        mode='picasso_compatible')

    return [params,locs_props]