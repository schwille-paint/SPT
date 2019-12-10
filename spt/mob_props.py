import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
# import warnings
# warnings.filterwarnings("ignore")

import picasso_addon.io as addon_io
import spt.immob_props as immob_props

#%%
def get_props(df,Ts,ignore):
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
    
    s_other=immob_props.get_other(df)
    s_NgT=s_other
    s_msd=s_other
    
        
    # Combine output
    s_out=pd.concat([s_NgT,s_msd,s_other])
    
    return s_out

#%%
def apply_props(df,
                Ts=np.concatenate((np.arange(1,10,1),
                                   np.arange(10,41,10),
                                   np.arange(50,1951,50),
                                   np.arange(2000,4801,200),
                                   np.arange(5000,50001,1000)),axis=0),
                ignore=1):
    """ 
          
    """
    tqdm.pandas() # For progressbar under apply
    df_props = df.groupby('group').progress_apply(lambda df: get_props(df,Ts,ignore))

    df_props.dropna(inplace=True)
    
    return df_props

#%%
def apply_props_dask(df,
                     Ts=np.concatenate((np.arange(1,10,1),
                                           np.arange(10,41,10),
                                           np.arange(50,1951,50),
                                           np.arange(2000,4801,200),
                                           np.arange(5000,50001,1000)),axis=0),
                    ignore=1,
                    NoPartitions=30): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
    """
    ########### Load packages
#    import dask
#    import dask.multiprocessing
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    ########### Globally set dask scheduler to processes
#    dask.config.set(scheduler='processes')
#    dask.set_options(get=dask.multiprocessing.get)
    ########### Partionate df using dask for parallelized computation
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!!
    df=dd.from_pandas(df,npartitions=NoPartitions) 
    ########### Define apply_props for dask which will be applied to different partitions of df
    def apply_props_2part(df,Ts,ignore): return df.groupby('group').apply(lambda df: get_props(df,Ts,ignore))
    ########### Map apply_props_2part to every partition of df for parallelized computing    
    with ProgressBar():
        df_props=df.map_partitions(apply_props_2part,Ts,ignore).compute(scheduler='processes')
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
    
    ### Set standard conditions if not set as input
    standard_params={'ignore':1,
                     'parallel':True,
                     'NoPartitions':30,
                     'filter':'paint'
                     }
    ### Remove keys in params that are not needed
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
        
    ### Procsessing marks: extension&generatedby
    try: extension=info[-1]['extension']+'_tprops'
    except: extension='_locs_xxx_picked_tprops'
    params['extension']=extension
    params['generatedby']='spt.immobile_props.main()'
    
    ### Get path of and number of frames
    path=info[0]['File']
    path=os.path.splitext(path)[0]
    NoFrames=info[0]['Frames']
    
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
    ### Filtering
    print('Filtering ..(%s)'%(params['filter']))
    params['NoGroups_nofilter']=len(locs_props) # Number of groups before filter
    locs_props=filter_(locs_props,NoFrames,params['filter']) # Apply filter
    params['NoGroups_filter']=len(locs_props) # Number of groups after filter


    print('Saving _tprops ...')
    info_props=info.copy()+[params]
    addon_io.save_locs(path+extension+'.hdf5',
                       locs_props,
                       info_props,
                       mode='picasso_compatible')

    return [params,locs_props]