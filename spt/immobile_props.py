'''
.. _picasso.addon:
    https://picasso-addon.readthedocs.io/en/latest/howto.html#autopick
.. _spt:
    https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import dask.dataframe as dd

import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore")

import picasso_addon.io as addon_io

#%%
def get_trace(df,NoFrames,field='net_gradient'):
    '''
    Get continuous field vs. time trace of length=NoFrames for one group. The function assumes that
    there is only one group in the localization list!
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        NoFrames(int):        Length of measurement in frames of corresponding raw movie.
        field(str):           Column name in df, e.g. 'photons' for intensity vs. time trace.
        
    Returns:
        numpy.array: Trace, e.g. continuous field vs. time trace of length=NoFrames for one group
    '''
    
    df[field]=df[field].abs() # Get absolute values of field
    df_sum=df[['frame',field]].groupby('frame').sum() # Sum multiple localizations in single frame
    trace=np.zeros(NoFrames) # Define trace of length=NoFrames with zero entries
    trace[df_sum.index.values]=df_sum[field].values # Add (summed) field entries to trace for each frame
    
    return trace

#%%
def get_taubs(df,ignore=1):
    '''
    Get bright time distribution, i.e. the intervals of continuous localizations only interrupted 
    by ``ignore`` for one group and additional properties of each bright time. I.e. a bright time 
    corresponds to one trajectory in terms of single particle tracking. The function assumes 
    that there is only one group in the localization list!
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e._picked.hdf5 as in `picasso.addon`_ 
        ignore(int=1):        Maximum interruption (frames) allowed to be regarded as one bright time.
    Returns:
        list:
        - [0](numpy.array): Bright times only interrupted by ``ignore``
        - [1](numpy.array): 1st frame of each bright time
        - [2](numpy.array): Mean photon values of each bright time
        - [3](numpy.array): Relative standard deviation of photons of each bright time
    '''
    locs_trace=df.loc[:,['frame','photons']].sort_values(by=['frame']) # Get subset of df sort by frame
    ### Get ...
    frames=locs_trace.frame.values # ... frame number per localization
    photons=locs_trace.photons.values # ... photon number per localization
    locs_idx=np.arange(0,len(frames),1).astype(int) # ... and index per localization
    
    dframes=frames[1:]-frames[0:-1] # Get frame distances i.e. dark times
    dframes=dframes.astype(float) # Convert to float values for later multiplications
    
    ################################################################ Get sorted tau_b_distribution
    dframes[dframes<=(ignore+1)]=0 # Set (bright) frames to 0 that have nnext neighbor distance <= ignore+1
    dframes[dframes>1]=1 # Set dark frames to 1
    dframes[dframes<1]=np.nan # Set bright frames to NaN
    
    ### Get 1st frame and index in locs of bright events
    mask_start=np.concatenate([[1],dframes],axis=0) # Mask for start of events, add one at start
    frames_start=frames*mask_start # Apply mask to frames to get start frames events
    locs_idx_start=locs_idx*mask_start
    locs_idx_start=locs_idx_start[~np.isnan(frames_start)] # Start index in locs coordinates
    frames_start=frames_start[~np.isnan(frames_start)] # get only non-NaN values, removal of bright frames
    
    ### Get last frame and index in locs of bright events
    mask_end=np.concatenate([dframes,[1]],axis=0) # Mask for end of events, add 1 at end
    frames_end=frames*mask_end # Apply mask to frames to get end frames of events
    locs_idx_end=locs_idx*mask_end
    locs_idx_end=locs_idx_end[~np.isnan(frames_end)]+1 # End index in locs coordinates
    frames_end=frames_end[~np.isnan(frames_end)].astype(int) # get only non-NaN values, removal of bright frames
    
    taubs=frames_end-frames_start+1 # get tau_b distribution
    
    ### Convert to integers
    taubs=taubs.astype(int)
    frames_start=frames_start.astype(int)
    locs_idx_end=locs_idx_end.astype(int)
    locs_idx_start=locs_idx_start.astype(int)
    
    ### Get photon values per taub
    taubs_photons=np.zeros(len(taubs)) #Init
    taubs_photons_err=np.zeros(len(taubs)) #Init
    for idx,start in enumerate(locs_idx_start):
        taubs_photons[idx]=np.mean(photons[start:locs_idx_end[idx]])
        taubs_photons_err[idx]=np.std(photons[start:locs_idx_end[idx]])
        taubs_photons_err[idx]=taubs_photons_err[idx]/taubs_photons[idx] # Relative error!
        
    return taubs,frames_start,taubs_photons,taubs_photons_err

#%%
def tracks_greaterT(track_length,
                    track_start,
                    track_photons,
                    track_photons_std):
    '''
    Get number of trajectories (bright times) per particle (group or pick) greater or equal to T, i.e. TPP in `spt`_.
    The function takes returns of get_taubs() as input.
    
    Args:
        track_length(numpy.array): Trajectory durations or bright times only interrupted by ``ignore``. See get_taubs().
        track_start(numpy.array):  1st frame of each trajectory. See get_taubs().
        track_photons(numpy.array): Mean photon values of each bright time. See get_taubs().
        track_photons_std(numpy.array): Relative standard deviation of photons of each bright time. See get_taubs().
    Returns:
        list:
        - [0](pandas.Series): Index consists of a combination of one of the following letters and T in frames.
        
            - n: Number of tracks longer or equal to T, i.e. TPP as in `spt`_
            - s: Mean 1st frame of tracks longer or equal to T.
            - p: Mean photons of tracks longer or equal to T.
            - e: Mean relative standard deviation of photons of tracks longer or equal to T.
        - [1](numpy.array): T in frames
    '''
    ### Define Ts
    Ts=np.concatenate((np.arange(1,50,1),
                       np.arange(50,91,10),
                       np.arange(100,381,20),
                       np.arange(400,1951,50),
                       np.arange(2000,4801,200),
                       np.arange(5000,50001,1000)),axis=0)
    
    ### Init observables
    gT=np.zeros(len(Ts))
    gT_start=np.zeros(len(Ts))
    gT_photons=np.zeros(len(Ts))
    gT_photons_std=np.zeros(len(Ts))
    
    for idx,T in enumerate(Ts):
        positives=track_length>=T # Which tracks are longer than T? -> positives
            
        gT[idx]=np.sum(positives) # How many positives
        gT_start[idx]=np.mean(track_start[positives]) # When did positives start?
        gT_photons[idx]=np.mean(track_photons[positives]) # How bright were positives on average?
        gT_photons_std[idx]=np.mean(track_photons_std[positives]) # How much scatter positives in brightness on average?
        
    ### Prepare output
    gT=pd.Series(gT)
    gT.index=['n%i'%(T) for T in Ts]
    
    gT_start=pd.Series(gT_start)
    gT_start.index=['s%i'%(T) for T in Ts]
    
    gT_photons=pd.Series(gT_photons)
    gT_photons.index=['p%i'%(T) for T in Ts]
    
    gT_photons_std=pd.Series(gT_photons_std)
    gT_photons_std.index=['e%i'%(T) for T in Ts]
    
    s_out=pd.concat([gT,gT_start,gT_photons,gT_photons_std])
    
    return [s_out,Ts]

#%%
def get_NgT(df,ignore=1):
    '''
    Combine get_taubs() and tracks_greaterT() to return TPP as pd.Series for one group.
    
        * Input equivalent to get_taubs().
        * Output equivalent to tracks_greaterT()[0].
        
    The function assumes that there is only one group in the localization list!
    
    Args:     
        df(pandas.DataFrame): Grouped localization list, i.e._picked.hdf5 as in `picasso.addon`_
        ignore(int=1):        Maximum interruption (frames) allowed to be regarded as one bright time.
    Returns:
        pandas.Series:
            Index consists of a combination of one of the following letters and T in frames.
            
            - n: Number of tracks longer or equal to T, i.e. TPP as in `spt`_
            - s: Mean 1st frame of tracks longer or equal to T.
            - p: Mean photons of tracks longer or equal to T.
            - e: Mean relative standard deviation of photons of tracks longer or equal to T.
    '''
    
    taubs,start_frames,taubs_photons,taubs_photons_err=get_taubs(df,ignore)
    gT=tracks_greaterT(taubs,
                       start_frames,
                       taubs_photons,
                       taubs_photons_err)[0]
        
    return gT
    
#%%
def get_start(df,ignore):
    '''
    For a list of ``ignore`` values, was there a bright time at the start of the measurement and of which duration?
    That means for
    
        - ``ignore=2`` and the first bright time starting at frame=3 there was NO bright time
        - ``ignore=3`` and the first bright time starting at frame=3 there was a bright time of finite duration.
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        ignore(list):         List of maximum interruption (frames) allowed to be regarded as one bright time.
    Returns:
        pandas.Series: 
            Index has format ``Tstart-i%i'%(ignore)``.
            The value is the duration of the bright time at start of measurement if it happened within ``ignore``.
    '''
    ### First frame in localizations
    min_frame=df['frame'].min() # First localization in group
    
    s_out=pd.Series([])
    for i in ignore:
        taubs=get_taubs(df,ignore=i)[0] # Get taub distribution
        if min_frame<=i:
            try:    
                Tstart = taubs[0]+min_frame
            except IndexError:
                Tstart = np.nan
        else:
            Tstart=0

        s_out = s_out.append(pd.Series({'Tstart-i%.0f'%(i):Tstart}))
  
    return s_out

#%%
def get_var(df):
    '''
    Get various properties for one group in _picked.hdf5 as in `picasso.addon`_.
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
    Returns:
        pandas.Series: 
            Indices are means of all columns in ``df`` plus ...
            
                - ``n_locs``:      Number of localizations
                - ``photons``:     Not mean but median!
                - ``std_photons``: Standard deviation of photons.
                - ``bg``:          Background photons. Not mean but median!
                - ``sx``:          Standard deviation of group in ``x``
                - ``sy``:          Standard deviation of group in ``y``
                - ``min_frame``:   Mimimum in frames
                - ``max_frame``:   Maximum in frames
                - ``len``:         max_frame-min_frame (see above)
    '''
    ### Get all mean values
    s_out=df.mean()
    ### Set photon and bg values to median
    s_out['bg']=df['bg'].median()
    s_out['photons']=df['photons'].median()
    
    ### Set sx and sy to  maximum spread in x,y (locs!) instead of mean of sx,sy (PSF width)
    s_out['sx']=np.percentile(df['x'],95)-np.percentile(df['x'],5)
    s_out['sy']=np.percentile(df['y'],95)-np.percentile(df['y'],5)
    
    ### Add std_photons
    s_out['std_photons']=df['photons'].std()
    
    ### Add min/max of frames
    s_out['min_frame']=df['frame'].min()
    s_out['max_frame']=df['frame'].max()
    s_out['len']=s_out['max_frame']-s_out['min_frame']+1
    s_out['n_locs']=len(df)
    
    return s_out

#%%
def get_props(df,ignore=1):
    """ 
    Combination of get_NgT(df,ignore) and get_start(df,ignore) and get_var(df).
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        ignore(int=1):        Maximum interruption (frames) allowed to be regarded as one bright time.
    Returns:
        pandas.Series:       Concatenated output of get_NgT(df,ignore) and get_start(df,ignore) and get_var(df).
        
    """
    
    # Call individual functions
    s_var=get_var(df)
    s_start=get_start(df,[0,1,2,3,4,5])
    s_gT=get_NgT(df,ignore)
    
    
    # Combine output
    s_out=pd.concat([s_var,s_start,s_gT])
    
    return s_out

#%%
def apply_props(df,
                ignore=1):
    """ 
    Group _picked.hdf5 by groups (i.e. picks in `picasso.addon`_) and 
    apply get_props() to each group to get immobile properties as in `spt`_.
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        ignore(int=1):        Maximum interruption (frames) allowed to be regarded as one bright time.
    Returns:
        pandas.DataFrame:     Output of get_props() for each group in ``df`` (groupby-apply approach).
    """
    tqdm.pandas() # For progressbar under apply
    df_props = df.groupby('group').progress_apply(lambda df: get_props(df,ignore))

    df_props.dropna(inplace=True)
    
    return df_props

#%%
def apply_props_dask(df,
                    ignore=1): 
    """
    Same as apply_props() but in parallelized version using DASK by partitioning df. 
    Local DASK cluster has to be started manually for efficient computation, see cluster_setup_howto().
    
    Args:
        df(pandas.DataFrame): Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        ignore(int=1):        Maximum interruption (frames) allowed to be regarded as one bright time.
    Returns:
        pandas.DataFrame:     Output of get_props() for each group in ``df`` (groupby-apply approach).
    """
    
    ### Define groupby.apply function for dask which will be applied to different partitions of df
    def apply_props_2part(df,ignore): return df.groupby('group').apply(lambda df: get_props(df,ignore))
    
    t0=time.time() # Timing
    ### Set up DataFrame for dask
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!! 
    NoPartitions=max(1,int(0.8 * mp.cpu_count()))
    df=dd.from_pandas(df,npartitions=NoPartitions)                
        
    ### Compute using running dask cluster, if no cluster is running dask will start one with default settings (maybe slow since not optimized for computation!)
    df_props=df.map_partitions(apply_props_2part,ignore).compute()
    dt=time.time()-t0
    print('... Computation time %.1f s'%(dt)) 
    return df_props

#%%
def cluster_setup_howto():
    '''
    Print instruction howto start a DASK local cluster for efficient computation of apply_props_dask().
    Fixed ``scheduler_port=8787`` is used to easily reconnect to cluster once it was started.
    
    '''

    print('Please first start a DASK LocalCluster by running following command in directly in IPython shell:')
    print()
    print('Client(n_workers=max(1,int(0.8 * mp.cpu_count())),')
    print('       processes=True,')
    print('       threads_per_worker=1,')
    print('       scheduler_port=8787,')
    print('       dashboard_address=":1234")') 
    return

#%%
def filter_fix(df):
    """ 
    Filter for immobilized single dye origami as described in `spt`_.
    Positives are groups 
    
        - with a trajectory within the first 5 frames after the start of the measurement
        - and number of trajectories within group lie in 90% interval of all groups
    
    Args:
        df(pandas.DataFrame): Immobile properties as calulated by apply_props()
    Returns:
        pandas.DataFrame: Positives in ``df`` according to single dye filter as described above.
    """            
    ### Localization within first 5 frames condition
    istrue = df.min_frame<=5
    
    ### Number of events upper percentile cut
    crit_nevents=np.percentile(df.n1,90) 
    istrue = istrue & (df.n1<=crit_nevents)
    
    df_filter=df.loc[istrue,:]

    return df_filter

#%%
def filter_nofix(df,NoFrames):
    """ 
    Filter for immobilized origami with DNA-PAINT based tracking handle (TH) as described in `spt`_.
    Positives are groups 
    
        - with a trajectory within the first 5 frames after the start of the measurement
        - and number localizations within group are greater or equal to 20% of total measurement duration (in frames)
    
    Args:
        df(pandas.DataFrame): Immobile properties as calulated by apply_props()
    Returns:
        pandas.DataFrame: Positives in ``df`` according to TH filter as described above.
    """ 
    istrue=df.min_frame<=5            
    istrue=istrue&(df.n_locs/NoFrames>=0.2) # Occupancy of more than 20%
    
    df_filter=df.loc[istrue,:]
    
    return df_filter

#%%
def filter_(df,NoFrames,apply_filter=None):
    '''
    Decide which filter to apply to the output of get_props(), either:
        
        - 'sd' as given by filter_fix()
        - 'th' as given by filter_nofix()
        - 'none' if no filter should be applied
        
    Args:
        df(pandas.DataFrame): Immobile properties as calulated by apply_props()
        NoFrames(int):        Length of measurement in frames of corresponding raw movie.
        apply_filter(str):    Either 'sd','th' or 'none'. See above.
    Returns:
        pandas.DataFrame: Positives in ``df`` according to chosen filter as described above.
    '''
    if apply_filter=='sd':
        df_filter=filter_fix(df)
    elif apply_filter=='th':
        df_filter=filter_nofix(df,NoFrames)
    elif apply_filter=='none':
        df_filter=df.copy()
    else:
        print('No filter criterium chosen. Please choose sd,th,none')
        sys.exit()
    
    return df_filter


#%%
def main(locs,info,path,**params):
    '''
    Get immobile properties for each group in _picked.hdf5 file (see `picasso.addon`_) and filter.
    
    
    Args:
        locs(pandas.DataFrame):    Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        info(list):                Info _picked.yaml to _picked.hdf5 localizations as list of dictionaries.
        path(str):                 Path to _picked.hdf5 file.
        
    Keyword Args:
        ignore(int=1):             Maximum interruption (frames) allowed to be regarded as one bright time.
        parallel(bool=True):       Apply parallel computing using DASK? Local cluster should be started before according to cluster_setup_howto()
        filter(string='th'):       Which filter to use, either None, 'th' or 'sd' or 'none'
    
    Returns:
        list:
            
        - [0](dict):             Dict of keyword arguments passed to function.
        - [1](pandas.DataFrame): Immobile properties of each group in ``locs`` as calulated by apply_props()
    '''
    
    ### Path of file that is processed and number of frames
    path=os.path.splitext(path)[0]
    NoFrames=info[0]['Frames']
    
    ### Define standard 
    standard_params={'ignore':1,
                     'parallel':True,
                     'filter':'th',
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
        
    ### Procsessing marks: extension&generatedby
    params['generatedby']='spt.immobile_props.main()'
    
    
    ##################################### Calculate kinetic properties
    print('Calculating kinetic information ...')
    if params['parallel']==True:
        print('... in parallel')
        locs_props=apply_props_dask(locs,
                                    ignore=params['ignore'],
                                    )
    else:
        locs_props=apply_props(locs,
                                    ignore=params['ignore'],
                                    )
        
    ##################################### Filtering
    print('Filtering ..(%s)'%(params['filter']))
    params['NoGroups_nofilter']=len(locs_props) # Number of groups before filter
    locs_props=filter_(locs_props,NoFrames,params['filter']) # Apply filter
    params['NoGroups_filter']=len(locs_props) # Number of groups after filter

    ##################################### Saving
    print('Saving _tprops ...')
    locs_props.reset_index(inplace=True) # Write group index into separate column
    info_props=info.copy()+[params]
    addon_io.save_locs(path+'_tprops.hdf5',
                       locs_props,
                       info_props,
                       mode='picasso_compatible')
           
    return [params,locs_props]