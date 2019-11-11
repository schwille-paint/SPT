import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

#%%
def get_trace(df,NoFrames):
    '''
    Get intensity vs time trace of length=NoFrames for group.
    '''
    df['photons']=df['photons'].abs() # Get absolute values of photons
    df_sum=df[['frame','photons']].groupby('frame').sum() # Sum multiple localizations in single frame
    trace=np.zeros(NoFrames) # Define trace of length=NoFrames with zero entries
    trace[df_sum.index.values]=df_sum['photons'].values # Add (summed) photons to trace for each frame
    
    return trace

#%%
def get_taubs(df,ignore=1):
    '''
    Return sorted array of bright times in trace for group.
    '''
    frames=df['frame'].values # Get sorted frames as numpy.ndarray
    frames.sort()
    dframes=frames[1:]-frames[0:-1] # Get frame distances i.e. dark times
    dframes=dframes.astype(float) # Convert to float values for later multiplications
    
    ################################################################ Get sorted tau_b_distribution
    dframes[dframes<=(ignore+1)]=0 # Set (bright) frames to 0 that have nnext neighbor distance <= ignore+1
    dframes[dframes>1]=1 # Set dark frames to 1
    dframes[dframes<1]=np.nan # Set bright frames to NaN
    
    mask_end=np.concatenate([dframes,[1]],axis=0) # Mask for end of events, add 1 at end
    frames_end=frames*mask_end # Apply mask to frames to get end frames of events
    frames_end=frames_end[~np.isnan(frames_end)] # get only non-NaN values, removal of bright frames
    
    mask_start=np.concatenate([[1],dframes],axis=0) # Mask for start of events, add one at start
    frames_start=frames*mask_start # Apply mask to frames to get start frames events
    frames_start=frames_start[~np.isnan(frames_start)] # get only non-NaN values, removal of bright frames
    
    taubs=frames_end-frames_start+1 # get tau_b distribution
    
    return taubs

#%%
def taubs_to_NgT(taubs,Ts):
    '''
    Return number of bright times greater than Ts as array of len(Ts).
    '''
    NgT=np.zeros(len(Ts))
    for idx,T in enumerate(Ts):
        NgT[idx]=np.sum(taubs>=T)
    
    NgT=pd.Series(NgT)
    NgT.index=Ts
    
    return NgT

#%%
def get_NgT(df,Ts,ignore=1):
    '''
    Combine get_taubs and taubs_to_NgT to return NgT as pd.Series.
    '''
    taubs=get_taubs(df,ignore)
    NgT=taubs_to_NgT(taubs,Ts)
    
    return NgT
    
#%%
def get_start(df,ignore):
    '''
    Get occurence of first localization in group and number of consecutive on frames starting from frame=0.
    '''
    
    min_frame=df['frame'].min() # First localization in group
    s_out = pd.Series({'min_frame':min_frame}) # Assign to output

    for i in ignore:
        taubs=get_taubs(df,ignore=i) # Get taub distribution
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
def get_other(df):
    """ 
    Get mean and std values for a single group.
    
    Parameters
    ---------
    df : pandas.DataFrame
        Picked localizations for single group. Required columns are 'frame'.

    Returns
    -------
    s_out : pandas.Series
        Length: 6
        Column:
            'group' : int
        Index: 
            'mean_frame' : float64
                Mean of frames for all localizations in group
            'mean_x' : float64
                Mean x position
            'mean_y' : float64
                Mean y position
            'mean_photons' : float64
                Mean of photons for all localizations in group
            'mean_bg' : float64
                Mean background
            'std_frame' : flaot64
                Standard deviation of frames for all localizations in group
            'std_x' : float64
                Standard deviation of x position
            'std_y' : float64
                Standard deviation of y position
            'std_photons' : float64
                Standard deviation of photons for all localizations in group
            'n_locs' : int
                Number of localizations in group
    """
    # Get mean values
    s_mean=df[['frame','x','y','photons','bg']].mean()
    # Get number of localizations
    s_mean['n_locs']=len(df)
    mean_idx={'frame':'mean_frame','x':'mean_x','y':'mean_y','photons':'mean_photons','mean_bg':'bg'}
    # Get std values
    s_std=df[['frame','x','y','photons']].std()
    std_idx={'frame':'std_frame','x':'std_x','y':'std_y','photons':'std_photons'}
    # Combine output
    s_out=pd.concat([s_mean.rename(mean_idx),s_std.rename(std_idx)])
    
    return s_out

#%%
def get_props(df,Ts,ignore=1):
    """ 
    Wrapper function to combine:
        - get_NgT(df,Ts,ignore)
        - get_start(df,ignore)
    
    Parameters
    ---------
    df : pandas.DataFrame
        'locs' of locs_picked.hdf5 as given by Picasso
    ignore: int
        Ignore as defined in props.get_tau
    Returns
    -------
    s : pandas.DataFrame
        Columns as defined in individual functions. Index corresponds to 'group'.
    """
    
    # Call individual functions
    s_other=get_other(df)
    s_start=get_start(df,[0,1,2,3,4,5])
    s_NgT=get_NgT(df,Ts,ignore)
    
    
    # Combine output
    s_out=pd.concat([s_other,s_start,s_NgT])
    
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
def filter_fix(df):
    """ 
    Filter for fixed single dye experiments    
    """            
    istrue = df.min_frame<=5
    
    df_filter=df.loc[istrue,:]

    return df_filter

#%%
def filter_nofix(df,NoFrames):
    """ 
    Filter for DNA-PAINT based tracking handle. Under progress...
    """
    istrue=df.min_frame<=5            
    istrue=istrue&(df.n_locs/NoFrames>=0.2) # Occupancy of more than 20%
    
    df_filter=df.loc[istrue,:]
    
    return df_filter

#%%
def filter_(df,NoFrames,apply_filter=None):
    '''
    Decide which filter to apply.
    '''
    if apply_filter=='fix':
        df_filter=filter_fix(df)
    elif apply_filter=='nofix':
        df_filter=filter_nofix(df,NoFrames)
    elif apply_filter==None:
        df_filter=df.copy()
    else:
        print('No filter criterium chosen. Please choose fix,nofix,None')
        sys.exit()
    
    return df_filter