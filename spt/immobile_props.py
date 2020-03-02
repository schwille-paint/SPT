import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

import picasso_addon.io as addon_io

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
    Return array of bright times and corresponding 1st frame of each bright event of for trace of a group.
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
    Return number of bright times greater than Ts as array of len(Ts).
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
    Combine get_taubs and taubs_to_NgT to return NgT as pd.Series.
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
    Get occurence of first localization in group and number of consecutive on frames starting from frame=0.
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
    Get various properties for each group of picked localizations.
    
    args:
        df(pandas.DataFrame):  Picked localizations (see picasso.render)

    returns:
        s_out(pandas.Series): index: Means of all columns in df, plus:
                                'n_locs':      Number of localizations
                                'photons':     Not mean but median!
                                'std_photons': Standard deviation of photons.
                                'bg':          Background photons. Not mean but median!
                                'sx':          Standard deviation of group in 'x'
                                'sy':          Standard deviation of group in 'y'
                                'min_frame':   Mimimum in frames
                                'max_frame':   Maximum in frames
                                'len':         max_frame-min_frame (see above)
    '''
    ### Get all mean values
    s_out=df.mean()
    ### Set photon values to median
    s_out['bg']=df['bg'].median()
    phot_crit=np.percentile(df['photons'],99)
    s_out['photons']=df.loc[df.photons>=phot_crit,'photons'].mean()
    
    ### Set sx and sy to standard deviation in x,y (locs!) instead of mean of sx,sy (PSF width)
    s_out[['sx','sy',]]=df[['x','y']].std()
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
    Wrapper function to combine:
        - get_NgT(df,Ts,ignore)
        - get_start(df,ignore)
    
    args:
        
    returns:
        
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
    def apply_props_2part(df,ignore): return df.groupby('group').apply(lambda df: get_props(df,ignore))
    ########### Map apply_props_2part to every partition of df for parallelized computing    
    with ProgressBar():
        df_props=df.map_partitions(apply_props_2part,ignore).compute(scheduler='processes')
    return df_props

#%%
def filter_fix(df):
    """ 
    Filter for fixed single dye experiments    
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
    elif apply_filter=='paint':
        df_filter=filter_nofix(df,NoFrames)
    elif apply_filter==None:
        df_filter=df.copy()
    else:
        print('No filter criterium chosen. Please choose fix,paint,None')
        sys.exit()
    
    return df_filter


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
                     'filter':'paint'
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