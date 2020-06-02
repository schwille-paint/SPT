import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import dask.dataframe as dd
import multiprocessing as mp
import time
# import hdbscan
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

import picasso_addon.io as addon_io
import spt.analytic_expressions as express


#%%
def get_trace(df,NoFrames,field='net_gradient'):
    '''
    Get intensity vs time trace of length=NoFrames for group.
    '''
    
    df[field]=df[field].abs() # Get absolute values of field
    df_sum=df[['frame',field]].groupby('frame').sum() # Sum multiple localizations in single frame
    trace=np.zeros(NoFrames) # Define trace of length=NoFrames with zero entries
    trace[df_sum.index.values]=df_sum[field].values # Add (summed) field entries to trace for each frame
    
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
    Combine get_taubs and taubs_to_NgT to return NgT as pd.Series. Additionaly return longest trajectory in trace as tau_max.
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
def fit_Ncomb(x,y,centers_init,N):
    
    ### Prepare start parameters for fit
    p0=[]
    p0.extend([centers_init[0]])
    p0.extend([p0[0]/4])
    p0.extend([max(y) for i in range(N)])
    
    ### Define comb of N gaussians
    def gauss_comb(x,*p): return express.gauss_Ncomb(x,p,N)
    
    def redchi(x,y,p):
        yopt=gauss_comb(x,*p) # Get reduced chi
        chi=np.divide((yopt-y)**2,y)
        chi=chi[np.isfinite(chi)]
        chi=np.sum(chi)/(len(chi)-len(p))
        return chi
    
    ### Fit
    levels=np.zeros(4+6)
    try:
        ### Try first cluster peak as start parameter
        popt0,pcov=curve_fit(gauss_comb,x,y,p0=p0)
        popt0=np.absolute(popt0)
        
        ### Try second cluster peak as start parameter
        p0[0]=centers_init[1]
        popt1,pcov=curve_fit(gauss_comb,x,y,p0=p0)
        popt1=np.absolute(popt1)
        
        ### Select better chi fit
        if redchi(x,y,popt0)<redchi(x,y,popt1): popt=popt0
        else: popt=popt1
        
        ### Try half of first cluster peak
        p0[0]=centers_init[0]/2
        popt3,pcov=curve_fit(gauss_comb,x,y,p0=p0)
        popt3=np.absolute(popt3)
        
        ### Select better chi fit
        if redchi(x,y,popt3)<redchi(x,y,popt): popt=popt3
        
    except:
        levels[0]=N
        levels[1]=np.nan # set chi to nan
        return levels
   
    ### Remove peaks outside of data range
    centers=np.array([(i+1)*popt[0] for i in range(N)]) # Location of ith peak
    popt[2:][centers>=max(x)]=0 #Assign zeros to out of range
    
    ### Set peaks with an amplitude lower than 1% than the peak amplitude to zero
    Acrit=0.01*max(popt[2:])
    popt[2:][popt[2:]<Acrit]=0
     
    ### Now calculate fit reduced chi
    chi=redchi(x,y,popt)
    
    ### Assign to end result
    levels[0]=N
    levels[1]=chi
    levels[2:2+len(popt)]=popt
    
    return levels
        
#%%
def fit_levels(data,centers_init):
     
    ### Prepare data for fit
    y,x=np.histogram(data,
                     bins='auto',
                     )
    x=x[:-1]+(x[1]-x[0])/2
    
    ### Fit comb of Gaussians for different number of levels
    fits=np.zeros((10,6))
    for N in range(1,7):
        fits[:,N-1]=fit_Ncomb(x,y,centers_init,N) 
    
    ########## Decide on the appropriate number of levels based on following criteria
    levels=fits.copy()
    
    ### 1. Set chi value of fits having two extrema within peak amplitudes to NaN
    for i in range(np.shape(levels)[1]):
        diff_sign=np.sign(np.diff(levels[4:,i]))
        diff_sign=diff_sign[diff_sign!=0]
        n_maxima=np.sum(np.absolute(np.diff(diff_sign))==2)
        if n_maxima>2: levels[1,i]=np.nan
    
    ### 2. Remove NaNs in chi, corresponding to not succesful fits
    levels=levels[:,np.isfinite(levels[1,:])]
    
    ### 3. Search for k with minimum reduced chisquare, go back in ks and return value where first jump bigger than 10% occurs
    k=np.argmin(levels[1,:])
    chis=levels[1,:]
    if k>0:
        for i in range(k,0,-1):
            if np.absolute((chis[i]-chis[i-1])/chis[i-1])>0.1:
                k=i
                break
    
    ### After decision was made prepare return
    N=int(levels[0,k])
    
    pN=levels[2:2+2+N,k]
    yopt=express.gauss_Ncomb(x,pN,N)
    
    N=np.sum(levels[4:,k]>0)
    
    return N,x,y,yopt,fits

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
                    ignore=1): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
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
    Cluster detection (pick) in localization list by thresholding in number of localizations per cluster.
    Cluster centers are determined by creating images of localization list with set oversampling.
    
    
    args:
        locs(pd.Dataframe):        Picked localizations as created by picasso render
        info(list(dict)):          Info to picked localizations
        path(str):                 Path to _picked.hdf5 file.
        
    **kwargs: If not explicitly specified set to default, also when specified as None
        ignore(int=1):             Ignore value for bright frame
        parallel(bool=True):       Apply parallel computing using dask? 
                                   Dask cluster should be set up and running for best performance!
        filter(string='paint'):    Which filter to use, either None, 'th' or 'sd' or 'none'
        save_picked(bool=False):   If true _picked file containing just groups that passed filter will be saved under _picked_valid
    
    return:
        list[0](dict):             Dict of **kwargs passed to function.
        list[1](pandas.DataFrame): Kinetic properties of all groups.
                                   Will be saved with extension '_picked_tprops.hdf5' for usage in picasso.filter
    '''
    
    ### Path of file that is processed and number of frames
    path=os.path.splitext(path)[0]
    NoFrames=info[0]['Frames']
    
    ### Define standard 
    standard_params={'ignore':1,
                     'parallel':True,
                     'filter':'paint',
                     'save_picked':False,
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
    
    
    ##################################### Optional saving of reduced _locs_picked file
    if params['save_picked']: 
        ### Reduce _locs to remaining groups in _props
        groups=locs_props.group.values
        locs_filter=locs.query('group in @groups')
        
        info_filter=info.copy()+[params]
        
        ### Save
        addon_io.save_locs(path+'_valid.hdf5',
                           locs_filter,
                           info_filter,
                           mode='picasso_compatible')
    else:
        locs_filter=locs.copy()
        
    return [params,locs_props,locs_filter]