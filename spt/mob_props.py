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
import spt.analytic_expressions as express

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
def get_msd(df,detail=False):
    '''
    Return mean-square-displacement of one group using trackpy implementation in units of px^2/frame.
    
    args:
        df(pd.DataFrame):     Localizations of single trajectory (see trackpy.motion.msd())
        detail(bool=False):   Detail as defined in trackpy.motion.msd().

    '''
    N=len(df)
    max_lagtime=int(np.floor(0.2*N)) # Only compute msd up to a quarter of trajectory length
    msd=motion.msd(df,
                   mpp=1,
                   fps=1,
                   max_lagtime=max_lagtime,
                   detail=detail,
                   pos_columns=None,
                   )
    
    msd=msd.msd # Get only mean square displacement, since aslo <x>, <x^2>, ... are computed
    
    ### Get absolute standard deviation of all points in msd
    msd_std=express.msd_abs_std(msd,N)
    
    return [msd,msd_std]

#%%
def fit_msd_free(msd,msd_std,offset=False):
    '''
    
    '''
    ### Prepare fit 
    x=msd.index.astype(float)
    y=msd
    y_std=msd_std
    ### Initial value
    p0=[(msd.iloc[-1]-msd.iloc[0])/len(msd),msd.iloc[0]]
    try:
        if offset==True:
            popt,pcov=curve_fit(express.msd_free,x,y,p0=p0)
            chi2=np.sum(np.square((y-express.msd_free(x,*popt))/y_std))/(len(msd)-2) # reduced chi square
        else:
            popt,pcov=curve_fit(express.msd_free,x,y,p0=p0[0])
            chi2=np.sum(np.square((y-express.msd_free(x,popt[0]))/y_std))/(len(msd)-1) # reduced chi square

    except:
        popt=np.full(2,np.nan)
        chi2=np.nan
    ### Assign to output
    if offset==True:
        s_out=pd.Series({'a':popt[0],'b':popt[1],'chi2':chi2,'mode':'free'})
    else:
        s_out=pd.Series({'a':popt[0],'b':0,'chi2':chi2,'mode':'free'})
        
    return s_out

#%%
def fit_msd_confined(msd,msd_std):
    '''
    
    '''
    ### Prepare fit 
    x=msd.index.astype(float)
    y=msd
    y_std=msd_std
    ### Initial value
    p0=[0,0]
    p0[0]=np.max(msd)
    halfmsd=(msd.max()-msd.min())*0.5+msd.min()
    p0[1]=msd[msd<=halfmsd].values[-1]
    try:
        popt,pcov=curve_fit(express.msd_confined,x,y,p0=p0)
        chi2=np.sum(np.square((y-express.msd_confined(x,*popt))/y_std))/(len(msd)-2) # reduced chi square
    except:
        popt=np.full(2,np.nan)
        chi2=np.nan
    ### Assign to output
    s_out=pd.Series({'a':popt[0],'b':popt[1],'chi2':chi2,'mode':'confined'})
    
    return s_out

#%%
def fit_msd_anomal(msd,msd_std):
    '''
    
    '''
    ### Prepare fit 
    x=msd.index.astype(float)
    y=msd
    y_std=msd_std
    ### Initial value
    p0=[0,0]
    p0[0]=(msd.iloc[-1]-msd.iloc[0])/len(msd)
    p0[1]=1.0
    try:
        popt,pcov=curve_fit(express.msd_anomal,x,y,p0=p0)
        chi2=np.sum(np.square((y-express.msd_anomal(x,*popt))/y_std))/(len(msd)-2) # reduced chi square
    except:
        popt=np.full(2,np.nan)
        chi2=np.nan
    ### Assign to output
    s_out=pd.Series({'a':popt[0],'b':popt[1],'chi2':chi2,'mode':'anomal'})
    
    return s_out

#%%
def fit_msd_free_iterative(msd,msd_std,lp,max_it=5):
    '''
    
    '''
    ### Set inital track length that will be fitted to half the msd, 
    ### which is already set to only 0.2 of full track length, hence only 10% are used!
    p=[int(np.floor(0.5*len(msd)))]
    
    i=0
    while i<max_it:
        ### Truncate msd up to p for optimal fitting result
        msd_iter=msd[:p[-1]]    
        msd_std_iter=msd_std[:p[-1]] 
        
        ### Fit truncated msd
        s_out=fit_msd_free(msd_iter,msd_std_iter,offset=True)
        
        ### Update x
        x=np.abs(lp/(s_out['a']/4))
        # x=np.abs(s_out['b']/(s_out['a']/4))
        
        ### Assign iteration and fitted track length
        s_out['p']=p[-1]
        s_out['max_it']=i+1
        
        ### Update optimal track length to be fitted
        try:
            p=p+[int(np.ceil(2+2.7*x**0.5))]
        except:
            break
            
        if np.abs(p[-1]-p[-2])<1:
            break
        i+=1
    
    if s_out['a']!=np.nan:
        x=msd_iter.index.astype(float)
        popt=[s_out['a'],s_out['b']]
        s_out['chi2']=np.sum(np.square((msd_iter-express.msd_free(x,*popt))/msd_std_iter))/(len(msd_iter)-2) # reduced chi square
    
    s_out['mode']='iterative'
    return s_out

#%%
def getfit_msd(df,lp,offset=False):
    '''
    Calculate msd of single trjecory using get_msd and apply all fit methods.
    '''
    
    ### Get msd
    msd=get_msd(df)
    
    ### Try every fitting method
    df_fit=pd.DataFrame([],columns=['a','b','chi2','mode','p','max_it'])
    df_fit.loc[0,:]=fit_msd_free(msd[0],msd[1],offset=offset)
    df_fit.loc[1,:]=fit_msd_confined(msd[0],msd[1])
    df_fit.loc[2,:]=fit_msd_anomal(msd[0],msd[1])
    df_fit.loc[3,:]=fit_msd_free_iterative(msd[0],msd[1],lp)
    
    ### Asign output series
    s_out=pd.concat([df_fit.iloc[0,0:3].rename({'a':'a_free','b':'b_free','chi2':'chi_free'}),
                     df_fit.iloc[1,0:3].rename({'a':'a_conf','b':'b_conf','chi2':'chi_conf'}),
                     df_fit.iloc[2,0:3].rename({'a':'a_anom','b':'b_anom','chi2':'chi_anom'}),
                     df_fit.iloc[3,[0,1,2,4,5]].rename({'a':'a_iter','b':'b_iter','chi2':'chi_iter','p':'p_iter','max_it':'max_iter'}),
                     ])
    
    return [s_out,msd,df_fit]

#%%
def decide_fit(msd,df_fit,choose='opt'):
    
    ### Sort by residual
    df_fit=df_fit.sort_values(by=['chi2'])
    
    if choose=='opt':
        s_fit=df_fit.iloc[0,:]
    else:
        s_fit=df_fit.iloc[choose,:]
    
    ### Return evaluation of fit
    x=msd.index.astype(float)
    if s_fit['mode']=='free':
        y=express.msd_free(x,s_fit['a'],s_fit['b'])
    elif s_fit['mode']=='confined':
        y=express.msd_confined(x,s_fit['a'],s_fit['b']) 
    elif s_fit['mode']=='anomal':
        y=express.msd_anomal(x,s_fit['a'],s_fit['b'])
    elif s_fit['mode']=='iterative':
        y=express.msd_free(x,s_fit['a'],s_fit['b']) 
        
    return [y,s_fit['mode']]

#%%
def get_fakeprops(df):
    msd=get_msd(df)
    
    s_out=pd.Series({'a_iter':0})
    
    return s_out
#%%
def get_props(df,lp,offset=False):
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
    s_msd=getfit_msd(df,lp,offset)[0]
    # s_fake=get_fakeprops(df)
        
    # Combine output
    s_out=pd.concat([s_var,s_msd])
    
    return s_out

#%%
def apply_props(df,
                lp,
                offset=False):
    """ 
          
    """
    tqdm.pandas() # For progressbar under apply
    df_props = df.groupby('group').progress_apply(lambda df: get_props(df,lp,offset))
    
    return df_props

#%%
def apply_props_dask(df,
                     lp,
                     offset=False,
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
    def apply_props_2part(df,lp,offset): return df.groupby('group').apply(lambda df: get_props(df,lp,offset))
    
    ### Create meta (dict) with correct column names so that dask knows output
    meta_dict={'a_free':'f8',
               'b_free' :'i8',
               'chi_free':'f8',
               'a_conf':'f8',
               'b_conf':'f8',
               'chi_conf':'f8',
               'a_anom':'f8',
               'b_anom':'f8',
               'chi_anom':'f8',
               'a_iter':'f8',
               'b_iter':'f8',
               'chi_iter':'f8',
               'p_iter':'i8',
               'max_iter':'f8',
               'frame':'f8',
               'x':'f8',
               'y':'f8',
               'photons':'f8',
               'sx':'f8',
               'sy':'f8',
               'bg':'f8',
               'lpx':'f8',
               'lpy':'f8',
               'ellipticity':'f8',
               'net_gradient':'f8',
               'n_locs':'f8',
               'std_photons':'f8', 
               'min_frame':'f8',
               'max_frame':'f8',
               'len':'f8',
               }
    
    ### Map apply_props_2part to every partition of df for parallelized computing    
    with ProgressBar():
        df_props=df.map_partitions(apply_props_2part,lp,offset,meta=meta_dict).compute(scheduler='processes')
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
    standard_params={'offset':False,
                     'parallel':True,
                     'NoPartitions':30,
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
      
    ### Processing marks: extension&generatedby
    try: extension=info[-1]['extension']+'_tmobprops'
    except: extension='_locs_pickedxxxx_tmobprops'
    params['extension']=extension
    params['generatedby']='spt.mob_props.main()'
    
    ### Get path of and number of frames
    path=info[0]['File']
    path=os.path.splitext(path)[0]
    
    ### Get median localization precision square for iterative fitting
    lp=np.median(locs.lpx**2+locs.lpy**2)
    
    ### Calculate kinetic properties
    print('Calculating kinetic information ...')
    if params['parallel']==True:
        print('... in parallel')
        locs_props=apply_props_dask(locs,
                                    lp,
                                    params['offset'],
                                    NoPartitions=params['NoPartitions'],
                                    )
    else:
        locs_props=apply_props(locs,
                               lp,
                               params['offset']
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