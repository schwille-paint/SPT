import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import spt.analytic_expressions as express

#%%
def tracks_per_frame(props,info):
    '''
    Count number of trajectories per frame.
    '''
    ### Get number of frames in measurement
    NoFrames=info[0]['Frames']
    
    ### Count number of tracks per frame
    n_tracks=np.zeros(NoFrames)
    print('Calculating number of tracks per frame...')
    print('')
    for f in range(NoFrames):
        ### Trajectories with min_frame<=frame<=max_frame
        positives=(props.min_frame<=f)&(props.max_frame>=f)
        n_tracks[f]=np.sum(positives)
    
    return n_tracks

#%%
def fit_tracks_per_frame(n_tracks):
    '''
    Fit number of trajectories per frame with deaying exponential
    '''
    y=n_tracks
    N=len(y)
    x=np.arange(0,N,1)
    
    if N>=3: # Try ftting if more than 2 points
        ### Init start values
        p0=[(y[-1]-y[0]),N/2,y[-1]] 
        try:
            popt,pcov=curve_fit(express.exp_tracks_per_frame,x,y,p0=p0)
        except:
            popt=np.full(3,np.nan)
    else:
        popt=np.full(3,np.nan)
    
    y_fit=express.exp_tracks_per_frame(x,*popt)
    return [popt,x,y_fit]

#%%
def get_half_time(df):
    '''
    Get half life time, i.e. 1-ecdf (means bigger than T) of start times for different ignore values.
    '''
    fields=df.columns.values[11:17]
    s_out=pd.Series(index=fields)

    for f in fields:
        ecdf=express.ecdf(df.loc[:,f])
        half_idx=np.where(ecdf[1]>0.5)[0][-1]
        s_out[f]=ecdf[0][half_idx]
    
    s_out.index=['half_i%i'%(i) for i in range(0,6)]
    return  s_out


#%%
def get_NgT(df):
    '''
    Return average and 25%/75% interquartile range of all NgT related values.
    '''
    fields=df.columns.values[17:]
    NgT_mean=df.loc[:,fields].mean(axis=0)
    NgT_std=df.loc[:,fields].std()
    NgT_iqr50=df.loc[:,fields].quantile(0.50,axis=0)
    NgT_iqr25=df.loc[:,fields].quantile(0.25,axis=0)
    NgT_iqr75=df.loc[:,fields].quantile(0.75,axis=0)
    
    NgT_stats=pd.DataFrame({'mean':NgT_mean,
                      'std':NgT_std,
                      '50%':NgT_iqr50,
                      '25%':NgT_iqr25,
                      '75%':NgT_iqr75})
    
    Ts=NgT_stats.index.values[17:]
    Ts=np.array([int(T[1:]) for T in Ts])
    Ts=np.unique(Ts)
    
    return NgT_stats,Ts

#%%
def get_T_with_N(df):
    '''
    Return critical times of NgT
    '''
    ### Define critical number of trajectories per particle
    Ncrit=[0.5,1,2,5,10,50,100]
    ### Get statistics of NgT
    NgT_stats,Ts=get_NgT(df)
   
    ### Numbers, rip of n in index
    idx=['n%i'%(T) for T in Ts]
    N=NgT_stats.loc[idx,'mean']
    N.index=Ts
    ### Starts, rip of n in index
    idx=['s%i'%(T) for T in Ts]
    S=NgT_stats.loc[idx,'50%']
    S.index=Ts
    ### Photons, rip of n in index
    idx=['p%i'%(T) for T in Ts]
    P=NgT_stats.loc[idx,'50%']
    P.index=Ts
    
    df_out=pd.DataFrame(index=[n for n in Ncrit],columns=['T','S']) # Init output
    for n in Ncrit:
        try:
            df_out.loc[n,'T']=N[N>=n].index.values[-1]
            df_out.loc[n,'S']=S[df_out.loc[n,'T']]
            df_out.loc[n,'P']=P[df_out.loc[n,'T']]
        except IndexError:
            df_out.loc[n,'T']=0
            
    ### Convert to Series
    s_T=df_out.loc[:,'T']
    s_T.index=['Tn=%.0e'%(n) for n in Ncrit]
    s_S=df_out.loc[:,'S']
    s_S.index=['Sn=%.0e'%(n) for n in Ncrit]
    s_P=df_out.loc[:,'P']
    s_P.index=['Pn=%.0e'%(n) for n in Ncrit]
    
    s_out=pd.concat([s_T,s_S,s_P])
    
    return s_out
#%%
def get_props(df):
    """ 
    Wrapper function to combine:
        - get_half_time(df)
    """
    # Call individual functions
    s_half=get_half_time(df)
    s_Tn=get_T_with_N(df)
    # Combine output
    s_out=pd.concat([s_half,s_Tn])
    
    return s_out

#%%
def get_result(df,name):
    df_result=df.groupby(axis=0,level=0).apply(get_props)
    
    fields=name.split('_')
    df_result=df_result.assign(dye=fields[0],
                               sample=fields[1],
                               buffer=fields[2],
                               exp=int(fields[3][3:]),
                               power=int(fields[4][1:-2]),
                               )
    
    return df_result