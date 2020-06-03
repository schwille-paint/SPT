import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import spt.analytic_expressions as express
import spt.immobile_props as improps

#%%
def tracks_per_frame(props,NoFrames):
    '''
    Count number of trajectories per frame.
    '''
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
        p0=[np.median(y[-11:-1]),N] 
        try:
            popt,pcov=curve_fit(express.exp_tracks_per_frame,x,y,p0=p0)
        except:
            popt=np.full(2,np.nan)
    else:
        popt=np.full(2,np.nan)
    
    y_fit=express.exp_tracks_per_frame(x,*popt)
    return [x,y_fit,popt]

#%%
def get_mobile_props(df_in,infos,px,CycleTime,remove_immob=True):
    '''
    
    ''' 
    expID=df_in.index.levels[0].values
    
    df_list=[]
    track_list=[]
    gT_list=[]
    result_list=[]
    
    for i in expID:
        df=df_in.loc[(expID[i],slice(None)),:]
        
        ### Conversion factor from px^2/frame to 1e-2 microns^2/s for a_iter to diffusion constant
        a_iter_convert=(px**2/(4*CycleTime))*100
        
        ### 1) Remove 'stuck' particles, assuming that diffusion is only governed by localization precision
        if remove_immob:
            # istrue=np.sqrt(df.sx*df.sy)>3
            D_hp=1 # Diffusion constant high pass in 1e-2 microns^2/s
            istrue = df.a_iter > D_hp/a_iter_convert
            df=df[istrue]
        
        df_list.extend([df]) # Assign to list
        
        ### 2) No. of tracks per frame and fit
        n_tracks=tracks_per_frame(df,infos[i][0]['Frames']) # Get tracks per frame
        fit=fit_tracks_per_frame(n_tracks) # Fit exponential
        track=[fit[0],n_tracks,fit[1],fit[2]] # x,y,y_fit,popt
        track_list.extend([track]) # Assign to list
        
        ### 3) NgT
        gT,Ts=improps.tracks_greaterT(df.len,
                                      df.min_frame,
                                      df.photons,
                                      df.std_photons)
        
        ### 4) Normalization of NgT to inital track number
        idx=['n%i'%(T) for T in Ts]
        gT[idx]=gT[idx]/track[-1][0]
        gT_list.extend([gT])
        
        ### 5) Get critical values
        Ns=gT[idx].values
        T_half=Ts[Ns>=0.5][-1] # Half time value
        photons_half=gT['p%i'%(T_half)] # Median photons of tracks longer than half time value
        tracks_init=track[-1][0] # Initial number of tracks
        tracks_loss=track[-1][1] # Loss of tracks per frame
        result=pd.Series({'T_half':T_half,
                          'photons_half':photons_half,
                          'tracks_init':tracks_init,
                          'tracks_loss':tracks_loss,
                          'CycleTime':CycleTime,
                          'a_iter_convert': a_iter_convert,
                          })
        result_list.extend([result])
        
    ### Re-concatenate dfs
    df_out=pd.concat([df for df in df_list])
    
    ### Combine gT Series to DataFrame
    gT_out=pd.DataFrame([gT for gT in gT_list])
    gT_out.index.name='expID'
    Ts=gT_out.columns.values
    Ts=list(np.unique([int(T[1:]) for T in Ts]))
    gT_out.loc['T',:]=Ts*4
    
    ### Combine result Series to DataFrame
    results=pd.DataFrame([result for result in result_list])
    result.index.name='expID'
    
    return df_out, results, track_list, gT_out

#%%
def get_half_time(df):
    '''
    Get half life time, i.e. 1-ecdf (means bigger than T) of start times for different ignore values.
    '''
    fields=['Tstart-i%i'%(i) for i in range(6)]
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
    fields=df.columns.values[23:]
    NgT_mean=df.loc[:,fields].mean(axis=0)
    NgT_std=df.loc[:,fields].std()
    NgT_iqr50=np.percentile(df.loc[:,fields],50,axis=0)
    NgT_iqr25=np.percentile(df.loc[:,fields],25,axis=0)
    NgT_iqr75=np.percentile(df.loc[:,fields],75,axis=0)
    
    NgT_stats=pd.DataFrame({'mean':NgT_mean,
                      'std':NgT_std,
                      '50%':NgT_iqr50,
                      '25%':NgT_iqr25,
                      '75%':NgT_iqr75})
    
    Ts=NgT_stats.index.values[22:]
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
    ### Starts, rip of s in index
    idx=['s%i'%(T) for T in Ts]
    S=NgT_stats.loc[idx,'50%']
    S.index=Ts
    ### Photons, rip of p in index
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