import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import trackpy as tp
import os

# own modules
import picasso.io as io

#%%
def annotate_filter(locs,movie,frame,photon_hp=0,c_min=0,c_max=1000):
    '''
    1) Plots frame of movie with annotated locs with photon values>photon_hp. Median proximity in frame indicated.
    2) Shows histogram of photons values for all locs and locs in frame
    
    Parameters
    ---------
    locs: pandas.Dataframe
    movie: pims.tiff_stack.TiffStack_tifffile
    frame: int
        Frame that will be plotted
    photon_hp: int
        Critical value for photon high pass acting on locs, defaults to 0
    
    Returns
    --------
    locs_filter:pandas.Dataframe
        locs that survived photon high pass
    '''
    from matplotlib.gridspec import GridSpec
      
    ### Apply photon 'high pass' to locs
    locs_filter=locs.drop(locs[locs.photons<photon_hp].index)
    
    ### Define plotting area
    f=plt.figure(num=11,figsize=[6,6])
    f.subplots_adjust(left=0.05,right=0.9,bottom=0.05,top=0.95,hspace=0.05,wspace=0.3)
    f.clear()
    gs = GridSpec(5,5,figure=f)
    ax_list=[]
    ### Annotate
    ax=f.add_subplot(gs[0:4,0:4])
    # tp.annotate(locs_filter[locs_filter.frame==frame],movie[frame],ax=ax,invert=True)
    mapp=ax.imshow(movie[frame],
                   cmap='gray',
                   vmin=c_min,
                   vmax=c_max,
                   interpolation='nearest',
                   origin='lower')
    plt.colorbar(mapp,
                 cax=f.add_subplot(gs[0:4,4]),
                 )
    ax.scatter(locs_filter[locs_filter.frame==frame].x,
               locs_filter[locs_filter.frame==frame].y,
               s=100,
               marker='o',
               facecolor='none',
               color='r'
               )
    med_prox=tp.proximity(locs.loc[locs.frame==frame,:]).median()
    ax.set_title('Median proximity: %.1f px'%(med_prox))
    ax.set_xticks([])
    ax.set_yticks([])
    ax_list.extend([ax])
    #### Photon histogram
    ax=f.add_subplot(gs[4,:5])
    bins=np.arange(0,locs.loc[:,'photons'].median()*4,10)   
    ax.hist(locs.loc[:,'photons'],
            bins=bins,
            density=True,
            histtype='step',
            edgecolor='k',
            lw=2,
            label='all')
    ax.hist(locs.loc[locs.frame==frame,'photons'],
            bins=bins,
            density=True,
            histtype='step',
            edgecolor='r',
            lw=2,
            label='frame=%i'%(frame))
    
    ax.axvline(photon_hp,ls='--',lw=3,c='k')
    ax.set_xticks(plt.xticks()[0][1:-2])
    ax.set_xticklabels(plt.xticks()[0]/1000)
    ax.set_yticks([])
    ax.legend()
    ax_list.extend([ax])
    return locs_filter,ax_list

#%%
def get_link(locs,locs_info,save_picked=False,**params):
    '''
    
    '''
    ### Set standard conditions if not set as input
    search_range=params['search_range']
    memory=params['memory']
    
#    standard_params={'search_range':search_range,
#                     'memory':memory,
#                     'save_picked':False
#                     }
    ### Procsessing marks: extension&generatedby
    try: extension=locs_info[-1]['extension']+'_picked'
    except: extension='_locs_xxx_picked'
    params['extension']=extension
    params['generatedby']='spt.trackpy_wrap.get_link()'
#
#    ### Remove keys in params that are not needed
#    for key, value in standard_params.items():
#        try:
#            params[key]
#            if params[key]==None: params[key]=standard_params[key]
#        except:
#            params[key]=standard_params[key]
#    ### Remove keys in params that are not needed
#    delete_key=[]
#    for key, value in params.items():
#        if key not in standard_params.keys():
#            delete_key.extend([key])
#    for key in delete_key:
#        del params[key]
    
    ### Get path of raw data      
    path=locs_info[0]['File']
    path=os.path.splitext(path)[0]
        
    #### Link locs
    link= tp.link_df(locs,search_range,memory=memory)
    link.sort_values(by=['particle','frame'],ascending=True,inplace=True)
    link.set_index('particle', inplace=True)
    
    #### Save linked locs as picks    
    if save_picked==True:
        
        info_picked=locs_info.copy()+[params]
        io.save_locs(path+extension+'.hdf5',
                     link.to_records(index=False),
                     info_picked,
                     )
    
    return link

#%%
def get_linkprops_noMSD(link,locs_info,length_hp):
    '''
    Calculates various trajectory properties without performing MSD analysis:
        1) Trajectory lengths
    
    Parameters
    ---------
    link : pandas.Dataframe  
        Same locs as locs, index set to 'particle' generated by trackpy.link,
        assigning each localization to a particle trajectory
    length_hp : int
        Cut-off value removing all trajectories shorter than length_hp. Set to
        100 by default.
    Returns
    -------
    link_props : pandas.DataFrame
    '''
    
    #### Get some properties
    print('Calculating means and lengths...')
    link_props=link.groupby('particle').mean()
    link_props['min_frame']=link.frame.groupby('particle').min() # get first frame of trajectory
    link_props['max_frame']=link.frame.groupby('particle').max() # get last frame of trajectory
    link_props['len']=np.subtract(link.frame.groupby('particle').max(),link.frame.groupby('particle').min()) + 1 # get trajectory length
    
    #### Reduce to traces above length high pass
    print('Applying track length high pass...')
    pass_particle=link_props[(link_props.len>length_hp)].index
    link=link.loc[pass_particle,:]
    link_props=link_props[link_props.len>length_hp] 
    
    #### Extract numTracks vs frames and fit to f(x)=a*exp(-x/b)+c
    print('Fitting decay of number of tracks per frame')   
    numtrack_fit=get_numtracks_fit(link_props,locs_info)
    
    #### Format link_props such that the output with key 'props' is identical to output from get_linkprops
    link_props=pd.concat([link_props,numtrack_fit],keys=['props','numtrack_fit'],axis=1)
     
    return link_props

#%%
def get_numtracks(link_props,frames):
    '''
    
    '''
    
    #### Initialize variables: iterable over all frames and numTrack set to 0 in each frame
    numTracks = pd.Series(np.zeros(len(frames)))
    
    print('Calculating number of tracks per frame...')
    for current_frame in frames:
        
        #### Compute sum of 'active' trajectories satisfying: min_frame <= current_frame <= max_frame
        larger_minFrame=link_props['min_frame']<=current_frame
        smaller_maxFrame=link_props['max_frame']>=current_frame
        istrue = larger_minFrame == smaller_maxFrame
        
        #### Store sum in numTracks
        numTracks[current_frame] = istrue.sum()
    
    return numTracks


#%%
def get_numtracks_fit(link_props,locs_info,saveplot=True):
    
    #### Calculate number of tracks per frame
    NoFrames=link_props['max_frame'].max()+1 # get total number of rames
    frames=range(0,NoFrames) 
    numtracks=get_numtracks(link_props,frames) # get number of tracks per frame
    
    #### Define fit model and fit to numtracks vs frames
    def exp_fit(x,a,b,c):
        f=a*np.exp(-x/b)+c
        return f

    popt,pcov=curve_fit(exp_fit,xdata=numtracks.index,
                          ydata=numtracks,
                          p0=[numtracks[0], # No of tracks in first frame
                              len(frames)/2, # decay constant set to half of acquisition length
                              numtracks[-(int(len(frames)*0.1)):].mean()]) # mean No of tracks over last 10 % of acquisition     
    
    #### Plot No of tracks vs frames incl fit
    f=plt.figure(num=20,figsize=[4,3])
    f.subplots_adjust(left=0.2,right=0.99,bottom=0.2,top=0.95)
    f.clear()
    ax=f.add_subplot(111)
    ax.plot(numtracks.index,numtracks,label='data')
    ax.plot(numtracks.index,exp_fit(numtracks.index,*popt),'r',linewidth=2,label='fit')
    ax.set_xlabel('Frames')
    ax.set_ylabel('# of Tracks')
    ax.legend(loc=1)
    
#    if saveplot==True:
#        ### Get path of file
#        path=info[0]['File']
#        path=os.path.splitext(path)[0]
#        plt.savefig(os.path.join(path+'N-tracks_vs_frames.pdf'))

        
    #### Prepare output            
    s_out=pd.DataFrame()
    s_out_init = pd.Series(np.ones(len(link_props)))
    s_out['a']=s_out_init*popt[0]
    s_out['b']=s_out_init*popt[1]
    s_out['c']=s_out_init*popt[2]
    s_out.set_index(link_props.index,inplace=True)
    
    return s_out    

#%%
def scan_sr_mem(locs,locs_info,locs_dir,sr,mem,length_hp,downsize='crop',save_scan=True):
    '''
    
    '''

    #### Downsize image stack either via cropping or time segmentation    
    #### for faster parameter scan
    if downsize=='crop':
        #### Cropping image stack to center 200px^2 ROI
        img_size=locs_info[0]['Height'] # get image size
        roi_range=100 # define max range for cropping
        locs=locs[(locs.x>(img_size/2-roi_range))&(locs.x<(img_size/2+roi_range))]
        locs=locs[(locs.y>(img_size/2-roi_range))&(locs.x<(img_size/2+roi_range))]
    if downsize=='segm':
        #### Creating time segment of image stack
        NoFrames=locs_info[0]['Frames'] # Original no of frames
        seg_size=0.2 # Determines fraction of remaining frames in segment
        locs=locs[locs.frame<=(np.ceil(NoFrames*seg_size))]
    
    save_picked=False
    #### Scan loop over tuple (search_range,memory) 
    #### Init output
    df_out=pd.DataFrame(columns=['len_mean','numtracks','sr','mem'])
    for s in sr:
        for m in mem:
            #### Init parameters for get link
            params={'search_range':s,'memory':m}
            
            #### Link localizations via trackpy
            link=get_link(locs,locs_info,save_picked,**params)
            #### Get link_props without MSD calculation and fitting
            link_props=get_linkprops_noMSD(link,locs_info,length_hp)
            #### Optional saving of scan results in directory
            if save_scan:
                savename='scan_sr%i_mem%i_len-hp%i_%s.h5'%(s,m,length_hp,downsize)
                link_props.to_hdf(os.path.join(locs_dir[0],savename),key='link')
            #### Calculating mean track length and number of tracks
            len_mean=link_props['props','len'].mean()
            numtracks=link_props['props','len'].size
            df_temp=pd.DataFrame({'len_mean':len_mean,
                                  'numtracks':numtracks,
                                  'sr':s,'mem':m},index=[(s,m)])
            df_out=pd.concat([df_out,df_temp])
    #### Plot results
    plot_scan_results(df_out)
           
    return df_out

#%%
def plot_scan_results(df):
    '''
    '''
    
    f=plt.figure(num=21,figsize=[5,3])
    f.subplots_adjust(left=0.15,right=0.84,bottom=0.17,top=0.83)
    f.clear()
    
    ax=f.add_subplot(111)
    ax.plot(np.arange(0,len(df),1),df.len_mean,'-o')
    ax.set_xticks(np.arange(0,len(df),1))
    ax.set_xticklabels(df.sr)
    ax.set_xlabel('Search range (px)')
    ax.set_ylabel('Track length (frames)')
    #### Second x-axis
    ax1=ax.twiny()
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.set_xticks(np.arange(0,len(df),1))
    ax1.set_xticklabels(df.mem)
    ax1.set_xlabel('Memory (frames)')
    ##### Second y axis
    ax2=ax.twinx()
    ax2.plot(np.arange(0,len(df),1),df.numtracks,'-o',c='grey')
    ax2.set_ylabel('# Tracks ()',color='grey')
    ax2.yaxis.set_tick_params(color='grey')
    plt.show()      

#%%    
def get_linkprops(link,locs_info,length_hp=100,max_lagtime=200):
    '''
    Calculates various trajectory properties:
        1) Trajectory lengths
        2) MSD analysis results
    
    Parameters
    ---------
    link : pandas.Dataframe  
        Same locs as locs, index set to 'particle' generated by trackpy.link,
        assigning each localization to a particle trajectory
    length_hp : int
        Cut-off value removing all trajectories shorter than length_hp. Set to
        100 by default.
    max_lagtime: int
        Maximal lagtime to which MSD is calculated. Set to 200 by default.
    Returns
    -------
    link_props : pandas.DataFrame
    '''
   
    #### Get some properties
    print('Calculating means and lengths...')
    link_props=link.groupby('particle').mean()
    link_props['min_frame']=link.frame.groupby('particle').min() # get first frame of trajectory
    link_props['max_frame']=link.frame.groupby('particle').max() # get last frame of trajectory
    link_props['len']=np.subtract(link.frame.groupby('particle').max(),link.frame.groupby('particle').min()) + 1 # get trajectory length
    
    #### Reduce to traces above length high pass
    print('Applying track length high pass...')
    pass_particle=link_props[(link_props.len>length_hp)].index
    link=link.loc[pass_particle,:]
    link_props=link_props[link_props.len>length_hp]
    
    #### Extract numTracks vs frames and fit to f(x)=a*exp(-x/b)+c
    print('Fitting decay of number of tracks per frame')   
    numtrack_fit=get_numtracks_fit(link_props,locs_info)
    
    #### MSD 
    print('Calculating msds...')
    if max_lagtime=='max':
        tau_max=link_props['len'].max()
    else:
        tau_max=max_lagtime
        
    imsd=tp.imsd(link,1,1,max_lagtime=tau_max)
       
    #### Fit individual msds
    print('Fitting msds...')
    imsd_logfit=imsd.apply(lambda df:fit_logiMSD(df),axis=0)
    imsd_fit_iter=imsd.apply(lambda df:fit_iMSD_free_iterate(df),axis=0)
    imsd_fit=pd.concat([imsd_logfit.T,imsd_fit_iter.T],axis=1)
    
    #### Combine link_props, imsd_fit & imsd
    link_props=pd.concat([link_props,numtrack_fit,imsd_fit,imsd.T],keys=['props','numtrack_fit','fit','msd'],axis=1)
     
    return link_props
#%%
def fit_logiMSD(msd,plot=False):
    '''
    Logarithmic weighted fitting of individual mean square displacements (msd) as given by trackpy.iMSD() 
    assuming following form for the msd: 
        
        msd=A*t^n
            A = Amplitude prop. to diffsuion coefficient D
            n = Determines diffusion mode, i.e. n=1 <-> Free diffusion
            t = Lagtime 
            
        -> log(msd)=log(A)+n*log(t)
    '''
    
    #### Absolute error of logarithmic msd
    def abs_err_logmsd(msd):
        #### Relative error of msd
        def rel_err_msd(N,n):
            sr_msd=(2*n**2+1)/((3*n)*(N-n+1))
            sr_msd=np.sqrt(sr_msd)
            return sr_msd
        
        sa_logmsd=np.log(msd)+np.log(rel_err_msd(len(msd),msd.index.astype(float)))
        #### Set last value to finite number
        sa_logmsd.iloc[-1]=sa_logmsd.iloc[-2]
        return sa_logmsd
    
    #### Fit function
    def fitfunc(logtau,A,n):
        logmsd=np.log(A)+n*logtau
        return logmsd
    
    #### Drop NaN
    msd.dropna(inplace=True)
    
    #### Prepare fit
    x=np.log(msd.index.astype(float))
    y=np.log(msd)
    yerr=abs_err_logmsd(msd)
    #### Cut off long lag times
    x=x[:int(len(x)*0.25)]
    y=y.iloc[:int(len(y)*0.25)]
    yerr=yerr.iloc[:int(len(yerr)*0.25)]
    #### Initial value
    p0=[msd.iloc[0],1.]
    try:
        popt,pcov=curve_fit(fitfunc,x,y,p0=p0,sigma=yerr,absolute_sigma=True)
        perr=np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt=np.full(2,np.nan)
        perr=np.full(2,np.nan)
    except ValueError:
        popt=np.full(2,np.nan)
        perr=np.full(2,np.nan)
    except TypeError:
        popt=np.full(2,np.nan)
        perr=np.full(2,np.nan)
    
    if plot:
        #### Plotting for checks
        f=plt.figure(num=12,figsize=[4,3])
        f.subplots_adjust(left=0.1,right=0.99,bottom=0.1,top=0.99)
        f.clear()
        ax=f.add_subplot(111)
        ax.errorbar(np.log(msd.index.astype(float)),np.log(msd),yerr=0.1*abs_err_logmsd(msd),fmt='-x',c='k',alpha=0.2)
        ax.plot(x,fitfunc(x,*popt),'-',c='r',lw=4)
        
    #### Assign tou output
    s_out=pd.Series({'A':popt[0],'A_err':perr[0],'n':popt[1],'n_err':perr[1]})
    
    return s_out

#%%
def fit_iMSD_free(msd,p,plot=False):
    '''

    '''

    #### Fit function
    def fitfunc(tau,a,b):
        msd=a+b*tau
        return msd
    
    #### Prepare fit
    x=msd.index.astype(float)
    y=msd
    #### Cut off long lag times
    x=x[:p]
    y=y.iloc[:p]
    #### Initial value
    p0=[msd.iloc[0],msd.iloc[10]-msd.iloc[0]]
    try:
        popt,pcov=curve_fit(fitfunc,x,y,p0=p0,sigma=None,absolute_sigma=False)
        perr=np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt=np.full(2,np.nan)
        perr=np.full(2,np.nan)
    except ValueError:
        popt=np.full(2,np.nan)
        perr=np.full(2,np.nan)
    except TypeError:
        popt=np.full(2,np.nan)
        perr=np.full(2,np.nan)
    
    if plot:
        #### Plotting for checks
        f=plt.figure(num=11,figsize=[4,3])
        f.subplots_adjust(left=0.1,right=0.99,bottom=0.1,top=0.99)
        f.clear()
        ax=f.add_subplot(111)
        ax.plot(msd.index.astype(float),msd,'-x',c='k',alpha=0.2)
        ax.plot(x,fitfunc(x,*popt),'-',c='r',lw=4)
        
    #### Assign tou output
    s_out=pd.Series({'a':popt[0],'a_err':perr[0],'b':popt[1],'b_err':perr[1]})
    
    return s_out

#%%
def fit_iMSD_free_iterate(msd,max_it=5,plot=False):
    '''
    
    '''
    
    #### Drop NaN
    msd.dropna(inplace=True)
    #### Set inital track length that will be fitted to 10% of full track length
    p=[int(len(msd)*0.2)]
    
    i=0
    while i<max_it:

        #### Fit
        s_out=fit_iMSD_free(msd,p[-1],plot=plot)
        x=np.abs(s_out['a']/s_out['b'])
        
        #### Assign iteration and fitted track length
        s_out['p']=p[-1]
        s_out['max_it']=i+1
        
        #### Update optimal track length to be fitted
        try:
            p=p+[int(np.ceil(2+2.7*x**0.5))]
        except:
            break
            
        if np.abs(p[-1]-p[-2])<1:
            break
        i+=1
        
    return s_out

 