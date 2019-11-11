def annotate_filter(locs,movie,frame,photon_hp=0):
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
        locs that survieved photon high pass
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import trackpy as tp
    
    #### Plot photons
    f=plt.figure(num=10,figsize=[4,3])
    f.subplots_adjust(left=0.05,right=0.95,bottom=0.08,top=0.99,hspace=0)
    f.clear()
    ax=f.add_subplot(211)
    bins=np.arange(0,locs.loc[:,'photons'].median()*4,10)
    ax.hist(locs.loc[:,'photons'],bins=bins,label='all')
    ax.axvline(photon_hp,ls='-',lw=3,c='r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax=f.add_subplot(212)
    ax.hist(locs.loc[locs.frame==frame,'photons'],bins=bins,label='frame=%i'%(frame))
    ax.axvline(photon_hp,ls='-',lw=3,c='r')
    ax.set_yticks([])
    ax.legend()
    
    #### Apply photon 'high pass' to locs
    locs_filter=locs.drop(locs[locs.photons<photon_hp].index)
    
    #### Plot annotate and median proximity
    f=plt.figure(num=11,figsize=[6,6])
    f.subplots_adjust(left=0.,right=0.99,bottom=0.,top=0.95)
    f.clear()
    ax=f.add_subplot(111)
    tp.annotate(locs_filter[locs_filter.frame==frame],movie[frame],ax=ax,invert=True)
    
    med_prox=tp.proximity(locs.loc[locs.frame==frame,:]).median()
    ax.set_title('Median proximity: %.1f px'%(med_prox))
    ax.set_xticks([])
    ax.set_yticks([])
    
    return locs_filter

#%%
def get_link(locs,search_range,memory):
    '''
    
    '''
    import trackpy as tp
      
    #### Link locs
    link= tp.link_df(locs,search_range,memory=memory)
    link.sort_values(by=['particle','frame'],ascending=True,inplace=True)
    link.set_index('particle', inplace=True)
    
    return link

#%%
def get_linkprops(link,length_hp=100,max_lagtime=200):
    '''
    
    '''
    import trackpy as tp
    import pandas as pd
      
    #### Get some properties
    print('Calculating means and lengths...')
    link_props=link.groupby('particle').mean()
    link_props['len']=link.groupby('particle').size()
    
    #### Reduce to traces above length high pass
    print('Applying track length high pass...')
    pass_particle=link_props[(link_props.len>length_hp)].index
    link=link.loc[pass_particle,:]
    link_props=link_props[link_props.len>length_hp]
    
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
    link_props=pd.concat([link_props,imsd_fit,imsd.T],keys=['props','fit','msd'],axis=1)
     
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
    from scipy.optimize import curve_fit as curve_fit
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
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
    from scipy.optimize import curve_fit as curve_fit
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
        
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
    import numpy as np
    
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
        p=p+[int(np.ceil(2+2.7*x**0.5))]
    
        if np.abs(p[-1]-p[-2])<1:
            break
        i+=1
        
    
    return s_out