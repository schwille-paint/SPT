import numpy as np
import pandas as pd

#%%
def get_ecdf(x):
    """
    Calculate experimental continuous distribution function (ECDF) of random variable x
    so that counts(value)=probability(x>=value). I.e last value of counts=1.
    
    Equivalent to :
        matplotlib.pyplot.hist(tau_dist,bins=numpy.unique(tau_dist),normed=True,cumulative=True)
    
    Parameters
    ---------
    x : numpy.ndarray
        1 dimensional array of random variable  
    Returns
    -------
    values : numpy.ndarray
         Bins of ECDF corresponding to unique values of x.
    counts : numpy.ndarray
        counts(value)=probability(x<=value).
    """
    x=x[x>0]
    values,counts=np.unique(x,return_counts=True) # Give unique values and counts in x
    counts_leq=np.cumsum(counts) # Empirical cfd counts(X<=x)
    counts_l=np.concatenate([[0],counts_leq[0:-1]]) # Empirical cfd counts(X<x)
    counts_l = counts_l/counts_leq[-1] # normalize that sum(counts) = 1, i.e. now P(X<x)
    counts_l_inv=1-counts_l # Empirical cfd inverse, i.e. P(X>=x)
    return (values,counts_l_inv)

#%%
def get_half_time(df):
    '''
    Get half life time, i.e. 1-ecdf (means bigger than T) of start times for different ignore values.
    '''
    fields=df.columns.values[11:17]
    s_out=pd.Series(index=fields)

    for f in fields:
        ecdf=get_ecdf(df.loc[:,f])
        half_idx=np.where(ecdf[1]>0.5)[0][-1]
        s_out[f]=ecdf[0][half_idx]
    
    s_out.index=['half_i%i'%(i) for i in range(0,6)]
    return  s_out

#%%
def get_NgT(df):
    '''
    Return average and standard deviation of number of trajectories per particle longer or equal than T.
    '''
    fields=df.columns.values[17:]
    NgT_mean=df.loc[:,fields].mean()
    NgT_std=df.loc[:,fields].std()
    NgT=pd.DataFrame({'avg':NgT_mean,'err':NgT_std})
    NgT.index=NgT.index.values.astype(float)
    return NgT

#%%
def get_T_with_N(df):
    '''
    Return critical times of NgT
    '''
    N=get_NgT(df).avg # Get NgT
    Ncrit=[0.5,1,2,5,10,20,50,100,1000] # Define critical points
    s_out=pd.Series(index=[n for n in Ncrit]) # Init output
    for n in Ncrit:
        try:
            s_out[n]=N[N>=n].index.values[-1]
        except IndexError:
            s_out[n]=0
            
    s_out.index=['Tn%.0e'%(n) for n in Ncrit] # Rename index
    
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
                               spacer=fields[1],
                               buffer=fields[2],
                               exp=int(fields[3][3:]),
                               power=int(fields[4][1:-2]),
                               )
    
    return df_result