# Some special functions used for SPT project
import numpy as np
from tqdm import tqdm
import numba 

#%%
def multiply_jumps(df,factor,segment,ratio):
    '''
    Multiply jumps by ``sqrt(factor)`` such that segments will have a diffusion contant multiplied by ``factor``. 
    Segments number of localizations is given by ``segment``. 
    Ratio indicates how many segments are mutiplied, i.e. for ``ratio=2`` every 2nd, 
    for ``ratio=3`` every third segment is multiplied.
    
    Args:
        df(pandas.DataFrame): One trajectory as returned by spt.mob_props.main() (i.e. one group in pickedXXXX.hdf5 files) 
        factor(int):          Diffusion constant multiplication factor, i.e. jumps will mutliplied by sqrt(factor) !!
        segment(int):         Number of localizations within segments.
        ratio(int):           Ratio of multiplied to normal segments (see above).
    Return:
        pandas.DataFrame: Same as ``df`` but with modified jumptimes.
    '''
    N=len(df)
    k=int(N/(segment*ratio))+1 # Number of full cycles
    
    ### Jumps in x-y for every position
    x=df.x.values
    dx=np.concatenate([np.array([0]),x[1:]-x[:-1]])
    y=df.y.values
    dy=np.concatenate([np.array([0]),y[1:]-y[:-1]])
    
    ### Create mask for segment multiplication
    ### 
    mask_unit=np.array([1]*(segment*(ratio-1))+[np.sqrt(factor)]*segment)
    mask=np.concatenate([mask_unit]*k)
    mask=mask[:N]
    
    ### Apply mask and sum up consecutive jumps starting from first position
    ### to get new x-y coordinates
    dx_mod=dx*mask
    x_mod=np.cumsum(dx_mod)+x[0]
    dy_mod=dy*mask
    y_mod=np.cumsum(dy_mod)+x[0]
    
    ### Assign new positions
    df.x=x_mod
    df.y=y_mod
    
    return df

#%%
def multiply_jumptimes(df,factor,segment,ratio):
    '''
    Multiply jumptimes by ``factor`` in segments. Segments number of localizations is given by ``segment``. 
    Ratio indicates how many segments are mutiplied, i.e. for ``ratio=2`` every 2nd, for ``ratio=3`` every third segment is multiplied
    leading to apparent slower diffusion within these segments.
    
    Args:
        df(pandas.DataFrame): One trajectory as returned by spt.mob_props.main() (i.e. one group in pickedXXXX.hdf5 files) 
        factor(int):          Jumptime multiplication factor
        segment(int):         Number of localizations within segments.
        ratio(int):           Ratio of slowed down to normal segments (see above).
    Return:
        pandas.DataFrame: Same as ``df`` but with modified jumptimes.
    '''
    N=len(df)
    k=int(N/(segment*ratio))+1 # Number of full cycles
    
    t=df.frame.values
    dt=np.concatenate([np.array([0]),t[1:]-t[:-1]])
    
    mask_unit=np.array([1]*(segment*(ratio-1))+[factor]*segment)
    mask=np.concatenate([mask_unit]*k)
    mask=mask[:N]
    
    dt_mod=dt*mask
    t_mod=np.cumsum(dt_mod)+t[0]
    
    df.frame=t_mod.astype(int)
    
    return df

#%%
def apply_multiply_jumps(df,factor,segment,ratio):
    '''
    Groupby apply approach of multiply_jumptimes(df,factor,segment,ratio) to each group in ``df``.

    Args:
        df(pandas.DataFrame): (Complete) linked localizations as returned by spt.mob_props.main() (pickedXXXX.hdf5 files) 
        factor(int):          Jumptime multiplication factor
        segment(int):         Number of localizations within segments.
        ratio(int):           Ratio of slowed down to normal segments (see above).

    Return:
        df_mod (TYPE): Same as ``df`` but with modified jumptimes.

    '''
    ### Print some statements
    print('Multiplying jumps ...')
    print('... by factor of %i'%(factor*100) +r'% ...')
    print('... every %i. segment ...'%ratio)
    print('... of %i localizations each.'%segment)
    df_mod=df.copy()
    
    tqdm.pandas()
    df_mod=df_mod.groupby('group').progress_apply(lambda df: multiply_jumps(df,factor,segment,ratio))
        
    df_mod.reset_index(inplace=True)
    
    return df_mod

#%%
def assign_subgroup(df,subN):
    '''
    Splitting of ``supgroup`` (full trajectory ID) into ``subgroup`` (subtrajectory ID).
    Each subgroup splits the supgroup in chunks of ``subN``, i.e. the subtrajectory duration.
    Subtrajectories with less then 10 localizations are dropped.
    
    Args:
        df(pandas.DataFrame):    One trajectory as returned by spt.mob_props.main() (i.e. one group in pickedXXXX.hdf5 files) 
        subN(int):               Maximum length of sub-trajectories
        
    Return:
        pandas.DataFrame: Same as ``df`` but with one new column ``subgroup`` (subtrajectory ID) in chunks of ``subN``. See also above.
    '''
    
    df=df.drop(columns=['supgroup']) # Drop supgroup, since it will appear in index through groupby!
    N=len(df)
    
    ### Create subgroup column
    k=int(N/subN) # Subgroups of length subN
    r=N%subN # Remainder subgroup
    if k>0:
        subgroups=np.hstack([np.ones(subN)*i] for i in range(k))
        if r!=0: 
            subgroups=np.hstack([subgroups,np.ones((1,r))*k]) # Assign last subgroup if there is a remainder
    else: # Length of trajectory smaller than subN, assign remainder
        subgroups=np.ones((1,r))*k
    
    subgroups=subgroups.flatten().astype(np.int64)
    
    ### Assign subgroups
    df_out=df.assign(subgroup=subgroups)
    
    ### Remove subgroups with less than 10 localizations
    last_subgroup=df_out.subgroup==df_out.subgroup.max() # Get bool with True at last subgroup
    last_subgroupN=np.sum(last_subgroup) # How long is last subgroup
    if last_subgroupN<10: df_out=df_out[~last_subgroup] # Remove subgroup if shorter than 10 localizations
    
    return df_out

#%%
def split_trajectories(df,subN):
    '''
    Groupby apply approach of assign_subgroup(df,subN) to each group in ``df``.
    Splits trajectories into shorter trajectories of length ``subN``. Three new columns are assigned to give track ID:
        ``supgroup``: Super-group = group column of original
        ``subgroup``: Sub-group = New trajectories of length subN or lower
        ``group``:    Unique new sub-group ID
    
    Args:
        df(pandas.DataFrame):    (Complete) linked localizations as returned by spt.mob_props.main() (pickedXXXX.hdf5 files)
        subN(int):               Maximum length of sub-trajectories
        
    Return:
        pandas.DataFrame: Same as ``df`` but with new columns indicating sup(er)- or subtrajectories. See above.               
    '''
    print('Start splitting ...')
    #### Rename group to supgroup (super-group)
    supdf=df.rename(columns={'group':'supgroup'})
    
    ### Convert groups in supgroups and assign subgroups of len subN to df
    tqdm.pandas() # For progressbar under apply
    subdf=supdf.groupby('supgroup').progress_apply(lambda df: assign_subgroup(df,subN))
    subdf.reset_index(inplace=True)
    subdf.drop(columns=['level_1'],inplace=True)
    
    print('Assign unique group index ...')
    ### Transform supgroup and subgroup into unique groupID
    i=subdf.loc[:,['supgroup','subgroup']].values
    value,count=np.unique(i,axis=0,return_counts=True)
    groups=[[np.ones(count[j])*j] for j in range(len(count))]
    groups=np.concatenate(groups,axis=1).flatten().astype(np.int64)
    subdf=subdf.assign(group=groups)
    
    return subdf

#%%
def assign_prop_to_picked(props,picked,field='a_iter'):
    ### Get property and number of localizations per group
    vals=props.loc[:,field].values 
    lens=props.loc[:,'n_locs'].values
    ### Combine both to assign to picked
    vals2picked=np.concatenate([np.ones(int(lens[i]))*vals[i] for i in range(len(vals))])
    
    picked_assigned=picked.assign(assign_field=vals2picked)
    
    return picked_assigned
    
#%%
def get_trackmap(locs,fov,oversampling,field='assign_field'):
    
    
    ### Define FOV and oversampling
    xmin=fov[0][0]
    xmax=fov[0][1]
    ymin=fov[1][0]
    ymax=fov[1][1]

    ### Reduce to locs within FOV, substract lower limits
    in_view=(locs.x>=xmin) & (locs.x<xmax) & (locs.y>=ymin) & (locs.y<ymax)
    locs_fov=locs[in_view]
    locs_fov.x=locs_fov.x-xmin
    locs_fov.y=locs_fov.y-ymin
    
    ### Prepare oversampled maps dimensions
    image_shape=(int((ymax-ymin)*oversampling),
                 int((xmax-xmin)*oversampling))
    
    ### Get x,y,and diffusion constant
    x=((locs_fov.x.values)*oversampling).astype(np.int32) # Like this we assign to x sub-pixels!
    y=((locs_fov.y.values)*oversampling).astype(np.int32) # Like this we assign to y sub-pixels!
    f=locs_fov.loc[:,field].values
    
    ### Turn y (rows) and x (columns) into flat index ans assign to real part
    ### Assign field value to imaginary part
    flatyxf=np.zeros((len(x),1)).astype(np.complex64)
    flatyxf=y*image_shape[1]+x
    flatyxf=flatyxf+1j*f
    flatyxf_u=np.unique(flatyxf)
    
    maps=render_trackmap(flatyxf_u,image_shape)
    
    maps[0][maps[0]==0]=np.nan
    
    return [maps[0],maps[1],maps[2]]
    
#%%
@numba.jit(nopython=True,nogil=True,cache=True)
def render_trackmap(flatyxf_u,image_shape):
    
    ## Prepare images for later assignment
    map_n=np.zeros(image_shape).astype(np.float32) # Map for number of tracks within sub-pixel
    map_f1=np.zeros(image_shape).astype(np.float32) # Map for first moment of attribute for tracks within sub-pixel
    map_f2=np.zeros(image_shape).astype(np.float32) # Map for second moment of attribute for tracks within sub-pixel
    
    ### Fill maps
    for val in flatyxf_u:
        ### Convert real part flat index back to two dimensions
        i=int(val.real/image_shape[1])
        j=int(val.real%image_shape[1])
        
        ### Assign
        map_n[i,j]+=1
        map_f1[i,j]+=val.imag
        map_f2[i,j]+=val.imag**2
      
    map_mean=map_f1/map_n
    map_std=np.sqrt(map_f2/map_n-map_mean**2)
    
    return map_n,map_mean,map_std
        
        