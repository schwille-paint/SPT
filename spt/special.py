# Some special functions used for SPT project
import numpy as np
from tqdm import tqdm
import numba 

#%%
def double_jumptime(df,segment,ratio):
    
    N=len(df)
    k=int(N/(segment*ratio))+1 # Number of full cycles
    
    t=df.frame.values
    dt=np.concatenate([np.array([0]),t[1:]-t[:-1]])
    
    mask_unit=np.array([1]*(segment*(ratio-1))+[2]*segment)
    mask=np.concatenate([mask_unit]*k)
    mask=mask[:N]
    
    dt_mod=dt*mask
    t_mod=np.cumsum(dt_mod)+t[0]
    
    df.frame=t_mod.astype(int)
    
    return df

#%%
def apply_double_jumptime(df,segment,ratio):
    
    print('Doubling jumptimes ...')
    print('... every %i. segment ...'%ratio)
    print('... of %i localizations each.'%segment)
    df_mod=df.copy()
    
    tqdm.pandas()
    df_mod=df_mod.groupby('group').progress_apply(lambda df: double_jumptime(df,segment,ratio))
        
    df_mod.reset_index(inplace=True)
    
    return df_mod

#%%
def assign_subgroup(df,subN):
    '''
    Renames group colum into supgroup and and assigns new column subgroup.
    Each subgroup splits the supgroup in chunks of subN, i.e. splitting of trajectories.
    
    args:
        df(pandas.DataFrame):    Picked localizations (see picasso.render)
        subN(int):               Maximum length of sub-trajectories
        
    return:
        subdf(pandas.DataFrame): Picked localizations (see picasso.render) with one new (subgroup)
                                 and one modified column (supgroup = group of original). See also above.

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
def pair_ids(x,y):
    ''' 
    Function to map two integers to one in a unique and deterministic manner.
    Python implementation of Matthew Szudzik's elegant pairing function (WolframAlpha)
    '''
    if x!=max(x,y):
        z=y**2+x
    elif x==max(x,y):
        z=x**2+x+y
    
    return int(z)

#%%
def split_trajectories(df,subN):
    '''
    Split trajectories into shorter trajectories of length subN. Three new columns are assigned to give track ID:
        supgroup: Super-group = group column of original
        subgroup: Sub-group = New trajectories of length subN or lower
        group:    Unique new sub-group ID
    
    args:
        df(pandas.DataFrame):    Picked localizations (see picasso.render)
        subN(int):               Maximum length of sub-trajectories
        
    return:
        subdf(pandas.DataFrame): Picked localizations (see picasso.render) with three new or modified columns (see above)              
    '''
    print('Start splitting ...')
    #### Rename group to supgroup (super-group)
    supdf=df.rename(columns={'group':'supgroup'})
    
    ### Convert groups in supgroups and assign subgroups of len subN to df
    tqdm.pandas() # For progressbar under apply
    subdf=supdf.groupby('supgroup').progress_apply(lambda df: assign_subgroup(df,subN))
    subdf.reset_index(inplace=True)
    subdf.drop(columns=['level_1'],inplace=True)
    
    ### Transform supgroup and subgroup into unique groupID
    ### using Matthew Szudzik's elegant pairing function (WolframAlpha)
    groups=subdf.apply(lambda df: pair_ids(df['supgroup'],df['subgroup']),axis=1)
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
        
        