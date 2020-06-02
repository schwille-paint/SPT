import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import os
### Load custom modules
import picasso.io as io

#%%
def get_link(locs,search_range,memory):
    '''
    Apply trackpy.link_df on localizations with given search_range and memory to get tracks sorted by group and frame.
    All tracks shorter or equal to 10 frames are removed.
    
    args:
        locs(pandas.DataFrame):        Localizations loaded with picasso.io.load_locs() as pandas.DataFrame
        info(picasso.io):              Info as loaded by picasso.io.load_locs()
        search_range(int):             Localizations within search_range (spatial) will be connetcted to tracks (see trackpy.link_df)
        memory(int):                   Localizations within memory (temporal) will be connetcted to tracks (see trackpy.link_df)
        
    return:
        link(pandas.DataFrame):         Linked trajecories using trackpy.link() with parameters. 
                                       All trajectories with <= 10 localizations are already removed!!  
    ''' 
    ### Link locs
    link= tp.link(locs,
                  search_range,
                  memory=memory,
                  link_strategy='hybrid',
                  )
    ### Sort and rename
    link.sort_values(by=['particle','frame'],ascending=True,inplace=True)
    link=link.rename(columns={'particle':'group'}) # Rename to groups for picasso compatibility
    
    ### Throw away tracks with only 10 localizations
    link=drop_shorttracks(link)
   
    return link

#%%
def drop_shorttracks(df,crit_len=10):
    '''
    Helper function to remove trajectories shorter than crit_len from picked.
    
    args:
        df(pandas.DataFrame): Trajectories DataFrame (see trackpy.link())
        crit_len(int=10):     Trajectories lower than crit_len (frames) will be removed from df
    '''
    ### Helper function to assign group length to all locs for later removal    
    def get_len(df,crit_len):
        s_out=pd.Series(np.ones(len(df))*len(df))
        return s_out
    
    ### Assign group length to locs
    len_group=df.groupby('group').apply(lambda df: get_len(df,crit_len))
    df_out=df.assign(n_locs=len_group.values)
    df_out=df_out[df_out.n_locs>=crit_len]
    
    return df_out

#%%
def scan_sr_mem(locs,info,path,sr,mem,roi=True,timewindow=True):
    '''
    Quick scan of the trackpy.link algorithm using tuples of searc_hranges and memory 
    over center ROI and starting time window of the video. Output and coresponding plot will be saved.
    
    args:
        locs(pandas.DataFrame):   Localizations loaded with picasso.io.load_locs() as pandas.DataFrame
        info(picasso.io):         Info as loaded by picasso.io.load_locs()
        path(str):                Path to _locs.hdf5 file for saving output
        sr(list(int)):            List of search_ranges to scan (see trackpy.link)
        mem(list(int)):           List of memory values to scan  (see trackpy.link)
        roi(bool=True):           If True scan is performed on cropped video to center 200^2px FOV
        timewindow(bool=True):    If True scan is performed on first 300 frames of video only

    return:
        df_out(pandas.DataFrame):                    
    '''
    
    #######
    ''' Get median critical proximity of all localizations in the first 100 frames
        Critical Proximity: 90% of all next-neighbor distances are greater than critical proximity
    '''
    ######
    if info[0]['Frames']>=100:
        prox=[np.percentile(tp.proximity(locs.loc[locs.frame==f,:]),10) for f in range(100)]
        prox=np.median(prox) 
    else:
        prox=[np.percentile(tp.proximity(locs.loc[locs.frame==f,:]),10) for f in range(info[0]['Frames'])]
        prox=np.median(prox)
    
    #######
    ''' 1) Crop locs to 200px^2 center FOV and 
        2) take only first 300 frames of _locs than
        3) perform scan over search_range and memory using get_link()
        4) Plot and save plot
    '''
    ######
    ### Crop stack to 200px^2 center FOV
    if roi==True:
        img_size=info[0]['Height'] # get image size
        roi_width=100
        locs=locs[(locs.x>(img_size/2-roi_width))&(locs.x<(img_size/2+roi_width))]
        locs=locs[(locs.y>(img_size/2-roi_width))&(locs.x<(img_size/2+roi_width))]
        
    ### Take only first 300 frames of stack
    if timewindow==True:
        locs=locs[locs.frame<=300]

    ### Scan loop over tuple (search_range,memory) 
    df_out=pd.DataFrame(columns=['len_mean','numtracks','sr','mem']) # Init output
    idx=0
    for s in sr:
        for m in mem:  
            #### Link localizations via trackpy
            link=get_link(locs,s,m) 
            ### Assign
            len_med=link.n_locs.median()
            num_tracks=len(link.group.unique())
            df_temp=pd.DataFrame({'len_med':len_med,
                                  'numtracks':num_tracks,
                                  'sr':s,
                                  'mem':m},index=[(s,m)])
            
            df_out=pd.concat([df_out,df_temp])
            idx+=idx
    
    plot_scan_results(df_out,prox)
    
    
    ### Save plot
    path=os.path.splitext(path)[0]
    plt.savefig(os.path.join(path+'_scan.pdf'),transparent=True)
    
    return df_out,prox

#%%
def plot_scan_results(df,prox):
    '''
    Quickly plot results of scan_sr_mem().
    
    args:
        df(pandas.DataFrame):  Return DataFrame of scan_sr_mem() 
    '''
    
    f=plt.figure(num=21,figsize=[6,3])
    f.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.75)
    f.clear()
    
    plt.title(r'90$\%$ of all NN distances > '+'%.1f px'%(prox))
    
    ax=f.add_subplot(111)
    ax.plot(np.arange(0,len(df),1),df.len_med,'-o',c='k')
    ax.set_xticks(np.arange(0,len(df),1))
    ax.set_xlim(-1,len(df))
    ax.set_xticklabels(df.sr)
    ax.set_xlabel('Search range (px)')
    ax.set_ylabel('Track length (frames)')
    
    #### Second x-axis
    ax1=ax.twiny()
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.set_xticks(np.arange(0,len(df),1))
    ax1.set_xlim(-1,len(df))
    ax1.set_xticklabels(df.mem)
    ax1.set_xlabel('Memory (frames)',color='k')
    
    ##### Second y axis
    ax2=ax.twinx()
    ax2.plot(np.arange(0,len(df),1),df.numtracks,'-o',c='grey')
    ax2.set_ylabel('# Tracks ()',color='grey')
    ax2.yaxis.set_tick_params(color='grey')
    plt.show()  

#%%
def main(locs,info,path,**params):
    '''
    Main function to call get_link in a convenient way with data saving.
    '''
    
    ##################################### Params and file handling
    
    ### Path of file that is processed and number of frames
    path=os.path.splitext(path)[0]  
    
    ### Define standard 
    standard_params={'search_range':5,
                     'memory':3,
                     }
    ### Remove keys in params that are not needed
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
        
    ### Processing marks
    params['generatedby']='spt.linklocs.get_link()'
    
    ### Prepare file extension for saving
    sr='%i%i'%(int(params['search_range']/10),
               np.mod(params['search_range'],10),
               )
    mr='%i%i'%(int(params['memory']/10),
               np.mod(params['memory'],10),
               )
    
    ##################################### Link
    link=get_link(locs,
                  search_range=params['search_range'],
                  memory=params['memory'],
                  )
    
    
    ##################################### Save
    #### Save complete link as _picked    
    info_picked=info.copy()+[params] 
    io.save_locs(path+'_picked%s%s.hdf5'%(sr,mr),
                 link.to_records(index=False),
                 info_picked,
                 )
    
    ### Save reduced version (only first 500 groups!) of link for viewing in render
    try:
        max_group=link.group.unique()[500]
    except:
        max_group=link.group.unique()[-1]
    link_view=link[link.group<=max_group]
    
    io.save_locs(path+'_picked%s%s'%(sr,mr)+'g500.hdf5', 
                 link_view.to_records(index=False),
                 info_picked,
                 )
    return [params,link]