import os
import pandas as pd
import importlib

#### Load own packages
import picasso.io as io
import spt.trackpy_wrap as track
importlib.reload(track)
############################################################## Define data
#### Path to locs.hdf5
locs_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-11-21_th_slb_origami-density-checks/s1_th_slb_ex200_p038uW_1']
locs_name=['s1_th_slb_ex200_p038uW_1_MMStack_Pos0.ome_locs.hdf5']
#### Path to tiff stack
movie_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-11-21_th_slb_origami-density-checks/s1_th_slb_ex200_p038uW_1']
movie_name=['s1_th_slb_ex200_p038uW_1_MMStack_Pos0.ome.tif']

############################################################## Read in data
#### Read in locs as DataFrame
locs=pd.DataFrame(io.load_locs(os.path.join(locs_dir[0],locs_name[0]))[0])
### Read in movie
movie=io.load_movie(os.path.join(movie_dir[0],movie_name[0]))[0]

#%%
############################################################## Annotate & Filter 
frame=1
photon_hp=800
locs_filter=track.annotate_filter(locs,movie,frame,photon_hp)

#%%
############################################################## Get link_props
search_range=[5,10,15]
memory=[1,2,3]

for s in search_range:
    for m in memory:
        #### Link
        link=track.get_link(locs_filter,s,m)
        #### MSDs and fitting
        link_props=track.get_linkprops(link,length_hp=100)
        #### Save to hdf5
        savename='link_sr%i_mem%i_php%i.h5'%(s,m,photon_hp)
        link_props.to_hdf(os.path.join(locs_dir[0],savename),key='link')

