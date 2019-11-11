import os
import importlib
import picasso.io

#### Load own packages
import lbfcs.io as io
import spt.analyze_old as track
importlib.reload(track)
############################################################## Define data
#### Path to locs.hdf5
locs_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/z.JB/19-03-01_Cholesterol-diffusion/id56-500_Pm2-40nM_exp-30ms_B+_50mM-MgCl2+POC_p250uW_4/19-03-02_JB']
locs_name=['id56-500_Pm2-40nM_exp-30ms_B+_50mM-MgCl2+POC_p250uW_4_MMStack_Pos0.ome_locs.hdf5']
#### Path to tiff stack
movie_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/z.JB/19-03-01_Cholesterol-diffusion/id56-500_Pm2-40nM_exp-30ms_B+_50mM-MgCl2+POC_p250uW_4']
movie_name=['id56-500_Pm2-40nM_exp-30ms_B+_50mM-MgCl2+POC_p250uW_4_MMStack_Pos0.ome.tif']

############################################################## Read in data
#### Read in locs
locs=io.load_locs(os.path.join(locs_dir[0],locs_name[0]))[0]
### Read in movie
movie,movie_info=picasso.io.load_movie(os.path.join(movie_dir[0],movie_name[0]))

#%%
############################################################## Annotate & Filter 
frame=100
photon_hp=800
locs_filter=track.annotate_filter(locs,movie[0],frame,photon_hp)

#%%
############################################################## Get link_props
search_range=[5,10,15]
memory=[1,2,3]

for s in search_range:
    for m in memory:
        #### Link
        link=track.get_link(locs,s,m)
        #### MSDs and fitting
        link_props=track.get_linkprops(link,length_hp=100)
        #### Save to hdf5
        savename='test_link_sr%i_mem%i_php%i.h5'%(s,m,photon_hp)
        link_props.to_hdf(os.path.join(locs_dir[0],savename),key='link')

