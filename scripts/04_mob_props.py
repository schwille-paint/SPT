import os
import pandas as pd
import importlib
import warnings
warnings.filterwarnings("ignore")

#### Load own packages
import picasso.io as io
import spt.trackpy_wrap as track
import spt.analyze as analyze

importlib.reload(track)
importlib.reload(analyze)
#%%
######################################## Define data
#### Path to locs.hdf5
locs_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-11-21_th_slb_origami-density-checks/s3_th_slb_ex200_p038uW_1/19-11-25_JS']
locs_name=['s3_th_slb_ex200_p038uW_1_MMStack_Pos0.ome_locs.hdf5']
#### Path to tiff stack
movie_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-11-21_th_slb_origami-density-checks/s3_th_slb_ex200_p038uW_1']
movie_name=['s3_th_slb_ex200_p038uW_1_MMStack_Pos0.ome.tif']

######################################## Read in data
#### Read in locs as DataFrame
locs,locs_info=io.load_locs(os.path.join(locs_dir[0],locs_name[0]))
locs=pd.DataFrame(locs)
#### Read in movie
movie=io.load_movie(os.path.join(movie_dir[0],movie_name[0]))[0]
#%%
######################################## Annotate (& Filter)
#### Visually inspect detected particle and median nearest neighbor distance
frame=1
locs=track.annotate_filter(locs,movie,frame)

#%%
######################################## Parameter scan for trackpy
param_scan=False
#### Define link parameters for scanning optimal search range    
if param_scan==True:
    search_range=[1,2,3,5]
    memory=[1,2]
    length_hp = 20
    #### Get scan results    
    scan_results=track.scan_sr_mem(locs,locs_info,
                                   locs_dir,search_range,memory,
                                   length_hp,downsize='segm',save_scan=False)
#%%
######################################## Get link_props with optimal parameters
search_range=2
memory=1
#### Link
link=track.get_link(locs,search_range,memory)
#### MSDs and fitting
#%%
link_props=track.get_linkprops(link,locs_info,length_hp=20)
#### Save linkprops
print('Saving linkprops...')
savename='linkprops_sr%i_mem%i_len_hp%i.h5'%(search_range,memory,length_hp)
link_props.to_hdf(os.path.join(locs_dir[0],savename),key='link')