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
locs_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-12-04_SLB_fix_and_th/140_wash_slb_B_exp200_p038uW_2_nd_1/19-12-05_JS']
locs_name=['140_wash_slb_B_exp200_p038uW_2_nd_1_MMStack_Pos1.ome_locs.hdf5']
#### Path to tiff stack
movie_dir=['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-12-04_SLB_fix_and_th/140_wash_slb_B_exp200_p038uW_2_nd_1']
movie_name=['140_wash_slb_B_exp200_p038uW_2_nd_1_MMStack_Pos1.ome.tif']

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
param_scan=True
#### Define link parameters for scanning optimal search range    
if param_scan==True:
    search_range=[1,2,3,5,7,9]
    memory=[1,2]
    length_hp = 20
    #### Get scan results    
    scan_results=track.scan_sr_mem(locs,locs_info,
                                   locs_dir,search_range,memory,
                                   length_hp,downsize='segm',save_scan=False)
#%%
######################################## Get link_props with optimal parameters
params={'search_range':2,
        'memory':1,
        'length_hp':20}
save_picked=False
#### Link
link=track.get_link(locs,locs_info,save_picked,**params)
#### MSDs and fitting
#%%
link_props=track.get_linkprops(link,locs_info,length_hp=20)
#%%
#### Save linkprops
print('Saving linkprops...')
savename='linkprops_sr%i_mem%i_len_hp%i.h5'%(params['search_range'],
                                             params['memory'],
                                             params['length_hp'])
link_props.to_hdf(os.path.join(locs_dir[0],savename),key='link')