import os
import pandas as pd
import importlib
import matplotlib.pyplot as plt
### Load custom modules
import picasso.io as io
import spt.linklocs as linklocs
### Update
importlib.reload(linklocs)

######################################## Define data (locs)
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_fix_slb_L_T21/slb_id140_L_exp200_p250uW_T21_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_fix_slb_L_T21/slb_id140_L_exp200_p250uW_T21_2'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_fix_slb_L_T21/slb_id140_L_exp200_p250uW_T21_3'])

file_names=[]
file_names.extend(['slb_id140_L_exp200_p250uW_T21_1_MMStack_Pos0.ome_locs.hdf5'])
file_names.extend(['slb_id140_L_exp200_p250uW_T21_2_MMStack_Pos0.ome_locs.hdf5'])
file_names.extend(['slb_id140_L_exp200_p250uW_T21_3_MMStack_Pos0.ome_locs.hdf5'])

#%%
######################################## Read in single data set
i=0

locs,info=io.load_locs(os.path.join(dir_names[i],file_names[i]))
locs=pd.DataFrame(locs)
movie=io.load_movie(info[0]['File'])[0]


######################################## Annotate (& Filter)
#### Visually inspect detected particle and median nearest neighbor distance and net_gradient
frame=0
mng=0

locs_filter,ax_list=linklocs.annotate_filter(locs,
                                             movie,
                                             frame=frame,
                                             mng=mng,
                                             c_min=70,
                                             c_max=300)
ax_list[0].set_xlim(250,450)
ax_list[0].set_ylim(250,450)
### Save plot
path=os.path.splitext(info[0]['File'])[0]     
plt.savefig(os.path.join(path+'_snap.pdf'),transparent=True)

#%%
######################################## Parameter scan for linking
search_range=[2,3,5,7]
memory=[1,2,3,5]
#### Get scan results    
scan_results=linklocs.scan_sr_mem(locs_filter,
                                  info,
                                  search_range,
                                  memory,
                                  )

#%%
######################################## Link all data sets 
params_all={'search_range':5,
            'memory':3,
            }

params_special={}

############################################ Main loop                   
failed_path=[]
for i in range(0,len(file_names)):
    ### Create path
    path=os.path.join(dir_names[i],file_names[i])
    ### Set paramters for each run
    params=params_all.copy()
    for key, value in params_special.items():
        params[key]=value[i]
    ### Run main function
    try:
        locs,info=io.load_locs(path)
        locs=pd.DataFrame(locs)
        out=linklocs.main(locs,info,**params)
    except:
        failed_path.extend([path])