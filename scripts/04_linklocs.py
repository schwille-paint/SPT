import os
import pandas as pd
import importlib
import traceback

import picasso.io as io
import spt.linklocs as linklocs

importlib.reload(linklocs)

######################################## Define data (locs)
dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p250uW_T21_1'])

file_names=[]
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p250uW_T21_1_MMStack_Pos0.ome_locs.hdf5'])

#%%
######################################## Read in single data set
i=0

path=os.path.join(dir_names[i],file_names[i])
locs,info=io.load_locs(path)
locs=pd.DataFrame(locs)

######################################## Parameter scan for linking
search_range=[1,3,5,6,7]
memory=[1,2,3]
#### Get scan results    
scan_results=linklocs.scan_sr_mem(locs,
                                  info,
                                  path,
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
        out=linklocs.main(locs,info,path,**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])
        
print()    
print('Failed attempts: %i'%(len(failed_path)))