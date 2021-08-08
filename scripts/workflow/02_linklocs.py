import os
import pandas as pd
import importlib
import traceback

import picasso.io as io
import spt.linklocs as linklocs

importlib.reload(linklocs)

######################################## Define data (locs)
dir_names=[]
file_names=[]

### Old file
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_th_slb_L_T21/slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1/21-03-15_FS_test'])
# file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs.hdf5'])

### Jan's file
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs.hdf5'])

#%%
######################################## Read in single data set
i=0

path=os.path.join(dir_names[i],file_names[i])
locs,info=io.load_locs(path)
locs=pd.DataFrame(locs)

######################################## Parameter scan for linking
search_range=[2,3,5]
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
params={'search_range':3,
        'memory':1,
        }


############################################ Main loop
failed_path=[]
for i in range(0,len(file_names)):
    ### Create path
    path=os.path.join(dir_names[i],file_names[i])
    
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