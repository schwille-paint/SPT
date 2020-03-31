#Script to call spt.immobile_props.main()
import os
import traceback
import importlib
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import spt.immobile_props as improps

importlib.reload(improps)

############################################# Load raw data
dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-16_immob-th_pseries_livecell\id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1'])

file_names=[]
file_names.extend(['id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])


############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'filter':'none',
            }

## Exceptions
params_special={}

############################################# Start dask parallel computing cluster 
try:
    client = Client('localhost:8787')
    print('Connecting to existing cluster...')
except OSError:
    improps.cluster_setup_howto()
    
#%%                   
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
        locs,info=addon_io.load_locs(path)
        out=improps.main(locs,info,path,**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))

