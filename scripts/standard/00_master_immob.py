#Script to call picasso_addon.localize.main()
import os
import traceback
import importlib
from dask.distributed import Client
import multiprocessing as mp

import picasso.io as io
import picasso_addon.localize as localize
import picasso_addon.autopick as autopick
import spt.immobile_props as improps

importlib.reload(localize)
importlib.reload(autopick)
importlib.reload(improps)

############################################# Load raw data
dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-03-11_pseries_fix_B21_rep\id140_B_exp200_p114uW_T21_1\test'])

file_names=[]
file_names.extend(['id140_B_exp200_p114uW_T21_1_MMStack_Pos0.ome.tif'])


############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'undrift':False,
            'min_n_locs':5,
            'filter':'fix',
            }

### Exceptions
params_special={}

############################################# Start dask parallel computing cluster 
try:
    client = Client('localhost:8787')
    print('Connecting to existing cluster...')
except OSError:
    improps.cluster_setup_howto()

#%%
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
        ### Load movie
        movie,info=io.load_movie(path)
        
        ### Localize and undrift
        out=localize.main(movie,info,path,**params)
        info=info+[out[0]] # Update info to used params
        path=out[-1] # Update path
        
        ### Autopick
        print()
        locs=out[1]
        out=autopick.main(locs,info,path,**params)
        info=info+[out[0]] # Update info to used params
        path=out[-1] # Update path
        
        ### Immobile kinetics analysis
        print()
        locs=out[1]
        out=improps.main(locs,info,path,**params)
        
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))
