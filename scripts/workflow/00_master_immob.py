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
dir_names.extend([r'C:\Users\flori\Downloads\picasso_tutorial\picasso_tutorial'])

file_names=[]
file_names.extend(['All_concatenated_cent256.tif'])


############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'undrift':True,
            'lbfcs':True,
            'filter':'none',
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
        info=info+[out[0][0]]+[out[0][1]] # Update info to used params 
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
