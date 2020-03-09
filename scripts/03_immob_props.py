#Script to call spt.immobile_props.main()
import os
import importlib
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import spt.immobile_props as improps

importlib.reload(improps)

############################################# Load raw data
dir_names=[]
dir_names.extend(['directory to locs_render_picked file'])

file_names=[]
file_names.extend(['file_name'])


############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'filter':'paint'}
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
        out=improps.main(locs,info,**params)
    except:
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))


#%%

### Query _picked for groups in _props
groups=out[1].index.values
locs_filter=locs.query('group in @groups')
filter_groups=locs_filter.group.unique()
