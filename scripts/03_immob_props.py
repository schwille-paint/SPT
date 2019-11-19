#Script to call spt.immobile_props.main()
import os
import importlib

import picasso_addon.io as addon_io
import spt.immobile_props as improps

importlib.reload(improps)

############################################# Load raw data
dir_names=[]
dir_names.extend(['directory to locs_render_picked file'])

file_names=[]
file_names.extend(['file name'])

############################################ Set non standard parameters 
### Valid for all evaluations
params_all={}
## Exceptions
params_special={}

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