#Script to call spt.mob_props.main()
import os
import importlib

import picasso_addon.io as addon_io

import spt.mob_props as mobprops
import spt.immobile_props as immobprops

importlib.reload(mobprops)
importlib.reload(immobprops)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_fix_slb_L_T21/slb_id140_L_exp200_p250uW_T21_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_fix_slb_L_T21/slb_id140_L_exp200_p250uW_T21_2'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-17_fix_slb_L_T21/slb_id140_L_exp200_p250uW_T21_3'])

file_names=[]
file_names.extend(['slb_id140_L_exp200_p250uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
file_names.extend(['slb_id140_L_exp200_p250uW_T21_2_MMStack_Pos0.ome_locs_picked0503.hdf5'])
file_names.extend(['slb_id140_L_exp200_p250uW_T21_3_MMStack_Pos0.ome_locs_picked0503.hdf5'])
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
        out=mobprops.main(locs,info,**params)
    except:
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))