#Script to call spt.mob_props.main()
import os
import traceback
import importlib
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import spt.mob_props as mobprops
import spt.immobile_props as improps

importlib.reload(mobprops)
importlib.reload(improps)

############################################# Load raw data
dir_names=[]
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1'])
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p250uW_T21_1'])

# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p038uW_T21_1'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p038uW_T21_2'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p038uW_T21_3'])

# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p114uW_T21_1'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p114uW_T21_2'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p114uW_T21_3'])

# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p250uW_T21_1'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p250uW_T21_2'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\slb_id140_L_exp200_p250uW_T21_3'])


file_names=[]
# file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503_split50.hdf5'])
# file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p250uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])

# file_names.extend(['slb_id140_L_exp200_p038uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id140_L_exp200_p038uW_T21_2_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id140_L_exp200_p038uW_T21_3_MMStack_Pos0.ome_locs_picked0503.hdf5'])

# file_names.extend(['slb_id140_L_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id140_L_exp200_p114uW_T21_2_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id140_L_exp200_p114uW_T21_3_MMStack_Pos0.ome_locs_picked0503.hdf5'])

# file_names.extend(['slb_id140_L_exp200_p250uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id140_L_exp200_p250uW_T21_2_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id140_L_exp200_p250uW_T21_3_MMStack_Pos0.ome_locs_picked0503.hdf5'])

############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'parallel':False}
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
        out=mobprops.main(locs,info,path,**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])
        
print()    
print('Failed attempts: %i'%(len(failed_path)))