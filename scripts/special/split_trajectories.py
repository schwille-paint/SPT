#Script to call picasso_addon.localize.main()
import os
import traceback
import importlib

import picasso_addon.io as addon_io
import spt.special as special

importlib.reload(special)

############################################# Load raw data
dir_names=[]
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1'])
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1'])
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p250uW_T21_1'])

file_names=[]
# file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
# file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p250uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])

############################################ Set maximum length of sub-trajectories
subN=25

#%%
############################################ Main loop                   
failed_path=[]
for i in range(0,len(file_names)):
    ### Create path
    path=os.path.join(dir_names[i],file_names[i])
          
    ### Run main function
    try:
        ### Load
        locs,info=addon_io.load_locs(path) 
        
        ### Split
        locs_split=special.split_trajectories(locs,subN) 
        
        ### Save
        path=os.path.splitext(path)[0]
        info_split=info.copy()+[{'subN':subN}]
        addon_io.save_locs(path+'_split%i.hdf5'%subN,
                           locs_split,
                           info_split,
                           mode='picasso_compatible')
        
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))








