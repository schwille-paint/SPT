#Script to call autopick main function
import os
import importlib
import picasso_addon.autopick as autopick
import picasso_addon.io as io

importlib.reload(autopick)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-11-12_th_infinity-tracks/id169_R1-54#_R1s1-8_40nM_exp400_p038uW_1/19-11-18_FS'])

file_names=[]
file_names.extend(['id169_R1-54#_R1s1-8_40nM_exp400_p038uW_1_MMStack_Pos0.ome_locs_render.hdf5'])
                   
NewNoFrames=4500                   
#%%
failed_path=[]

for i in range(0,len(file_names)):
    ### Create full path
    path=os.path.join(dir_names[i],file_names[i])
    try:
        ### Load
        locs,info=io.load_locs(path)
        ### Modify
        locs_filter=locs[locs.frame<=NewNoFrames]
        ### Save
        info_filter=info.copy()
        info_filter[0]['Frames']=NewNoFrames
        io.save_locs(path.replace('.hdf5','_filter.hdf5'),
                           locs_filter,
                           info_filter+[{'NewNoFrames':NewNoFrames,'extension':'_locs_render_filter'}],
                           mode='picasso_compatible')
    except:
        failed_path.extend([path])
    
    
print()    
print('Failed attempts: %i'%(len(failed_path)))