#Script to generate time segments 
import os
import importlib
import picasso_addon.autopick as autopick
import picasso_addon.io as io

importlib.reload(autopick)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-01-30_POC_scans/id169_BPOC_R1S1-8-40nM-exp200_p114uW_T21_1'])

file_names=[]
file_names.extend(['id169_BPOC_R1S1-8-40nM-exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_render_picked_filter.hdf5'])
                   
NewFirstFrame_list=[i*1500 for i in range(8,10)]
NewNoFrames_list=[4500]*2
#%%
failed_path=[]

for i in range(len(file_names)):
    ### Create full path
    path=os.path.join(dir_names[i],file_names[i])
    
    for j in range(len(NewFirstFrame_list)):
        NewFirstFrame=NewFirstFrame_list[j]
        NewNoFrames=NewNoFrames_list[j]
        try:
            ### Load
            locs,info=io.load_locs(path)
            
            ### Modify
            istrue=locs.frame>=NewFirstFrame
            istrue=istrue & (locs.frame<(NewFirstFrame+NewNoFrames))
            locs_filter=locs[istrue]
            locs_filter.frame=locs_filter.frame-NewFirstFrame # Set first frame to zero
            
            ### Save
            info_filter=info.copy()
            info_filter[0]['Frames']=NewNoFrames
            extension='_locs_render_picked_filter_f%i-%i'%(NewFirstFrame,NewFirstFrame+NewNoFrames)
            io.save_locs(path.replace('.hdf5',extension[26:]+'.hdf5'),
                         locs_filter,
                         info_filter+[{'NewNoFrames':NewNoFrames,
                                       'NewFirstFrame':NewFirstFrame,
                                       'extension':extension}],
                         mode='picasso_compatible')
        except:
            failed_path.extend([path])
    
    
print()    
print('Failed attempts: %i'%(len(failed_path)))