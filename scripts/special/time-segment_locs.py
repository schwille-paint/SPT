#Script to generate time segments 
import os
import picasso_addon.io as io

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/03_w6-500pM_Cy3B-C-c1500_561-p40uW-s50_Pos1_1/mng1x'])

file_names=[]
file_names.extend(['03_w6-500pM_Cy3B-C-c1500_561-p40uW-s50_Pos1_1_MMStack_Pos0.ome_locs_render.hdf5'])
                   
NewFirstFrame_list=[i*1500 for i in range(1,2)]
NewNoFrames_list=[4500]*1
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
            extension='_f%i-%i'%(NewFirstFrame,NewFirstFrame+NewNoFrames)
            io.save_locs(path.replace('.hdf5',extension+'.hdf5'),
                         locs_filter,
                         info_filter+[{'NewNoFrames':NewNoFrames,
                                       'NewFirstFrame':NewFirstFrame,
                                       'extension':extension}],
                         mode='picasso_compatible')
        except:
            failed_path.extend([path])
    
    
print()    
print('Failed attempts: %i'%(len(failed_path)))