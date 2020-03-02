#Script to call picasso_addon.localize.main()
import os
import importlib

import picasso.io as io
import picasso_addon.localize as localize
import picasso_addon.autopick as autopick
import spt.immobile_props as improps

importlib.reload(localize)
importlib.reload(autopick)
importlib.reload(improps)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/20-03-02_pseries_fix_B23_rep/id140_B_exp200_p038uW_T23_1']*3)

file_names=[]
file_names.extend(['id140_B_exp200_p038uW_T23_1_MMStack_Pos0.ome.tif'])
file_names.extend(['id140_B_exp200_p038uW_T23_1_MMStack_Pos1.ome.tif'])
file_names.extend(['id140_B_exp200_p038uW_T23_1_MMStack_Pos2.ome.tif'])


############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'undrift':False,
            'min_n_locs':5,
            'filter':'fix',
            #'parallel':False,
            }

### Exceptions
params_special={}

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
        ### Localize and undrift
        movie,info=io.load_movie(path)
        out=localize.main(movie,info,**params)
        info=info+[out[0][0]]+[out[1][0]] # Update info to used params
        
        ### Autopick
        print()
        locs=out[1][1]
        out=autopick.main(locs,info,**params)
        info=info+[out[0]] # Update info to used params
        
        ### Immobile kinetics analysis
        print()
        locs=out[2]
        out=improps.main(locs,info,**params)
        info=info+[out[0]] # Update info to used params
        
    except:
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))
