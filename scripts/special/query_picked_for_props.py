#Script to call spt.immob_props.main() and to checkout single groups
import os
import numpy as np
import picasso_addon.io as addon_io

############################################# Define data
# dir_names=[]
# dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-16_immob-th_pseries_livecell\id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1']*2)

# file_names=[]
# file_names.extend(['id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_render_render_picked_tprops.hdf5'])
# file_names.extend(['id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_render_render_picked.hdf5'])


dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_fix_slb_L_T21\id140_L_exp200_p114uW_T21_1']*2)

file_names=[]
file_names.extend(['id140_L_exp200_p114uW_T21_1_MMStack_Pos2.ome_locs_render_picked_tprops.hdf5'])
file_names.extend(['id140_L_exp200_p114uW_T21_1_MMStack_Pos2.ome_locs_render_picked.hdf5'])

############################################ Load data              
path=[os.path.join(dir_names[i],f) for i,f in enumerate(file_names)] # Path

props,info_props=addon_io.load_locs(path[0]) # Load _props
locs,info=addon_io.load_locs(path[1]) # Load _picked

#%%
X=locs.copy()
Y=props.copy()

##############################################################################
''' Define criteria for ensemble selection based on _props.
    Rewrite this section as needed ...
'''
### Only get center FOV
istrue=np.sqrt((Y.x-350)**2+(Y.y-350)**2)<=200

### Number of localizations threshold
istrue=istrue & (Y.n_locs>=100) & (Y.n_locs<=200) 

##############################################################################
''' Query both _props and _picked for positive groups (istrue!) and save picked
'''
### Query
Y=Y[istrue]
groups=Y.group.unique() # Positives
X=X.query('group in @groups')

### Only first localizations...
istrue=X.frame<=100

#%%
### Save
info_query=info.copy()
addon_io.save_locs(path[1].replace('.hdf5','_query.hdf5'),
                   X,
                   info_query,
                   mode='picasso_compatible')