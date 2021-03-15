#Script to call spt.immob_props.main() and to checkout single groups
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import picasso_addon.io as addon_io
import spt.immobile_props as improps
import spt.special as special

importlib.reload(improps)
plt.style.use('~/lbFCS/styles/paper.mplstyle')
# plt.style.use(r'C:\Users\flori\Documents\mpi\repos\lbFCS\styles\paper.mplstyle')

############################################# Define data
dir_names=[]
dir_names.extend([r'/fs/pool/pool-schwille-spt/P1_origamiXlink_SPT/Data/21-03-12_id208_id219_T23/id208+id219_R16ntCy3B_312.5pm_P13A488_20nM_p1000uW_488nm_2/21-03-15_FS_test']*2)

file_names=[]
file_names.extend(['id208+id219_R16ntCy3B_312.5pm_P13A488_20nM_p1000uW_488nm_2_MMStack_Pos0.ome_locs_picked0301.hdf5'])
file_names.extend(['id208+id219_R16ntCy3B_312.5pm_P13A488_20nM_p1000uW_488nm_2_MMStack_Pos0.ome_locs_picked0301_tmobprops.hdf5'])

############################################ Set parameters
px=0.13 # px in microns
CycleTime=0.2 
a_iter_convert=(px**2/(4*CycleTime))*100

############################################ Load data              
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))] # Path

locs,info=addon_io.load_locs(path[0]) # Load _picked
props,info_props=addon_io.load_locs(path[1]) # Load _props

#%%
X=locs.copy()
Y=props.copy()
### Unit conversion
Y.a_iter=Y.a_iter*a_iter_convert

############################################ Define criteria for ensemble selection

### Remove 'stuck' particles
D_hp=0
istrue=Y.a_iter > D_hp 

### Duration longer than ...
istrue = (Y.n_locs>=10) & istrue

### Get positives
Y=Y[istrue]
groups=Y.group.unique()
X=X.query('group in @groups')

### Assign diffusion constant to _picked -> assign_field
X=special.assign_prop_to_picked(Y,X,field='a_iter')


#%%
####################################### Plotting
### Render settings
fov=[[0,700],[0,700]]
oversampling=0.5

### Counts number of unique trajectories per sub-pixel as given vy oversampling
### maps[0] = Number of unique trajetories per sub-pixel
### maps[1] = Mean of 
maps=special.get_trackmap(X,fov,oversampling)


#################################### Number of tracks     
f=plt.figure(1,figsize=[5,5])
f.subplots_adjust(bottom=0,top=1,left=0,right=1)
f.clear()
ax=f.add_subplot(111)
mappable=ax.imshow(maps[0],
                   vmin=0,
                   vmax=10,
                   cmap='hot_r',
                   origin='lower',
                   )
ax.set_facecolor('none')
ax.set_xticks([])
ax.set_yticks([])

scalebar=plt.Rectangle([15,15], # position
                       20/(px/oversampling), # 20um lenght
                       10, # height
                       fc='w',
                       ec='k',
                       )
ax.add_patch(scalebar)



f=plt.figure(2,figsize=[1,5])
f.subplots_adjust(bottom=0.1,top=0.9,left=0.1,right=0.3)
f.clear()
ax=f.add_subplot(111)
plt.colorbar(mappable,
             cax=ax,
             )

