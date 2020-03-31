#Script to call spt.mob_props.main() and to checkout single groups
import os
import importlib
import numpy as np
import time
import matplotlib.pyplot as plt

import picasso_addon.io as addon_io
import spt.mob_props as mobprops
import spt.motion_metrics as metrics

importlib.reload(mobprops)

############################################# Load raw data
dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1'])

file_names=[]
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])


#%%
### Which dataset?
i=0

### Load dataset
path=os.path.join(dir_names[i],file_names[i])
locs,info=addon_io.load_locs(path)

#%%
importlib.reload(mobprops)
importlib.reload(metrics)

### Select group
g=137

g=locs.group.unique()[g]
### Track
df=locs[locs.group==g]

### Get displacement moments
moments=metrics.displacement_moments(df.frame.values,
                                     df.x.values,
                                     df.y.values)

    
########################################## Plotting

########################################## Plot traces
f=plt.figure(num=1,figsize=[4,3])
f.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.95)
f.clear()
ax=f.add_subplot(111)

ax.scatter(df.x,
           df.y,
           c=df.frame,
           s=7,
           cmap='magma',
           marker='o',
           alpha=0.4,
           label=g)
ax.legend()

########################################## Plot MSD and MME
f=plt.figure(num=2,figsize=[4,3])
f.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.95)
f.clear()
ax=f.add_subplot(111)
x=moments[:,0]
y=moments[:,1]
ax.plot(x,
        y,
        '-',
        c='r',
        label='MSD')

y=moments[:,3]
ax.plot(x,
        y,
        '-',
        c='b',
        label='MME')
ax.legend()
