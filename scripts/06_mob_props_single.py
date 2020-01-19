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
dir_names.extend(['//fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-12-16_SLB_newstock_fix_vs_th/s2_LC+Str+40nM-R1S1-8_exp100_p114uW_1/19-12-18_FS'])

file_names=[]
file_names.extend(['s2_LC+Str+40nM-R1S1-8_exp100_p114uW_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])


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
g=102

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

########################################## Plot MSD and MME ratios
f=plt.figure(num=3,figsize=[4,3])
f.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.95)
f.clear()
ax=f.add_subplot(111)
y=moments[:,2]/moments[:,1]**2
ax.plot(x,
        y,
        '-',
        c='r',
        label='MSD ratio')
ax.axhline(np.mean(y),c='r')

y=moments[:,4]/moments[:,3]**2
ax.plot(x,
        y,
        '-',
        c='b',
        label='MME ratio')
ax.axhline(np.mean(y),c='b')
ax.legend()