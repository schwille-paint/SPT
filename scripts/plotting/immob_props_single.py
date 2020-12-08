#Script to call spt.immob_props.main() and to checkout single groups
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

import picasso_addon.io as addon_io
import spt.immobile_props as improps
import spt.analyze as analyze

importlib.reload(improps)

plt.style.use('~/lbFCS/styles/paper.mplstyle')
############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-spt/Data/D042_TIRFM/20201125_GAB/Trackinghandle_ID183_a20_40nM_Pm2Cy3B_20uW_1']*2)

file_names=[]
file_names.extend(['Trackinghandle_ID183_a20_40nM_Pm2Cy3B_20uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['Trackinghandle_ID183_a20_40nM_Pm2Cy3B_20uW_1_MMStack_Pos0.ome_locs_render_picked_tprops.hdf5'])

############################################ Set parameters
CycleTime=0.2 #Aquisition cycle time [s]

#%%
############################################ Load _picked and _tprops            
paths = [os.path.join(dir_names[i],file_name) for i, file_name in enumerate(file_names)]
locs_init,info=addon_io.load_locs(paths[0]) # Load _picked
props,info=addon_io.load_locs(paths[1]) # Load _tprops

#%%
############################################ Query _picked for groups in _tprops
groups = props.group.values
locs = locs_init.query('group in @groups')

#%%
############################################ Select one group
g=102
picked=locs[locs.group==g]

### Trace
trace=improps.get_trace(picked,info[0]['Frames'])

### gT (individual)
gT=improps.get_NgT(picked,ignore=1) # Values
idx=list(gT.index.values)
T=np.unique([int(i[1:]) for i in idx]) # Index

### gT (ensemble)
gT_mean,T=analyze.get_NgT(props)


############################################ Plotting

### Trace
f=plt.figure(1,figsize=[5,2])
f.subplots_adjust(bottom=0.26,top=0.95,left=0.12,right=0.95)
f.clear()
ax=f.add_subplot(111)
ax.plot(np.arange(0,len(trace),1)*CycleTime,
        trace/100,
        c='b',
        lw=2)

ax.set_xlabel('Time (s)')
ax.set_ylabel('#Photons x100')


### NgT
field='n'
idx=['%s%i'%(field,t) for t in T]

f=plt.figure(2,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)
ax.plot(T*CycleTime,
        gT[idx],
        c='b',
        lw=2,
        label='individual')

ax.plot(T*CycleTime,
        gT_mean.loc[idx,'mean'],
        '--',
        c='k',
        lw=2,
        label='ensemble')


ax.set_xscale('log')
ax.set_xlim(0.2,1e3)
ax.set_xticks([1,10,100])
ax.set_xlabel('Time (s)')

ax.set_yscale('log')
ax.set_ylim(0.1,1e3)
ax.set_ylabel(r'TPP($\tau\geq T$)')

ax.legend()