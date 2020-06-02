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
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1']*2)

file_names=[]
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503_tmobprops.hdf5'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503_split50_tmobprops.hdf5'])

############################################ Set parameters for call to spt.immob_props.main() and dataset
params={'filter':'fix'} # Parameters
i=0 # Select dataset
CycleTime=0.2 #Aquisition cycle time [s]

#%%
############################################ Run spt.immob_props.main()               
path=os.path.join(dir_names[i],file_names[i]) # Path
locs,info=addon_io.load_locs(path) # Load data
out=improps.main(locs,info,**params) # Run main

#%%
############################################ Select one group
g=3
picked=locs[locs.group==g]

### Trace
trace=improps.get_trace(picked,info[0]['Frames'])

### gT (individual)
gT=improps.get_NgT(picked,ignore=1) # Values
idx=list(gT.index.values)
T=np.unique([int(i[1:]) for i in idx]) # Index

### gT (ensemble)
gT_mean,T=analyze.get_NgT(out[1])


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

ax.set_xlim(0,120)
ax.set_xticks([0,30,60,90,120])
ax.set_xlabel('Time (s)')

ax.set_ylim(-0.5,7)
ax.set_yticks([0,3,6])
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

# ax.plot(T*CycleTime,
#         gT_mean.loc[idx,'50%'],
#         '--',
#         c='r',
#         lw=2,
#         label='ensemble')

ax.set_xscale('log')
ax.set_xlim(0.2,120)
ax.set_xticks([1,10,100])
ax.set_xlabel('Time (s)')

# ax.set_yscale('log')
ax.set_ylim(0.1,10)
ax.set_ylabel(r'N($\tau\geq T$)')

ax.legend()