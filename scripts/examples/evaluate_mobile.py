import sys
import os
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import picasso_addon.io as io

import spt.analyze as analyze
import spt.immobile_props as immobprops

importlib.reload(analyze)
importlib.reload(immobprops)
# plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
############################################################## Parameters
savedir='/fs/pool/pool-schwille-paint/Analysis/p06.SP-tracking/mobile/z.plots'
savename=os.path.splitext(os.path.basename(sys.argv[0]))[0]

############################################################## Define data
dir_names=[]
dir_names.extend([r'C:\Users\flori\Documents\data\SPT\mobile\th\L21_exp200_p038uW'])

file_names=[]
file_names.extend([r'slb_id169_R1-54#_R1s1-8_40nM_exp200_p038uW_T21_1_MMStack_Pos0.ome_locs_picked0503_tmobprops.hdf5'])

############################################################## Read in data
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]
labels=[i for i in range(0,len(path))]
#### Read in locs
locs_props=pd.concat([io.load_locs(p)[0] for p in path],keys=labels,names=['rep'])
infos=[io.load_locs(p)[1] for p in path]

# CycleTime=float(savename.split('_')[3][3:])*1e-3
CycleTime=0.2
px=0.13 # px size in microns

############################################################## Inspect dataset
rep=0

### No. of tracks per frame and fit
n_tracks=analyze.tracks_per_frame(locs_props.loc[(rep,slice(None)),:],
                                  infos[rep])
popt,frames,n_tracks_fit=analyze.fit_tracks_per_frame(n_tracks)
#%%
### Filter 
X=locs_props.loc[(rep,slice(None)),:].copy()
istrue=(X.sx**2+X.sy**2)>(np.median(X.lpx*2+X.lpy**2)*10) # Spatial filter
X=X[istrue]
istrue=X.n_locs>50 # Track length filter
X=X[istrue]

### Tracks longer than T properties
gT,Ts=immobprops.tracks_greaterT(X.len,
                                 X.min_frame,
                                 X.photons,
                                 X.std_photons)


############################################################# Plotting
f=plt.figure(15,figsize=[8,9])
f.subplots_adjust(bottom=0.1,top=0.98,left=0.1,right=0.97,hspace=0.4,wspace=0.3)
f.clear()

############################################################# No. tracks per frame
ax=f.add_subplot(321)
ax.plot(frames,
        n_tracks,
        '-',
        c='gray',
        label='data')
ax.plot(frames,
        n_tracks_fit,
        '-',
        c='k',
        lw=2,
        label='fit')
ax.legend()
ax.set_xlabel('Frame')
ax.set_ylabel('# Tracks per frame')

############################################################# No. tracks greater than T
ax=f.add_subplot(322)
idx=['n%i'%(T) for T in Ts]
y=gT[idx]/(popt[0]+popt[2])
Tcrit=Ts[y>=0.5][-1]*CycleTime
ax.plot(Ts*CycleTime,
        y,
        c='k',
        lw=2)
ax.axvline(Tcrit,ls='--',lw=1)
ax.set_xscale('log')
ax.set_xlim(1,5e2)
ax.set_xlabel(r'T [s]')
ax.set_yscale('log')
ax.set_ylim(1e-1,1e2)
ax.set_ylabel(r'N (length$\geq$T)')

############################################################# Photons of tracks greater than T
ax=f.add_subplot(323)
idx=['p%i'%(T) for T in Ts]
y=gT[idx]/(CycleTime*1e3)
ax.plot(Ts*CycleTime,
        y,
        c='k',
        lw=2)
ax.axvline(Tcrit,ls='--',lw=1)
ax.set_xscale('log')
ax.set_xlim(1,5e2)
ax.set_xlabel(r'T [s]')
ax.set_ylim(1,30)
ax.set_ylabel(r'Photons (length$\geq$T) [1/ms]')

############################################################# MSD ratio
ax=f.add_subplot(324)
ax.hist(X.meanmoment_ratio,
        bins=np.linspace(0,3,50),
        color='gray',
        edgecolor='k')
ax.set_xlabel('MSD moment ratio')
ax.set_ylabel('# Tracks')

############################################################# Anomaloues MSD diffusion mode
ax=f.add_subplot(325)
ax.hist(X.b_mme_anom,
        bins=np.linspace(0,2,25),
        color='gray',
        edgecolor='k')
ax.set_xlabel('Diffusion mode')
ax.set_ylabel('# Tracks')

############################################################# Diffusion constant)
ax=f.add_subplot(326)
ax.hist(X.a_iter*(px**2/(4*CycleTime)),
        bins=np.linspace(0,0.15,50),
        color='gray',
        edgecolor='k')
ax.set_xlabel(r'Diffusion constant $[\mu m^2/s]$')
ax.set_ylabel('# Tracks')

# plt.savefig(os.path.join(savedir,savename+'_rep%i.pdf'%(rep)),transparent=True)