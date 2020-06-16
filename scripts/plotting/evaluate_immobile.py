import sys
import os
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#### Load own packages
import picasso_addon.io as io
import spt.analyze as analyze
import spt.analytic_expressions as anexpress
importlib.reload(analyze)
# plt.style.use('~/lbFCS/styles/paper.mplstyle')

############################################################## Parameters
save_results=True

savedir='/fs/pool/pool-schwille-paint/Analysis/p06.SP-tracking/immobile/tracking-handle/z.datalog'
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
X=locs_props.copy()

############################################################## Result prep
Y=analyze.get_result(X,savename)
#### Save
if save_results:
    Y.to_hdf(os.path.join(savedir,savename+'_stats.h5'),key='result')

CycleTime=float(savename.split('_')[3][3:])*1e-3

#%%
############################################################## Plotting

############################################################## 
''' 1st event length distribution
'''

ignore=1
field='Tstart-i%i'%(ignore)

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)
for l in labels:
    ecdf=anexpress.ecdf(X.loc[(l,slice(None)),field])
    ax.plot(ecdf[0]*CycleTime,ecdf[1],'-',lw=2,label=l)
    print('Rep%i, half-time of 1st event: %.1f'%(l,Y.loc[l,'half_i%i'%(ignore)]*CycleTime))

ax.axhline(0.5,ls='--',c='k')   
ax.set_xlim(0.2,1000)
ax.set_xlabel('T (s)')
ax.set_xscale('log')

ax.set_ylim(0,1.3)
ax.set_ylabel(r'P[1st event $\geq$ T] (%)')
ax.legend()    

############################################################## 
''' Number of tracks per particle with duration >= T
'''

field='n'
f=plt.figure(2,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)

for l in labels:
    NgT,Ts=analyze.get_NgT(X.loc[(l,slice(None)),:])
    
    idx=['%s%i'%(field,T) for T in Ts]
    y=NgT.loc[idx,'mean']
    ax.plot(Ts*CycleTime,y,'-',lw=2,label=l)
    print('Rep%i, 0.5 tracks per particle with duration >= %.1f s'%(l,Y.loc[l,'Tn=5e-01']*CycleTime))
   
ax.axhline(0.5,ls='--',c='k')
ax.set_xlim(0.2,1000)
ax.set_xlabel(r'T [s]')
ax.set_xscale('log')

ax.set_ylim(0.1,1000)
ax.set_yscale('log')
ax.set_ylabel(r'# Tracks per particle $\geq$ T')
ax.legend()
    
############################################################## 
''' Photons of tracks with duration >= T
'''

field='p'
f=plt.figure(3,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)

for l in labels:
    NgT,Ts=analyze.get_NgT(X.loc[(l,slice(None)),:])
    
    idx=['%s%i'%(field,T) for T in Ts]
    y=NgT.loc[idx,'50%']
    ax.plot(Ts*CycleTime,y,'-',lw=2,label=l)
    print('Rep%i, Tracks with T >= %.1f s have >= %.1f photons'%(l,Y.loc[l,'Tn=5e-01']*CycleTime,Y.loc[l,'Pn=5e-01']))

ax.axvline(Y.loc[l,'Tn=5e-01']*CycleTime,ls='--',c='k')
ax.set_xlim(0.2,1000)
ax.set_xlabel(r'T [s]')
ax.set_xscale('log')

ax.set_ylim(100,10000)
ax.set_yscale('log')
ax.set_ylabel(r'# Photons for tracks $\geq$ T')
ax.legend(loc='upper left')