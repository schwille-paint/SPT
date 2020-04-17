#Script to call spt.mob_props.main() and to checkout single groups
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import picasso_addon.io as addon_io
import spt.mob_props as mobprops
import spt.motion_metrics as metrics
import spt.analytic_expressions as express

importlib.reload(mobprops)
importlib.reload(metrics)
importlib.reload(express)

############################################################## Parameters
savedir=r'C:\Users\flori\ownCloud\Paper\p06.SPT\figures\main\figure4\v01\plots\partial-subdiffusive-tracks\p114uW'
savefig=False

CycleTime=0.2#float(savename.split('_')[3][3:])*1e-3 # Aquisition cycle time [s]
px=0.13 # px size [microns]
# Unit conversion factor from px^2/frame to 1e-2 microns^2/s for diffusion constant
a_iter_convert=(px**2/(4*CycleTime))*100 

############################################# Load raw data
dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-17_th_slb_L_T21\slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1']*4)

file_names=[]
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503.hdf5'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503_split50.hdf5'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503_tmobprops.hdf5'])
file_names.extend(['slb_id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_picked0503_split50_tmobprops.hdf5'])


### Load datasets
path=[os.path.join(dir_names[i],file_names[i]) for i in range(len(file_names))]

locs,inf=addon_io.load_locs(path[0]) # Full_picked
locs_sub,info_sub=addon_io.load_locs(path[1]) # Sub _picked
props,info_props=addon_io.load_locs(path[2]) # Full _props
props_sub,info_props_sub=addon_io.load_locs(path[3]) # Sub _props

#%%
############################################################## Prepare
X=locs.copy()
Xsub=locs_sub.copy()
Y=props.copy()
Ysub=props_sub.copy()

### Unit conversion
Y.a_iter=Y.a_iter*a_iter_convert
Ysub.a_iter=Ysub.a_iter*a_iter_convert

############################################################## General filter and reduce _props and sub _props to positives

### Remove 'stuck' particles, assuming that diffusion is only governed by localization precision
LocPrec=(Y.lpx.mean()+Y.lpy.mean())/2
Dcrit=((LocPrec**2*px**2)/CycleTime)*100
istrue=Y.a_iter>=Dcrit
Y=Y[istrue]

### Only select trajectories longer than ...
Tcrit=120 # in seconds
Y=Y[Y.len>=Tcrit/CycleTime]

### Now reduce sub _props to remaining supgroups
supgroups=Y.group.unique()
Ysub=Ysub.query('supgroup in @supgroups')
print(len(supgroups))
############################################################## Search for groups with interesting properties

### Search within remaining groups for trajectories with low diffusive sub-track
supgroups=Ysub.loc[Ysub.a_iter<2*Dcrit,'supgroup'].unique()
print(len(supgroups))

#%%
importlib.reload(mobprops)
############################################################## Get one interesting group and extract
g=54
g=supgroups[g]

# for g in supgroups:
### Get corresponding full and sub _props and _picked
gY=Y[Y.group==g]
gYsub=Ysub[Ysub.supgroup==g]
gX=X[X.group==g] 
gXsub=Xsub[Xsub.supgroup==g]

### Sub-tracks diffusion constant for every localization
Dsub=gYsub.a_iter.values 
Nsub=gYsub.n_locs.values
D_locs=np.concatenate([np.ones(int(Nsub[i]))*Dsub[i] for i in range(len(Dsub))])

### Get moments of full track
moments=metrics.displacement_moments(gX.frame.values,
                                     gX.x.values,
                                     gX.y.values)
fits=mobprops.getfit_moments(gX)
x_fit=moments[:,0]
y_fit=express.msd_free(x_fit,fits.a_iter,fits.b_iter)

############################################################### Plotting
f=plt.figure(num=1,figsize=[8,9])
f.subplots_adjust(left=0.12,right=0.97,bottom=0.06,top=0.975)
f.clear()
gs=gridspec.GridSpec(3,2,figure=f,hspace=0.3,wspace=0.3)

################################ Plot trajectory solor coded with sub-track diffusion
ax=f.add_subplot(gs[0:2,:])
mappable=ax.scatter(gXsub.x,
                    gXsub.y,
                    c=D_locs,
                    s=100,
                    cmap='magma',
                    vmin=0,
                    vmax=12,
                    marker='o',
                    edgecolor='k',
                    )
ax.plot(gX.x,
        gX.y,
        '-',
        c='k',
        lw=0.5,
        alpha=0.7,
        )
ax.text(gX.x.iloc[0],
        gX.y.iloc[0],
        'Start',
        c='w',
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(facecolor='k',edgecolor='w', alpha=0.5),
        )
ax.text(gX.x.iloc[-1],
        gX.y.iloc[-1],
        'End',
        c='w',
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(facecolor='k',edgecolor='w', alpha=0.5),
        )
plt.colorbar(mappable,
              ax=ax,
              fraction=0.1,
              shrink=1,
              ticks=np.arange(0,13,2)
              )
ax.set_xlabel('x-position [px]')
ax.set_xticklabels(ax.get_xticks()-ax.get_xticks()[1])
ax.set_ylabel('y-position [px]')
ax.set_yticklabels(ax.get_yticks()-ax.get_yticks()[1])

################################ Plot sub-track diffusion constant vs time
ax=f.add_subplot(gs[2,0])
mappable=ax.scatter((gYsub.min_frame-gYsub.min_frame.iloc[0])*CycleTime,
                    gYsub.a_iter,
                    c=gYsub.a_iter,
                    s=150,
                    cmap='magma',
                    vmin=0,
                    vmax=12,
                    marker='o',
                    edgecolor='k',
                    )
ax.plot((gYsub.min_frame-gYsub.min_frame.iloc[0])*CycleTime,
        gYsub.a_iter,
        '-',
        c='k')

ax.axhline(gY.a_iter.values,
           ls='--',
           color='r',
           lw=3)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Diffusion constant $[10^{-2} \mu m^2/s]$')

################################ Plot MSD
ax=f.add_subplot(gs[2,1])

ax.plot(moments[:,0]*CycleTime,
        moments[:,1],
        '-',
        c='gray',
        lw=3,
        label='data',
        )
ax.plot(x_fit*CycleTime,
        y_fit,
        '--',
        c='r',
        lw=3,
        label='fit',
        )

ax.legend(loc='lower right')
ax.set_xlabel('Lag-time [s]')
ax.set_ylabel(r'MSD [$px^2$]')
ax.set_ylim(0,1.1*max(moments[:,1]))

if savefig: plt.savefig(os.path.join(savedir,'%i.pdf'%g),transparent=True)