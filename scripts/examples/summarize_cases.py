#################################################### Load packages
import matplotlib.pyplot as plt #plotting
import os #platform independent paths
import pandas as pd 

plt.style.use('~/lbFCS/styles/paper.mplstyle')
############################################################## Parameters

savepath='/fs/pool/pool-schwille-paint/Analysis/p06.SP-tracking/immobile/tracking-handle/analysis/plots'
############################################################## Define data paths
labels=[]
########## Buffer B+ @10mM MgCl2
### exp200
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_B_exp200_p038uW'])
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_B_exp200_p114uW'])
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_B_exp200_p250uW'])

########## Buffer L  @3mM MgCl2 + 150 NaCl
### exp200
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_L_exp200_p038uW'])
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_L_exp200_p114uW'])
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_L_exp200_p250uW'])

########## Buffer L T21 @3mM MgCl2 + 150 NaCl
### exp200
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_L21_exp200_p038uW'])
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_L21_exp200_p114uW'])
labels.extend(['Cy3b_R1-36#:R1s1-8:40nM_L21_exp200_p250uW'])
                            
dir_names=['/fs/pool/pool-schwille-paint/Analysis/p06.SP-tracking/immobile/tracking-handle/z.datalog']*len(labels)
############################################################## Read in data
#### Load & Sorting
path=[os.path.join(dir_names[i],labels[i]+'_stats.h5') for i in range(0,len(labels))]
X=pd.concat([pd.read_hdf(p,key='result') for p in path])

X.sort_index(axis=0,ascending=True, inplace=True)
X.sort_index(axis=1,ascending=True, inplace=True)

#%%
############################################################## Quick selection
field='Tn=1e+00'
istrue=(X.power==38)&(X.exp==400)
Xred=X.loc[istrue,field]
#%%
exp=200
buffers=['B','L','L21']
colors=['r','b','k','magenta']

############################################################## Plotting half
field='half_i3'

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.25,right=0.95)
f.clear()
ax=f.add_subplot(111)

i=0
for b in buffers:
    Xred=X.loc[(X.exp==exp)&(X.buffer==b)]
    
    y=Xred.groupby('power').mean()[field]*exp*1e-3
    yerr=Xred.groupby('power').std()[field]*exp*1e-3
    x=y.index.values
    
    ax.errorbar(x,
                y,
                yerr=yerr,
                fmt='o-',
                color=colors[i],
                label='Buffer '+b,
                )
    i+=1
    
ax.set_xlim(0,300)
ax.set_xlabel('Reflected power (uW)')

ax.set_ylim(1,700)
ax.set_ylabel(r'$t_{1/2}$ of 1st track')

ax.legend()
#plt.savefig(os.path.join(savepath,'halftime_TH_B_Cy3B.pdf'),transparent=True)
#%%
############################################################## Plotting t>=T: Number of tracks 
field='Tn=5e-01'

f=plt.figure(2,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.25,right=0.95)
f.clear()
ax=f.add_subplot(111)

i=0
for b in buffers:
    Xred=X.loc[(X.exp==exp)&(X.buffer==b)]
    
    y=Xred.groupby('power').mean()[field]*exp*1e-3
    yerr=Xred.groupby('power').std()[field]*exp*1e-3
    x=y.index.values
    
    ax.errorbar(x,
                y,
                yerr=yerr,
                fmt='o-',
                color=colors[i],
                label='Buffer '+b,
                )
    i+=1
    
ax.set_xlim(0,300)
ax.set_xlabel('Reflected power (uW)')

ax.set_ylim(0,700)
ax.set_ylabel(r'$t_{min}$ for $\frac{tracks}{particle}=$'+'%.1f'%(float(field[3:])))

ax.legend()
#plt.savefig(os.path.join(savepath,'Tn1_TH_B_Cy3B.pdf'),transparent=True)

#%%
############################################################## Plotting t>=T: Photons/ms
field='Pn=5e-01'

f=plt.figure(3,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.25,right=0.95)
f.clear()
ax=f.add_subplot(111)

i=0
for b in buffers:
    buffer='B'
    Xred=X.loc[(X.exp==exp)&(X.buffer==b)]
    
    y=Xred.groupby('power').mean()[field]/(exp)
    yerr=Xred.groupby('power').std()[field]/(exp)
    x=y.index.values
    
    ax.errorbar(x,
                y,
                yerr=yerr,
                fmt='o-',
                color=colors[i],
                label='Buffer '+b,
                )
    i+=1
    
ax.set_xlim(0,300)
ax.set_xlabel('Reflected power (uW)')


ax.set_ylabel(r'$\frac{photons}{ms}$ for $\frac{tracks}{particle}=$'+'%.1f'%(float(field[3:])))

ax.legend(loc='upper left')
#plt.savefig(os.path.join(savepath,'Pn1_TH_B_Cy3B_normed.pdf'),transparent=True)
    
