#Script to call spt.immob_props.main() and to checkout single groups
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import picasso.io as io
import picasso.render as render

# plt.style.use('~/lbFCS/styles/paper.mplstyle')
plt.style.use(r'C:\Users\flori\Documents\mpi\repos\lbFCS\styles\paper.mplstyle')
############################################# Define data
dir_names=[]
dir_names.extend([r'C:\Data\p06.SP-tracking\20-01-16_immob-th_pseries_livecell\id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1'])

file_names=[]
file_names.extend(['id169_R1-54#_R1s1-8_40nM_exp200_p114uW_T21_1_MMStack_Pos0.ome_locs_render_picked_query_avg3_render.hdf5'])

############################################ Load data              
path=[os.path.join(dir_names[i],f) for i,f in enumerate(file_names)] # Path
locs,info=io.load_locs(path[0]) # Load _locs

#%%
################ Get rendered image
xy_lim=[349.7,350.3]
oversampling=40
image=render.render(locs,
                    info=None,
                    oversampling=oversampling,
                    viewport=[(xy_lim[0],xy_lim[0]),(xy_lim[1],xy_lim[1])],
                    blur_method=None,
                    min_blur_width=0,
                    )[1]
################ Lineplot histogram through center and fitting
### Get line
xy_c=[np.mean(locs.x),np.mean(locs.y)]
istrue=np.abs(locs.y-xy_c[1])<=1/oversampling
istrue=(np.abs(locs.x-xy_c[0])<=0.3) & istrue
line=locs[istrue].x
line=line-xy_c[0] # Center to zero!
line*=130 # Convert px to nm

### Fit line
bins=50
y,x=np.histogram(line,bins=bins)
x=x[:-1]
x+=(x[1]-x[0])/2

def gauss(x,a,x0,dx): return a*np.exp(-((x-x0)/dx)**2)
p0=[np.max(y),0,20]
popt,pcov=curve_fit(gauss,x,y,p0=p0)
yfit=gauss(x,*popt)

#################################################### Plotting

############### Rendered image
f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.02,top=0.98,left=0,right=0.9)
f.clear()
ax=f.add_subplot(111)
mapp=ax.imshow(image,
               cmap='magma',
               vmin=0,
               vmax=100e3,
               origin='lower'
               )
plt.colorbar(mapp,
             ax=ax,
             shrink=0.97,
             ticks=np.arange(0,1e5+1,2e4).astype(int),
             )

ax.set_xlim(0,22)
ax.set_xticks([])
ax.set_ylim(0.5,22.5)
ax.set_yticks([])

############### Lineplot
f=plt.figure(2,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)
ax.hist(line,
        bins=bins,
        color='tomato',
        edgecolor='k',
        lw=1,
        label='TH',
        )
ax.plot(x,
        yfit,
        '-',
        c='k',
        lw=2,
        label='fit')

ax.legend()
ax.set_xlabel('x [nm]')
ax.set_ylabel(r'Counts x $10^4$')
ax.set_yticklabels((ax.get_yticks()/1e4).astype(int))

