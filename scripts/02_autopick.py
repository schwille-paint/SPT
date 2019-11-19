#Script to call autopick main function
import os
import importlib
import matplotlib.pyplot as plt

import picasso.io as io
import picasso.render as render
import picasso_addon.autopick as autopick

importlib.reload(autopick)

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p06.SP-tracking/19-11-12_th_infinity-tracks/id169_R1-54#_R1s1-8_40nM_exp400_p038uW_1/19-11-18_FS_30min'])

file_names=[]
file_names.extend(['id169_R1-54#_R1s1-8_40nM_exp400_p038uW_1_MMStack_Pos0.ome_locs_render_filter.hdf5'])

############################################ Set non standard parameters 
### Valid for all evaluations
params_all={}
## Exceptions
params_special={}

#%%                   
failed_path=[]
for i in range(0,len(file_names)):
    ### Create path
    path=os.path.join(dir_names[i],file_names[i])
    ### Set paramters for each run
    params=params_all.copy()
    for key, value in params_special.items():
        params[key]=value[i]
    ### Run main function
    try:
        locs,info=io.load_locs(path)
        out=autopick.main(locs,info,**params)
    except:
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))





#%%
############################################ Checkout single files
i=0

params={'oversampling':5,
        'pick_box':11, # Usually 2*oversampling+1
        'min_n_locs':100,
        'fit_center':False,
        'pick_diameter':2,
        }

### Load file
path=os.path.join(dir_names[i],file_names[i])
locs,info=io.load_locs(path)
### Render
image=render.render(locs,
                    info,
                    params['oversampling'],
                    )[1] 
### Pick detection
centers=autopick.spotcenters_in_image(image,
                                      params['pick_box'],
                                      params['min_n_locs'],
                                      params['fit_center'])[0]
#%%
### Preview
f=plt.figure(num=1,figsize=[4,4])
f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
f.clear()
ax=f.add_subplot(111)

ax.imshow(image,cmap='gray',vmin=0,vmax=100,interpolation='nearest',origin='lower')
for i in range(0,len(centers)):
    circle=plt.Circle((centers.loc[i,'x'],centers.loc[i,'y']),
               params['pick_diameter']/2*params['oversampling'],
               facecolor='None',
               edgecolor='y',
               lw=2)
    ax.add_artist(circle)

ax.grid(False)
