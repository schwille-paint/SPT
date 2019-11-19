#Script to call picasso_addon.localize.main()
import os
import importlib
import matplotlib.pyplot as plt

import picasso.io as io
import picasso.localize
import picasso_addon.localize as localize

importlib.reload(localize)
############################################# Load raw data
dir_names=[]
dir_names.extend(['directory to .ome.tif file'])

file_names=[]
file_names.extend(['file name'])

############################################ Set non standard parameters 
### Valid for all evaluations
params_all={}
### Exceptions
params_special={}

#%%
############################################ Main loop                   
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
        movie,info=io.load_movie(path)
        out=localize.main(movie,info,**params)
    except:
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))






#%%
############################################ Checkout single files

### Load movie
i=0
path=os.path.join(dir_names[i],file_names[i])
movie,info=io.load_movie(path)
#%%
### Spot detection
frame=0
params={'mng':800,
        'box':5}

y,x,ng= picasso.localize.identify_in_frame(movie[frame],
                                           params['mng'],
                                           params['box'])
### Preview
f=plt.figure(num=1,figsize=[4,4])
f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
f.clear()
ax=f.add_subplot(111)
ax.imshow(movie[frame],cmap='gray',vmin=40,vmax=500,interpolation='nearest',origin='lower')
ax.scatter(x,y,s=150,marker='o',color='None',edgecolor='r',lw=2)   

ax.grid(False)
ax.set_xlim(200,400)
ax.set_ylim(200,400)