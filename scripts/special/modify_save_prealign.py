#Script to shift second render to first render for pre-alignment if automatic alignment does not work
import os
import matplotlib.pyplot as plt

import picasso.render as render
import picasso.io as io



############################################# Load data
### Define data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p15.RiboCounting/19-11-22/04_FOV1_Cy3B_p038uW_R1s1-8-100pM_POC_1/19-11-22_FS'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p15.RiboCounting/19-11-22/09_FOV1_Cy3B_p038uW_R1s1-8-050pM_POC_1/19-11-22_FS'])

file_names=[]
file_names.extend(['04_FOV1_Cy3B_p038uW_R1s1-8-100pM_POC_1_MMStack_Pos0.ome_locs_render.hdf5'])
file_names.extend(['09_FOV1_Cy3B_p038uW_R1s1-8-050pM_POC_1_MMStack_Pos0.ome_locs_render.hdf5'])
                                    
### Create paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(len(file_names))]

### Load data
locs_ref,info_ref=io.load_locs(path[0])       
locs,info=io.load_locs(path[1])

#%%
############################################# Shift
### Define shift
dx=21
dy=-4
oversampling=1
c_min=0
c_max=500
x_center=300
x_width=100
y_center=300
y_width=100

locs_shift=locs.copy()
xlim=info[0]['Width']
ylim=info[0]['Height']

### Shift locs
locs_shift.x=locs_shift.x+dx
locs_shift.y=locs_shift.y+dy

### Adjust locss to FOV
inFOV=(locs_shift.x>0)&(locs_shift.x<xlim)&(locs_shift.y>0)&(locs_shift.y<ylim)
locs_shift=locs_shift[inFOV]

### Show
image_ref=render.render(locs_ref,
                        info,
                        oversampling,
                        )[1]
image_shift=render.render(locs_shift,
                          info,
                          oversampling,
                          )[1]

f=plt.figure(num=1,figsize=[5,8])
f.subplots_adjust(bottom=0.1,top=1.,left=0.1,right=1.)
f.clear()
### Images
ax_ref=f.add_subplot(211)
ax_ref.imshow(image_ref,cmap='magma',vmin=c_min,vmax=c_max,interpolation='nearest',origin='lower')
ax_shift=f.add_subplot(212)
ax_shift.imshow(image_shift,cmap='magma',vmin=c_min,vmax=c_max,interpolation='nearest',origin='lower')
### Reference lines
ax_ref.axvline(x_center)
ax_ref.axhline(y_center)
ax_shift.axvline(x_center)
ax_shift.axhline(y_center)
### Limits
ax_ref.set_xlim(x_center-x_width/2,x_center+x_width/2)
ax_ref.set_ylim(y_center-y_width/2,y_center+y_width/2)
ax_shift.set_xlim(x_center-x_width/2,x_center+x_width/2)
ax_shift.set_ylim(y_center-y_width/2,y_center+y_width/2)

#%%
############################################# Save
info_shift=info.copy()
addDict={'reference':path[0],
         'dx':dx,
         'dy':dy,
         'extension':'_locs_render'}
io.save_locs(path[1].replace('.hdf5','_prealign.hdf5'),
                   locs_shift,
                   info_shift+[addDict],
                   )
