import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn= r'C:\Users\flori\ownCloud\Paper\p06.SPT\figures\main\figure4\v01\plots\diffusion-maps\video-test'

fps = 1

### Get list of all .png files in directory
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files=[f for f in files if os.path.splitext(f)[-1]=='.png']

### Proper sorting of files
def sort_files(x): return int(os.path.splitext(x)[0].split('_')[-1])
files.sort(key = sort_files)

### Read in files and append to frame_array
frame_array = []
for f in files:
    img = cv2.imread(join(pathIn,f))
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)

#%%
pathOut = join(pathIn,'video.mp4')
### Create video
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MPJG'), fps, size)
for f in frame_array:
    out.write(f)
out.release()
