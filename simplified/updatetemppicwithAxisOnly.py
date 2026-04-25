import os
import numpy as np
import onnx
import onnxruntime as ort
# from onnxcustom.utils.onnx_split import split_onnx
import time
from matplotlib import pyplot as plt
from matplotlib import cm
import function_file
import math

import cdsapi # type: ignore
import numpy as np
import netCDF4 as nc # type: ignore
import os
from datetime import datetime

## this is from the detect data
surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
with nc.Dataset( './sourcedata/2018-10-23-06.nc') as nc_file:
    surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
    surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
    surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
    surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
# input = np.load(os.path.join('.\input_data\store\input_surface.npy')).astype(np.float32)
input = surface_data
x1 = range(0,101,20)
y1 = range(0,81,20)
xx1 = []
xx2 = []
for i in range(0,9):
    temp1 = 137.5 + i*2.5
    temp2 = (90-72.5) - i*2.5
    if temp1<180:
        xx1.append(str(temp1)+'E')
    else:
        xx1.append(str(360-temp1)+'W')
    if temp2>0:
        xx2.append(str(temp2)+'N')
    else:
        xx2.append(str(-temp2)+'S')

plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=10,weight='bold')
plt.figure(dpi=600)
plt.imshow(input[3,290:370, 550:650], cmap='jet')
plt.clim(298.0, 304.0)
hhhh = plt.colorbar(orientation = 'vertical')
plt.xticks(x1,xx1[0:6])
plt.yticks(y1,xx2[0:5])
# hhhh.set_ticklabels(barbar)
hhhh.set_label('unit(K)',fontsize=10,weight='bold')
plt.grid()

# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 


## this is for forcase data
input1 = np.load(os.path.join('output_surface6.npy')).astype(np.float32)
# input = np.load(os.path.join('.\input_data\store\input_surface.npy')).astype(np.float32)
plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=10,weight='bold')
plt.figure(dpi=600)
plt.imshow(input1[3,290:370, 550:650], cmap='jet')
plt.clim(298.0, 304.0)

hhhh = plt.colorbar(orientation = 'vertical')
hhhh.set_label('unit(K)',fontsize=10,weight='bold')
plt.xticks(x1,xx1[0:6])
plt.yticks(y1,xx2[0:5])
# hhhh.set_ticklabels(barbar)
plt.grid()

plt.show() 

data = abs(input-input1)/abs(input)

plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=10,weight='bold')
plt.figure(dpi=600)

plt.imshow(data[3,290:370, 550:650], cmap='jet')
hhhh = plt.colorbar(orientation = 'vertical')
plt.xticks(x1,xx1[0:6])
plt.yticks(y1,xx2[0:5])

plt.grid()

# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 
