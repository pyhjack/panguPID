#%%
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


## download data from internet 
# c = cdsapi.Client()

# # The date and time of the initial field
# date_time = datetime(
#     year=2018, 
#     month=10, 
#     day=23,
#     hour=6,
#     minute=0)


# # The variables required
# surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
# upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

# # Area to download
# area = [90, 0, -90, 359.75]

# # Pressure levels required
# pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']


# # Download the surface data
# c.retrieve('reanalysis-era5-single-levels', {
# 		'product_type': 'reanalysis',
# 		'format': 'netcdf',
# 		'variable': surface_variables,
# 		'date': date_time.strftime("%Y-%m-%d"),
# 		'time': date_time.strftime("%H:%M"),
# 		'area': area,
# 	},  'surf1809300.nc' )

# c.retrieve('reanalysis-era5-pressure-levels', {
#     'product_type': 'reanalysis',
#     'format': 'netcdf',
#     'variable': upper_variables,
#     'pressure_level': pressure_levels,
#     'date': date_time.strftime("%Y-%m-%d"),
#     'time': date_time.strftime("%H:%M"),
#     'area': area,
# },  'upper1809300.nc')
                                           
# # Convert the surface data to npy          
# surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
# with nc.Dataset( './draw/surf1809300.nc') as nc_file:
#     surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
#     surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
#     surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
#     surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)

# np.save( './draw/input_surf1809300.npy', surface_data)

# upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
# with nc.Dataset('./draw/upper1809300.nc') as nc_file:
#     upper_data[0] = (nc_file.variables['z'][:]).astype(np.float32)
#     upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
#     upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
#     upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
#     upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
# np.save('./draw/input_upper1809300.npy', upper_data)

Val = time.time()
# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
model_24 = onnx.load('pangu_weather_6.onnx')

print('read model: %.2f s' % (time.time()-Val))
Val = time.time()

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 0

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested'}

# Initialize onnxruntime session for Pangu-Weather Models


ort_session_24 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
print('initialize model: %.2f s' % (time.time()-Val))


#%%
# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)


# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
y_location_ori,x_location_ori = function_file.findtyphoneclone_single_MSLP(input_surface[0, 250:400, 500:700]/1e5)

# y = np.linspace(90, -90, 721)
# x = np.linspace(0, 359.75,1440)

Val = time.time()
    # Run the inference session
output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
# Save the results
print('inference %.2f s' % (time.time()-Val))

    

# np.save(os.path.join(output_data_dir, 'output_upper'+str(i)), output)
# np.save(os.path.join(output_data_dir, 'output_surface'+str(i)), output_surface)
y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    # plt.figure(dpi=300)
    # plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # # 
    # plt.clim(0.98, 1.02)
    # plt.colorbar(orientation = 'horizontal')
    # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    #plt.show() 
print('original x direction change is %3d ,original  y direction change is %3d' %( x_location - x_location_ori , y_location - y_location_ori))
dx = x_location - x_location_ori
dy = y_location - y_location_ori
cosalpha = (dx) / (math.sqrt(dx*dx + dy*dy))
aplha = 180 * math.acos(cosalpha)/math.pi
pressure_original = output_surface[0, 250+y_location, 500+x_location]/1e5
print('original typhonn direction %.2f' %aplha)
print('typoon original lowest pressure %.3f' %pressure_original)


#%% update the figure with the axis lable

input = np.load(os.path.join('10230100.npy')).astype(np.float32)


x1 = range(0,1441,120)
y1 = range(0,721,120)
# barV =[0.98,0.985,0.99,0.995,1.00,1.005,1.01,1.015,1.02]
# barbar =[]
# for i in range(0,9):
#     barbar.append(str(barV[i]) + 'bar')



xx = ['','30E','60E','90E','120E','150E','180','150E','120E','90W','60W','30W','']
yy = ['','60N','30N','0','30S','60S','']
plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=11,weight='bold')
plt.figure(dpi=600)
# plt.imshow(input_surface[0,:,:]/1e5, cmap='jet')
# plt.clim(0.98, 1.02)
plt.imshow(input_surface[0,:,:]/1e5, cmap='jet')
hhhh = plt.colorbar(orientation = 'horizontal')
plt.xticks(x1,xx)
plt.yticks(y1,yy)
# hhhh.set_ticklabels(barbar)
plt.grid()

hhhh.set_label('unit(km)',fontsize=11,weight='bold')
# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 

# %%
input = np.load(os.path.join('10230600.npy')).astype(np.float32)


x1 = range(0,201,25)
y1 = range(0,150,20)
# barV =[0.98,0.985,0.99,0.995,1.00,1.005,1.01,1.015,1.02]
# barbar =[]
xx1 = []
xx2 = []
for i in range(0,9):
    temp1 = 125 + i*6.25
    temp2 = (90-62.5) - i*5
    if temp1<180:
        xx1.append(str(temp1)+'E')
    else:
        xx1.append(str(360-temp1)+'W')
    if temp2>0:
        xx2.append(str(temp2)+'N')
    else:
        xx2.append(str(-temp2)+'S')
print(xx1)

# yy = ['','60N','30N','0','30S','60S','']
plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=10,weight='bold')
plt.figure(dpi=600)
# plt.imshow(input_surface[0,:,:]/1e5, cmap='jet')
# plt.clim(0.98, 1.02)
plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
hhhh = plt.colorbar(orientation = 'vertical')
plt.xticks(x1,xx1)
plt.yticks(y1,xx2[0:8])
# hhhh.set_ticklabels(barbar)
plt.grid()

hhhh.set_label('unit(bar)',fontsize=10,weight='bold')
# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 
# %%
input = np.load(os.path.join('10230600.npy')).astype(np.float32)


x1 = range(0,81,10)
y1 = range(0,81,10)
xx1 = []
xx2 = []
for i in range(0,9):
    temp1 = 137.5 + i*2.5
    temp2 = (90-70) - i*2.5
    xx1.append(str(temp1)+'E')
    if temp2 !=0 :
        xx2.append(str(temp2)+'S')
    else: 
        xx2.append('0')

# yy = ['','60N','30N','0','30S','60S','']
plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=11,weight='bold')
plt.figure(dpi=600)
# plt.imshow(input_surface[0,:,:]/1e5, cmap='jet')

plt.imshow(input_surface[0, 280:360, 550:630]/1e5, cmap='jet')
plt.clim(1.006, 1.02)
hhhh = plt.colorbar(orientation = 'vertical')
plt.xticks(x1,xx1[0:8])
plt.yticks(y1,xx2[0:6])
# hhhh.set_ticklabels(barbar)
plt.grid()

hhhh.set_label('unit(bar)',fontsize=11,weight='bold')
# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 


# %%
input = np.load(os.path.join('.\input_data\store\input_upper.npy')).astype(np.float32)

input_surface = np.load(os.path.join('.\input_data\store\input_surface.npy')).astype(np.float32)
output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
x1 = range(0,81,10)
y1 = range(0,81,10)
xx1 = []
xx2 = []
for i in range(0,9):
    temp1 = 137.5 + i*2.5
    temp2 = (90-70) - i*2.5
    xx1.append(str(temp1)+'E')
    if temp2 !=0 :
        xx2.append(str(temp2)+'S')
    else: 
        xx2.append('0')
print(xx1)
plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=10,weight='bold')
plt.figure(dpi=600)
plt.imshow(output_surface[0, 280:360, 550:630]/1e5, cmap='jet')
plt.clim(1.006, 1.02)
hhhh = plt.colorbar(orientation = 'vertical')
plt.xticks(x1,xx1)
plt.yticks(y1,xx2)
# hhhh.set_ticklabels(barbar)
plt.grid()
hhhh.set_label('unit(bar)',fontsize=10,weight='bold')
# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 
# %%
