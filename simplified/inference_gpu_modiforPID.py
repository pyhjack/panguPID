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

x1 = range(0,1441,120)
y1 = range(0,721,120)
# barV =[0.98,0.985,0.99,0.995,1.00,1.005,1.01,1.015,1.02]
# barbar =[]
# for i in range(0,9):
#     barbar.append(str(barV[i]) + 'bar')



xx = ['','30E','60E','90E','120E','150E','180W','150E','120E','90W','60W','30W','']
yy = ['','60N','30N','0','30S','60S','']
plt.figure(figsize=(5,3))
plt.rc('font',family='Times New Roman',size=11,weight='bold')
plt.figure(dpi=600)
# plt.imshow(input_surface[0,:,:]/1e5, cmap='jet')
# plt.clim(0.98, 1.02)
plt.imshow(input[3,0,:,:], cmap='jet')
hhhh = plt.colorbar(orientation = 'horizontal')
plt.xticks(x1,xx)
plt.yticks(y1,yy)
# hhhh.set_ticklabels(barbar)
plt.grid()

# plt.savefig('./output_data/vis/' + str(i) + '.jpg')
plt.show() 

 # %% define the taming function
def taming_on_U10(x):
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
    for j in range(5):
        for k in range(20):
            input_surface = function_file.fluct_on_U10(input_surface,330 - k ,595 - j,x)
            input = function_file.fluct_on_U10_upper(input,330  -k ,595 - j,x)
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    plt.figure(dpi=300)
    output_surface = function_file.fluct_on_Pressure(output_surface,250+y_location ,500+ x_location)
    plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    plt.clim(0.98, 1.02)
    plt.show() 
    plt.imshow(output_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    plt.clim(0.98, 1.02)
    # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    plt.show() 
    dx = x_location - x_location_ori
    dy = y_location - y_location_ori
    cosalpha = (dx) / (math.sqrt(dx*dx + dy*dy))
    aplha = 180 * math.acos(cosalpha)/math.pi
    print('taming typhonn direction %.2f' %aplha)
    print(' x direction : %3d ,y direction : %3d' %( x_location - x_location_ori , y_location - y_location_ori))
    return aplha

def taming_on_U10_for_typoon_pressure(x):
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
    for j in range(5):
        for k in range(20):
            input_surface = function_file.fluct_on_U10(input_surface,330 - k ,595 - j,x)
            input = function_file.fluct_on_U10_upper(input,330  -k ,595 - j,x)
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    pressure_low = output_surface[0, 250 + y_location, 500 + x_location]/1e5
    plt.figure(dpi=300)
    output_surface = function_file.fluct_on_Pressure(output_surface,250+y_location ,500+ x_location)
    plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    plt.clim(0.98, 1.02)
    plt.show() 
    plt.imshow(output_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    plt.clim(0.98, 1.02)
    # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    plt.show() 
    
    print('taming typhoon  %.5f and difference is %.5f' %(pressure_low, pressure_original-pressure_low))
    return pressure_low    

def taming_on_Temp(x):
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
    for j in range(5):
        for k in range(20):
            input_surface = function_file.fluct_on_Temperature(input_surface,330 - k ,595 - j,x)
            input = function_file.fluct_on_Temperature1(input,330  -k ,595 - j,x)
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    plt.figure(dpi=300)
    output_surface = function_file.fluct_on_Pressure(output_surface,250+y_location ,500+ x_location)
    plt.imshow(input_surface[3, 250:400, 500:700], cmap='jet')
    plt.colorbar()
    plt.show() 
    plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    plt.clim(0.98, 1.02)
    plt.show() 
    plt.imshow(output_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    plt.clim(0.98, 1.02)
    # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    plt.show() 
    dx = x_location - x_location_ori
    dy = y_location - y_location_ori
    cosalpha = (dx) / (math.sqrt(dx*dx + dy*dy))
    aplha = 180 * math.acos(cosalpha)/math.pi
    print('taming typhonn direction %.2f' %aplha)
    print(' x direction : %3d ,y direction : %3d' %( x_location - x_location_ori , y_location - y_location_ori))
    return aplha
# %% taming range from 116.57 - 180 and let the target angle to be 175
target = 150
kp = 1.5
ki = 0.0001
kd = 1
modified = 1
modified_store = []
result = []
error_int = 0
error_previous = 0
for index in range(30):
    aplha = taming_on_U10(-1 * modified)
    result.append(aplha)
    error = target - aplha
    p_out = kp * error
    error_int += error
    i_out = ki * error_int
    d_out = kd * (error - error_previous)
    modified += (p_out + i_out + d_out)
    print(modified)
    error_previous = error
    modified_store.append(modified)

plt.plot(result)
# %%
aplha = taming_on_U10_for_typoon_pressure(-50)
# %%
# %% taming range from 1.002 - 1.00484 and let the central pressure to be 1.004
target = 0.978
kp = 10000
ki = 100
kd = 10
modified =0
modified_store = []
result = []
error_int = 0
error_previous = 0
for index in range(30):
    aplha = taming_on_U10_for_typoon_pressure(-1 * modified)
    print(aplha)
    result.append(aplha)
    error = target - aplha
    p_out = kp * error
    error_int += error
    i_out = ki * error_int
    d_out = kd * (error - error_previous)
    modified += (p_out + i_out + d_out)
    print(modified)
    error_previous = error
    modified_store.append(modified)

plt.plot(result)
# %%

aplha = taming_on_Temp(-5)
# %%
