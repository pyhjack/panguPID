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
# model_24 = onnx.load('pangu_weather_6.onnx')
model_1 = onnx.load('pangu_weather_1.onnx')
# model_3 = onnx.load('pangu_weather_3.onnx')

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


# ort_session_24 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
ort_session_1 = ort.InferenceSession('pangu_weather_1.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
# ort_session_3 = ort.InferenceSession('pangu_weather_3.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

print('initialize model: %.2f s' % (time.time()-Val))


#%%
# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)


# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
y_location_ori,x_location_ori = function_file.findtyphoneclone_single_MSLP(input_surface[0, 250:400, 500:700]/1e5)

# y = np.linspace(90, -90, 721)
# x = np.linspace(0, 359.75,1440)

#%%

Val = time.time()
    # Run the inference session



# output1, output_surface1 = ort_session_1.run(None, {'input':input, 'input_surface':input_surface}) # 1hour

# output2, output_surface2 = ort_session_1.run(None, {'input':output1, 'input_surface':output_surface1}) # 2hour

# output3, output_surface3 = ort_session_3.run(None, {'input':input, 'input_surface':input_surface}) # 3hour
# np.save( 'output3.npy', output3)
# np.save( 'output_surface3.npy', output_surface3)

# output3 = np.load('output3.npy')
# output_surface3 = np.load('output_surface3.npy')
# output4, output_surface4 = ort_session_1.run(None, {'input':output3, 'input_surface':output_surface3}) # 4hour

# output6, output_surface6 = ort_session_24.run(None, {'input':input, 'input_surface':input_surface}) # 6hour
# np.save( 'output6.npy', output6)
# np.save( 'output_surface6.npy', output_surface6)

output6 = np.load('output6.npy')
output_surface6 = np.load('output_surface6.npy')
output7, output_surface7 = ort_session_1.run(None, {'input':output6, 'input_surface':output_surface6}) # 7hour



# Save the results
print('inference %.2f s' % (time.time()-Val))

    





# %%
list = ['10230100','10230200','10230300','10230400','10230600','10230700']
print(str(list[1])+'.npy')

a = str(list[5])+'.npy'

aa = np.load(a)

v_mea = aa[1, 250:400, 500:700]**2 + aa[2, 250:400, 500:700]**2
plt.figure(dpi=1000)
plt.imshow(aa[3, 250:400, 500:700], cmap='jet')
# plt.clim(1.006, 1.02)
plt.colorbar()
plt.show()


v_pre = output_surface6[1, 250:400, 500:700]**2 + output_surface6[2, 250:400, 500:700]**2
plt.figure(dpi=1000)
plt.imshow(v_pre**(0.5), cmap='jet')
plt.clim(1.006, 1.02)
plt.colorbar()
plt.show()

a = str(list[4])+'.npy'

aa = np.load(a)

error_rate = abs ( output_surface7[0, 250:400, 500:700]/1e5 - aa[0, 250:400, 500:700]/1e5 )
error_rate = error_rate /  (aa[0, 250:400, 500:700]/1e5)
plt.figure(dpi=1000)
plt.imshow(error_rate,cmap='jet')
plt.colorbar()
plt.show()


# %%
list = ['10230100','10230200','10230300','10230400','10230600','10230700']
print(str(list[1])+'.npy')

a = str(list[5])+'.npy'

aa = np.load('input_surface6.npy')

v_mea = aa[1, 250:400, 500:700]**2 + aa[2, 250:400, 500:700]**2
plt.figure(dpi=1000)
plt.imshow(v_mea**(0.5), cmap='jet')
# plt.clim(1.006, 1.02)
plt.colorbar()
plt.show()


v_pre = output_surface6[1, 250:400, 500:700]**2 + output_surface6[2, 250:400, 500:700]**2
plt.figure(dpi=1000)
plt.imshow(v_pre**(0.5), cmap='jet')
# plt.clim(1.006, 1.02)
plt.colorbar()
plt.show()

valueError_1 = (np.mean(v_pre) - np.mean(v_mea) ) / np.mean(v_mea)
print(valueError_1)
# error_rate = abs ( v_pre - v_mea )
# error_rate = error_rate / (v_pre)
# error_rate = error_rate ** (0.5)

# plt.figure(dpi=1000)
# plt.imshow(error_rate, cmap='jet')
# plt.clim(0, 1)
# plt.colorbar()
# plt.show()
# %%
