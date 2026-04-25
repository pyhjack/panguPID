#%%
import os
import numpy as np
import onnx
import onnxruntime as ort
import time
from matplotlib import pyplot as plt
from matplotlib import cm

Val = time.time()
# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
model_24 = onnx.load('pangu_weather_24.onnx')

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
ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
print('initialize model: %.2f s' % (time.time()-Val))


#%%
# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)


# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

# y = np.linspace(90, -90, 721)
# x = np.linspace(0, 359.75,1440)

for i in range(40):
    Val = time.time()
    # Run the inference session
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
# Save the results
    print('inference %.2f s' % (time.time()-Val))

    np.save(os.path.join(output_data_dir, 'output_upper'+str(i)), output)
    np.save(os.path.join(output_data_dir, 'output_surface'+str(i)), output_surface)
    input, input_surface = output, output_surface

    plt.figure(dpi=300)
    plt.imshow(input_surface[0, 50:400, 300:700]/1e5, cmap='jet')
    # 
    plt.clim(0.95, 1.05)
    plt.colorbar(orientation = 'horizontal')
    plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    plt.show()
    



input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)    



# %%
