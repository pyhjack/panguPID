# panguPID
The Pangu Weather Model combined with PID control, used for a simple test of artificial typhoons modification
# Project Introduction
This is a project based on the Pangu meteorological large model combined with PID control, mainly used for SISO control testing after typhoon modeling.
Project Structure
# Main Files
tkintertest_GUI.py - Main GUI Interface Program
inference_gpu_modiforPID.py - GPU Inference Module (PID Optimized Version)
inference_gpu.py - GPU Inference Module
inference_cpu.py - CPU Inference Module
calculate_series.py - Sequence Calculation Module
function_file.py - Function file
recheckfunctioncode.py - Function Code Inspection
# Data Processing
drawpicturenewaxis.py - Drawing function (new coordinate axis)
processtime06data.py - Time Data Processing
updatedrawOnly.py - Image update
updatetemppicwithAxisOnly.py - Temperature Chart Update
updatevelocitypicwithAxisOnly.py - Speed Chart Update
# Profile
pangu_weather.yaml - Weather Model Configuration File
tkintertest_GUI.spec - PyInstaller Packaging Configuration
# Data Catalog
input_data/ - Input data
output_data/ - Output data
sourcedata/ - Source data
# Environmental Requirements
Python 3.8+
Essential scientific computing libraries: numpy, matplotlib, tkinter, etc
GPU support (optional, for accelerating inference)
# How to Use
Run GUI
python tkintertest_GUI.py
Or directly run the executable file (if it has been packaged)
./tkintertest_GUI.exe
Data preprocessing
python processtime06data.py
# Reasoning computation
python inference_gpu_modifor PID.py  # GPU version
python inference_cpu.py             # CPU version
# Data File Description
Main Data Files
*.npy - NumPy Data Files (Meteorological Data)
*.nc - NetCDF meteorological data file
*.png - Result Chart
# Data file size
Due to the large size of the data files (each .npy file is approximately 16MB), these files have been excluded via .gitignore. It is recommended to：
1.Managing large files with Git LFS
2.or store data files separately
3.Or use an external storage service
# Project Features
1.PID Control Optimization - Optimization of PID Algorithm for Meteorological Data
2.GUI interface - A user interface based on Tkinter
3.Multi-platform support - Supports CPU and GPU inference
4.Data Visualization - Rich chart output
# Precautions
1.Large files (e.g., .npy, .nc, .exe) are not included in the Git repository
2.Dependencies need to be configured according to the actual environment
3.The GPU version requires CUDA environment support
# License
MIT License
# Contact Information
Project Maintainer: Peng Yuehua

