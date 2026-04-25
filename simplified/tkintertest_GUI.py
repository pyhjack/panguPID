#%%
import tkinter
import sys
sys.path.append("C:\\Users\\ADMIN\\desktop\\simplifiedcode>")
import tkinter.messagebox
import tkinter.ttk as ttk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import onnx
import onnxruntime as ort
import os
import function_file
import math
import matplotlib.colors
import time
import netCDF4 as nc
from threading import Thread
from PIL import Image , ImageTk

import matplotlib.cm as cm
Val = time.time()

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

# Val = time.time()
#     # Run the inference session
# output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
# # Save the results
# print('inference %.2f s' % (time.time()-Val))

    

# # np.save(os.path.join(output_data_dir, 'output_upper'+str(i)), output)
# # np.save(os.path.join(output_data_dir, 'output_surface'+str(i)), output_surface)
# y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
#     # plt.figure(dpi=300)
#     # plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
#     # # 
#     # plt.clim(0.98, 1.02)
#     # plt.colorbar(orientation = 'horizontal')
#     # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
#     #plt.show() 
# print('original x direction change is %3d ,original  y direction change is %3d' %( x_location - x_location_ori , y_location - y_location_ori))
# dx = x_location - x_location_ori
# dy = y_location - y_location_ori
# cosalpha = (dx) / (math.sqrt(dx*dx + dy*dy))
# aplha = 180 * math.acos(cosalpha)/math.pi
# pressure_original = output_surface[0, 250+y_location, 500+x_location]/1e5
# print('original typhonn direction %.2f' %aplha)-
# print('typoon original lowest pressure %.3f' %pressure_original)


# y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
#     # plt.figure(dpi=300)
#     # plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
#     # # 
#     # plt.clim(0.98, 1.02)
#     # plt.colorbar(orientation = 'horizontal')
#     # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
#     #plt.show() 
# print('original x direction change is %3d ,original  y direction change is %3d' %( x_location - x_location_ori , y_location - y_location_ori))
# dx = x_location - x_location_ori
# dy = y_location - y_location_ori
# cosalpha = (dx) / (math.sqrt(dx*dx + dy*dy))
# aplha = 180 * math.acos(cosalpha)/math.pi
# pressure_original = output_surface[0, 250+y_location, 500+x_location]/1e5
# print('original typhonn direction %.2f' %aplha)
# print('typoon original lowest pressure %.3f' %pressure_original)


 # %% define the taming function
def taming_on_U10(x):
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
    for j in range(10):
        for k in range(30):
            input_surface = function_file.fluct_on_U10(input_surface,330 - k ,595 - j,x)
            input = function_file.fluct_on_U10_upper(input,330  -k ,595 - j,x)
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    # plt.figure(dpi=300)
    output_surface = function_file.fluct_on_Pressure(output_surface,250+y_location ,500+ x_location)
    # plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # plt.clim(0.98, 1.02)
    # plt.show() 
    # plt.imshow(output_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # plt.clim(0.98, 1.02)
    # # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    # plt.show() 
    dx = x_location - x_location_ori
    dy = y_location - y_location_ori
    cosalpha = (dx) / (math.sqrt(dx*dx + dy*dy))
    aplha = 180 * math.acos(cosalpha)/math.pi
    # print('taming typhonn direction %.2f' %aplha)
    # print(' x direction : %3d ,y direction : %3d' %( x_location - x_location_ori , y_location - y_location_ori))
    return aplha

def taming_on_U10_for_typoon_pressure(x):
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
    for j in range(10):
        for k in range(30):
            input_surface = function_file.fluct_on_U10(input_surface,330 - k ,595 - j,x)
            input = function_file.fluct_on_U10_upper(input,330  -k ,595 - j,x)
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    pressure_low = output_surface[0, 250 + y_location, 500 + x_location]/1e5
    # plt.figure(dpi=300)
    output_surface = function_file.fluct_on_Pressure(output_surface,250+y_location ,500+ x_location)
    # plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # plt.clim(0.98, 1.02)
    # plt.show() 
    # plt.imshow(output_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # plt.clim(0.98, 1.02)
    # # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    # plt.show() 
    
    # print('taming typhoon  %.5f and difference is %.5f' %(pressure_low, pressure_original-pressure_low))
    return pressure_low    

def taming_on_Temp(x):
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
    for j in range(10):
        for k in range(30):
            input_surface = function_file.fluct_on_Temperature(input_surface,330 - k ,595 - j,x)
            input = function_file.fluct_on_Temperature1(input,330  -k ,595 - j,x)
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    y_location,x_location = function_file.findtyphoneclone_single_MSLP(output_surface[0, 250:400, 500:700]/1e5)
    # plt.figure(dpi=300)
    pressure_low = output_surface[0, 250 + y_location, 500 + x_location]/1e5
    output_surface = function_file.fluct_on_Pressure(output_surface,250+y_location ,500+ x_location)
    # plt.imshow(input_surface[3, 250:400, 500:700], cmap='jet')
    # plt.colorbar()
    # plt.show() 
    # plt.imshow(input_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # plt.clim(0.98, 1.02)
    # plt.show() 
    # plt.imshow(output_surface[0, 250:400, 500:700]/1e5, cmap='jet')
    # plt.clim(0.98, 1.02)
    # # plt.savefig('./output_data/vis/' + str(i) + '.jpg')
    # plt.show() 
    # aplha = 180 * math.acos(cosalpha)/math.pi
    # print('taming typhonn direction %.2f' %aplha)
    # print(' x direction : %3d ,y direction : %3d' %( x_location - x_location_ori , y_location - y_location_ori))
    return pressure_low



def run_simu():

    target = float(entry6.get())
    kp = float(wp.get())
    ki = float(wi.get())
    kd = float(wd.get())
    iter_time = int(entry5.get())
    modified = 1
    modified_store = []
    result = []
    error_int = 0
    error_previous = 0
    for index in range(iter_time):
        if combo.get()==values[0]:
            aplha = taming_on_U10(-1 * modified)
        elif combo.get()==values[1]:
            aplha = taming_on_U10_for_typoon_pressure(-1 * modified)
        elif combo.get()==values[2]:
            aplha = taming_on_Temp(-1 * modified)
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
        fig2.clear()
        ax = fig2.add_subplot()
        ax.plot(result)    
        canvas2.draw()
        labelbtn.config(text=f"仿真进行中... ({index+1}/{iter_time})")    
    labelbtn.config(text="仿真已完成")  

background_thread = None
def calculate_background():
    global background_thread
    if background_thread is None or not background_thread.is_alive():
        background_thread = Thread(target=run_simu)
        background_thread.start()





    # ax = fig.add_subplot()
    # ax.plot(np.arange(100),np.arange(100)**3)


def subwindows():
    
    # def changepic():
        
        # ax.set_xticklabels(xx1[0:6])
    # sub = tkinter.Tk()
    second_windows = tkinter.Tk()
    second_windows.title("误差分析")

   

    labelxl = tkinter.Label(second_windows,text='选取范围X最小值')
    labelxh = tkinter.Label(second_windows,text='选取范围X最大值')
    labelyl = tkinter.Label(second_windows,text='选取范围Y最小值')
    labelyh = tkinter.Label(second_windows,text='选取范围Y最大值')

        
    label = tkinter.Label(second_windows,text="check")
    labeltop = tkinter.Label(second_windows,text="误差种类")
    labelbotm = tkinter.Label(second_windows,text="时间分析")
    labelsource = tkinter.Label(second_windows,text='实际测量值')
    labelpredict = tkinter.Label(second_windows,text='预测物理值')
    valuesana = ['风速误差分析','气压值误差','气温值误差']
    combo = ttk.Combobox(second_windows,values=valuesana,font=("黑体",12))
    values1time = ['6小时']
    combo1 = ttk.Combobox(second_windows,values=values1time,font=("黑体",12))
    drawsource = ttk.Frame(second_windows)   # real figure used in 1,1
    drawpredict = ttk.Frame(second_windows)
    figtop = Figure(figsize=(5,3),dpi=100)
    canvastop = FigureCanvasTkAgg(figtop,drawsource)

    wxl = tkinter.Scale(second_windows,orient="horizontal",length=500,from_=0,to=1440,resolution=2,tickinterval=200,activebackground="red")
    wxh = tkinter.Scale(second_windows,orient="horizontal",length=500,from_=0,to=1440,resolution=2,tickinterval=200,activebackground="green")
    wyl = tkinter.Scale(second_windows,orient="horizontal",length=500,from_=0,to=721,resolution=1,tickinterval=200,activebackground="blue")
    wyh = tkinter.Scale(second_windows,orient="horizontal",length=500,from_=0,to=721,resolution=1,tickinterval=200,activebackground="yellow")
    combo.current(1)
    combo1.current(0)
    figbotm = Figure(figsize=(5,3),dpi=100)
    canvasbotm = FigureCanvasTkAgg(figbotm,drawpredict)
    canvastop.draw()
    canvasbotm.draw()
    canvastop.get_tk_widget().pack()
    canvasbotm.get_tk_widget().pack()



    def changemod():
        print('button functional')
        xlow = int(wxl.get())
        xhigh = int(wxh.get())
        ylow = int(wyl.get())
        yhigh = int(wyh.get())
        x1 = range(0,xhigh-xlow+1,20)
        y1 = range(0,yhigh-ylow+1,20)
        xx1 = []
        xx2 = []
        a = np.maximum(len(x1),len(y1))
        for i in range(0,a):
            temp1 = 137.5 + i*2.5
            temp2 = (90-72.5) - i*2.5
            if temp1<180:
                xx1.append(str(str(temp1)+"E"))
            else:
                xx1.append(str(str(360-temp1)+"W"))
            if temp2>0:
                xx2.append(str(str(temp2)+"N"))
            else:
                xx2.append(str(str(-temp2)+"S"))
        
        if combo.get() == valuesana[1]:
            data1 = input[0,:,:]/1e5
            data2 = input1[0,:,:]/1e5
        elif combo.get() == valuesana[2]:
            data1 = input[3,:,:]
            data2 = input1[3,:,:]
        elif combo.get() == valuesana[0]:
            data1 = (input[1,:,:]**2+input[2,:,:]**2)**0.5
            data2 = (input1[1,:,:]**2+input1[2,:,:]**2)**0.5

        figtop.clear()
        ax = figtop.add_subplot()
        figbotm.clear()
        bx = figbotm.add_subplot()

        hldbar = ax.imshow(data1[ylow:yhigh, xlow:xhigh], cmap='jet')
        ax.set_xticks(x1)
        ax.set_xticklabels(xx1[0:len(x1)])
        ax.set_yticks(y1)
        ax.set_yticklabels(xx2[0:len(y1)])

        max1 = np.amax(data1[ylow:yhigh, xlow:xhigh])
        min1 = np.amin(data1[ylow:yhigh, xlow:xhigh])
        norm1 = matplotlib.colors.Normalize(vmin=min1,vmax=max1)
        

        holdbar1 = bx.imshow(data2[ylow:yhigh, xlow:xhigh], cmap='jet')
        bx.set_xticks(x1)
        bx.set_xticklabels(xx1[0:len(x1)])
        bx.set_yticks(y1)
        bx.set_yticklabels(xx2[0:len(y1)])
        max2 = np.amax(data2[ylow:yhigh, xlow:xhigh])
        min2 = np.amin(data2[ylow:yhigh, xlow:xhigh])

        minmax = np.maximum(min1,min2)
        maxmin = np.minimum(max1,max2)
        norm1 = matplotlib.colors.Normalize(vmin=minmax,vmax=maxmin)
        figtop.colorbar(cm.ScalarMappable(norm=norm1,cmap='jet'),ax=ax)
        figbotm.colorbar(cm.ScalarMappable(norm=norm1,cmap='jet'),ax=bx)

        hldbar.set_clim(minmax,maxmin)
        holdbar1.set_clim(minmax,maxmin)

        canvastop.draw()
        canvasbotm.draw()
        canvastop.get_tk_widget().pack()
        canvasbotm.get_tk_widget().pack()
    
    btnchangemode = tkinter.Button(second_windows,command=changemod,text='显示',bg='cyan',font=("黑体",20))

    def errorwindows():

        third_windows = tkinter.Tk()
        third_windows.title("误差定量分析")

        drawerror = ttk.Frame(third_windows)
        figerror = Figure(figsize=(5,3),dpi=100)
        canvaerror = FigureCanvasTkAgg(figerror,drawerror)
        canvaerror.draw()
        canvaerror.get_tk_widget().pack()


        ax = figerror.add_subplot()

        print('button functional')
        xlow = int(wxl.get())
        xhigh = int(wxh.get())
        ylow = int(wyl.get())
        yhigh = int(wyh.get())
        x1 = range(0,xhigh-xlow+1,20)
        y1 = range(0,yhigh-ylow+1,20)
        xx1 = []
        xx2 = []
        a = np.maximum(len(x1),len(y1))
        for i in range(0,a):
            temp1 = 137.5 + i*2.5
            temp2 = (90-72.5) - i*2.5
            if temp1<180:
                xx1.append(str(str(temp1)+"E"))
            else:
                xx1.append(str(str(360-temp1)+"W"))
            if temp2>0:
                xx2.append(str(str(temp2)+"N"))
            else:
                xx2.append(str(str(-temp2)+"S"))
        
        def Normalprocess():
            drawerror.pack()
            dataerror = abs(data2-data1)/abs(data1)
            holdbar1 = ax.imshow(dataerror[ylow:yhigh, xlow:xhigh], cmap='jet')
            ax.set_xticks(x1)
            ax.set_xticklabels(xx1[0:len(x1)])
            ax.set_yticks(y1)
            ax.set_yticklabels(xx2[0:len(y1)])
            max1 = np.amax(dataerror[ylow:yhigh, xlow:xhigh])
            min1 = np.amin(dataerror[ylow:yhigh, xlow:xhigh])
            norm1 = matplotlib.colors.Normalize(vmin=min1,vmax=max1)
            figerror.colorbar(cm.ScalarMappable(norm=norm1,cmap='jet'),ax=ax)
        
        if combo.get() == valuesana[1]:
            data1 = input[0,:,:]/1e5
            data2 = input1[0,:,:]/1e5
            Normalprocess()
        elif combo.get() == valuesana[2]:
            data1 = input[3,:,:]
            data2 = input1[3,:,:]
            Normalprocess()
        elif combo.get() == valuesana[0]:
            data1 = (input[1,290:370,550:650]**2+input[2,290:370,550:650]**2)**0.5
            data2 = (input1[1,290:370,550:650]**2+input1[2,290:370,550:650]**2)**0.5
            relative_error=100*(np.mean(data1)-np.mean(data2))/np.mean(data1)
            tkinter.messagebox.showinfo("Alter","由于计算的是风速，需使用台风周围速度平均值计算，结算相对误差为为"+str(relative_error)+"%")
            third_windows.destroy()
        
       

        third_windows.mainloop()

    btnshowerrorvalue = tkinter.Button(second_windows,command=errorwindows,text='误差具体值',bg='cyan',font=("黑体",20))
    ## build in grid form
    label.grid(row=1,column=2)
    labeltop.grid(row=2,column=0)
    combo.grid(row=2,column=1)
    labelbotm.grid(row=3,column=0)
    combo1.grid(row=3,column=1)
    labelsource.grid(row=1,column=3)
    drawsource.grid(row=2,column=3,rowspan=3)
    labelpredict.grid(row=5,column=3)
    drawpredict.grid(row=6,column=3,rowspan=3)
    

    btnshowerrorvalue.grid(row=3,column=2)
    btnchangemode.grid(row=2,column=2)
    labelxl.grid(row=4,column=0)
    wxl.grid(row=4,column=1,columnspan=2)
    labelxh.grid(row=5,column=0)
    wxh.grid(row=5,column=1,columnspan=2)
    labelyl.grid(row=6,column=0)
    wyl.grid(row=6,column=1,columnspan=2)
    labelyh.grid(row=7,column=0)
    wyh.grid(row=7,column=1,columnspan=2)

    wxl.set(550)
    wxh.set(650)
    wyl.set(290)
    wyh.set(370)

    # initial boundary
    # xlow = 550
    # xhigh = 650
    # ylow = 290
    # yhigh = 370

    xlow = int(wxl.get())
    xhigh = int(wxh.get())
    ylow = int(wyl.get())
    yhigh = int(wyh.get())

    figtop.clear()
    ax = figtop.add_subplot()
    figbotm.clear()
    bx = figbotm.add_subplot()

    input = np.zeros((4, 721, 1440), dtype=np.float32)
    with nc.Dataset( './sourcedata/2018-10-23-06.nc') as nc_file:
        input[0] = nc_file.variables['msl'][:].astype(np.float32)
        input[1] = nc_file.variables['u10'][:].astype(np.float32)
        input[2] = nc_file.variables['v10'][:].astype(np.float32)
        input[3] = nc_file.variables['t2m'][:].astype(np.float32)

    x1 = range(0,xhigh-xlow+1,20)
    y1 = range(0,yhigh-ylow+1,20)
    xx1 = []
    xx2 = []
    a = np.maximum(len(x1),len(y1))
    for i in range(0,a):
        temp1 = 137.5 + i*2.5
        temp2 = (90-72.5) - i*2.5
        if temp1<180:
            xx1.append(str(str(temp1)+"E"))
        else:
            xx1.append(str(str(360-temp1)+"W"))
        if temp2>0:
            xx2.append(str(str(temp2)+"N"))
        else:
            xx2.append(str(str(-temp2)+"S"))
    hldbar = ax.imshow(input[0,ylow:yhigh, xlow:xhigh]/1e5, cmap='jet')
    hldbar.set_clim(1.006,1.02)
    # hldbar.colorbar.orientation
    ax.set_xticks(x1)
    ax.set_xticklabels(xx1[0:len(x1)])
    ax.set_yticks(y1)
    ax.set_yticklabels(xx2[0:len(y1)])


    input1 = np.load(os.path.join('output_surface6.npy')).astype(np.float32)
    hldbar1 = bx.imshow(input1[0,ylow:yhigh, xlow:xhigh]/1e5, cmap='jet')
    hldbar1.set_clim(1.006,1.02)
    bx.set_xticks(x1)
    bx.set_xticklabels(xx1[0:len(x1)])
    bx.set_yticks(y1)
    bx.set_yticklabels(xx2[0:len(y1)])




    second_windows.mainloop()

        

top = tkinter.Tk()  # this is the mainwindows

######################## define objects in the windows

draw1 = ttk.Frame(top)   # real figure used in 1,1
draw2 = ttk.Frame(top)   # plot figure used in 1,10

labellogo= tkinter.Label(top)   # figure at 5,1

label1 = tkinter.Label(top,text="年",width=3,font=("黑体",12))
label2 = tkinter.Label(top,text="月",width=3,font=("黑体",12))
label3 = tkinter.Label(top,text="日",width=3,font=("黑体",12))
label4 = tkinter.Label(top,text="时",width=3,font=("黑体",12))
label5 = tkinter.Label(top,text="循环",width=4,font=("黑体",12))
label6 = tkinter.Label(top,text="次",width=3,font=("黑体",12))
label7 = tkinter.Label(top,text="目标值",width=5,font=("黑体",12))



labelp = tkinter.Label(top,text="P(比例系数)",bg='red',width=12,font=("黑体",12),fg='white')
labeli = tkinter.Label(top,text="I(积分系数)",bg='green',width=12,font=("黑体",12),fg='white')
labeld = tkinter.Label(top,text="D(差分系数)",bg='blue',width=12,font=("黑体",12),fg='white')

labelli = tkinter.Label(top,text="台风周围压力分布图",width=30,font=("黑体",10))
labelri = tkinter.Label(top,text="系统输出随迭代次数变化趋势",width=30,font=("黑体",10))

entry1 = tkinter.Entry(top,width=10)
entry2 = tkinter.Entry(top,width=10)
entry3 = tkinter.Entry(top,width=10)
entry4 = tkinter.Entry(top,width=10)
entry5 = tkinter.Entry(top,width=10)
entry6 = tkinter.Entry(top,width=10)

btn1 = tkinter.Button(top,text="点击进行PID仿真",command=calculate_background,bg='cyan',font=("黑体",20))
labelbtn = tkinter.Label(top,text='待进行仿真',bg='orchid',font=("黑体",20))
btn2 = tkinter.Button(top,text='误差分析',command=subwindows,bg='cyan',font=("黑体",20))

wp = tkinter.Scale(top,orient="horizontal",length=500,from_=0,to=10000,resolution=0.0001,tickinterval=2000,activebackground="red")
wi = tkinter.Scale(top,orient="horizontal",length=500,from_=0,to=10000,resolution=0.0001,tickinterval=2000,activebackground="green")
wd = tkinter.Scale(top,orient="horizontal",length=500,from_=0,to=10000,resolution=0.1,tickinterval=2000,activebackground="blue")





labelopt = tkinter.Label(top,text="选择算例",width=10,font=("黑体",12))
values = ['速度扰动控制方向','速度扰动控制强度','温度扰动控制强度']
combo = ttk.Combobox(top,values=values,font=("黑体",12))
combo.current(1) # default use 速度对强度




label8 = tkinter.Label(top,text="系统输入:速度扰动值",width=30,height=5,font=("黑体",15,"bold"))
label9 = tkinter.Label(top,text="系统输出:台风最低压值",width=30,height=5,font=("黑体",15,"bold"))


####################### place the object in the windows using grid()

labelli.grid(row=0,column=1)
labelri.grid(row=0,column=10)


draw1.grid(row=1,column=1,rowspan=3)
draw2.grid(row=1,column=10,rowspan=3)
labellogo.grid(row=7,column=10,sticky="E")

######## first row input
entry1.grid(row=1,column=2)
entry2.grid(row=1,column=4)
entry3.grid(row=1,column=6)
entry4.grid(row=1,column=8)

label1.grid(row=1,column=3)
label2.grid(row=1,column=5)
label3.grid(row=1,column=7)
label4.grid(row=1,column=9)


############ second row input
label5.grid(row=2,column=2,columnspan=4)
entry5.grid(row=2,column=6,columnspan=3)
label6.grid(row=2,column=8)

############## third row input

label7.grid(row=3,column=2,columnspan=4)
entry6.grid(row=3,column=6,columnspan=3)


############### place the PID scale 
labelp.grid(row=4,column=8,columnspan=2)
labeli.grid(row=5,column=8,columnspan=2)
labeld.grid(row=6,column=8,columnspan=2)

wp.grid(row=4,column=10)
wi.grid(row=5,column=10)
wd.grid(row=6,column=10)


labelopt.grid(row=5,column=2)
combo.grid(row=5,column =3, columnspan=4)

label8.grid(row=4,column=1)
btn2.grid(row=5,column=1)
label9.grid(row=6,column=1)

labelbtn.grid(row=6,column=2,columnspan=7)
btn1.grid(row=7,column=2,columnspan=7)


def changelabel8_9(event):
    entry5.delete(0,100)
    entry6.delete(0,100)
    wp.config(to=10000,tickinterval=2000)
    if combo.get()==values[0]:
        wp.set(0.2)
        wi.set(0.03)
        wd.set(0.01)
        entry5.insert(0,"20")
        entry6.insert(0,"160")
        label8.config(text="系统输入:速度扰动值")
        label9.config(text="系统输出:台风最低压位置")
    elif combo.get()==values[1]:
        wp.config(to=100000,tickinterval=20000)
        wp.set(10000)
        wi.set(100)
        wd.set(10)
        entry5.insert(0,"20")
        entry6.insert(0,"1.0032")
        label8.config(text="系统输入:速度扰动值")
        label9.config(text="系统输出:台风最低压值")
    elif combo.get()==values[2]:
        wp.set(10000)
        wi.set(100)
        wd.set(10)
        entry5.insert(0,"20")
        entry6.insert(0,"1.00245")
        label8.config(text="系统输入:温度扰动值")
        label9.config(text="系统输出:台风最低压值")
combo.bind("<<ComboboxSelected>>",changelabel8_9)


#################### initialize the object

entry1.insert(0,"2018")
entry2.insert(0,"10")
entry3.insert(0,"23")
entry4.insert(0,"0")

entry5.insert(0,"20")
entry6.insert(0,"1.0032")

wp.set(10000)
wi.set(100)
wd.set(10)

fig1 = Figure(figsize=(5,3),dpi=100)
canvas1 = FigureCanvasTkAgg(fig1,draw1)

fig2 = Figure(figsize=(5,3),dpi=100)
canvas2 = FigureCanvasTkAgg(fig2,draw2)

image1 = Image.open('logoLeftmid.jpg')
photo = ImageTk.PhotoImage(image=image1)

labellogo.config(image=photo)

datattt = np.load('output_surface3.npy')
ax = fig1.add_subplot()

ax.imshow(datattt[0, 250:400, 500:700]/1e5,cmap='jet')

max = np.amax(datattt[0, 250:400, 500:700]/1e5)
min = np.amin(datattt[0, 250:400, 500:700]/1e5)
norm1 = matplotlib.colors.Normalize(vmin=min,vmax=max)
fig1.colorbar(cm.ScalarMappable(norm=norm1,cmap='jet'),ax=ax)


dx = np.arange(1,20)
ax = fig2.add_subplot()
ax.plot(dx,dx**2)



canvas1.draw()
canvas2.draw()


canvas1.get_tk_widget().pack()
canvas2.get_tk_widget().pack()



top.title("                                                      人工调控台风仿真系统")

top.state("normal")
# top.config(bg='white')
top.mainloop()


# %%
