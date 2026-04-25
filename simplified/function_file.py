# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:02:02 2023

@author: ZRS
"""

import numpy as np
from scipy import ndimage as ndi

def fluct_on_Pressure(input_surface,Y_location,X_location):
    
    intermi_surface = input_surface 
    # keep the same surface input
    
    intermi_surface[0][Y_location][X_location] = intermi_surface[0][Y_location][X_location] * 0.5
    
    # in such mesh increase the pressure at 1.1 times for surface
    
    return intermi_surface

def fluct_on_Pressure1(input_upper,Y_location,X_location):
    
    intermi_surface = input_upper 
    # keep the same surface input
    
    intermi_surface[0,:,Y_location,X_location] = intermi_surface[0,:,Y_location,X_location]*0.5
    
    # in such mesh increase the pressure by make it change to 1.1 times for upper 
    
    return intermi_surface


def fluct_on_Temperature(input_surface,Y_location,X_location,T):
    intermi_surface = input_surface 
    # keep the same surface input
    
    intermi_surface[3][Y_location][X_location] = intermi_surface[3][Y_location][X_location] + T
    
    # in such mesh set the temperature at 3000K
    
    return intermi_surface


def fluct_on_Temperature1(input_upper,Y_location,X_location,T):
    
    intermi_upper = input_upper
    for i in range(intermi_upper[2,:,Y_location,X_location].ndim):
        intermi_upper[2,i,Y_location,X_location] = intermi_upper[2,i,Y_location,X_location] + T
    
    return intermi_upper
    
    

def fluct_on_U10(input_surface,Y_location,X_location,val):
    intermi_surface = input_surface 
    # keep the same surface input
    # print("hello")
    intermi_surface[1][Y_location][X_location] = val
    # intermi_surface[2][Y_location][X_location] = 0
    
    # in such mesh set the temperature at 3000K
    
    return intermi_surface

def fluct_on_U10_upper(input_upper,Y_location,X_location,val):
    
    intermi_upper = input_upper
    

    intermi_upper[3,:,Y_location,X_location] = val
    # intermi_upper[4,:,Y_location,X_location] = 0
    
    return intermi_upper
    
    
def fluct_on_V10(input_surface,Y_location,X_location,val):
    intermi_surface = input_surface 
    # keep the same surface input

    #intermi_surface[1][Y_location][X_location] = 0
    intermi_surface[2][Y_location][X_location] = val
    
    
    # in such mesh set the temperature at 3000K
    
    return intermi_surface

def fluct_on_V10_upper(input_upper,Y_location,X_location,val):
    
    intermi_upper = input_upper
    

    #intermi_upper[3,:,Y_location,X_location] = 0
    intermi_upper[4,:,Y_location,X_location] = val
    
    return intermi_upper

def TwoarrayNorm(a,b):
    
    output = np.sqrt(a[:][:]**2 + b[:][:]**2)
    return output

def findtyphoneclone_single_MSLP(input_surface):
    # this is a subsection area of surface input

    min_pressure,max_pressure,min_loc,max_loc = ndi.extrema(input_surface)
    
    Y_location = min_loc[0]
    
    X_location = min_loc[1]
    
    
    # THE ORDER OF the location is min_loc[1] is the first axis, and min_loc[0] is the second axis
    return Y_location , X_location
    