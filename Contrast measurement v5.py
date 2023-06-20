# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:34:21 2023

@author: Maxime Meghnagi
"""



import napari
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



pixel_width = 0.0586 #µm
pixel_height = 0.0586 #µm
camera_noise = 640



@magicgui(call_button="Calculate contrast")
def init(Image: Image, Rectangle: Shapes) -> Labels:
    global viewer
    Image_data = np.array(Image.data[5])
    print(Image_data)
    viewer.add_image(Image_data)
    print('\n1')
    mask = Rectangle.to_masks()[0]
    print('\n2\n')
    print(mask.shape[0],mask.shape[1])
    print('\n')
    usefull_mask=mask[5,:,:]
    print(usefull_mask)
    viewer.add_image(usefull_mask)
    y_pos,x_pos=area_real_size(usefull_mask)[0],area_real_size(usefull_mask)[1]
    print('\n')
    y_pix_pos,x_pix_pos=y_pos*pixel_height,x_pos*pixel_width
    print(y_pix_pos,x_pix_pos)
    y_size,x_size=usefull_mask.shape[0]-y_pos,usefull_mask.shape[1]-x_pos
    print('\n')
    print(x_size,y_size)
    new_image=part_of_image(Image_data,x_pos,y_pos,x_size,y_size,camera_noise)
    print('\n')
    print(new_image)
    print('\n')
    av=average(new_image,y_size,x_size)
    print(av)
    print('\n')
    peaks, _ = find_peaks(av,distance=8)
    print(peaks)
    print('\n')
    valley, _  = find_peaks(-av,distance=8)
    print(valley)
    print('\n')
    print(x_pos,y_pos)
    print('\n')
    peaks_val,valley_val = peaks_value(av,peaks),peaks_value(av,valley)
    print(peaks_val,valley_val)
    print('\n')
    Imax = np.mean(peaks_val)
    Imin = np.mean(valley_val)
    print(Imax,Imin)
    print('\n')
    C = (Imax-Imin)/(Imax+Imin)
    print('contrast:', C) 
    print('\n')
    x_axis_av = np.zeros(len(av))
    for i in range (len(av)):
        x_axis_av[i]= pixel_width*(x_pos+i)
    x_axis_peak = np.zeros(len(peaks_val))
    for i in range (len(peaks_val)):
        x_axis_peak[i]= pixel_width*(x_pos+peaks[i])
    x_axis_valley = np.zeros(len(valley_val))
    for i in range (len(valley_val)):
        x_axis_valley[i]= pixel_width*(x_pos+valley[i])
    plt.figure()    
    plt.plot(x_axis_av, av, color='black')
    plt.plot(x_axis_peak,peaks_val,marker='o',markersize=4,linewidth=0,linestyle='solid',color='red')
    plt.plot(x_axis_valley,valley_val,marker='o',markersize=4,linewidth=0,linestyle='solid',color='blue')
    plt.plot(x_axis_av, Imax*np.ones_like(x_axis_av), linewidth=1.5,linestyle='dotted',color='red')
    plt.plot(x_axis_av, Imin*np.ones_like(x_axis_av), linewidth=1.5,linestyle='dotted',color='blue')
    plt.title("Contrast {}".format(round(C,2)),size=14,fontweight='bold')
    plt.xlabel("Position [µm]", size=14)
    plt.ylabel("Intensity [a.u.]", size=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim((np.amin(av)*0.9,np.amax(av)*1.1))
    plt.xlim((np.amin(x_axis_av),np.amax(x_axis_av)))
    fig = plt.gcf() 
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer._renderer)
    viewer.add_image(data,name='Contrast',scale=(4, 4))    
    print('\nEnd')
    return Labels

def area_real_size(mask_):
    for i in range (mask_.shape[0]):
        for j in range (mask_.shape[1]):
            if mask_[i][j] == True :
                return (i,j)

def part_of_image(im,x_pos,y_pos,x_size,y_size,cam_noise):
    im_area = [[None]*y_size for _ in range (x_size)]
    for i in range (x_size):
        for j in range (y_size):
            im_area[i][j]=im[y_pos+i][x_pos+j]-cam_noise
    return im_area

def average(n_i,x_size,y_size):
    av = np.zeros(x_size)
    if (y_size==1):
        return n_i[0]
    else:
        for i in range (y_size):
            av += n_i[i]
    av = av / y_size
    return av

def peaks_value(av, peaks):
    peaks_val=np.zeros(len(peaks))
    for i in range (len(peaks)):
        peaks_val[i]=av[peaks[i]]
    return peaks_val



viewer = napari.Viewer()
viewer.window.add_dock_widget(init, name='Contrast tool')
napari.gui_qt()