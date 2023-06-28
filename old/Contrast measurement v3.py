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


@magicgui(call_button="Calculate contrast")
def segment(Image: Image, Area: Shapes) -> Labels:
    global viewer
    Image_data = np.array(Image.data[0])
    print(Image_data)
    viewer.add_image(Image_data)
    print('\n1\n')
    mask = Area.to_masks()[0]
    print('\n2\n')
    print(mask.shape[0],mask.shape[1])
    print('\n')
    viewer.add_image(mask[5,:,:])
    print(mask[5,:,:])
    #y_pos,x_pos=area_real_size(mask)[0],area_real_size(mask)[1]
    #print(x_pos,y_pos)
    # y_size,x_size=mask.shape[0]-y_pos,mask.shape[1]-x_pos
    # print(x_size,y_size)
    # new_image=part_of_image(Image_data,x_pos,y_pos,x_size,y_size)
    # print(new_image)
    # viewer.add_image(new_image)
    # print('Test 2')
    return Labels



def area_real_size(mask_):
    for i in range (mask_.shape[0]):
        for j in range (mask_.shape[1]):
            if mask_[i][j] == True :
                return (i,j)
        
            
    


def part_of_image(im,x_pos,y_pos,x_size,y_size):
    im_area = [[None]*y_size for _ in range (x_size)]
    for i in range (x_size):
        for j in range (y_size):
            im_area[i][j]=im[y_pos+i][x_pos+j]
    return im_area



viewer = napari.Viewer()
viewer.window.add_dock_widget(segment, name='Contrast tool')
napari.gui_qt()