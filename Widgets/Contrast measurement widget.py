# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:57:16 2023

@author: Maxime Meghnagi
"""


import napari
from qtpy.QtWidgets import QWidget, QPushButton, QDoubleSpinBox,  QFormLayout, QLabel
from magicgui.widgets import ComboBox
from magicgui import magic_factory
from napari.layers import Image, Labels, Shapes
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math



color = ['blue','red','green','cyan','magenta','yellow','black']



def cam_noise_av(mask_, image_data):
    s = 0
    c = 0
    for i in range(mask_.shape[0]):
        for j in range(mask_.shape[1]):
            if (mask_[i][j] == True):
                s += image_data[i][j]
                c += 1
    return (s/c)

def area_real_size(mask_):
    for i in range(mask_.shape[0]):
        for j in range(mask_.shape[1]):
            if (mask_[i][j] == True):
                sum_ = 0
                if (((i-1)>0) and ((j-1)>0) and (mask_[i-1][j-1]==True)):
                    sum_ += 1
                if (((i-1)>0) and (mask_[i-1][j]==True)):
                    sum_ += 1
                if (((i-1)>0) and ((j+1)<mask_.shape[1]) and (mask_[i-1][j+1]==True)):
                    sum_ += 1
                if (((j-1)>0) and (mask_[i][j-1]==True)):
                    sum_ += 1
                if (((j+1)<mask_.shape[1]) and (mask_[i][j+1]==True)):
                    sum_ += 1
                if (((i+1)<mask_.shape[0]) and ((j-1)>0) and (mask_[i+1][j-1]==True)):
                    sum_ += 1
                if (((i+1)<mask_.shape[0]) and (mask_[i+1][j]==True)):
                    sum_ += 1
                if (((i+1)<mask_.shape[0]) and ((j+1)<mask_.shape[1]) and (mask_[i+1][j+1]==True)):
                    sum_ += 1
                if (sum_<2):
                    return (i, j)


def lin_size(list,x_pos,y_pos):
    nb_pix_x = 0
    nb_pix_y = 0
    c = True
    c_right = True
    c_left = True
    c_down = True
    c_up = True
    while(c == True):
        if(x_pos == 0):
            c_left = False
        if(x_pos == len(list[0])-1):
            c_right = False
        if(y_pos == 0):
            c_up = False
        if(y_pos == len(list)-1):
            c_down = False
        if((c_up==True) and ((int(y_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos)]==True)) and ((int(y_pos-2)<0) or ((list[int(y_pos-2)][int(x_pos)]==True) or ((int(x_pos+1)<list.shape[1]) and (list[int(y_pos-2)][int(x_pos+1)]==True)) or ((int(x_pos-1)>-1) and (list[int(y_pos-2)][int(x_pos-1)]==True)) or ((int(x_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos-1)]==True)) or ((int(x_pos+1)<list.shape[1]) and (list[int(y_pos-1)][int(x_pos+1)]==True))))):
            y_pos -= 1 
            nb_pix_y += 1
            c_down = False
        elif((c_up==True) and (c_left==True) and (int(y_pos-1)>-1) and (int(x_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos-1)]==True)):
            y_pos -= 1
            x_pos -=1 
            nb_pix_y += 1
            nb_pix_x += 1
            c_down = False
            c_right = False    
        elif((c_up==True) and (c_right==True) and (int(y_pos-1)>-1) and (int(x_pos+1)<list.shape[1]) and (list[int(y_pos-1)][int(x_pos+1)]==True)):
            y_pos -= 1
            x_pos +=1 
            nb_pix_y += 1
            nb_pix_x += 1
            c_down = False
            c_left = False
        elif((c_left==True) and ((int(x_pos-1)>-1) and (list[int(y_pos)][int(x_pos-1)]==True)) and ((int(x_pos-2)<0) or ((list[int(y_pos)][int(x_pos-2)]==True) or ((int(y_pos+1)<list.shape[0]) and (list[int(y_pos+1)][int(x_pos-2)]==True)) or ((int(y_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos-2)]==True)) or ((int(y_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos-1)]==True))  or ((int(y_pos+1)<list.shape[0]) and (list[int(y_pos+1)][int(x_pos-1)]==True))))):
            x_pos -=1 
            nb_pix_x += 1
            c_right = False
        elif((c_right==True) and (int(x_pos+1)<list.shape[1]) and (list[int(y_pos)][int(x_pos+1)]==True)  and ((int(x_pos+2)>=list.shape[1]) or ((list[int(y_pos)][int(x_pos+2)]==True) or ((int(y_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos+2)==True])) or ((int(y_pos+1)<list.shape[0]) and (list[int(y_pos+1)][int(x_pos+2)] == True)) or ((int(y_pos-1)>-1) and (list[int(y_pos-1)][int(x_pos+1)] == True)) or ((int(y_pos+1)<list.shape[0]) and (list[int(y_pos+1)][int(x_pos+1)] == True))))):
            x_pos += 1 
            nb_pix_x += 1
            c_left = False
        elif((c_down == True) and (int(y_pos+1)<list.shape[0]) and (list[int(y_pos+1)][int(x_pos)] == True) and ((int(y_pos+2)>=list.shape[0]) or ((list[int(y_pos+2)][int(x_pos)] == True) or ((int(x_pos+1)<list.shape[1]) and (list[int(y_pos+2)][int(x_pos+1)] == True)) or ((int(x_pos+1)>-1) and (list[int(y_pos+2)][int(x_pos-1)] == True)) or ((int(x_pos+1)<list.shape[1]) and (list[int(y_pos+1)][int(x_pos+1)] == True)) or ((int(x_pos+1)>-1) and (list[int(y_pos+1)][int(x_pos-1)] == True))))):
            y_pos += 1 
            nb_pix_y += 1
            c_up = False
        elif((c_down == True) and (c_left ==  True) and (int(y_pos+1)<list.shape[0]) and (int(x_pos-1)>-1) and (list[int(y_pos+1)][int(x_pos-1)] == True)):
            y_pos += 1
            x_pos -=1 
            nb_pix_y += 1
            nb_pix_x += 1
            c_up = False
            c_right = False    
        elif((c_down == True) and (c_right ==  True) and (int(y_pos+1)<list.shape[0]) and (int(x_pos+1)<list.shape[1]) and (list[int(y_pos+1)][int(x_pos+1)] == True)):
            y_pos += 1
            x_pos +=1 
            nb_pix_y += 1
            nb_pix_x += 1
            c_up = False
            c_left = False 
        else:
            c = False
    return (nb_pix_x+1,nb_pix_y+1)


def peaks_value(y_val, peaks_pos):
    peaks_val = np.zeros(len(peaks_pos))
    for i in range(len(peaks_pos)): 
        peaks_val[i] = y_val[int(peaks_pos[i])]
    return peaks_val


def find_index(t, a):
    for i, element in enumerate(t):
        if a in element:
            return i
    return -1



@magic_factory
def choose_image(Image: Image):
       pass #TODO: substitute with a qtwidget without magic functions


@magic_factory
def choose_shape(Shape: Shapes):
       pass #TODO: substitute with a qtwidget without magic functions


@magic_factory
def area_cam_noise(Shape: Shapes):
       pass #TODO: substitute with a qtwidget without magic functions


class ContrastWidget(QWidget):

    
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.create_ui()


    def create_ui(self):
        layout = QFormLayout()
        self.setLayout(layout)
        pixel_width_box = QDoubleSpinBox()
        pixel_width_box.setDecimals(4)
        pixel_width_box.setValue(0.0586)
        pixel_width_box.setSingleStep(0.0001)
        pixel_width_layout = QFormLayout()
        pixel_width_layout.addRow('Pixel width (µm)', pixel_width_box)
        layout.addRow(pixel_width_layout)
        self.pixel_width_box = pixel_width_box
        pixel_height_box = QDoubleSpinBox()
        pixel_height_box.setDecimals(4)
        pixel_height_box.setValue(0.0586)
        pixel_height_box.setSingleStep(0.0001)
        pixel_height_layout = QFormLayout()
        pixel_height_layout.addRow('Pixel height (µm)', pixel_height_box)
        layout.addRow(pixel_height_layout)
        self.camera_noise_label = QLabel()
        layout.addRow('Camera Noise :  ', self.camera_noise_label)
        self.pixel_height_box = pixel_height_box
        images_magiccombo = ComboBox()
        shapes_magiccombo = ComboBox()
        self.choose_image_widget = choose_image(call_button=False)
        self.choose_shape_widget = choose_shape(call_button=False)
        self.area_cam_noise_widget = area_cam_noise(call_button=False)
        self.add_magic_function(self.choose_image_widget, layout)
        self.add_magic_function(self.choose_shape_widget, layout)
        self.add_magic_function(self.area_cam_noise_widget, layout)
        self.images_magiccombo = images_magiccombo 
        self.shapes_magiccombo = shapes_magiccombo 
        self.area_cam_noise_magiccombo = shapes_magiccombo 
        calculate_btn = QPushButton('Calculate contrast')
        calculate_btn.clicked.connect(lambda: self.calculate_contrast(self.pixel_width_box.value(),self.pixel_height_box.value()))
        layout.addWidget(calculate_btn)


    def add_magic_function(self, widget, _layout):
       self.viewer.layers.events.inserted.connect(widget.reset_choices)
       self.viewer.layers.events.removed.connect(widget.reset_choices)
       _layout.addWidget(widget.native)


    def calculate_contrast(self, pixel_width, pixel_height) -> Labels:
        self.image_set()
        self.data_analysis(pixel_width, pixel_height)
        self.graph_view()
        self.camera_noise_label.setText('%.2f' % round(camera_noise, 2))


    def image_set(self):
        global image_data, nb_of_line, shape_layer, camera_noise, step        
        selected_image = str(self.choose_image_widget.Image.value)
        selected_shape = str(self.choose_shape_widget.Shape.value)
        area_cam_noise_shape = str(self.area_cam_noise_widget.Shape.value)
        choices=str(self.viewer.layers).split(">, <")
        image_layer = self.viewer.layers[find_index(choices,selected_image)]
        shape_layer = self.viewer.layers[find_index(choices,selected_shape)]
        cam_noise_layer = self.viewer.layers[find_index(choices,area_cam_noise_shape)]   
        step = viewer.dims.current_step[0]
        image_data = np.array(image_layer.data[step])
        cam_noise_data = cam_noise_layer.to_masks()[:,step][0]
        camera_noise = cam_noise_av(cam_noise_data, image_data)
        nb_of_line = len(shape_layer.to_masks())
        camera_noise = cam_noise_av(cam_noise_data, image_data)

    def data_analysis(self, pixel_width, pixel_height):
        global values_x, values_y, peaks, valley, peaks_val, valley_val, Imin, Imax, Contrast, x_axis_peak, x_axis_valley
        x_pos = np.zeros(nb_of_line)
        y_pos = np.zeros(nb_of_line)
        x_size = np.zeros(nb_of_line)
        y_size = np.zeros(nb_of_line)
        size = np.zeros(nb_of_line)
        sim_x = np.empty((nb_of_line,), dtype=object)
        sim_y = np.empty((nb_of_line,), dtype=object)
        values_x = [[] for _ in range(nb_of_line)]
        values_y = [[] for _ in range(nb_of_line)]
        peaks = [[] for _ in range(nb_of_line)]
        valley = [[] for _ in range(nb_of_line)]
        peaks_val = [[] for _ in range(nb_of_line)]
        valley_val = [[] for _ in range(nb_of_line)]
        x_axis_peak = [[] for _ in range(nb_of_line)]
        x_axis_valley = [[] for _ in range(nb_of_line)]
        Imax = np.zeros(nb_of_line)
        Imin = np.zeros(nb_of_line)
        Contrast = np.zeros(nb_of_line)
        for i in range (nb_of_line):
            mask = shape_layer.to_masks()[i][step]
            (y_pos[i], x_pos[i]) = area_real_size(mask)
            x_size[i], y_size[i] = lin_size(mask,x_pos[i],y_pos[i])
            if (x_size[i]<y_size[i]):
                sim_y[i] = np.arange(int(y_pos[i]), int(y_pos[i] + y_size[i]) + 1, 1) 
                sim_x[i] = np.round(np.linspace(x_pos[i],x_pos[i]+x_size[i]+1,int(y_size[i]+1))).astype(int)
            else :
                sim_x[i] = np.arange(int(x_pos[i]), int(x_pos[i] + x_size[i]) + 1, 1)
                sim_y[i] = np.round(np.linspace(y_pos[i],y_pos[i]+y_size[i]+1,int(x_size[i]+1))).astype(int)
            size[i] = np.size(sim_x[i])
            for j in range (int(size[i])):
                values_x[i].append(math.sqrt((pixel_width*j)**2 + (pixel_height*j)**2))
                values_y[i].append(image_data[sim_y[i][j],sim_x[i][j]] - camera_noise)
            peaks[i] = find_peaks(values_y[i], distance=8)
            for j in range (len(values_y[i])):
                values_y[i][j] = (-int(values_y[i][j]))
            valley[i] = find_peaks(values_y[i], distance=8)
            for j in range (len(values_y[i])):
                values_y[i][j] = (-int(values_y[i][j]))
            peaks_val[i] = peaks_value(values_y[i], peaks[i][0])
            valley_val[i] = peaks_value(values_y[i], valley[i][0])
            Imax[i] = np.mean(peaks_val[i])
            Imin[i] = np.mean(valley_val[i])
            Contrast[i] = (Imax[i] - Imin[i]) / (Imax[i] + Imin[i])
            for j in range(len(peaks_val[i])):
                x_axis_peak[i].append(math.sqrt((pixel_width*peaks[i][0][j])**2 + (pixel_height*peaks[i][0][j])**2))
            x_axis_peak[i] = x_axis_peak[i][1:]
            for j in range(len(valley_val[i])): 
                x_axis_valley[i].append(math.sqrt((pixel_width*valley[i][0][j])**2 + (pixel_height*valley[i][0][j])**2))
            x_axis_valley[i] = x_axis_valley[i][1:]


    def graph_view(self):
        if (nb_of_line==1):
            shape_layer.edge_color = [0., 0., 1., 1.] # Blue
        elif (nb_of_line==2):
            shape_layer.edge_color = [[0., 0., 1., 1.],[1., 0., 0., 1.]] # Red
        elif (nb_of_line==3):
            shape_layer.edge_color = [[0., 0., 1., 1.],[1., 0., 0., 1.],[0., 1., 0., 1.]] # Green
        elif (nb_of_line==4):
            shape_layer.edge_color = [[0., 0., 1., 1.],[1., 0., 0., 1.],[0., 1., 0., 1.],[0., 1., 1., 1.]] # Cyan
        elif (nb_of_line==5):
            shape_layer.edge_color = [[0., 0., 1., 1.],[1., 0., 0., 1.],[0., 1., 0., 1.],[0., 1., 1., 1.],[1., 0., 1., 1.]] # Magenta
        elif (nb_of_line==6):
            shape_layer.edge_color = [[0., 0., 1., 1.],[1., 0., 0., 1.],[0., 1., 0., 1.],[0., 1., 1., 1.],[1., 0., 1., 1.],[1., 1., 0., 1.]] # Yellow
        elif (nb_of_line==7):
            shape_layer.edge_color = [[0., 0., 1., 1.],[1., 0., 0., 1.],[0., 1., 0., 1.],[0., 1., 1., 1.],[1., 0., 1., 1.],[1., 1., 0., 1.],[0., 0., 0., 1.]] # Black
        plt.figure()
        for i in range (nb_of_line):
            plt.plot(values_x[i], values_y[i], color=color[i],label="C={}".format(round(Contrast[i], 2)))
        plt.title("Contrast", size=14, fontweight='bold')
        plt.xlabel("Position [µm]", size=14)
        plt.ylabel("Intensity [a.u.]", size=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.show()



if __name__ == '__main__':
    viewer = napari.Viewer()
    mywidget = ContrastWidget(viewer)
    mywidget.setMinimumSize(225, 200)
    viewer.window.add_dock_widget(mywidget, name='Contrast')
    napari.run() 