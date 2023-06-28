# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:57:16 2023

@author: Maxime Meghnagi
"""


import napari
from qtpy.QtWidgets import QWidget, QPushButton, QDoubleSpinBox, QComboBox, QFormLayout
from magicgui.widgets import ComboBox
from magicgui import magic_factory
from napari.layers import Image, Labels, Shapes
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math



color = ['blue','red','green','cyan','magenta','yellow','black']



def area_real_size(mask_):
    for i in range(mask_.shape[0]):
        for j in range(mask_.shape[1]):
            if mask_[i][j] == True:
                return (i, j)


def part_of_image(im, x_pos, y_pos, x_size, y_size, cam_noise):
    im_area = [[None] * y_size for _ in range(x_size)]
    for i in range(x_size):
        for j in range(y_size):
            im_area[i][j] = im[y_pos + i][x_pos + j] - cam_noise
    return im_area


def peaks_value(y_val, peaks_pos):
    peaks_val = np.zeros(len(peaks_pos))
    for i in range(len(peaks_pos)):
        peaks_val[i] = y_val[peaks_pos[i]]
    return peaks_val


def find_index(t, a):
    for i, element in enumerate(t):
        if a in element:
            return i
    return -1


def det2_pix_arr(mask,i,j):
    if (i==0 or i==(len(mask[0]-1)) or j==0 or j==(len(mask[:,0]-1))):
        return True
    else:
        sum=mask[j-1,i-1]+mask[j-1,i]+mask[j-1,i+1]+mask[j,i-1]+mask[j,i+1]+mask[j+1,i-1]+mask[j+1,i]+mask[j+1,i+1]
        if (sum<2):
            return False
        else:
            return True



@magic_factory
def choose_image(Image: Image):
       pass #TODO: substitute with a qtwidget without magic functions


@magic_factory
def choose_shape(Shape: Shapes):
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
        self.pixel_height_box = pixel_height_box
        cam_noise_box = QDoubleSpinBox()
        cam_noise_box.setMaximum(999999)
        cam_noise_box.setDecimals(0)
        cam_noise_box.setValue(640)
        cam_noise_box.setSingleStep(1)
        cam_noise_layout = QFormLayout()
        cam_noise_layout.addRow('Camera noise', cam_noise_box)
        layout.addRow(cam_noise_layout)
        self.cam_noise_box = cam_noise_box
        images_magiccombo = ComboBox()
        shapes_magiccombo = ComboBox()
        self.choose_image_widget = choose_image(call_button=False)
        self.choose_shape_widget = choose_shape(call_button=False)
        self.add_magic_function(self.choose_image_widget, layout)
        self.add_magic_function(self.choose_shape_widget, layout)
        self.images_magiccombo = images_magiccombo 
        self.shapes_magiccombo = shapes_magiccombo 
        calculate_btn = QPushButton('Calculate contrast')
        calculate_btn.clicked.connect(lambda: self.calculate_contrast(self.cam_noise_box.value(), self.pixel_width_box.value(),self.pixel_height_box.value()))
        layout.addWidget(calculate_btn)


    def add_magic_function(self, widget, _layout):
       self.viewer.layers.events.inserted.connect(widget.reset_choices)
       self.viewer.layers.events.removed.connect(widget.reset_choices)
       _layout.addWidget(widget.native)


    def calculate_contrast(self, camera_noise, pixel_width, pixel_height) -> Labels:
        self.image_set()
        self.data_analysis(camera_noise, pixel_width, pixel_height)
        self.graph_view()


    def image_set(self):
        global image_data, nb_of_line, shape_layer, step        
        selected_image = str(self.choose_image_widget.Image.value)
        selected_shape = str(self.choose_shape_widget.Shape.value)
        choices=str(self.viewer.layers).split(">, <")
        image_layer = self.viewer.layers[find_index(choices,selected_image)]
        shape_layer = self.viewer.layers[find_index(choices,selected_shape)]        
        step = viewer.dims.current_step[0]
        image_data = np.array(image_layer.data[step])
        nb_of_line = len(shape_layer.to_masks())


    def data_analysis(self, camera_noise, pixel_width, pixel_height):
        global values_x, values_y, peaks, valley, peaks_val, valley_val, Imin, Imax, Contrast, x_axis_peak, x_axis_valley
        x_pos = np.zeros(nb_of_line)
        y_pos = np.zeros(nb_of_line)
        x_size = np.zeros(nb_of_line)
        y_size = np.zeros(nb_of_line)
        sim_x = np.empty((nb_of_line,), dtype=object)
        sim_y = np.empty((nb_of_line,), dtype=object)
        size = np.zeros(nb_of_line)
        values_x = np.empty((nb_of_line,), dtype=object)
        values_y = np.empty((nb_of_line,), dtype=object)
        peaks = np.empty((nb_of_line,), dtype=object)
        valley = np.empty((nb_of_line,), dtype=object)
        peaks_val = np.empty((nb_of_line,), dtype=object)
        valley_val = np.empty((nb_of_line,), dtype=object)
        Imax = np.zeros(nb_of_line)
        Imin = np.zeros(nb_of_line)
        Contrast = np.zeros(nb_of_line)
        x_axis_peak = np.empty((nb_of_line,), dtype=object)
        x_axis_valley = np.empty((nb_of_line,), dtype=object)
        for i in range (nb_of_line):
            mask = shape_layer.to_masks()[i]
            mask = mask[step,:,:]
            y_pos[i], x_pos[i] = area_real_size(mask)[0], area_real_size(mask)[1]
            y_size[i], x_size[i] = mask.shape[0] - y_pos[i], mask.shape[1] - x_pos[i]
            sim_x[i] = np.arange(int(x_pos[i]), int(x_pos[i] + x_size[i]) + 1, 1)
            sim_y[i] = np.round(np.linspace(y_pos[i],y_pos[i]+y_size[i]+1,int(x_size[i]+1))).astype(int)
            size[i] = np.size(sim_x[i])
            for j in range (int(size[i])):
                values_x[i] = np.append(values_x[i], math.sqrt((pixel_width*j)**2 + (pixel_height*j)**2))
                values_y[i] = np.append(values_y[i], image_data[sim_y[i][j],sim_x[i][j]] - camera_noise) 
            values_x[i] = values_x[i][1:]
            values_y[i] = values_y[i][1:]
            peaks[i] = np.append(peaks[i], find_peaks(values_y[i], distance=8))
            valley[i] = np.append(valley[i], find_peaks(-values_y[i], distance=8))
            peaks_val[i] = np.append(peaks_val[i], peaks_value(values_y[i],peaks[i][1]))
            valley_val[i] = np.append(valley_val[i], peaks_value(values_y[i],valley[i][1]))
            peaks_val[i] = peaks_val[i][1:]
            valley_val[i] = valley_val[i][1:]
            Imax[i] = np.mean(peaks_val[i])
            Imin[i] = np.mean(valley_val[i])
            Contrast[i] = (Imax[i] - Imin[i]) / (Imax[i] + Imin[i])
            for j in range(len(peaks_val[i])):
                x_axis_peak[i] = np.append(x_axis_peak[i], math.sqrt((pixel_width*peaks[i][1][j])**2 + (pixel_height*peaks[i][1][j])**2))
            x_axis_peak[i] = x_axis_peak[i][1:]
            for j in range(len(valley_val[i])): 
                x_axis_valley[i] = np.append(x_axis_valley[i], math.sqrt((pixel_width*valley[i][1][j])**2 + (pixel_height*valley[i][1][j])**2))
            x_axis_valley[i] = x_axis_valley[i][1:]


    def graph_view(self):
        plt.figure()
        for i in range (nb_of_line):
            plt.plot(values_x[i], values_y[i], color=color[i],label="C={}".format(round(Contrast[i], 2)))
            plt.plot(x_axis_peak[i], peaks_val[i], marker='o', markersize=4, linewidth=0, linestyle='solid', color=color[i])
            plt.plot(x_axis_valley[i], valley_val[i], marker='o', markersize=4, linewidth=0, linestyle='solid', color=color[i])
            plt.plot(values_x[i], Imax[i] * np.ones_like(values_x[i]), linewidth=1.5, linestyle='dotted', color=color[i])
            plt.plot(values_x[i], Imin[i] * np.ones_like(values_x[i]), linewidth=1.5, linestyle='dotted', color=color[i])
        plt.title("Contrast", size=14, fontweight='bold')
        plt.xlabel("Position [µm]", size=14)
        plt.ylabel("Intensity [a.u.]", size=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        fig = plt.gcf()
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer)
        self.viewer.add_image(data, name='Contrast', scale=(4, 4))



if __name__ == '__main__':
    viewer = napari.Viewer()
    mywidget = ContrastWidget(viewer)
    mywidget.setMinimumSize(225, 200)
    viewer.window.add_dock_widget(mywidget, name='Contrast')
    napari.run()