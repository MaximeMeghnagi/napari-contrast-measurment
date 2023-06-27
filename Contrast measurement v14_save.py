# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:57:16 2023

@author: Maxime Meghnagi
"""


import napari
from qtpy.QtWidgets import QWidget, QPushButton, QDoubleSpinBox, QComboBox, QFormLayout, QLineEdit
from magicgui.widgets import SpinBox, Label, Container, ComboBox, FloatSpinBox, LineEdit, RadioButtons, PushButton
from magicgui import magic_factory
from napari.layers import Image, Labels, Shapes
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import h5py
import pathlib
import os



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


def average(n_i, x_size, y_size):
    av = np.zeros(x_size)
    if y_size == 1:
        return n_i[0]
    else:
        for i in range(y_size):
            av += n_i[i]
    av = av / y_size
    return av


def peaks_value(av, peaks):
    peaks_val = np.zeros(len(peaks))
    for i in range(len(peaks)):
        peaks_val[i] = av[peaks[i]]
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
def choose_h5(Import: pathlib.Path = os.getcwd()+'\\temp.h5'):
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
        self.choose_h5_widget = choose_h5(call_button=False)
        self.add_magic_function(self.choose_image_widget, layout)
        self.add_magic_function(self.choose_shape_widget, layout)
        self.add_magic_function_h5(self.choose_h5_widget, layout)
        self.images_magiccombo = images_magiccombo 
        self.shapes_magiccombo = shapes_magiccombo 
        calculate_btn = QPushButton('Calculate contrast')
        calculate_btn.clicked.connect(lambda: self.calculate_contrast(self.cam_noise_box.value(), self.pixel_width_box.value()))
        layout.addWidget(calculate_btn)


    def add_magic_function(self, widget, _layout):
       self.viewer.layers.events.inserted.connect(widget.reset_choices)
       self.viewer.layers.events.removed.connect(widget.reset_choices)
       _layout.addWidget(widget.native)


    def add_magic_function_h5(self, widget, _layout):
        self.viewer.layers.events.changed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)


    def calculate_contrast(self, camera_noise, pixel_width) -> Labels:
        selected_image = str(self.choose_image_widget.Image.value)
        selected_shape = str(self.choose_shape_widget.Shape.value)
        choices=str(self.viewer.layers).split(">, <")
        image_layer = self.viewer.layers[find_index(choices,selected_image)]
        shape_layer = self.viewer.layers[find_index(choices,selected_shape)]        
        step = viewer.dims.current_step[0]
        image_data = np.array(image_layer.data[step]) 
        mask = shape_layer.to_masks()[0] # Make that with different lines
        usefull_mask = mask[5, :, :]
        y_pos, x_pos = area_real_size(usefull_mask)[0], area_real_size(usefull_mask)[1]
        y_size, x_size = usefull_mask.shape[0] - y_pos, usefull_mask.shape[1] - x_pos
        new_image = part_of_image(image_data, x_pos, y_pos, x_size, y_size, camera_noise)
        av = average(new_image, y_size, x_size)
        peaks, _ = find_peaks(av, distance=8)
        valley, _ = find_peaks(-av, distance=8)
        peaks_val, valley_val = peaks_value(av, peaks), peaks_value(av, valley)
        Imax = np.mean(peaks_val)
        Imin = np.mean(valley_val)
        C = (Imax - Imin) / (Imax + Imin)
        x_axis_av = np.zeros(len(av))
        for i in range(len(av)):
            x_axis_av[i] = pixel_width * (x_pos + i)
        x_axis_peak = np.zeros(len(peaks_val))
        for i in range(len(peaks_val)):
            x_axis_peak[i] = pixel_width * (x_pos + peaks[i])
        x_axis_valley = np.zeros(len(valley_val))
        for i in range(len(valley_val)):
            x_axis_valley[i] = pixel_width * (x_pos + valley[i])
        plt.figure()
        plt.plot(x_axis_av, av, color='black')
        plt.plot(x_axis_peak, peaks_val, marker='o', markersize=4, linewidth=0, linestyle='solid', color='red')
        plt.plot(x_axis_valley, valley_val, marker='o', markersize=4, linewidth=0, linestyle='solid', color='blue')
        plt.plot(x_axis_av, Imax * np.ones_like(x_axis_av), linewidth=1.5, linestyle='dotted', color='red')
        plt.plot(x_axis_av, Imin * np.ones_like(x_axis_av), linewidth=1.5, linestyle='dotted', color='blue')
        plt.title("Contrast {}".format(round(C, 2)), size=14, fontweight='bold')
        plt.xlabel("Position [µm]", size=14)
        plt.ylabel("Intensity [a.u.]", size=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim((np.amin(av) * 0.9, np.amax(av) * 1.1))
        plt.xlim((np.amin(x_axis_av), np.amax(x_axis_av)))
        fig = plt.gcf()
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer)
        self.viewer.add_image(data, name='Contrast', scale=(4, 4))
        return Labels



if __name__ == '__main__':
    viewer = napari.Viewer()
    mywidget = ContrastWidget(viewer)
    mywidget.setMinimumSize(225, 200)
    viewer.window.add_dock_widget(mywidget, name='Contrast')
    napari.run()

