# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:00:24 2023

@author: Maxime Meghnagi
"""

import napari
from qtpy.QtWidgets import QWidget, QPushButton, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem
from qtpy.QtWidgets import QVBoxLayout, QFormLayout
import numpy as np
from numpy.fft import fft2, ifftshift, fftshift, fftfreq
from skimage import data
from napari.layers import Image,Labels
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects
from magicgui import magicgui





class MyWidget(QWidget):
    
    def __init__(self, viewer:napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.create_ui()
        
    def create_ui(self):
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # create list widget for selecting layers
        Image_box = QListWidget()
        Image_box_Layout = QFormLayout()
        Image_box_Layout.addRow('Image :',Image_box)
        layout.addLayout(Image_box_Layout)
        self.Image_box = Image_box

        # populate list widget with layer names
        for layer in self.viewer.layers:
            item = QListWidgetItem(layer.name)
            Image_box.addItem(item)
        
            
    
if __name__ == '__main__':
   
    viewer = napari.view_image(data.astronaut(), rgb=True)
    mywidget = MyWidget(viewer)
    viewer.window.add_dock_widget(mywidget, name = 'Contrast widget')
    napari.run()
    
    