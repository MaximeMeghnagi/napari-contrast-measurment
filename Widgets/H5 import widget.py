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
import h5py
import pathlib






@magic_factory
def choose_h5(Import: pathlib.Path = ''):
        pass #TODO: substitute with a qtwidget without magic functions


class H5Widget(QWidget):

    
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.create_ui()


    def create_ui(self):
        layout = QFormLayout()
        self.setLayout(layout)
        self.choose_h5_widget = choose_h5(call_button=False)
        self.add_magic_function_h5(self.choose_h5_widget, layout)



    def add_magic_function_h5(self, widget, _layout):
        self.viewer.layers.events.changed.connect(widget.reset_choices)
        _layout.addWidget(widget.native)




if __name__ == '__main__':
    viewer = napari.Viewer()
    mywidget = H5Widget(viewer)
    mywidget.setMinimumSize(225, 200)
    viewer.window.add_dock_widget(mywidget, name='H5 import')
    napari.run()

