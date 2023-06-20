# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:34:21 2023

@author: Maxime Meghnagi
"""

import napari
from magicgui import magicgui
from napari.layers import Image,Labels,Shapes

@magicgui(call_button="Calculate contrast")
def segment(Image : Image, Area : Shapes)->Labels:
    return Labels()

viewer = napari.Viewer()

viewer.window.add_dock_widget(segment, name = 'Contrast tool')
napari.run()