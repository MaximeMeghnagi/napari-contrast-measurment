# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:34:21 2023

@author: Maxime Meghnagi
"""



import napari
from magicgui import magicgui
from napari.layers import Image,Labels,Shapes
import numpy as np



@magicgui(call_button="Calculate contrast")
def segment(Image : Image, Area : Shapes)->Labels:
    Image_data = np.array(Image.data)
    print(Image_data)
    mask = Area.to_masks()[0]
    print(mask)
    masked_data = Image_data * mask  # Appliquer le masque à l'image
    print(masked_data)
    return Labels

viewer = napari.Viewer()

viewer.window.add_dock_widget(segment, name = 'Contrast tool')

napari.gui_qt()    



    # print('Test 3')
    # masked_data = Image_data * mask  # Appliquer le masque à l'image
    # print('Test 4')
    # masked_image = Image(data=masked_data)  # Créer une nouvelle couche d'image avec les données masquées
    # print('Test 5')
    # viewer.layers.append(masked_image)  # Ajouter la couche d'image à la visionneuse Napari
    # # Faites quelque chose avec les données d'image correspondant à la zone tracée
    # print('Test 6')