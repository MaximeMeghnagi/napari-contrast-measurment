# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:41:49 2023

@author: Maxime Meghnagi
"""

import napari
import h5py

file_path = 'C:/Maxime/Etudes/Stage Polimi 2023/Contrast/230606_110320_FLIRcamerameasurement_Contrast with beam splitter 2mW_level1500.h5'

dataset_path = '/measurement/FLIRcamerameasurement/t0/c0/image'  # Chemin vers le jeu de données contenant l'image

with h5py.File(file_path, 'r') as f:
    try:
        image_data = f[dataset_path][:]
    except KeyError:
        raise KeyError(f"Le jeu de données '{dataset_path}' n'existe pas dans le fichier H5.")

viewer = napari.Viewer()
viewer.add_image(image_data)

napari.run()



layers = image_data.shape[0]
height = image_data.shape[1]
width = image_data.shape[2]


print(layers,height,width)

print(image_data[0])

for i in range (layers):
    if (i==0):
        intensity = image_data[0]
    else :
        intensity += image_data[i]
        
print(intensity)

print(intensity/layers)

print(intensity.shape[0],intensity.shape[1])