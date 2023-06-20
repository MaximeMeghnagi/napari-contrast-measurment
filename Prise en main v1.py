# -*- coding: utf-8 -*-
"""
Created on Sat Jun  10 10:23:54 2023

@author: Maxime Meghnagi
"""

import napari
import numpy as np
from skimage import data

def create_rectangle(center, sy, sx, color, name):
    cz=center[0]
    cy=center[1]
    cx=center[2]
    hsx = sx//2
    hsy = sy//2
    rectangle = [ [cz, cy+hsy, cx-hsx], # up-left
                  [cz, cy+hsy, cx+hsx], # up-right
                  [cz, cy-hsy, cx+hsx], # down-right
                  [cz, cy-hsy, cx-hsx]  # down-left
                  ]
    
    return rectangle
    
def create_line(start, end, color, name):
    line = [start, end]
    return line
        
viewer = napari.view_image(data.astronaut(), rgb=True)


rectangle= create_rectangle([0,100,100], 50, 70, 'red', 'test') 
rectangle1= create_rectangle([0,200,100], 50, 70, 'red', 'test') 
rectangle2= create_rectangle([0,100,300], 50, 70, 'red', 'test') 

name = 'test'

viewer.add_shapes([np.array(rectangle)],
                          edge_width=2,
                          edge_color='red',
                          face_color=[1,1,1,0],
                          name = name
                          )

viewer.layers[name].add_rectangles(np.array([rectangle,rectangle1,rectangle2]), edge_color=np.array(['green','yellow','blue']))  


start_point = [50, 50]  # Coordonnées du point de départ de la ligne
end_point = [200, 200]  # Coordonnées du point d'arrivée de la ligne
line_color = 'pink'  # Couleur de la ligne
line_name = 'line'  # Nom de la ligne

line = create_line(start_point, end_point, line_color, line_name)

viewer.add_shapes([np.array(line)],
                  shape_type='line',
                  edge_width=2,
                  edge_color=line_color,
                  name=line_name)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    