# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:57:16 2023

@author: Maxime Meghnagi
"""

import napari
from qtpy.QtWidgets import QWidget, QFormLayout, QPushButton, QFileDialog
import h5py



class H5Widget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.create_ui()

    def create_ui(self):
        layout = QFormLayout()
        self.setLayout(layout)
        self.select_button = QPushButton('Select file')
        self.select_button.clicked.connect(self.handle_file_selection)
        layout.addRow(self.select_button)

    def handle_file_selection(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self)
        if file_path:
            print(file_path)
            dataset_path = '/measurement/FLIRcamerameasurement/t0/c0/image'
            with h5py.File(file_path, 'r') as f:
                image_data = f[dataset_path][:]
            self.viewer.add_image(image_data)


if __name__ == '__main__':
    viewer = napari.Viewer()
    mywidget = H5Widget(viewer)
    mywidget.setMinimumSize(225, 200)
    viewer.window.add_dock_widget(mywidget, name='H5 import')
    napari.run()
