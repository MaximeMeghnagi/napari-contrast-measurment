import napari
from qtpy.QtWidgets import QWidget, QPushButton, QDoubleSpinBox, QComboBox, QVBoxLayout, QFormLayout, QMessageBox
from napari.layers import Image, Labels, Shapes
from napari.utils.events import Event
import numpy as np


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


class ContrastWidget(QWidget):

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        super().__init__()
        self.create_ui()

        # Connecter le signal pour mettre à jour les options des combobox
        self.viewer.layers.events.changed.connect(self.update_combobox_options)

    def create_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        pixel_width_box = QDoubleSpinBox()
        pixel_width_box.setDecimals(4)
        pixel_width_box.setValue(0.0586)
        pixel_width_box.setSingleStep(0.0001)
        pixel_width_layout = QFormLayout()
        pixel_width_layout.addRow('Pixel width (µm)', pixel_width_box)
        layout.addLayout(pixel_width_layout)
        self.pixel_width_box = pixel_width_box
        cam_noise_box = QDoubleSpinBox()
        cam_noise_box.setMaximum(999999)
        cam_noise_box.setDecimals(0)
        cam_noise_box.setValue(640)
        cam_noise_box.setSingleStep(1)
        cam_noise_layout = QFormLayout()
        cam_noise_layout.addRow('Camera noise', cam_noise_box)
        layout.addLayout(cam_noise_layout)
        self.cam_noise_box = cam_noise_box
        self.shapes_combo = QComboBox()
        self.image_combo = QComboBox()
        layout.addWidget(self.shapes_combo)
        layout.addWidget(self.image_combo)
        calculate_btn = QPushButton('Calculate contrast')
        calculate_btn.clicked.connect(lambda: self.calculate_contrast(
            self.cam_noise_box.value(), self.pixel_width_box.value()))
        layout.addWidget(calculate_btn)

        # Mettre à jour les options des combobox au démarrage
        self.update_combobox_options()

    def update_combobox_options(self, event=None):
        # Récupérer la liste des couches actuelles
        layers = self.viewer.layers

        # Récupérer les noms des couches d'image et de forme
        image_names = [layer.name for layer in layers if isinstance(layer, Image)]
        shape_names = [layer.name for layer in layers if isinstance(layer, Shapes)]

        # Récupérer les options actuellement sélectionnées
        current_image = self.image_combo.currentText()
        current_shape = self.shapes_combo.currentText()

        # Mettre à jour les options des combobox
        self.image_combo.clear()
        self.image_combo.addItems(image_names)
        self.shapes_combo.clear()
        self.shapes_combo.addItems(shape_names)

        # Rétablir les options sélectionnées précédemment
        self.image_combo.setCurrentText(current_image)
        self.shapes_combo.setCurrentText(current_shape)

    def calculate_contrast(self, camera_noise, pixel_width):
        selected_shape = self.shapes_combo.currentText()
        selected_image = self.image_combo.currentText()

        image_layer = None
        shape_layer = None

        # Trouver les couches correspondantes dans Napari
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.name == selected_image:
                image_layer = layer
            elif isinstance(layer, Shapes) and layer.name == selected_shape:
                shape_layer = layer

        # Calculer le contraste en utilisant les couches trouvées
        if image_layer is not None and shape_layer is not None:
            # Effectuer le calcul du contraste ici
            contrast = calculate_contrast(image_layer.data, shape_layer.data, camera_noise, pixel_width)
            QMessageBox.information(self, 'Contrast', f'The calculated contrast is: {contrast}')
        else:
            QMessageBox.warning(self, 'Error', 'Please select an image and a shape layer')

        return Labels()


viewer = napari.Viewer()
mywidget = ContrastWidget(viewer)
viewer.window.add_dock_widget(mywidget)
napari.run()
