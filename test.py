import napari
from napari.layers import Image, Shapes
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QComboBox, QApplication
import sys


class ImageShapeSelector(QWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.layout = QVBoxLayout()

        self.image_label = QLabel('Select Image:')
        self.layout.addWidget(self.image_label)

        self.image_combobox = QComboBox()
        self.layout.addWidget(self.image_combobox)

        self.shape_label = QLabel('Select Shape:')
        self.layout.addWidget(self.shape_label)

        self.shape_combobox = QComboBox()
        self.layout.addWidget(self.shape_combobox)

        self.setLayout(self.layout)

        self.populate_comboboxes()
        self.image_combobox.currentIndexChanged.connect(self.image_selected)
        self.shape_combobox.currentIndexChanged.connect(self.shape_selected)

    def populate_comboboxes(self):
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        shape_layers = [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]

        self.image_combobox.clear()
        self.shape_combobox.clear()

        for layer in image_layers:
            self.image_combobox.addItem(layer.name)

        for layer in shape_layers:
            self.shape_combobox.addItem(layer.name)

    def image_selected(self, index):
        if index >= 0:
            self.viewer.layers.unselect_all()
            selected_layer = self.viewer.layers[index]
            selected_layer.selected = True

    def shape_selected(self, index):
        if index >= 0:
            self.viewer.layers.unselect_all()
            selected_layer = self.viewer.layers[index]
            selected_layer.selected = True

    def update_comboboxes(self):
        self.populate_comboboxes()


def main():
    with napari.gui_qt():
        viewer = napari.Viewer()
        selector = ImageShapeSelector(viewer)
        viewer.window.add_dock_widget(selector, area='right')

        # Ajouter des images et des formes à la visionneuse pour tester
        viewer.add_image(data=[[1, 2, 3], [4, 5, 6]])
        viewer.add_shapes([[100, 100, 50], [200, 200, 100]])

        # Mettre à jour les menus déroulants lorsque des changements sont effectués dans la visionneuse
        def on_layer_change(event):
            selector.update_comboboxes()

        viewer.layers.events.changed.connect(on_layer_change)

        viewer.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()
    sys.exit(app.exec_())
