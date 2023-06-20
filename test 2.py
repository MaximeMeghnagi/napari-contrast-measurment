import napari
from qtpy.QtWidgets import QComboBox, QWidget, QVBoxLayout, QApplication, QPushButton

class ChoicesWidget(QWidget):
    def __init__(self, choices):
        super().__init__()
        self.layout = QVBoxLayout()
        self.combobox = QComboBox()
        self.choices = choices
        self.layout.addWidget(self.combobox)

        # Créer un bouton
        self.button = QPushButton('Activer/Désactiver choix n°4')
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

        # Connecter le signal-événement 'clicked' du bouton à la fonction de mise à jour
        self.button.clicked.connect(self.toggle_choice)

    def toggle_choice(self):
        if self.button.isChecked():
            self.combobox.addItem('Choix n°4')
            self.choices.append('Choix n°4')
        else:
            self.combobox.removeItem(self.choices.index('Choix n°4'))
            self.choices.remove('Choix n°4')

# Créer une liste de choix initiale
choices = ["Option 1", "Option 2", "Option 3"]

# Créer une instance de l'application Napari
with napari.gui_qt():
    viewer = napari.Viewer()

    # Créer une instance de ChoicesWidget avec la liste de choix initiale
    widget = ChoicesWidget(choices)

    # Ajouter le widget contenant la liste de choix à Napari
    viewer.window.add_dock_widget(widget, area='right')