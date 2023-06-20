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