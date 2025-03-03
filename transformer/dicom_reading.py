import numpy as np
import os

import pydicom
import scipy.ndimage
"""
Zmodyfikowany kod ze strony "https://www.kaggle.com/code/neuerfrhling/read-ct-dicom", dotyczący odczytywania plików dicom skanów tomografii komputerowej.
Zaimplementowany w celu zapoznania się ze strukturą plików dcm.
"""

#Funkcja do ładowania skanów jako tablice numpy
def load_single_dicom(slice_path):
    slice_data = pydicom.dcmread(slice_path)  # Wczytujemy wybrany plik DICOM
    pixel_array = slice_data.pixel_array  # Dane pikselowe (macierz 2D)

    # Pobieranie metadanych
    orientation = np.transpose(slice_data.ImageOrientationPatient)  # Orientacja obrazu
    spacing_xy = np.array(slice_data.PixelSpacing, dtype=float)  # Odstępy między pikselami w X i Y
    spacing_z = float(slice_data.SliceThickness)  # Grubość cięcia w osi Z
    spacing = np.array([spacing_z, spacing_xy[1], spacing_xy[0]])  # Przestrzeń w formacie ZYX

    origin = slice_data.ImagePositionPatient  # Pozycja obrazu w przestrzeni
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.array([origin[2], origin[1], origin[0]])  # Konwersja do Z, Y, X

    return pixel_array, spacing, orientation, origin, slice_data

slice_path = 'tutaj umieszczona była ścieżka do pojedynczego pliku dcm'

# Załaduj jedno cięcie
pixel_array, spacing, orientation, origin, slice_data = load_single_dicom(slice_path)

# Wyświetl szczegóły
print("Wymiary obrazu (y, x):", pixel_array.shape)
print("Spacing (Z, Y, X):", spacing)
print("Orientacja obrazu:", orientation)
print("Pozycja początkowa (origin):", origin)
print("Metadane cięcia:", slice_data)
