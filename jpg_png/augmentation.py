import Augmentor

# Ścieżka do folderu z obrazami
folder_path = r"C:\Users\julia\Desktop\aaa"

# Inicjalizacja pipeline z folderem
p = Augmentor.Pipeline(folder_path)

# Definiowanie operacji augmentacji
p.flip_left_right(0.5)
p.rotate(0.7, 15, 15)
p.skew(0.4, 0.1)
p.zoom(0.8, 0.85, 1.1)
p.random_contrast(0.5, 0.5, 1.5)

# Generowanie 2 nowych obrazów dla każdego pliku wejściowego
p.sample(2)

print("Augmentacja zakończona.")