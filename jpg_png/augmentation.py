import os
import shutil
import Augmentor

# Ścieżki
before = "C:\\Users\\julia\\Desktop\\jpg_png\\validating\\healthy"
after = "C:\\Users\\julia\\Desktop\\agu\\valid\\healthy"

# Tworzenie folderu docelowego, jeśli nie istnieje
os.makedirs(after, exist_ok=True)

# Inicjalizacja pipeline
p = Augmentor.Pipeline(source_directory=before)  # Dodanie całego folderu do pipeline
p.flip_left_right(0.5)
p.rotate(0.7, 15, 15)
p.skew(0.4, 0.1)
p.zoom(0.8, 0.85, 1.1)
p.random_contrast(0.5, 0.5, 1.5)

# Generowanie 2 nowych obrazów dla każdego pliku wejściowego
p.sample(len(os.listdir(before)) * 3)  # Liczba augmentacji = 3 * liczba obrazów

# Przeniesienie wygenerowanych obrazów do folderu `after`
for generated_file in os.listdir(p.output_directory):
    shutil.move(os.path.join(p.output_directory, generated_file), after)

# Usuwanie folderu `output` utworzonego przez Augmentor (opcjonalne)
if os.path.exists(p.output_directory):
    shutil.rmtree(p.output_directory)

print("Augmentacja zakończona.")

