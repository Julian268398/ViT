import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from main import VisionTransformer
from PIL import Image

# Funkcja walidacji rozmiaru
def validate_image_size(image, target_size=(256, 256)):
    if image.size != target_size:
        print(f"Warning: Resizing image from {image.size} to {target_size}")
        return image.resize(target_size, Image.BICUBIC)
    return image

# Konfiguracja
DATA_DIR = "C:\\Users\\julia\\Desktop\\praca inżynierska\\photos"  # Zmień na ścieżkę do swojego folderu TEST
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Przygotowanie danych
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: validate_image_size(img)),  # Dodana walidacja rozmiaru
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_dataset = ImageFolder(root=f"{DATA_DIR}/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Załaduj model
model = VisionTransformer(image_size=256, n_classes=2).to(DEVICE)
model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))
model.eval()

# Testowanie
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Obliczanie miar
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')  # Zakłada binary classification (2 klasy)
recall = recall_score(y_true, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Szczegółowy raport
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
