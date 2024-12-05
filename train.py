import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from main import VisionTransformer
from PIL import Image

# Funkcja walidacji rozmiaru
def validate_image_size(image, target_size=(256, 256)):
    if image.size != target_size:
        print(f"Warning: Resizing image from {image.size} to {target_size}")
        return image.resize(target_size, Image.BICUBIC)
    return image

# Konfiguracja
DATA_DIR = "C:\\Users\\julia\\Desktop\\praca inżynierska\\photos\\suma"  # Zmień na ścieżkę do swojego folderu
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Przygotowanie danych
transform = transforms.Compose([
    transforms.Lambda(lambda img: validate_image_size(img)),  # Dodana walidacja rozmiaru
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = ImageFolder(root=f"{DATA_DIR}/train", transform=transform)
val_dataset = ImageFolder(root=f"{DATA_DIR}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = VisionTransformer(image_size=256, n_classes=2).to(DEVICE)

# Optymalizator i funkcja kosztu
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()

# Trening
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "model_weights.pth")
