import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import ImageFolder
from main import VisionTransformer

# Konfiguracja
DATA_DIR = "ścieżka do folderu zawierającego przygotowane dane w folderach traningowych, walidacyjnych oraz testowych"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")  # służy weryfikacji podczas uruchomienia programu na różnych urządzeniach

# Transformacje dla obrazów .jpg
# W przypadku plików dcm wszystkie miały jednakowy rozmiar 512x512
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Zmiana rozmiaru na 512x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizacja dla obrazów RGB
])

# Ładowanie danych z ImageFolder
train_dataset = ImageFolder(root=f"{DATA_DIR}/train", transform=transform)
val_dataset = ImageFolder(root=f"{DATA_DIR}/valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#  Inicjalizacja Modelu
model = VisionTransformer(img_size=512, in_chans=3, n_classes=2).to(DEVICE)

# Optymalizator i funkcja kosztu
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)
criterion = CrossEntropyLoss()

# Early Stopping Configuration
patience = 5
patience_counter = 0
best_val_loss = float('inf')  # Ustawiony jako nieskończoność

# Trening
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # Przenoszenie obrazów i etykiet na urządzenie

        optimizer.zero_grad()  # Zeruje gradienty dla wszystkich parametrów, w celu zapobiegania kumulowaniu się ich,
        # co powodowałoby błędy w zmianach parametrów modelu
        outputs = model(images)
        loss = criterion(outputs, labels)  # Obliczanie wartości funkcji strat
        loss.backward()  # Obliczanie gradientu funkcji strat względem parametrów modelu za pomocą wstecznej propagacji
        optimizer.step()  # Aktualizacja parametrów modelu zgodnie z obliczonymi gradientami i algorytmem optymalizacji

        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():  # Wyłączenie obliczania gradientów, ponieważ w fazie walidacji gradienty są zbędne
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)  # Normalizacja całkowitej straty treningowej przez liczbę partii danych
    # w zbiorze treningowym, aby uzyskać średnią stratę.
    val_loss /= len(val_loader)  # Analogicznie ze stratą walidacyjną

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # mechanizm wczesnego zatrzymania
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Zapisywana jest najlepsza wersja modelu w razie wczesnego zatrzymania
        torch.save(model.state_dict(), "model_jpg.pth")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break
