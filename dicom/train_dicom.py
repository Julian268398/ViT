import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from main_dicom import VisionTransformer, get_dataloaders  # Zaimportuj funkcję do ładowania danych

# Konfiguracja
DATA_DIR = "C:\\Users\\julia\\Desktop\\VIT"
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Ładowanie danych
train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

# Model
model = VisionTransformer(img_size=512, in_chans=1, n_classes=2).to(DEVICE)

# Optymalizator i funkcja kosztu
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)
criterion = CrossEntropyLoss()

# Early Stopping Configuration
patience = 5
patience_counter = 0
best_val_loss = float('inf')  # Ustawiamy jako nieskończoność

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

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), "../model_dicom.pth")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")

    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break
