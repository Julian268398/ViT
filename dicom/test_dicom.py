import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from main_dicom import VisionTransformer  # Importujemy model

# Konfiguracja
DATA_DIR = "C:\\Users\\julia\\Desktop\\VIT"
MODEL_PATH = "../model_dicom.pth"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Funkcja do ładowania danych testowych
def get_test_loader(data_dir, batch_size):
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(root=f"{data_dir}/testing", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


# Załaduj dane testowe
test_loader = get_test_loader(DATA_DIR, BATCH_SIZE)

# Załaduj model
model = VisionTransformer(img_size=512, in_chans=1, n_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Testowanie
all_labels = []
all_preds = []

print("Starting testing...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # Pobranie przewidywań

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Obliczanie metryk
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)

# Wyniki
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall (Sensitivity): {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
