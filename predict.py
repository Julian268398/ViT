import torch
from torchvision import transforms
from PIL import Image
from main import VisionTransformer

# Funkcja walidacji rozmiaru
def validate_image_size(image, target_size=(512, 512)):
    if image.size != target_size:
        print(f"Warning: Resizing image from {image.size} to {target_size}")
        return image.resize(target_size, Image.BICUBIC)
    return image

MODEL_PATH = "model_weights.pth"
IMAGE_PATH = "24.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacje obrazu
transform = transforms.Compose([
    transforms.Lambda(lambda img: validate_image_size(img)),  # Dodana walidacja rozmiaru
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Wczytaj model
model = VisionTransformer(image_size=512, n_classes=2)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# Wczytaj obraz
image = Image.open(IMAGE_PATH)
image = validate_image_size(image)  # Walidacja przed transformacją
image = transform(image).unsqueeze(0).to(DEVICE)

# Predykcja
with torch.no_grad():
    outputs = model(image)
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

    print(f"Prediction: {'Cancer' if pred == 1 else 'Healthy'} (Confidence: {probs[0][pred]:.4f})")
