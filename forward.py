from PIL import Image
import numpy as np
import torch

k = 10

# Wczytanie etykiet z obsługą błędów
with open("ml.txt") as f:
    imagenet_labels = dict(enumerate(f))

model = torch.load("model.pth")
model.eval()

img = (np.array(Image.open("skarpeta.jpg")) / 128) - 1
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels.get(ix, "Unknown").strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")