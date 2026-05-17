import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import *
from model import build_mobilenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)
idx_to_class = {v: k for k, v in class_map.items()}

test_df = pd.read_csv(f"{DATA_DIR}/test_split_mobilenet.csv")
model   = build_mobilenet(NUM_CLASSES, freeze_backbone=False).to(device)
model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))
model.eval()

# Last conv block before adaptive pooling
target_layer = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layer)

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

os.makedirs(f"{OUTPUT_DIR}/gradcam", exist_ok=True)
samples = test_df.sample(GRADCAM_SAMPLES, random_state=42)

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = f"{IMAGE_DIR}/{row['image_path']}"
    true_cls = class_map[str(row["label_code_id"])]
    raw_img  = np.array(Image.open(img_path).convert("RGB").resize((224, 224))) / 255.0
    input_t  = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    pred_cls  = model(input_t).argmax(1).item()
    correct   = "CORRECT" if pred_cls == true_cls else "WRONG"
    grayscale = cam(input_tensor=input_t, targets=[ClassifierOutputTarget(pred_cls)])
    overlay   = show_cam_on_image(raw_img.astype(np.float32), grayscale[0], use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"True: {idx_to_class[true_cls]}")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Pred: {idx_to_class[pred_cls]} [{correct}]")
    axes[1].axis("off")
    plt.savefig(f"{OUTPUT_DIR}/gradcam/sample_{i}.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"GradCAM images saved to {OUTPUT_DIR}/gradcam/")
