import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import PillDataset
from model import build_mobilenet, unfreeze_backbone


# ── Mixup helpers ──────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Data ───────────────────────────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILTERED_CSV)
with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

labels = [class_map[str(r)] for r in df["label_code_id"]]

train_df, temp_df, y_train, y_temp = train_test_split(
    df, labels, test_size=(1 - TRAIN_SPLIT), random_state=RANDOM_SEED, stratify=labels)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

train_ds     = PillDataset(train_df, "train")
val_ds       = PillDataset(val_df,   "val")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
class_weights = torch.tensor(cw, dtype=torch.float)

# ── Model, loss, optimizer ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model     = build_mobilenet(NUM_CLASSES, freeze_backbone=True).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS, pct_start=0.1)

# ── Training loop ──────────────────────────────────────────────────────────
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    if epoch == UNFREEZE_EPOCH:
        print(f"\n[Epoch {epoch}] Unfreezing backbone for full fine-tuning...")
        unfreeze_backbone(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE / 20, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE / 20,
            steps_per_epoch=len(train_loader),
            epochs=NUM_EPOCHS - epoch + 1, pct_start=0.05)

    # Train with Mixup
    model.train()
    t_loss, t_correct = 0.0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        imgs_m, ya, yb, lam = mixup_data(imgs, lbls, MIXUP_ALPHA)
        optimizer.zero_grad()
        out  = model(imgs_m)
        loss = mixup_criterion(criterion, out, ya, yb, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += (out.argmax(1) == lbls).sum().item()

    # Validate (no Mixup)
    model.eval()
    v_loss, v_correct = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (out.argmax(1) == lbls).sum().item()

    t_acc = t_correct / len(train_ds)
    v_acc = v_correct / len(val_ds)
    history["train_loss"].append(t_loss / len(train_ds))
    history["val_loss"].append(v_loss / len(val_ds))
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
        print(f"  -> Saved best model (val_acc={v_acc:.4f})")

# Save test split for evaluate.py
test_df.to_csv(f"{DATA_DIR}/test_split_mobilenet.csv", index=False)

# Learning curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"], label="Train"); ax2.plot(history["val_acc"], label="Val")
ax2.set_title("Accuracy"); ax2.legend()
plt.savefig(f"{OUTPUT_DIR}/learning_curves.png", dpi=150, bbox_inches="tight")
print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print("Training complete.")
