# smoke_test_train.py
# runs a mini version of the full training loop to confirm nothing is broken
# uses 2 classes, 2 epochs, small batches -- does NOT touch config.py or any saved files

import sys
import time
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# --- patch config in memory only, nothing on disk changes ---
cfg.NUM_CLASSES = 2
cfg.NUM_EPOCHS  = 2
cfg.BATCH_SIZE  = 8

print("=" * 55)
print("  SMOKE TEST -- 2 classes, 2 epochs, batch size 8")
print("=" * 55)
print()

# --- load filtered metadata and slice to 2 classes ---
df = pd.read_csv(cfg.FILTERED_METADATA_CSV)
with open(cfg.CLASS_MAP_JSON) as f:
    full_class_map = json.load(f)

df[cfg.COL_LABEL] = df[cfg.COL_LABEL].astype(str)
counts      = df.groupby(cfg.COL_LABEL).size().sort_values(ascending=False)
two_classes = counts.index[:2].tolist()
df_small    = df[df[cfg.COL_LABEL].isin(two_classes)].reset_index(drop=True)

smoke_class_map = {cls: idx for idx, cls in enumerate(sorted(two_classes))}
print("smoke classes: %s" % smoke_class_map)
print("total rows   : %d" % len(df_small))
print()

# --- split ---
train_df, val_df = train_test_split(
    df_small,
    test_size=0.20,
    stratify=df_small[cfg.COL_LABEL],
    random_state=cfg.RANDOM_SEED,
)
print("train=%d  val=%d" % (len(train_df), len(val_df)))
print()

# --- build datasets ---
from dataset import PillDataset, get_train_transform, get_eval_transform

train_ds = PillDataset(train_df, smoke_class_map, transform=get_train_transform())
val_ds   = PillDataset(val_df,   smoke_class_map, transform=get_eval_transform())

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

# --- build MLP-Mixer with 2 output classes ---
from model import PillMixer, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = PillMixer(num_classes=2).to(device)
print("model device : %s" % device)
print("parameters   : %d" % count_parameters(model))
n_patches = (cfg.IMAGE_SIZE // cfg.PATCH_SIZE) ** 2
print("patches/img  : %d" % n_patches)
print()

class_weights = train_ds.get_class_weights().to(device)
print("class weights: %s" % class_weights.tolist())
print()

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

# --- mini training loop ---
print("%-6s  %-10s  %-9s  %-9s  %-8s  %-6s" % (
    "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Time"
))
print("-" * 55)

for epoch in range(1, cfg.NUM_EPOCHS + 1):
    t0 = time.time()

    # train
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)
    train_loss = total_loss / total
    train_acc  = correct / total

    # val
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_correct  += (logits.argmax(1) == labels).sum().item()
            val_total    += images.size(0)
    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total

    print("%6d  %10.4f  %9.4f  %9.4f  %8.4f  %5.1fs" % (
        epoch, train_loss, train_acc, val_loss, val_acc, time.time() - t0
    ))

print()
print("smoke test passed -- no errors, training loop works end to end")
print("config.py is unchanged (2-class / 2-epoch override was in memory only)")
print()
print("original settings still in effect:")
import importlib
import config as cfg2
importlib.reload(cfg2)
print("  NUM_CLASSES      = %d" % cfg2.NUM_CLASSES)
print("  NUM_EPOCHS       = %d" % cfg2.NUM_EPOCHS)
print("  NUM_MIXER_LAYERS = %d" % cfg2.NUM_MIXER_LAYERS)
print("  PATCH_SIZE       = %d" % cfg2.PATCH_SIZE)
