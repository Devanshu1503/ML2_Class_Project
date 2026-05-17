# patch_importance.py
# visualises which patches most influenced the MLP-Mixer's prediction.
# uses input-gradient attribution: backprop the predicted class score and
# show the L2 norm of the gradient per patch position.

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from dataset import build_datasets
from model import PillMixer


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError("no checkpoint -- run train.py first")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = PillMixer(
        num_classes=     ckpt.get("num_classes",      cfg.NUM_CLASSES),
        patch_size=      ckpt.get("patch_size",       cfg.PATCH_SIZE),
        hidden_dim=      ckpt.get("hidden_dim",       cfg.HIDDEN_DIM),
        num_layers=      ckpt.get("num_layers",       cfg.NUM_MIXER_LAYERS),
        tokens_mlp_dim=  ckpt.get("tokens_mlp_dim",  cfg.TOKENS_MLP_DIM),
        channels_mlp_dim=ckpt.get("channels_mlp_dim",cfg.CHANNELS_MLP_DIM),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def patch_importance_map(model, image_tensor, device):
    P  = model.patch_size
    x  = image_tensor.unsqueeze(0).to(device)
    B, C, H, W = x.shape
    nH, nW = H // P, W // P

    patches = x.unfold(2, P, P).unfold(3, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(1, nH * nW, C * P * P)
    patches = patches.detach().requires_grad_(True)

    embedded = model.patch_embed(patches)
    mixed    = model.mixer_layers(embedded)
    normed   = model.norm(mixed)
    pooled   = normed.mean(dim=1)
    logits   = model.classifier(pooled)

    pred_class = logits.argmax(dim=1).item()
    logits[0, pred_class].backward()

    grad       = patches.grad.squeeze(0)                   # (num_patches, patch_dim)
    importance = grad.norm(dim=1).detach().cpu().numpy()   # (num_patches,)
    return importance.reshape(nH, nW), pred_class


def unnormalise(tensor):
    mean = np.array(cfg.IMAGENET_MEAN)
    std  = np.array(cfg.IMAGENET_STD)
    img  = tensor.permute(1, 2, 0).numpy()
    return np.clip(img * std + mean, 0, 1)


def visualise_sample(model, image_tensor, label, class_names, out_path):
    device    = next(model.parameters()).device
    grid, pred = patch_importance_map(model, image_tensor, device)
    P          = model.patch_size

    img_np   = unnormalise(image_tensor)
    imp_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    heatmap  = np.kron(imp_norm, np.ones((P, P)))   # upsample to full image resolution

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(imp_norm, cmap="hot", interpolation="nearest")
    axes[1].set_title("patch importance grid")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap, cmap="hot", alpha=0.5, interpolation="bilinear")
    axes[2].set_title("overlay")
    axes[2].axis("off")

    true_name = class_names[label]
    pred_name = class_names[pred]
    correct   = "correct" if pred == label else "WRONG"
    fig.suptitle("true: %s  |  pred: %s  (%s)" % (true_name, pred_name, correct), fontsize=11)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved %s" % out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg.CLASS_MAP_JSON) as f:
        class_map = json.load(f)
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    model         = load_model(cfg.BEST_MODEL_PATH, device)
    _, _, test_ds = build_datasets()

    out_dir = cfg.OUTPUT_DIR / "patch_importance"
    for i in range(min(cfg.NUM_VIZ_SAMPLES, len(test_ds))):
        image_tensor, label = test_ds[i]
        visualise_sample(model, image_tensor, label, class_names, out_dir / ("sample_%02d.png" % i))

    print("\ndone -- images saved to %s" % out_dir)


if __name__ == "__main__":
    main()
