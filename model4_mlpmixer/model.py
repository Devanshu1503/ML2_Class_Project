import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


class MlpBlock(nn.Module):
    """Linear → GELU → Dropout → Linear → Dropout"""
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MixerLayer(nn.Module):
    """
    One Mixer layer: token-mixing MLP then channel-mixing MLP, both with
    LayerNorm pre-normalisation and residual connections.
    """
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1       = nn.LayerNorm(hidden_dim)
        self.norm2       = nn.LayerNorm(hidden_dim)
        self.token_mlp   = MlpBlock(num_patches,  tokens_mlp_dim,   dropout)
        self.channel_mlp = MlpBlock(hidden_dim,   channels_mlp_dim, dropout)

    def forward(self, x):
        # x: (B, num_patches, hidden_dim)

        # token mixing — MLP operates across the patch dimension
        y = self.norm1(x).transpose(1, 2)    # (B, hidden_dim, num_patches)
        x = x + self.token_mlp(y).transpose(1, 2)

        # channel mixing — MLP operates across the embedding dimension
        x = x + self.channel_mlp(self.norm2(x))
        return x


class PillMixer(nn.Module):
    """
    MLP-Mixer for pill classification (Tolstikhin et al., 2021).

    Pipeline:
      1. Split image into non-overlapping patches
      2. Project each patch linearly to hidden_dim  (patch embedding)
      3. Pass through N Mixer layers
      4. Global average pool across patches
      5. Linear classifier
    """
    def __init__(
        self,
        num_classes=cfg.NUM_CLASSES,
        image_size=cfg.IMAGE_SIZE,
        patch_size=cfg.PATCH_SIZE,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_MIXER_LAYERS,
        tokens_mlp_dim=cfg.TOKENS_MLP_DIM,
        channels_mlp_dim=cfg.CHANNELS_MLP_DIM,
        dropout_rate=cfg.DROPOUT_RATE,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        num_patches     = (image_size // patch_size) ** 2
        patch_dim       = patch_size * patch_size * 3

        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        self.mixer_layers = nn.Sequential(*[
            MixerLayer(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.norm       = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # split image into patches: (B, num_patches, patch_dim)
        x = x.unfold(2, P, P).unfold(3, P, P)           # (B, C, nH, nW, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * P * P)

        x = self.patch_embed(x)     # (B, num_patches, hidden_dim)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)           # global average pool over patches
        return self.classifier(x)   # raw logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model     = PillMixer(num_classes=cfg.NUM_CLASSES)
    dummy     = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    out       = model(dummy)
    n_patches = (cfg.IMAGE_SIZE // cfg.PATCH_SIZE) ** 2
    print("output shape :", out.shape)
    print("num patches  :", n_patches)
    print("parameters   :", count_parameters(model))
