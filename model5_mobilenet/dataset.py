import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import IMAGE_DIR, CLASS_MAP_PATH, IMAGE_SIZE


def get_transforms(split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return T.Compose([
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            T.RandomCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # must be after ToTensor
        ])
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class PillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        self.df        = df.reset_index(drop=True)
        self.transform = get_transforms(split)
        with open(CLASS_MAP_PATH) as f:
            self.class_map = json.load(f)   # {str(label_code_id): int_index}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(IMAGE_DIR + "/" + str(row["image_path"])).convert("RGB")
        label = self.class_map[str(row["label_code_id"])]
        return self.transform(img), label
