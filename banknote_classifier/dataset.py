import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AugmentationStrategy(ABC):
    """Abstract Strategy interface for input data preprocessing."""
    @abstractmethod
    def build_pipeline(self) -> transforms.Compose:
        pass

class BaseValidationTransform(AugmentationStrategy):
    """Standard ImageNet scaling for validation and inference phases."""
    def build_pipeline(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class RegularTrainAugmentation(AugmentationStrategy):
    """Data augmentation technique to regularize baseline models during training."""
    def build_pipeline(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def fetch_image_manifest(target_dir: str, labels_map: dict[str, int]) -> tuple[list[str], list[int]]:
    """Scan file directories dynamically to build image path array and encoded integer targets."""
    file_paths, corresponding_labels = [], []
    for root, directories, _ in os.walk(target_dir):
        for folder in directories:
            if folder in labels_map:
                folder_path = os.path.join(root, folder)
                valid_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))]
                file_paths.extend(valid_images)
                corresponding_labels.extend([labels_map[folder]] * len(valid_images))
    return file_paths, corresponding_labels

class BanknoteDataset(Dataset):
    """Custom PyTorch dataset handling both training and explicit testing pipelines."""
    def __init__(self, paths: list[str], targets: list[int] = None, policy: AugmentationStrategy = None) -> None:
        self.paths = paths
        self.targets = targets
        self.transform = policy.build_pipeline() if policy else None

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        if self.targets is not None:
            return img, self.targets[idx]
        return img

    def __len__(self) -> int:
        return len(self.paths)

def invert_normalization(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert standard tensor maps back to displayable RGB channels."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    raw_array = img_tensor.cpu().numpy().transpose((1, 2, 0))
    raw_array = std * raw_array + mean
    return np.clip(raw_array, 0.0, 1.0)