import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Clean imports because of Src Layout & Editable Install
from banknote_classifier import (
    BanknoteDataset,
    BaseValidationTransform,
    RegularTrainAugmentation,
    fetch_image_manifest,
    net_registry
)

class AppConfig:
    EPOCHS = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 0
    DATA_ROOT = "./data"
    TRAIN_DIR = "./data/bangla/Training"
    LABELS = {"1": 0, "2": 1, "5": 2, "10": 3, "20": 4, "50": 5, "100": 6, "500": 7, "1000": 8}

def main():
    # 1. Hardware Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training Pipeline on {device}...")

    # 2. Prepare Strategies & Data
    train_aug = RegularTrainAugmentation()
    valid_aug = BaseValidationTransform()

    if not os.path.exists(AppConfig.TRAIN_DIR):
        print("⚠️ Data directory not found! Please ensure Bangla-Money-Dataset is downloaded into ./data/bangla")
        return

    all_paths, all_labels = fetch_image_manifest(AppConfig.TRAIN_DIR, AppConfig.LABELS)
    train_paths, val_paths, train_lbls, val_lbls = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42
    )

    # 3. Inject Strategies into Datasets
    train_loader = DataLoader(
        BanknoteDataset(train_paths, train_lbls, policy=train_aug),
        batch_size=AppConfig.BATCH_SIZE, shuffle=True, num_workers=AppConfig.NUM_WORKERS
    )
    val_loader = DataLoader(
        BanknoteDataset(val_paths, val_lbls, policy=valid_aug),
        batch_size=AppConfig.BATCH_SIZE, shuffle=False, num_workers=AppConfig.NUM_WORKERS
    )

    # 4. Instantiate Model via Registry
    model = net_registry.getattr("vgg16_classifier")(target_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=AppConfig.LEARNING_RATE)

    # 5. Quick Verification Loop (1 Epoch)
    model.train()
    print("\n--- Testing 1 Epoch (CPU verification) ---")
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        
    print("\n✅ Verification Complete! Architecture and pipeline are fully functional.")

if __name__ == "__main__":
    if os.name != "nt":
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()