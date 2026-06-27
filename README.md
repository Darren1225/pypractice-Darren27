# Bangla Banknote Recognition (VGG-16)

This project implements a deep learning pipeline using a custom VGG-16 architecture to recognize and classify Bangladeshi banknotes (denominations: 1, 2, 5, 10, 20, 50, 100, 500, 1000).

## Requirements
- Python 3.12+
- Compatible hardware (NVIDIA GPU, AMD GPU, Apple Silicon, or CPU)

## Hardware & OS Compatibility
This codebase is designed for high portability and cross-platform compatibility, automatically selecting the optimal hardware acceleration backend:
- **NVIDIA GPU (CUDA)**: Automatically used if CUDA drivers and a compatible GPU are detected.
- **AMD GPU (ROCm)**: Supported on Linux via ROCm-enabled PyTorch (maps CUDA calls to AMD HIP backend automatically).
- **Apple Silicon (M1/M2/M3 Mac)**: Automatically utilizes Metal Performance Shaders (MPS) for local GPU acceleration.
- **CPU Mode**: Safely falls back to CPU execution if no GPU backend or drivers are available.
- **Cross-OS Support**: Optimized for Windows, Linux, and macOS. The multiprocessing startup method is dynamically adjusted to prevent errors and crashes on Windows.

## Installation & Setup

Follow these steps to set up the virtual environment and install the required dependencies:

### 1. Create a Virtual Environment
In the project root directory, run:
```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment
- **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies
Ensure you have activated the virtual environment, then run:
```bash
pip install -r requirements.txt
```

---

## Running the Project

To start training the VGG-16 model and run test inference:
```bash
python main.py
```

### What happens when you run `main.py`?
1. **Dataset Auto-download**: The script checks if the Bangla banknote dataset is present in `./data/`. If missing, it automatically clones the **Bangla-Money-Dataset** from GitHub.
2. **Model Training**: The custom VGG-16 model is trained for 15 epochs with Cosine Annealing learning rate scheduling.
3. **Save Model Weights**: The model saves the state dict with the lowest validation loss to `./Saving_Path/model_weight.pth`.
4. **Learning Curves**: Saves training/validation loss and accuracy curves to `./Saving_Path/loss_accuracy_curves.png`.
5. **Test Inference**: Loads the best model weights, predicts classes for the test set, calculates and prints test set accuracy, and saves predictions to `./data/example.csv`.
