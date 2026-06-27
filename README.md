# Lab Instruction: Modern Python ML Engineering (uv, Project Layouts, and Design Patterns)

Welcome to the Python Machine Learning Engineering Lab. In this lab, you will learn how to transition from a single-file prototype (`main.py`) to a production-grade, highly structured, type-safe, and modular machine learning project.

---

## Project Baseline Overview

This baseline project implements a deep learning pipeline using a custom VGG-16 architecture to recognize and classify Bangladeshi banknotes (denominations: 1, 2, 5, 10, 20, 50, 100, 500, 1000).

### Hardware & OS Compatibility
This codebase is designed for high portability and cross-platform compatibility, automatically selecting the optimal hardware acceleration backend:
- **NVIDIA GPU (CUDA)**: Automatically used if CUDA drivers and a compatible GPU are detected.
- **AMD GPU (ROCm)**: Supported on Linux via ROCm-enabled PyTorch (maps CUDA calls to AMD HIP backend automatically).
- **Apple Silicon (M1/M2/M3 Mac)**: Automatically utilizes Metal Performance Shaders (MPS) for local GPU acceleration.
- **CPU Mode**: Safely falls back to CPU execution if no GPU backend or drivers are available.
- **Cross-OS Support**: Optimized for Windows, Linux, and macOS. The multiprocessing startup method is dynamically adjusted to prevent errors and crashes on Windows.

---

## Lab Tasks

You will learn and implement:
1. **Modern dependency management with `uv`** (Astral's fast Python toolchain).
2. **Flat Layout vs. Src Layout** architectural packaging standards.
3. **Registry Pattern & Strategy Pattern** for modular ML software design.
4. **Code Quality and Type Checking** with `ruff` and `mypy`.

---

## Part 1: Environment & Dependency Management with `uv`

In modern Python engineering, standard `pip` and virtualenv can be slow and hard to keep reproducible. We will use `uv`, an extremely fast Python packaging tool written in Rust.

### Step 1.1: Install `uv`
Install `uv` globally on your machine:
- **Linux/macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows**:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Alternatively (via `pip`)**:
  ```bash
  pip install uv
  ```

### Step 1.2: Initialize the Project and Manage Dependencies
We will use a `pyproject.toml` file to declare dependencies rather than a traditional `requirements.txt`:
1. Run `uv init` in the root of the project to create `pyproject.toml`.
2. Add project dependencies using `uv add`:
   ```bash
   uv add torch torchvision numpy pandas matplotlib scikit-learn tqdm
   ```
3. Add development dependencies (linters and type checkers):
   ```bash
   uv add --dev ruff mypy
   ```
4. Run `uv sync` to generate the `uv.lock` file and create the `.venv` virtual environment.
   - *Concept*: `pyproject.toml` defines abstract dependencies, whereas `uv.lock` locks the exact resolved dependency tree to guarantee absolute reproducibility.
5. Execute python scripts using `uv run`:
   ```bash
   uv run python main.py
   ```

---

## Part 2: Project Layouts (Flat Layout vs. Src Layout)

You will split the monolithic `main.py` into separate modules (`main.py`, `models.py`, `dataset.py`) and organize them into two standard Python layouts.

### Task 2A: Implement Flat Layout
Organize your project such that all code files are in the root directory:
```
pypractice-exercise/
├── main.py
├── models.py
├── dataset.py
├── pyproject.toml
├── uv.lock
└── data/
```
In this layout, modules can be imported directly via `import models` or `from dataset import CustomDataset`. 
- **Run the model** inside the Flat Layout using:
  ```bash
  uv run python main.py
  ```

### Task 2B: Migrate to Src Layout (Production Standard)
The Flat Layout suffers from "accidental imports" (modules in the root can be imported even when the package is not installed). Src Layout prevents this by forcing all source code into a `src/` directory.

Reorganize the project to look like this:
```
pypractice-exercise/
├── src/
│   └── banknote_classifier/
│       ├── __init__.py
│       ├── main.py
│       ├── models.py
│       └── dataset.py
├── pyproject.toml
├── uv.lock
└── data/
```
To run the project in Src Layout:
1. Declare your package in `pyproject.toml` under `[build-system]` and `[project]`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "banknote-classifier"
   version = "0.1.0"
   description = "Bangla Banknote Classifier"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       "torch",
       "torchvision",
       "numpy",
       "pandas",
       "matplotlib",
       "scikit-learn",
       "tqdm"
   ]
   ```
2. Install the package in editable mode:
   ```bash
   uv pip install -e .
   ```
3. Run the project using the package namespace:
   ```bash
   uv run python -m banknote_classifier.main
   ```

---

## Part 3: Design Patterns in Machine Learning Code

To make your codebase extensible and clean, you will refactor it using two common software design patterns.

### Task 3A: Implement the Registry Pattern
In large-scale ML systems, we want to configure model architectures dynamically via string names (e.g. from config files) without hardcoding imports.

1. Implement a decorator-based registration system in a new file `src/banknote_classifier/registry.py`:
   ```python
   import torch.nn as nn
   from typing import Dict, Type

   class ModelRegistry:
       def __init__(self) -> None:
           self._registry: Dict[str, Type[nn.Module]] = {}

       def register(self, name: str):
           def decorator(cls: Type[nn.Module]):
               self._registry[name] = cls
               return cls
           return decorator

       def get(self, name: str) -> Type[nn.Module]:
           if name not in self._registry:
               raise ValueError(f"Model '{name}' not found in registry.")
           return self._registry[name]

   MODEL_REGISTRY = ModelRegistry()
   ```
2. Register your `VGG16` class in `models.py`:
   ```python
   from banknote_classifier.registry import MODEL_REGISTRY

   @MODEL_REGISTRY.register("vgg16")
   class VGG16(nn.Module):
       # ...
   ```
3. Initialize the model in `main.py` dynamically using `MODEL_REGISTRY.get("vgg16")()`.

### Task 3B: Implement the Strategy Pattern
The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. We will use it to easily swap data preprocessing/augmentation strategies at runtime.

1. Create a base strategy class and concrete strategies in `src/banknote_classifier/dataset.py`:
   ```python
   from abc import ABC, abstractmethod
   from torchvision import transforms

   class PreprocessStrategy(ABC):
       @abstractmethod
       def get_transforms(self) -> transforms.Compose:
           pass

   class StandardPreprocess(PreprocessStrategy):
       def get_transforms(self) -> transforms.Compose:
           return transforms.Compose([
               transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           ])

   class HeavyAugmentation(PreprocessStrategy):
       def get_transforms(self) -> transforms.Compose:
           return transforms.Compose([
               transforms.Resize((224, 224)),
               transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(15),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           ])
   ```
2. Configure the dataset to accept a `PreprocessStrategy` strategy object.
3. Switch strategies in `main.py` by simply passing either `StandardPreprocess()` or `HeavyAugmentation()` to your dataset loader.

---

## Part 4: Code Quality & Type Safety

### Step 4.1: Code Linting & Formatting with `ruff`
Add Ruff settings to your `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I"] # Pycodestyle (E), Pyflakes (F), Isort (I)
```
Run Ruff to check and format your code:
```bash
uv run ruff check src/
uv run ruff format src/
```

### Step 4.2: Static Type Checking with `mypy`
Add the Mypy configurations to `pyproject.toml`:
```toml
[tool.mypy]
ignore_missing_imports = true
strict = true
```
Annotate all function parameters and return types (e.g. `model: nn.Module`, `device: torch.device`, `epochs: int -> None`), and verify the code passes the strict type check:
```bash
uv run mypy src/
```
Ensure there are no type errors!
