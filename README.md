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
2. Add project dependencies using `uv add` (e.g. `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`).
3. Add development dependencies (linters and type checkers) using `uv add --dev` (e.g. `ruff`, `mypy`).
4. Run `uv sync` to generate the `uv.lock` file and create the `.venv` virtual environment.
   - *Concept*: Understand the difference between `pyproject.toml` (declares abstract dependencies) and `uv.lock` (locks exact resolved dependencies).
5. Execute python scripts using `uv run`.

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
- **Goal**: Verify your model runs inside this Flat Layout using `uv run python main.py`.

### Task 2B: Migrate to Src Layout (Production Standard)
The Flat Layout suffers from "accidental imports" (modules in the root can be imported even when the package is not installed). Src Layout prevents this by forcing all source code into a `src/` directory.

Reorganize the project to look like this:
```
pypractice-exercise/
├── main.py               ← entry point, stays in root
├── src/
│   └── banknote_classifier/
│       ├── __init__.py
│       ├── models.py
│       └── dataset.py
├── pyproject.toml
├── uv.lock
└── data/
```

> **Why keep `main.py` at root?**  
> `main.py` acts as the entry point and imports from the `banknote_classifier` package (e.g. `from banknote_classifier.models import ...`). When the package is **not installed**, Python cannot resolve the import and will throw a `ModuleNotFoundError`. This directly demonstrates why `uv pip install -e .` (editable install) is required — it registers the `src/` directory so the package becomes importable.

To run the project in Src Layout, configure `pyproject.toml` for packaging and installation:
1. Define the build system and project metadata in `pyproject.toml` using the following skeleton template:
   ```toml
   [build-system]
   # TODO: Declare the Hatchling build-backend and requirements
   
   [project]
   name = "banknote-classifier"
   version = "0.1.0"
   description = "Bangla Banknote Classifier"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       # TODO: List all your production packages (torch, torchvision, etc.)
   ]
   ```
2. Try running `uv run python main.py` **before** installing — observe the `ModuleNotFoundError`.
3. Install the package in editable mode using `uv pip install -e .`
4. **Goal**: Run `uv run python main.py` again — this time it should succeed, importing modules from `src/banknote_classifier/`.

---

## Part 3: Design Patterns in Machine Learning Code

To make your codebase extensible and clean, you will refactor it using two common software design patterns.

### Task 3A: Implement the Registry Pattern
In large-scale ML systems, we want to configure model architectures dynamically via string names (e.g. from config files) without hardcoding imports.

1. Implement a decorator-based registration system in a new file `src/banknote_classifier/registry.py` matching the class structure:
   ```python
   import torch.nn as nn
   from typing import Dict, Type

   class ModelRegistry:
       def __init__(self) -> None:
           # TODO: Initialize internal storage (e.g. dict) for model mappings
           pass

       def register(self, name: str):
           # TODO: Implement decorator function to register class mappings
           pass

       def get(self, name: str) -> Type[nn.Module]:
           # TODO: Retrieve and return the registered class, or raise ValueError if not found
           pass

   MODEL_REGISTRY = ModelRegistry()
   ```
2. Register your `VGG16` class in `models.py` using the `@MODEL_REGISTRY.register("vgg16")` decorator.
3. In `main.py`, dynamically initialize the model using `MODEL_REGISTRY.get("vgg16")()`.

### Task 3B: Implement the Strategy Pattern
The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. We will use it to easily swap data preprocessing/augmentation strategies at runtime.

1. Create a base strategy class and concrete strategies in `src/banknote_classifier/dataset.py`:
   ```python
   from abc import ABC, abstractmethod
   from torchvision import transforms

   class PreprocessStrategy(ABC):
       @abstractmethod
       def get_transforms(self) -> transforms.Compose:
           # TODO: Define abstract method signature
           pass

   class StandardPreprocess(PreprocessStrategy):
       def get_transforms(self) -> transforms.Compose:
           # TODO: Implement basic transforms (Resize, ToTensor, standard Normalize)
           pass

   class HeavyAugmentation(PreprocessStrategy):
       def get_transforms(self) -> transforms.Compose:
           # TODO: Implement heavy transforms (add flip, rotation, etc.)
           pass
   ```
2. Configure your dataset or dataloader setup to accept a `PreprocessStrategy` strategy object.
3. **Goal**: Switch between augmentation pipelines in your training setup simply by passing either `StandardPreprocess()` or `HeavyAugmentation()` at runtime.

---

## Part 4: Code Quality & Type Safety

### Step 4.1: Code Linting & Formatting with `ruff`
Add Ruff configurations to `pyproject.toml` (e.g. line-length setting and selecting basic rules `["E", "F", "I"]`).
1. Run `uv run ruff check src/` to check for style violations and import sorting issues.
2. Run `uv run ruff format src/` to automatically format all files.

### Step 4.2: Static Type Checking with `mypy`
Add Mypy configuration section to `pyproject.toml` enabling strict checks:
```toml
[tool.mypy]
ignore_missing_imports = true
strict = true
```
1. Add strict Python type annotations (e.g. parameter types and return types) to all your refactored modules (such as your `ModelRegistry` class, training functions, data loader functions, etc.).
2. Run `uv run mypy src/` and make sure the codebase compiles without any static type check warnings.
