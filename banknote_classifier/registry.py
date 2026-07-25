from typing import Callable, Dict, Type
import torch.nn as nn

class ModelRegistry:
    """Registry pattern to dynamically manage and instantiate deep learning models."""
    
    def __init__(self) -> None:
        self._models: Dict[str, Type[nn.Module]] = {}

    def register(self, model_name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        """Decorator to register a neural network class under a unique name."""
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            self._models[model_name] = cls
            return cls
        return decorator

    def getattr(self, model_name: str) -> Type[nn.Module]:
        """Retrieve the registered model class; raises ValueError if missing."""
        if model_name not in self._models:
            active_models = ", ".join(sorted(self._models.keys())) or "None"
            raise ValueError(f"Model '{model_name}' not found. Registered variants: [{active_models}]")
        return self._models[model_name]

# Global single instance for project-wide usage
net_registry = ModelRegistry()