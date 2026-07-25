# Expose core modules directly at the package root level
from .dataset import BanknoteDataset, BaseValidationTransform, RegularTrainAugmentation, fetch_image_manifest
from .registry import net_registry
from .models import CustomVGG16

__version__ = "0.1.0"