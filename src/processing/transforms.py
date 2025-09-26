#src/processing/transforms.py

"""
Defines the transformation classes that form the processing pipeline.
Each class is a callable that processes an image and its corresponding mask.
"""
from typing import List, Dict, Any
import numpy as np

# DEV: BaseTransform спроектирован так, чтобы возвращать СПИСОК.
# Это ключевой момент для поддержки one-to-many трансформаций, таких как SmartROI (тайлинг),
# где одно входное изображение порождает несколько выходных.
# Для one-to-one трансформаций (как Resize) список будет просто содержать один элемент.

class BaseTransform:
    """Abstract base class for all transformations."""
    
    def __init__(self, **kwargs):
        # kwargs allows us to pass any parameters from the config
        pass

    def __call__(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applies the transformation.

        Args:
            image: The input image as a NumPy array.
            mask: The input mask as a NumPy array, or None.
            context: A dictionary with metadata about the item being processed.

        Returns:
            A list of dictionaries, where each dictionary contains the
            processed 'image', 'mask', and updated 'context'.
        """
        raise NotImplementedError("Each transform must implement the `__call__` method.")


class Passthrough(BaseTransform):
    """
    A null transformation that does nothing.
    Useful for testing the pipeline structure or for creating a raw, copied dataset.
    """
    def __call__(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Returns the input data unchanged, wrapped in a list."""
        # DEV: Просто заворачиваем исходные данные в формат, который ожидает Pipeline.
        return [{'image': image, 'mask': mask, 'context': context}]