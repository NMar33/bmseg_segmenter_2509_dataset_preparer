#src/writers/base_writer.py

"""
Abstract base class for writers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import numpy as np


class BaseWriter(ABC):
    """Defines the interface for writing processed data to disk."""

    def __init__(self, **params):
        pass

    @abstractmethod
    def write(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
        """
        Saves the image and mask to disk.

        Args:
            image: The processed image to save.
            mask: The processed mask to save.
            context: Metadata about the item.
            output_root: The root directory for the prepared dataset.

        Returns:
            A dictionary containing information about the saved files,
            e.g., their relative paths. This is used by the ManifestBuilder.
        """
        pass