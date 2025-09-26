# src/preparers/volume_preparer.py

"""
A preparer for datasets where data is stored in 3D volumes (e.g., TIFF stacks).
"""
from typing import Generator, Dict, Any
from pathlib import Path
import tifffile as tiff
import numpy as np

from src.preparers.base_preparer import BasePreparer


class VolumePreparer(BasePreparer):
    """Discovers items by slicing 3D volumes into 2D frames."""

    def _discover_items(self, extracted_root: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Loads 3D TIFF stacks, slices them, and yields each slice as a data item.
        """
        splits_config = self.prep_config.get("splits")
        if not splits_config:
            raise ValueError("'splits' configuration is missing for VolumePreparer.")

        print(f"Slicing volumes from: {extracted_root}")
        
        for split_name, files in splits_config.items():
            volume_path = extracted_root / files['volume']
            labels_path = extracted_root / files.get('labels')

            if not volume_path.exists():
                print(f"Warning: Volume file not found for split '{split_name}': {volume_path}")
                continue
            
            volume = tiff.imread(str(volume_path))
            labels = tiff.imread(str(labels_path)) if labels_path and labels_path.exists() else None

            if volume.ndim != 3:
                raise ValueError(f"Expected a 3D volume for '{volume_path.name}', but got {volume.ndim} dimensions.")
            
            if labels is not None and volume.shape != labels.shape:
                 raise ValueError(f"Shape mismatch between volume {volume.shape} and labels {labels.shape} "
                                  f"for split '{split_name}'.")

            # DEV: Мы не передаем пути к файлам, а передаем сами np.ndarray.
            # Это хак, чтобы не читать один и тот же большой файл много раз.
            # BasePreparer должен быть немного модифицирован, чтобы поддерживать это.
            # Но для начала сделаем более простой вариант: yield путей.
            # UPD: Давайте сделаем это правильно сразу. Модифицируем BasePreparer,
            # чтобы он мог принимать либо пути, либо уже загруженные данные.
            # Для этого в `yield` будем добавлять флаг `is_loaded: True`.
            
            num_slices = volume.shape[0]
            for i in range(num_slices):
                yield {
                    "image": volume[i],
                    "mask": labels[i] if labels is not None else None,
                    "is_loaded": True, # Flag to indicate data is already in memory
                    "split": split_name,
                    "original_id": f"{volume_path.stem}_s{i:03d}"
                }