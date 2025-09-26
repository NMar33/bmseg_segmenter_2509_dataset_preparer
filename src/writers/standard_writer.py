#src/writers/standard_writer.py

"""
A writer that saves files in the standard canonical structure:
- <output_root>/images/{split}/{filename}
- <output_root>/masks/{split}/{filename}
"""
from typing import Dict, Any
from pathlib import Path
import numpy as np

from src.writers.base_writer import BaseWriter
from src.utils import image_utils


class StandardWriter(BaseWriter):
    """Writes data to a standard, canonical directory structure."""

    def __init__(self, image_format: Dict[str, Any], mask_format: Dict[str, Any]):
        """
        Initializes the writer with format configurations.

        Args:
            image_format: Dictionary from config specifying image save options
                          (e.g., {'ext': '.tif', 'dtype': 'uint16'}).
            mask_format: Dictionary from config specifying mask save options
                         (e.g., {'ext': '.png', 'dtype': 'uint8'}).
        """
        super().__init__()
        self.image_format = image_format
        self.mask_format = mask_format

    def write(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
        """
        Saves the image and mask to disk according to the standard layout.

        Args:
            image: The processed image to save.
            mask: The processed mask to save.
            context: Metadata about the item, including 'split' and 'original_id'.
            output_root: The root directory for the prepared dataset.

        Returns:
            A dictionary containing information about the saved files, which will be
            used by the ManifestBuilder.
        """
        split = context['split']
        
        # DEV: Генерируем новое, каноническое имя файла.
        # Это важно для унификации. Имя зависит от исходного ID и,
        # возможно, индекса тайла, если он был создан (например, в SmartROI).
        # Это делает каждое имя файла уникальным и отслеживаемым.
        base_id = context['original_id']
        tile_index = context.get('tile_index')
        filename_stem = f"{base_id}" if tile_index is None else f"{base_id}_t{tile_index:03d}"
        
        # --- Save image ---
        img_ext = self.image_format['ext']
        img_dtype = self.image_format['dtype']
        img_filename = f"{filename_stem}{img_ext}"
        img_path = output_root / "images" / split / img_filename
        
        # DEV: Здесь мы явно приводим к нужному типу перед сохранением,
        # как указано в конфиге. `to_uint16` содержит логику нормализации,
        # что важно для консистентности.
        if img_dtype == 'uint16':
            img_to_save = image_utils.to_uint16(image)
            image_utils.save_tiff_uint16(img_path, img_to_save)
        elif img_dtype == 'uint8':
            img_to_save = image_utils.to_uint8_visualization(image)
            image_utils.save_png_uint8(img_path, img_to_save)
        else:
             raise ValueError(f"Unsupported image dtype '{img_dtype}' in writer config.")
            
        # --- Save mask if it exists ---
        mask_path_rel = None
        if mask is not None:
            mask_ext = self.mask_format['ext']
            mask_filename = f"{filename_stem}{mask_ext}"
            mask_path = output_root / "masks" / split / mask_filename
            
            # Masks are always uint8 binary images (0 or 255)
            mask_to_save = image_utils.labels_to_binary_mask(mask)
            image_utils.save_png_uint8(mask_path, mask_to_save)
            mask_path_rel = str(mask_path.relative_to(output_root))

        # --- Prepare output info for the manifest ---
        # DEV: Этот словарь - ключевой результат работы writer'а.
        # Он передается в ManifestBuilder, чтобы записать, что и куда было сохранено.
        # `id` здесь - это уникальный идентификатор *сохраненного* файла.
        output_info = {
            "id": filename_stem,
            "image_path": str(img_path.relative_to(output_root)),
            "mask_path": mask_path_rel
        }
            
        return output_info