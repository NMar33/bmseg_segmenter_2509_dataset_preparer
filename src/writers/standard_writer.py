#src/writers/standard_writer.py

"""
A writer that saves files in the standard canonical structure:
- images/{split}/{filename}
- masks/{split}/{filename}
"""
from typing import Dict, Any
from pathlib import Path
import numpy as np

from src.writers.base_writer import BaseWriter
from src.utils import image_utils


class StandardWriter(BaseWriter):
    """Writes data to a standard directory structure."""

    def __init__(self, image_format: Dict[str, Any], mask_format: Dict[str, Any]):
        super().__init__()
        self.image_format = image_format
        self.mask_format = mask_format

    def write(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
        """Saves the image and mask according to the standard layout."""
        split = context['split']
        
        # DEV: Генерируем новое, каноническое имя файла.
        # Это важно для унификации. Имя зависит от ID датасета, исходного ID и,
        # возможно, индекса тайла, если он есть.
        base_id = context['original_id']
        tile_index = context.get('tile_index')
        filename_stem = f"{base_id}" if tile_index is None else f"{base_id}_t{tile_index:03d}"
        
        # Save image
        img_ext = self.image_format['ext']
        img_dtype = self.image_format['dtype']
        img_filename = f"{filename_stem}{img_ext}"
        img_path = output_root / "images" / split / img_filename
        
        # DEV: Здесь мы явно приводим к нужному типу перед сохранением,
        # как указано в конфиге.
        if img_dtype == 'uint16':
            img_to_save = image_utils.to_uint16(image)
            image_utils.save_tiff_uint16(img_path, img_to_save)
        else: # Default to uint8
            img_to_save = image.astype(np.uint8)
            image_utils.save_png_uint8(img_path, img_to_save)
            
        output_info = {
            "image_path": str(img_path.relative_to(output_root))
        }

        # Save mask if it exists
        if mask is not None:
            mask_ext = self.mask_format['ext']
            mask_filename = f"{filename_stem}{mask_ext}"
            mask_path = output_root / "masks" / split / mask_filename
            mask_to_save = mask.astype(np.uint8) # Masks are always uint8
            image_utils.save_png_uint8(mask_path, mask_to_save)
            output_info["mask_path"] = str(mask_path.relative_to(output_root))
        else:
            output_info["mask_path"] = None
            
        return output_info