#src/preparers/folder_preparer.py

"""
A preparer for datasets organized in folders (e.g., images/train, masks/train).
"""
from typing import Generator, Dict, Any
from pathlib import Path

from src.preparers.base_preparer import BasePreparer


class FolderPreparer(BasePreparer):
    """Discovers items in a classic folder-based layout."""

    def _discover_items(self, extracted_root: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Finds image-mask pairs in subdirectories like 'train', 'val', 'test'.
        """
        img_base_rel = Path(self.prep_config['img_base_rel'])
        msk_base_rel = Path(self.prep_config['msk_base_rel'])
        splits = self.prep_config.get('splits', ['train', 'val', 'test'])
        
        print(f"Scanning for items in splits: {splits}")

        for split in splits:
            img_dir = extracted_root / img_base_rel / split
            msk_dir = extracted_root / msk_base_rel / split

            if not img_dir.is_dir():
                print(f"Warning: Image directory not found for split '{split}': {img_dir}")
                continue

            # DEV: Ищем все распространенные форматы изображений.
            for img_path in sorted(img_dir.glob('*.*')):
                if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                    continue
                
                # Find corresponding mask
                # DEV: Логика поиска маски из твоего ноутбука. Она гибкая и
                # обрабатывает случаи вроде `_m` суффикса.
                mask_path = self._find_mask(msk_dir, img_path.stem)
                
                if mask_path is None:
                    print(f"Warning: Mask not found for image: {img_path.name} in split {split}")

                yield {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "split": split,
                    "original_id": img_path.stem
                }

    @staticmethod
    def _find_mask(mask_dir: Path, stem: str) -> Path | None:
        """Tries to find a mask file matching the image stem."""
        if not mask_dir.is_dir():
            return None
            
        # Common extensions for masks
        extensions = ['.png', '.tif', '.jpg', '.jpeg']
        
        # Exact match
        for ext in extensions:
            if (p := mask_dir / f"{stem}{ext}").exists(): return p
        
        # Match with common suffixes
        for suffix in ['_m', '_mask', '_gt', '_label']:
            for ext in extensions:
                if (p := mask_dir / f"{stem}{suffix}{ext}").exists(): return p
        
        return None