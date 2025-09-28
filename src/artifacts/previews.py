# src/artifacts/previews.py

"""
Generates visual previews for a sample of the dataset.
"""
from typing import Dict, Any, List
from pathlib import Path
import random
import imageio.v2 as iio
import numpy as np
import cv2

from src.utils import image_utils

class PreviewGenerator:
    """Creates and saves image previews (image, mask, overlay)."""

    def __init__(self, per_split: int, seed: int, **kwargs):
        self.per_split = per_split
        self.seed = seed
        self.kwargs = kwargs # color_rgb, alpha, etc.
        self._candidates: Dict[str, List[Dict[str, Path]]] = {"train": [], "val": [], "test": []}

    def add_candidate(self, output_info: Dict[str, Any], split: str):
        """Adds a processed item as a candidate for preview generation."""
        # DEV: Мы не храним сами изображения, а только пути к ним, чтобы экономить память.
        # Превью будут генерироваться в самом конце, путем чтения файлов с диска.
        candidate = {
            "image_path": output_info.get("image_path"),
            "mask_path": output_info.get("mask_path"),
        }
        if candidate["image_path"]:
             self._candidates.setdefault(split, []).append(candidate)

    def generate(self, prepared_root: Path):
        """Selects candidates, creates previews, and saves them."""
        print("Generating previews...")
        rng = random.Random(self.seed)
        
        for split, candidates in self._candidates.items():
            if not candidates:
                continue
            
            preview_dir = prepared_root / "previews" / split
            preview_dir.mkdir(parents=True, exist_ok=True)
            
            num_to_sample = min(self.per_split, len(candidates))
            sample = rng.sample(candidates, num_to_sample)
            
            for item in sample:
                try:
                    self._create_single_preview(item, prepared_root, preview_dir)
                except Exception as e:
                    print(f"\nWarning: Failed to create preview for {item['image_path']}. Error: {e}")

    def _create_single_preview(self, item: Dict[str, Path], prepared_root: Path, preview_dir: Path):
        img_path = prepared_root / item['image_path']
        img = image_utils.read_gray_safe(img_path)
        img_u8 = image_utils.to_uint8_visualization(img)
        
        mask = None
        if item.get('mask_path'):
            mask_path = prepared_root / item['mask_path']
            if mask_path.exists():
                mask = image_utils.read_gray_safe(mask_path)
        
        overlay = image_utils.create_overlay(img_u8, mask, **self.kwargs) if mask is not None else None
        
        # Create a panel: [Image | Mask | Overlay]
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask is not None else np.zeros_like(img_bgr)
        overlay_bgr = overlay if overlay is not None else img_bgr
        panel = np.concatenate([img_bgr, mask_bgr, overlay_bgr], axis=1)

        preview_filename = f"{img_path.stem}_preview.png"
        iio.imwrite(preview_dir / preview_filename, panel)