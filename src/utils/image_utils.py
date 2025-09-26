#src/utils/image_utils.py
"""
A collection of standalone utility functions for image processing tasks.
These functions primarily operate on NumPy arrays.
"""
from typing import Tuple
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import tifffile as tiff

from src.config import SETTINGS


def read_gray_safe(path: Path) -> np.ndarray:
    """Safely reads an image and converts it to grayscale if necessary."""
    # DEV: Эта функция объединяет два подхода из ноутбука: tifffile для TIFF
    # и OpenCV для всего остального, чтобы избежать проблем с Pillow (DecompressionBomb).
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at: {path}")

    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        arr = tiff.imread(str(path))
    else:
        # Using imdecode to be robust with non-ASCII paths
        data = np.fromfile(str(path), dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    
    if arr is None:
        raise IOError(f"Failed to read or decode image file: {path}")

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def to_uint16(arr: np.ndarray) -> np.ndarray:
    """Converts a NumPy array to uint16, scaling values appropriately."""
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return (arr.astype(np.uint16) * (SETTINGS.IMAGE.UINT16_MAX // SETTINGS.IMAGE.UINT8_MAX))
    
    # Scale float arrays from their range to [0, 65535]
    arr_f32 = arr.astype(np.float32)
    min_val, max_val = arr_f32.min(), arr_f32.max()
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.uint16)
    
    scale = max_val - min_val
    scaled_arr = (arr_f32 - min_val) / scale
    return (scaled_arr * SETTINGS.IMAGE.UINT16_MAX).astype(np.uint16)


def to_uint8_visualization(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Converts an array to uint8 for visualization, clipping percentiles."""
    arr_f32 = arr.astype(np.float32)
    low, high = np.percentile(arr_f32, [p_low, p_high])
    if high <= low:
        low, high = arr_f32.min(), arr_f32.max()
    
    scale = high - low if high > low else 1.0
    arr_f32 = np.clip((arr_f32 - low) / scale, 0, 1)
    return (arr_f32 * SETTINGS.IMAGE.UINT8_MAX).astype(np.uint8)


def labels_to_binary_mask(labels: np.ndarray) -> np.ndarray:
    """Converts a label map (0 for background) to a binary mask (0 or 255)."""
    return ((labels > 0).astype(np.uint8) * SETTINGS.IMAGE.UINT8_MAX)


def save_tiff_uint16(path: Path, arr: np.ndarray) -> None:
    """Saves a NumPy array as a uint16 TIFF image with compression."""
    create_dir_if_not_exists(path.parent)
    tiff.imwrite(str(path), arr, compression="deflate", predictor=True)


def save_png_uint8(path: Path, arr: np.ndarray) -> None:
    """Saves a NumPy array as a uint8 PNG image."""
    create_dir_if_not_exists(path.parent)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    iio.imwrite(str(path), arr)


def create_overlay(
    img_u8: np.ndarray,
    mask_u8: np.ndarray,
    color_rgb: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """Creates a color overlay of a mask on top of an image."""
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) # Convert RGB to BGR for OpenCV
    
    color_layer = np.zeros_like(img_bgr)
    color_layer[:] = color_bgr
    
    mask_3ch = (mask_u8 > 0).astype(np.uint8)[:, :, np.newaxis]
    
    overlayed = (img_bgr * (1 - alpha) + color_layer * alpha * mask_3ch)
    return np.clip(overlayed, 0, 255).astype(np.uint8)