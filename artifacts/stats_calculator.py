# src/artifacts/stats_calculator.py

"""
Calculates statistics for images and masks.
Supports both in-memory and memory-safe streaming modes.
"""
# DEV: Этот модуль самый сложный. Здесь много математики из твоего ноутбука.
# Мы просто обернем ее в классы.
from typing import Dict, Any, List
import numpy as np
import gc

from src.config import SETTINGS

# Welford's algorithm for online variance
def _welford_init(): return {"n": 0, "mean": 0.0, "M2": 0.0}
def _welford_update(st, x): st["n"] += 1; delta = x - st["mean"]; st["mean"] += delta / st["n"]; st["M2"] += delta * (x - st["mean"])
def _welford_finalize(st):
    if st["n"] < 2: return st["mean"], 0.0
    return st["mean"], (st["M2"] / (st["n"] - 1))**0.5

class StatsCalculator:
    """Calculates and stores statistics for the dataset."""
    
    def __init__(self, mode: str = "streaming", percentiles: List[float] = [1.0, 99.0]):
        if mode not in ["streaming", "in_memory"]:
            raise ValueError("StatsCalculator mode must be 'streaming' or 'in_memory'")
        self.mode = mode
        self.percentiles = percentiles
        self._data: Dict[str, Dict[str, List[np.ndarray]]] = {"train": {"images": [], "masks": []}, 
                                                              "val": {"images": [], "masks": []}, 
                                                              "test": {"images": [], "masks": []}}
        # For streaming mode
        self._stream_stats: Dict[str, Any] = {}

    def update(self, image: np.ndarray, mask: np.ndarray | None, split: str):
        """Updates stats with a new image/mask pair."""
        if self.mode == "in_memory":
            self._data[split]["images"].append(image)
            if mask is not None:
                self._data[split]["masks"].append(mask)
        else: # streaming
            self._update_stream(image, mask, split)

    def _update_stream(self, image: np.ndarray, mask: np.ndarray | None, split: str):
        """Update streaming statistics."""
        if split not in self._stream_stats:
            self._stream_stats[split] = {
                "img_welford": _welford_init(),
                "img_hist": np.zeros(SETTINGS.IMAGE.UINT16_MAX + 1, dtype=np.int64),
                "mask_welford": _welford_init(),
            }
        
        # Image stats
        img_norm = image.astype(np.float32) / SETTINGS.IMAGE.UINT16_MAX
        _welford_update(self._stream_stats[split]["img_welford"], img_norm.mean())
        
        hist, _ = np.histogram(image, bins=SETTINGS.IMAGE.UINT16_MAX + 1, range=(0, SETTINGS.IMAGE.UINT16_MAX))
        self._stream_stats[split]["img_hist"] += hist
        
        # Mask stats
        if mask is not None:
            pos_frac = (mask > 0).mean()
            _welford_update(self._stream_stats[split]["mask_welford"], pos_frac)

    def calculate(self) -> Dict[str, Any]:
        """Calculates final statistics and returns them as a dictionary."""
        print("Calculating statistics...")
        if self.mode == "in_memory":
            # DEV: Эта ветка будет очень требовательна к RAM
            # Реализация будет аналогична streaming, но на собранных np.stack
            raise NotImplementedError("In-memory stats calculation is not implemented yet. Use 'streaming' mode.")
        
        final_stats = {"images": {}, "masks": {}}
        for split, data in self._stream_stats.items():
            # Image stats
            img_mean, img_std = _welford_finalize(data["img_welford"])
            p_values = self._percentiles_from_hist(data["img_hist"], self.percentiles)
            final_stats["images"][split] = {"mean": img_mean, "std": img_std}
            for p, val in zip(self.percentiles, p_values):
                final_stats["images"][split][f"p{p}"] = val

            # Mask stats
            mask_mean, mask_std = _welford_finalize(data["mask_welford"])
            final_stats["masks"][split] = {"positive_fraction_mean": mask_mean, "positive_fraction_std": mask_std}
        
        # Clean up memory
        del self._stream_stats
        gc.collect()
        
        return final_stats
        
    def _percentiles_from_hist(self, hist: np.ndarray, percentiles: List[float]) -> List[float]:
        """Calculates percentiles from a histogram."""
        cum_hist = np.cumsum(hist)
        total = cum_hist[-1]
        if total == 0:
            return [0.0] * len(percentiles)
        
        results = []
        for p in percentiles:
            target_count = (p / 100.0) * total
            idx = np.searchsorted(cum_hist, target_count)
            # Normalize back to [0, 1] range
            results.append(idx / SETTINGS.IMAGE.UINT16_MAX)
        return results