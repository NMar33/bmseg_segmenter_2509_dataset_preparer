#src/processing/transforms.py

"""
Defines the transformation classes that form the processing pipeline.
Each class is a callable that processes an image and its corresponding mask.
"""
from typing import List, Dict, Any
import numpy as np
import cv2
import random

# DEV: BaseTransform спроектирован так, чтобы возвращать СПИСОК.
# Это ключевой момент для поддержки one-to-many трансформаций, таких как SmartROI (тайлинг),
# где одно входное изображение порождает несколько выходных.
# Для one-to-one трансформаций (как Resize) список будет просто содержать один элемент.

class BaseTransform:
    """Abstract base class for all transformations."""

    def __init__(self, **kwargs):
        """
        Initializes the transform with parameters from the config.
        
        Args:
            **kwargs: Catches all parameters defined in the 'params' block
                      of a transform step in the YAML config.
        """
        # DEV: kwargs позволяет нам быть гибкими. Если в конфиге появятся
        # лишние параметры для какой-то трансформации, программа не упадет.
        self.params = kwargs

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


class Resize(BaseTransform):
    """
    Resizes an image and its mask to a target size or scales them.
    Supports resizing to a fixed (width, height) or scaling so that the
    longest side equals `max_dim`.
    """
    def __init__(self, width: int = 0, height: int = 0, max_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        if not (width and height) and not max_dim:
            raise ValueError("Resize transform requires either (width, height) or max_dim to be set.")
        if (width or height) and max_dim:
            raise ValueError("Resize transform cannot have both (width, height) and max_dim set.")

        self.width = width
        self.height = height
        self.max_dim = max_dim
        # DEV: Сопоставление строковых значений из YAML с флагами OpenCV
        self.inter_map = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
        }
        self.inter_img = self.inter_map.get(kwargs.get("interpolation_img", "area"), cv2.INTER_AREA)
        self.inter_mask = self.inter_map.get(kwargs.get("interpolation_mask", "nearest"), cv2.INTER_NEAREST)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]

        if self.max_dim > 0:
            scale = self.max_dim / max(h, w)
            if scale >= 1.0: # Do not upscale
                return [{'image': image, 'mask': mask, 'context': context}]

            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            target_size = (new_w, new_h)
        else:
            target_size = (self.width, self.height)

        resized_image = cv2.resize(image, target_size, interpolation=self.inter_img)
        resized_mask = None
        if mask is not None:
            # DEV: Важно использовать INTER_NEAREST для масок, чтобы не создавать
            # новые значения классов/цветов на границах.
            resized_mask = cv2.resize(mask, target_size, interpolation=self.inter_mask)
            # Ensure mask remains binary after interpolation, just in case
            if resized_mask.max() > 0:
                 resized_mask = ((resized_mask > (resized_mask.max() / 2)).astype(np.uint8) * 255)

        return [{'image': resized_image, 'mask': resized_mask, 'context': context}]


class SmartROI(BaseTransform):
    """
    Extracts one or more 'smart' regions of interest (tiles) from a larger image.
    The selection is based on a configurable strategy, e.g., preferring tiles with
    a high fraction of positive mask pixels.
    """
    def __init__(self, tile_size: int, stride: int, max_tiles_per_image: int, selection_strategy: List[Dict], **kwargs):
        super().__init__(**kwargs)
        if tile_size <= 0 or stride <= 0 or max_tiles_per_image <= 0:
            raise ValueError("tile_size, stride, and max_tiles_per_image must be positive integers.")
        
        self.tile_size = tile_size
        self.stride = stride
        self.max_tiles_per_image = max_tiles_per_image
        self.selection_strategy = selection_strategy
        self.seed = kwargs.get('seed') # Optional seed for reproducibility

    def __call__(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if mask is None:
            # DEV: Если нет маски, "умный" ROI невозможен. Можно либо падать с ошибкой,
            # либо брать случайные тайлы. Выберем второй вариант для гибкости.
            print(f"Warning: SmartROI running without a mask for item {context['original_id']}. Falling back to random selection.")

        # 1. Generate all possible tile candidates and score them
        scored_tiles = self._score_tiles(image, mask)

        if not scored_tiles:
            return [] # Image is smaller than tile size, skip it

        # 2. Select the best tiles based on the strategy
        best_tiles = self._select_best_tiles(scored_tiles)

        # 3. Crop and return the selected tiles
        output_items = []
        for i, tile in enumerate(best_tiles):
            x, y = tile['x'], tile['y']
            tile_image = image[y:y+self.tile_size, x:x+self.tile_size]
            tile_mask = mask[y:y+self.tile_size, x:x+self.tile_size] if mask is not None else None

            new_context = context.copy()
            new_context['roi_coords'] = (x, y)
            new_context['tile_index'] = i
            new_context['tile_scores'] = tile['scores']

            output_items.append({'image': tile_image, 'mask': tile_mask, 'context': new_context})

        return output_items

    def _score_tiles(self, image: np.ndarray, mask: np.ndarray | None) -> List[Dict[str, Any]]:
        """Generates tile positions and calculates scores for each."""
        h, w = image.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            return []

        scored_tiles = []
        xs = list(range(0, w - self.tile_size + 1, self.stride))
        if not xs or xs[-1] != w - self.tile_size: xs.append(w - self.tile_size)
        ys = list(range(0, h - self.tile_size + 1, self.stride))
        if not ys or ys[-1] != h - self.tile_size: ys.append(h - self.tile_size)

        for y in ys:
            for x in xs:
                scores = {}
                if mask is not None:
                    tile_mask = mask[y:y+self.tile_size, x:x+self.tile_size]
                    scores['positive_fraction'] = float((tile_mask > 0).mean())
                    # DEV: Можно легко добавить другие метрики, если они появятся в `selection_strategy`
                    # scores['boundary_fraction'] = self._calculate_boundary_fraction(tile_mask)

                # DEV: Можно добавить скоринг по самому изображению, например, контраст
                # tile_image = image[y:y+self.tile_size, x:x+self.tile_size]
                # scores['contrast'] = float(tile_image.std())

                scored_tiles.append({'x': x, 'y': y, 'scores': scores})
        return scored_tiles

    def _select_best_tiles(self, scored_tiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Selects tiles according to the configured multi-step strategy."""

        # DEV: Эта логика позволяет делать многоуровневый отбор.
        # Например: "Сначала попробуй найти тайлы с маской > 10%. Если не нашел,
        # тогда ищи тайлы с контрастом > 50. Если и таких нет, возьми случайный".

        rng = random.Random(self.seed) if self.seed is not None else random
        
        candidates = list(scored_tiles)
        
        for step in self.selection_strategy:
            strategy_type = step.get('type')
            
            if strategy_type == 'positive_fraction':
                min_frac = step.get('min_fraction', 0.0)
                filtered = [t for t in candidates if t['scores'].get('positive_fraction', 0.0) >= min_frac]
                if filtered:
                    # If found, select from this pool and finish
                    rng.shuffle(filtered)
                    return filtered[:self.max_tiles_per_image]

            elif strategy_type == 'random':
                # This should usually be the last step in the strategy
                rng.shuffle(candidates)
                return candidates[:self.max_tiles_per_image]
                
            # DEV: Можно добавить другие стратегии, например, отбор по самому высокому скору, а не по порогу.
            # elif strategy_type == 'highest_score':
            #     score_key = step.get('key') # e.g., 'contrast'
            #     candidates.sort(key=lambda t: t['scores'].get(score_key, 0.0), reverse=True)
            #     return candidates[:self.max_tiles_per_image]
            
        # Fallback: if no strategy matched or returned results, return random from the initial pool
        rng.shuffle(scored_tiles)
        return scored_tiles[:self.max_tiles_per_image]