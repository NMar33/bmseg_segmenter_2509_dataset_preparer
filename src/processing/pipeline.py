#src/processing/pipeline.py
"""
The Pipeline class orchestrates the execution of a series of transformations.
"""
from typing import List, Dict, Any
import numpy as np


# DEV: Мы делаем "фабрику" прямо здесь, чтобы не усложнять run_preparation.py.
# В будущем, если трансформаций станет очень много, можно вынести это в отдельный
# модуль `factories.py`.
from src.processing import transforms


TRANSFORM_CATALOG = {
    "Passthrough": transforms.Passthrough,
    "Resize": transforms.Resize,       # <-- ДОБАВЛЕНО
    "SmartROI": transforms.SmartROI,   # <-- ДОБАВЛЕНО
}


class Pipeline:
    """Manages a sequence of transformations."""

    def __init__(self, config: List[Dict[str, Any]]):
        """
        Initializes the pipeline from a configuration list.

        Args:
            config: A list of dictionaries, where each dict represents a
                    transformation step (e.g., {'name': 'Resize', 'params': {...}}).
        """
        self.transforms: List[transforms.BaseTransform] = []
        for step_config in config:
            name = step_config['name']
            params = step_config.get('params', {})
            
            transform_class = TRANSFORM_CATALOG.get(name)
            if not transform_class:
                raise ValueError(f"Unknown transformation '{name}'. "
                                 f"Available: {list(TRANSFORM_CATALOG.keys())}")
            
            self.transforms.append(transform_class(**params))
            
        # DEV: Если конвейер пустой (в конфиге `processing_pipeline: []`),
        # мы добавляем Passthrough по умолчанию. Это упрощает логику в Preparer,
        # так как ему не нужно проверять, пустой ли пайплайн. Он всегда может
        # его вызывать.
        if not self.transforms:
            print("Processing pipeline is empty. Using Passthrough transform by default.")
            self.transforms.append(transforms.Passthrough())

    def process(self, image: np.ndarray, mask: np.ndarray | None, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes an image and mask through the entire transformation pipeline.

        Args:
            image: The input image.
            mask: The input mask.
            context: Initial metadata.

        Returns:
            A list of processed items. The list may contain more than one item
            if a one-to-many transformation was used.
        """
        # DEV: Это сердце one-to-many. Каждый шаг пайплайна принимает список
        # и возвращает новый список, применяя свою логику к каждому элементу.
        # Например, Resize применится к каждому из тайлов, которые создал SmartROI.
        
        # Start with a list containing the single input item
        items_to_process = [{'image': image, 'mask': mask, 'context': context}]

        for transform in self.transforms:
            processed_items = []
            for item in items_to_process:
                # Apply the transform and extend the results list
                processed_items.extend(transform(item['image'], item['mask'], item['context']))
            items_to_process = processed_items
        
        return items_to_process