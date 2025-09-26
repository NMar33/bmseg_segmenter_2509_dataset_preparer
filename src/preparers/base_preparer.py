#src/preparers/base_preparer.py

"""
Abstract base class for preparers. Contains the main orchestration logic.
"""
from abc import ABC, abstractmethod
from typing import Generator, Dict, Any
from pathlib import Path
from tqdm import tqdm

from src.processing.pipeline import Pipeline
from src.writers.standard_writer import StandardWriter # Assuming standard writer for now
from src.utils import image_utils


class BasePreparer(ABC):
    """Orchestrates the data preparation process."""

    def __init__(self, prep_config: Dict[str, Any], pipeline_config: List[Dict[str, Any]], writer_config: Dict[str, Any]):
        self.prep_config = prep_config
        self.pipeline = Pipeline(pipeline_config)
        # DEV: Пока что мы жестко завязываемся на StandardWriter. В будущем можно
        # сделать фабрику и для writer'ов, если их станет несколько.
        self.writer = StandardWriter(**writer_config)
        
    @abstractmethod
    def _discover_items(self, extracted_root: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Discovers raw data items (images, masks) and yields them one by one.
        This method must be implemented by subclasses.

        Yields:
            A dictionary containing metadata for a single data item, e.g.,
            {'image_path': Path, 'mask_path': Path, 'split': str, 'original_id': str}
        """
        pass

    def run(self, extracted_root: Path, output_root: Path) -> None:
        """
        Executes the full preparation pipeline.
        This method should not be overridden by subclasses.
        """
        print(f"Starting preparation. Output will be saved to: {output_root}")
        
        # DEV: Здесь на Шаге 6 мы будем инициализировать генераторы артефактов
        # manifest_builder = ManifestBuilder()
        # stats_calculator = StatsCalculator()

        item_generator = self._discover_items(extracted_root)
        
        # Use tqdm for a progress bar. We need to count items first if the generator
        # doesn't have a known length. For now, let's assume we can convert to a list.
        items = list(item_generator)
        print(f"Found {len(items)} items to process.")
        
        for item_context in tqdm(items, desc="Processing items"):
            try:
                # 1. Read
                image = image_utils.read_gray_safe(item_context['image_path'])
                mask = image_utils.read_gray_safe(item_context['mask_path']) if item_context.get('mask_path') else None

                # 2. Process
                processed_items = self.pipeline.process(image, mask, item_context)

                # 3. Write
                for processed in processed_items:
                    output_info = self.writer.write(
                        processed['image'], 
                        processed['mask'], 
                        processed['context'], 
                        output_root
                    )
                    
                    # DEV: Здесь на Шаге 6 мы будем обновлять артефакты
                    # manifest_builder.add_entry(item_context, processed['context'], output_info)
                    # stats_calculator.update(...)

            except (FileNotFoundError, IOError) as e:
                print(f"\nWARNING: Skipping item {item_context.get('original_id')} due to an error: {e}")
            except Exception as e:
                print(f"\nERROR: Failed to process item {item_context.get('original_id')}. Error: {e}")

        # DEV: Здесь на Шаге 6 мы будем сохранять финальные артефакты
        # manifest_builder.save(output_root / "manifest.jsonl")
        # dataset_map_generator.generate(...)
        
        print("\nPreparation finished successfully.")