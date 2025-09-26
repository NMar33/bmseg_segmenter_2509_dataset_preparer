#src/preparers/base_preparer.py

"""
Abstract base class for preparers. Contains the main orchestration logic for
transforming raw datasets into a canonical format.
"""
from abc import ABC, abstractmethod
from typing import Generator, Dict, Any, List
from pathlib import Path

from tqdm import tqdm

from src.processing.pipeline import Pipeline
from src.writers.base_writer import BaseWriter
from src.utils import image_utils

# Import artifact generators
from src.artifacts.manifest import ManifestBuilder
from src.artifacts.stats_calculator import StatsCalculator
from src.artifacts.previews import PreviewGenerator
from src.artifacts.dataset_map import DatasetMapGenerator


class BasePreparer(ABC):
    """
    Orchestrates the data preparation process.

    This class is responsible for:
    1. Finding raw data items using a discovery strategy (implemented by subclasses).
    2. Reading the raw data (image and mask).
    3. Passing the data through a processing pipeline.
    4. Writing the processed results to disk using a writer.
    5. Coordinating the generation of all artifacts (manifest, stats, previews, dataset_map).
    """

    def __init__(self,
                 prep_config: Dict[str, Any],
                 pipeline_config: List[Dict[str, Any]],
                 writer_config: Dict[str, Any],
                 artifacts_config: Dict[str, Any] = None,
                 full_config: Dict[str, Any] = None):
        """
        Initializes the preparer.

        Args:
            prep_config: Configuration specific to the preparer (e.g., paths).
            pipeline_config: Configuration for the processing pipeline.
            writer_config: Configuration for the writer.
            artifacts_config: Configuration for all artifact generators.
            full_config: The complete YAML configuration, needed for the dataset_map.
        """
        self.prep_config = prep_config
        self.pipeline = Pipeline(pipeline_config)
        self.writer: BaseWriter = self._create_writer(writer_config)
        
        self.artifacts_config = artifacts_config or {}
        self.full_config = full_config

    def _create_writer(self, writer_config: Dict[str, Any]) -> BaseWriter:
        """Factory method for creating a writer instance."""
        # DEV: Пока что мы жестко завязываемся на StandardWriter, так как он единственный.
        # Если в будущем появятся другие writer'ы, здесь будет логика выбора
        # на основе `writer_config['type']`. Это делает код готовым к расширению.
        from src.writers.standard_writer import StandardWriter
        return StandardWriter(**writer_config)

    @abstractmethod
    def _discover_items(self, extracted_root: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Discovers raw data items (images, masks) and yields them one by one.
        This method MUST be implemented by subclasses.

        Yields:
            A dictionary containing metadata for a single data item.
            It can yield either paths to files:
            {
                'image_path': Path('/path/to/img.png'),
                'mask_path': Path('/path/to/mask.png'),
                'split': 'train',
                'original_id': 'img_001'
            }
            Or data already loaded into memory (for performance with large volume files):
            {
                'image': np.ndarray,
                'mask': np.ndarray,
                'is_loaded': True,
                'split': 'train',
                'original_id': 'vol_slice_001'
            }
        """
        pass

    def run(self, extracted_root: Path, output_root: Path) -> None:
        """
        Executes the full preparation pipeline.
        This is the main orchestration method and should not be overridden by subclasses.
        """
        print(f"Starting preparation. Output will be saved to: {output_root}")
        
        # --- 1. INITIALIZE ARTIFACT GENERATORS ---
        # DEV: Мы создаем генераторы артефактов в самом начале.
        # Они будут накапливать информацию в цикле и сохранять ее в конце.
        manifest = ManifestBuilder()
        
        stats_calc = None
        stats_config = self.artifacts_config.get('stats', {})
        if stats_config.get('enabled', True):
            # Просто берем вложенный словарь 'params' и передаем его
            constructor_params = stats_config.get('params', {})
            stats_calc = StatsCalculator(**constructor_params)

        preview_gen = None
        preview_config = self.artifacts_config.get('previews', {})
        if preview_config.get('enabled', True):
            constructor_params = preview_config.get('params', {})
            preview_gen = PreviewGenerator(**constructor_params)

        # # Conditionally create generators based on config to allow disabling them
        # stats_config = self.artifacts_config.get('stats', {})
        # stats_calc = StatsCalculator(**stats_config) if stats_config.get('enabled', True) else None

        # preview_config = self.artifacts_config.get('previews', {})
        # preview_gen = PreviewGenerator(**preview_config) if preview_config.get('enabled', True) else None

        # --- 2. DISCOVER AND PROCESS ITEMS ---
        item_generator = self._discover_items(extracted_root)
        
        # DEV: Превращаем генератор в список, чтобы получить общее количество для tqdm.
        # Это может потреблять память, если список очень большой. Альтернатива —
        # передавать total в tqdm, если subclass может его посчитать заранее.
        # Для наших задач это пока не критично.
        items = list(item_generator)
        if not items:
            print("Warning: No items discovered to process. Finishing early.")
            return
            
        print(f"Found {len(items)} items to process.")
        
        for item_context in tqdm(items, desc="Processing items"):
            try:
                # Read Data
                if item_context.get('is_loaded', False):
                    image = item_context.get('image')
                    mask = item_context.get('mask')
                    if image is None:
                        raise ValueError("Context is marked as 'is_loaded' but contains no 'image' data.")
                else:
                    image_path = item_context.get('image_path')
                    mask_path = item_context.get('mask_path')
                    if not image_path:
                        raise ValueError("Context must contain 'image_path' if not pre-loaded.")
                    image = image_utils.read_gray_safe(image_path)
                    mask = image_utils.read_gray_safe(mask_path) if mask_path and mask_path.exists() else None

                # Process Data
                processed_items = self.pipeline.process(image, mask, item_context)

                # Write Data and Update Artifacts
                for processed in processed_items:
                    output_info = self.writer.write(
                        image=processed['image'], 
                        mask=processed.get('mask'), 
                        context=processed['context'], 
                        output_root=output_root
                    )
                    
                    # Update artifacts with info from this specific processed item
                    manifest.add_entry(
                        original_context=item_context,
                        processed_context=processed['context'],
                        output_info=output_info
                    )
                    if stats_calc:
                        stats_calc.update(processed['image'], processed.get('mask'), processed['context']['split'])
                    if preview_gen:
                        preview_gen.add_candidate(output_info, processed['context']['split'])

            except (FileNotFoundError, IOError) as e:
                print(f"\n[WARNING] Skipping item '{item_context.get('original_id')}' due to a file error: {e}")
            except Exception as e:
                import traceback
                print(f"\n[ERROR] Failed to process item '{item_context.get('original_id')}'. Error: {e}")
                # DEV: Включаем traceback для облегчения отладки непредвиденных ошибок.
                traceback.print_exc()

        # --- 3. SAVE FINAL ARTIFACTS ---
        print("\nFinalizing and saving artifacts...")
        
        # Save the detailed manifest of all processed files
        manifest.save(output_root / "manifest.jsonl")

        # Calculate final statistics from collected data
        final_stats = stats_calc.calculate() if stats_calc else {}
        
        # Generate visual previews from a sample of saved files
        if preview_gen:
            preview_gen.generate(output_root)
            
        # The final act: create the main dataset_map.yaml file
        if self.full_config:
            map_gen = DatasetMapGenerator(self.full_config)
            map_gen.generate(output_root, final_stats)
        else:
            print("[WARNING] full_config was not provided to the preparer. Skipping dataset_map.yaml generation.")
        
        print("\nPreparation finished successfully.")