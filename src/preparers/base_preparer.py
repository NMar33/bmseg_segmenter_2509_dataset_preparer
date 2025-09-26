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
from src.writers.standard_writer import StandardWriter
from src.utils import image_utils


class BasePreparer(ABC):
    """
    Orchestrates the data preparation process.

    This class is responsible for:
    1. Finding raw data items using a discovery strategy (implemented by subclasses).
    2. Reading the raw data (image and mask).
    3. Passing the data through a processing pipeline.
    4. Writing the processed results to disk using a writer.
    5. (Future) Coordinating the generation of artifacts like manifests and stats.
    """

    def __init__(self, prep_config: Dict[str, Any], pipeline_config: List[Dict[str, Any]], writer_config: Dict[str, Any]):
        """
        Initializes the preparer.

        Args:
            prep_config: Configuration specific to the preparer (e.g., paths).
            pipeline_config: Configuration for the processing pipeline.
            writer_config: Configuration for the writer.
        """
        self.prep_config = prep_config
        self.pipeline = Pipeline(pipeline_config)
        
        # DEV: Пока что мы жестко завязываемся на StandardWriter. В будущем можно
        # сделать фабрику и для writer'ов, если их станет несколько.
        # Это приемлемое упрощение на данном этапе.
        self.writer = StandardWriter(**writer_config)
        
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
            Or data already loaded into memory:
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
        
        # DEV: Здесь на Шаге 6 мы будем инициализировать генераторы артефактов.
        # manifest_builder = ManifestBuilder()
        # stats_calculator = StatsCalculator()
        # preview_generator = PreviewGenerator()

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
                # --- 1. Read Data ---
                # DEV: Это ключевое изменение для поддержки VolumePreparer.
                # Мы проверяем флаг 'is_loaded'. Если он есть, мы берем данные
                # прямо из item_context. Иначе — читаем с диска по путям.
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

                # --- 2. Process Data ---
                processed_items = self.pipeline.process(image, mask, item_context)

                # --- 3. Write Data ---
                for processed in processed_items:
                    output_info = self.writer.write(
                        image=processed['image'], 
                        mask=processed.get('mask'), 
                        context=processed['context'], 
                        output_root=output_root
                    )
                    
                    # DEV: Здесь на Шаге 6 мы будем обновлять артефакты.
                    # Например, передавать информацию о сохраненных файлах в manifest.
                    # manifest_builder.add_entry(
                    #     original_context=item_context,
                    #     processed_context=processed['context'],
                    #     output_info=output_info
                    # )
                    # stats_calculator.update(processed['image'], processed.get('mask'))

            except (FileNotFoundError, IOError) as e:
                # DEV: Ловим ожидаемые ошибки чтения файлов и просто пропускаем элемент,
                # выводя предупреждение. Это делает пайплайн более устойчивым.
                print(f"\n[WARNING] Skipping item '{item_context.get('original_id')}' due to a file error: {e}")
            except Exception as e:
                # DEV: Ловим все остальные, более серьезные ошибки. Выводим ошибку,
                # но продолжаем работу со следующим элементом.
                print(f"\n[ERROR] Failed to process item '{item_context.get('original_id')}'. Error: {e}")
                # Для отладки можно добавить: import traceback; traceback.print_exc()

        # DEV: Здесь на Шаге 6 мы будем сохранять финальные артефакты.
        # print("Saving artifacts...")
        # manifest_builder.save(output_root / "manifest.jsonl")
        # dataset_map_generator.generate(output_root, stats_calculator.get_results(), ...)
        
        print("\nPreparation finished successfully.")