# src/artifacts/manifest.py

"""
Builds the manifest.jsonl file, which logs every processed item.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

class ManifestBuilder:
    """Collects and saves metadata for each processed item."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    def add_entry(self, original_context: Dict[str, Any], processed_context: Dict[str, Any], output_info: Dict[str, Any]):
        """
        Adds a new record to the manifest.

        Args:
            original_context: The context before the pipeline (contains source paths).
            processed_context: The context after the pipeline (contains ROI info, etc.).
            output_info: Information from the writer (contains output paths).
        """
        # DEV: Мы собираем самую полезную информацию из всех этапов,
        # чтобы можно было полностью отследить происхождение каждого файла.
        entry = {
            "id": output_info.get("id", processed_context['original_id']),
            "split": original_context['split'],
            "source": {
                "image_path": str(original_context.get('image_path', 'N/A')),
                "mask_path": str(original_context.get('mask_path', 'N/A')),
            },
            "prepared": {
                "image_path": output_info.get('image_path'),
                "mask_path": output_info.get('mask_path'),
            },
            "processing_info": {
                "roi_coords": processed_context.get('roi_coords'),
                "tile_index": processed_context.get('tile_index'),
                # DEV: Добавляем сюда любые другие полезные данные из контекста
            }
        }
        self._entries.append(entry)

    def save(self, path: Path):
        """Saves all collected entries to a .jsonl file."""
        print(f"Saving manifest with {len(self._entries)} entries to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for entry in self._entries:
                f.write(json.dumps(entry) + '\n')