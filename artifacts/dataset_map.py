# src/artifacts/dataset_map.py

"""
Generates the final dataset_map.yaml file.
"""
import yaml
import time
from pathlib import Path
from typing import Dict, Any

class DatasetMapGenerator:
    """Assembles all metadata into the final dataset_map.yaml."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate(self, prepared_root: Path, stats: Dict[str, Any]):
        """Creates and saves the dataset_map.yaml file."""
        print(f"Generating dataset_map.yaml at {prepared_root}...")
        
        # DEV: Собираем всю информацию из конфига и результатов других артефактов
        # в одну большую, красивую структуру.
        dataset_map = {
            "dataset_id": self.config['dataset_id'],
            "prepared_info": {
                "root": str(prepared_root),
                "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "source_info": self.config.get('acquisition', self.config.get('source')),
            "processing_pipeline": self.config.get('processing_pipeline', []),
            "writer_config": self.config.get('writer'),
            "statistics": stats,
            # DEV: Добавим сюда другие полезные секции из конфига
            "citation": self.config.get('citation', 'N/A'),
            "notes": self.config.get('notes', []),
        }
        
        # Use a custom dumper to avoid ugly YAML aliases
        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        with open(prepared_root / "dataset_map.yaml", 'w') as f:
            yaml.dump(dataset_map, f, Dumper=NoAliasDumper, sort_keys=False, indent=2)