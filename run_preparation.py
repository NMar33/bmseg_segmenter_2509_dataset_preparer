#run_preparation.py

"""
Main entry point for the dataset preparation utility.
This script reads a YAML configuration file and orchestrates the entire
data acquisition and preparation pipeline.
"""
import argparse
import pprint
from pathlib import Path
from typing import Dict, Any

import yaml

# DEV: Импортируем все необходимые классы для фабрик.
# Это центральная точка, где регистрируются все "строительные блоки" системы.
from src.acquisition.base_acquirer import BaseAcquirer
from src.acquisition.http_acquirer import HttpAcquirer
from src.acquisition.urisc_acquirer import UriscAcquirer

from src.preparers.base_preparer import BasePreparer
from src.preparers.folder_preparer import FolderPreparer
from src.preparers.volume_preparer import VolumePreparer

# DEV: Каталог (словарь) — это простой и эффективный способ реализовать "фабрику".
# Он сопоставляет строковое значение из конфига (e.g., `strategy: http_archive`)
# с классом в коде. Это позволяет нам легко добавлять новые стратегии,
# не меняя основную логику.
ACQUIRER_CATALOG = {
    "http_archive": HttpAcquirer,
    "urisc_gdrive_rar": UriscAcquirer,
}

PREPARER_CATALOG = {
    "folder": FolderPreparer,
    "volume": VolumePreparer,
}


def create_acquirer(config: Dict[str, Any]) -> BaseAcquirer | None:
    """
    Factory function to create an acquirer instance based on the configuration.

    Args:
        config: The 'acquisition' block from the main configuration.

    Returns:
        An initialized instance of a BaseAcquirer subclass, or None if the
        acquisition step is not defined.
    """
    strategy_name = config.get("strategy")
    if not strategy_name:
        # DEV: Если в конфиге нет блока `acquisition` или поля `strategy`,
        # мы просто возвращаем None. Это означает, что пользователь
        # должен предоставить данные локально.
        return None

    acquirer_class = ACQUIRER_CATALOG.get(strategy_name)
    if not acquirer_class:
        raise ValueError(f"Unknown acquisition strategy '{strategy_name}'. "
                         f"Available strategies are: {list(ACQUIRER_CATALOG.keys())}")

    return acquirer_class(**config)


def create_preparer(config: Dict[str, Any]) -> BasePreparer:
    """
    Factory function to create a preparer instance based on the configuration.

    Args:
        config: The full application configuration dictionary.

    Returns:
        An initialized instance of a BasePreparer subclass.
    """
    # DEV: Мы передаем в конструктор Preparer'а только те части конфига,
    # которые ему действительно нужны. Это хороший принцип — не передавать
    # весь гигантский конфиг в каждый объект, за исключением случаев, когда
    # он нужен целиком (как `full_config` для DatasetMap).
    preparation_config = config.get('preparation', {})
    pipeline_config = config.get('processing_pipeline', [])
    writer_config = config.get('writer', {})
    artifacts_config = config.get('artifacts', {})

    preparer_name = preparation_config.get('preparer')
    if not preparer_name:
        raise ValueError("Configuration error: 'preparation.preparer' key is missing.")

    preparer_class = PREPARER_CATALOG.get(preparer_name)
    if not preparer_class:
        raise ValueError(f"Unknown preparer type '{preparer_name}'. "
                         f"Available types are: {list(PREPARER_CATALOG.keys())}")

    return preparer_class(
        prep_config=preparation_config,
        pipeline_config=pipeline_config,
        writer_config=writer_config,
        artifacts_config=artifacts_config,
        full_config=config  # Pass the entire config for the dataset map
    )


def main():
    """Parses command line arguments, loads config, and starts the process."""
    parser = argparse.ArgumentParser(description="Dataset Preparation Utility")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiment."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Load the YAML configuration with UTF-8 encoding for safety
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}") from e

    print("--- Configuration Loaded ---")
    pprint.pprint(config)
    print("-" * 50)

    # Define common paths from the configuration
    output_root = Path(config['prepared_root']) / config['dataset_id']
    cache_root = Path(config.get('cache_root', './cache'))

    # --- Stage 1: Acquisition ---
    print("--- Stage 1: ACQUISITION ---")

    acquirer = create_acquirer(config.get('acquisition', {}))

    if acquirer:
        extracted_root = acquirer.run(cache_root=cache_root)
        print(f"Data acquired and available at: {extracted_root}")
    elif 'source' in config and 'extracted_root' in config['source']:
        # Fallback for local, pre-existing data if no acquirer is defined
        extracted_root = Path(config['source']['extracted_root'])
        print(f"Using pre-existing local data at: {extracted_root}")
        if not extracted_root.is_dir():
            raise FileNotFoundError(f"The specified 'source.extracted_root' does not exist: {extracted_root}")
    else:
        raise ValueError("Configuration must contain either an 'acquisition' block "
                         "or a 'source.extracted_root' path to the data.")

    # --- Stage 2: Preparation ---
    print("-" * 50)
    print("--- Stage 2: PREPARATION ---")

    # Create the appropriate preparer using the factory
    preparer = create_preparer(config)

    # Run the main preparation process
    preparer.run(extracted_root=extracted_root, output_root=output_root)

    print("-" * 50)
    print(f"Process finished successfully. Prepared dataset is available at: {output_root}")


if __name__ == "__main__":
    main()