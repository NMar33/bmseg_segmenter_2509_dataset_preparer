#run_preparation.py
"""
Main entry point for the dataset preparation utility.
This script reads a YAML configuration file and orchestrates the entire
data acquisition and preparation pipeline.
"""
import argparse
import pprint
from pathlib import Path

import yaml


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

    # Load the YAML configuration
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}") from e

    print("Successfully loaded configuration:")
    pprint.pprint(config)
    print("-" * 50)

    # DEV: Здесь будет начинаться основная логика.
    # На следующих шагах мы добавим вызовы Acquirer'а и Preparer'а.
    # acquirer = create_acquirer(config['acquisition'])
    # extracted_root = acquirer.run(...)
    # ...и так далее.

    print("Foundation is ready. Main logic will be added in the next steps.")


if __name__ == "__main__":
    main()