# src/acquisition/base_acquirer.py

"""
Abstract base class for all data acquirers.
"""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseAcquirer(ABC):
    """
    Defines the interface for acquiring raw dataset files.
    Implementations are responsible for downloading, extracting, and placing
    the data into a local cache directory.
    """

    def __init__(self, **params):
        """
        Initializes the acquirer with parameters from the config.
        
        Args:
            params: A dictionary of parameters from the 'acquisition' block
                    of the YAML config.
        """
        self.params = params

    @abstractmethod
    def run(self, cache_root: Path) -> Path:
        """
        Ensures the raw data is available locally and returns the path to it.
        This method must be idempotent: if data is already acquired, it should
        do nothing and just return the path.

        Args:
            cache_root: The root directory for storing all downloaded and
                        extracted data.

        Returns:
            The path to the root directory of the extracted dataset.
        """
        pass