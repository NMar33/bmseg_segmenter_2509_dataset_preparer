"""
An acquirer for the U-RISC dataset, which involves downloading from Google Drive
and extracting multiple RAR archives.
"""
from pathlib import Path
import os
import shutil

from src.acquisition.base_acquirer import BaseAcquirer
from src.utils.file_utils import create_dir_if_not_exists, run_subprocess

class UriscAcquirer(BaseAcquirer):
    """
    Acquires the U-RISC dataset.
    - Downloads a folder from Google Drive using `gdown`.
    - Extracts `imgs.rar` and `label.rar` using `unrar`.
    """

    def run(self, cache_root: Path) -> Path:
        gdrive_url = self.params.get("gdrive_url")
        if not gdrive_url:
            raise ValueError("'gdrive_url' parameter is required for urisc_gdrive_rar acquirer.")

        # Define paths
        # DEV: Мы создаем уникальные подпапки, чтобы кеш от разных датасетов
        # не перемешивался.
        download_dir = cache_root / "downloads" / "urisc_gdrive"
        extract_dir = cache_root / "extracted" / "urisc"

        # Idempotency check
        # DEV: Проверяем наличие ключевых подпапок, которые создаются после распаковки.
        # Это более надежно, чем просто проверять существование `extract_dir`.
        if (extract_dir / "imgs").is_dir() and (extract_dir / "label").is_dir():
            print(f"U-RISC data already extracted at '{extract_dir}'. Skipping acquisition.")
            return extract_dir

        # Ensure system dependencies are met
        self._check_dependencies()
        
        # 1. Download from Google Drive
        create_dir_if_not_exists(download_dir)
        print(f"Downloading U-RISC dataset from Google Drive...")
        # DEV: `gdown` может быть капризным. Оборачиваем в try/except, чтобы дать
        # пользователю понятное сообщение об ошибке.
        try:
            # Note: gdown needs the path as a string
            run_subprocess(["gdown", "--folder", gdrive_url, "-O", str(download_dir), "--fuzzy"])
        except RuntimeError as e:
            print("\nERROR: Failed to download from Google Drive using gdown.")
            print("Please ensure 'gdown' is installed and you have access to the URL.")
            raise e

        # 2. Extract RAR archives
        create_dir_if_not_exists(extract_dir)
        print("Extracting RAR archives...")
        
        # DEV: Здесь мы "хардкодим" знание о структуре скачанного архива.
        # Это и есть "дзен" - прячем специфику в одном месте.
        archives_to_extract = {
            "imgs.rar": extract_dir / "imgs",
            "label.rar": extract_dir / "label",
        }

        try:
            for rar_name, dest_path in archives_to_extract.items():
                archive_path = download_dir / rar_name
                if not archive_path.exists():
                    raise FileNotFoundError(f"Required archive '{rar_name}' not found after download.")
                
                print(f"Extracting '{rar_name}' to '{dest_path}'...")
                create_dir_if_not_exists(dest_path)
                # Note: `unrar x` extracts with full path. `-o+` overwrites existing files.
                run_subprocess(["unrar", "x", "-o+", str(archive_path), str(dest_path)])
        except (RuntimeError, FileNotFoundError) as e:
            print("\nERROR: Failed to extract RAR archives.")
            print("Please ensure 'unrar' command is available in your system's PATH.")
            print("On Debian/Ubuntu, you can install it with: sudo apt-get install unrar")
            raise e

        print("U-RISC dataset acquired successfully.")
        return extract_dir
        
    def _check_dependencies(self):
        """Checks if required command-line tools are available."""
        if not shutil.which("gdown"):
            raise RuntimeError("`gdown` command not found. Please install it via `pip install gdown`.")
        if not shutil.which("unrar"):
            raise RuntimeError(
                "`unrar` command not found. Please install it using your system's package manager "
                "(e.g., `sudo apt-get install unrar` on Debian/Ubuntu)."
            )