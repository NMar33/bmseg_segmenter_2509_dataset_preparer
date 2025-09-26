# src/acquisition/http_acquirer.py

"""
An acquirer for downloading and extracting standard archives (zip, tar) from an HTTP(S) URL.
"""
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

from src.acquisition.base_acquirer import BaseAcquirer
from src.utils.file_utils import create_dir_if_not_exists


class HttpAcquirer(BaseAcquirer):
    """Acquires data by downloading and extracting a simple web archive."""

    def run(self, cache_root: Path) -> Path:
        url = self.params.get("url")
        if not url:
            raise ValueError("'url' parameter is required for http_archive acquirer.")

        filename = url.split("/")[-1]
        download_dir = cache_root / "downloads"
        extract_dir = cache_root / "extracted" / Path(filename).stem
        
        create_dir_if_not_exists(download_dir)

        # Idempotency check: if the data is already extracted, return the path.
        # DEV: Простая проверка на существование папки. Для надежности можно
        # добавить проверку наличия определенных файлов внутри, если нужно.
        if extract_dir.is_dir() and any(extract_dir.iterdir()):
            print(f"Data already extracted at '{extract_dir}'. Skipping acquisition.")
            return extract_dir

        # Download the file
        archive_path = download_dir / filename
        self._download(url, archive_path)
        
        # Extract the file
        print(f"Extracting '{archive_path.name}' to '{extract_dir}'...")
        create_dir_if_not_exists(extract_dir)
        self._unpack(archive_path, extract_dir)
        
        print("Extraction complete.")
        return extract_dir

    def _download(self, url: str, dest_path: Path):
        """Downloads a file with a progress bar."""
        if dest_path.exists():
            print(f"File '{dest_path.name}' already downloaded. Skipping.")
            return

        print(f"Downloading from {url} to {dest_path}...")
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(dest_path, "wb") as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=dest_path.name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except requests.RequestException as e:
            # Clean up partially downloaded file on error
            if dest_path.exists():
                dest_path.unlink()
            raise IOError(f"Failed to download file from {url}. Error: {e}") from e

    def _unpack(self, archive_path: Path, extract_to: Path):
        """Unpacks a zip or tar archive."""
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_to)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}. "
                             "Only .zip and .tar.* are supported by HttpAcquirer.")