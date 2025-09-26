#src/config.py
"""
Application-wide constants and settings.
Using Pydantic for type validation and clear structure.
"""
from pathlib import Path
from pydantic import BaseModel

# DEV: Используем Pydantic, а не dataclasses, потому что он автоматически
# валидирует типы. Если в будущем мы будем загружать сюда конфиг из файла,
# это спасет от многих ошибок. Плюс, его модели легко расширять.

class ImageConstants(BaseModel):
    """Constants related to image properties and processing."""
    UINT16_MAX: int = 65535
    UINT8_MAX: int = 255


class AppSettings(BaseModel):
    """Main application settings container."""
    # DEV: Все глобальные "магические числа" и константы должны жить здесь.
    # Если понадобится что-то еще, добавляем сюда новый под-класс.
    IMAGE: ImageConstants = ImageConstants()
    DEFAULT_CACHE_DIR: Path = Path("./cache")


# Create a single, importable instance of the settings
SETTINGS = AppSettings()