import numpy as np
import cv2
import random
from pathlib import Path
import shutil

# ==============================================================================
# Parameters - Настройте генератор здесь!
# ==============================================================================
ROOT_DIR = Path("./tmp/dummy_dataset")
IMAGE_SIZE = (2048, 2048)  # H, W
SPLITS = {
    "train": 30,  # Количество изображений в 'train'
    "val": 10      # Количество изображений в 'val'
}

# DEV: Новые параметры для управления "реалистичностью"
BACKGROUND_NOISE_INTENSITY = 20.0  # Интенсивность Гауссова шума на фоне
SHAPE_COUNT_RANGE = (1, 2)         # Генерировать от 1 до 4 фигур на маске
SHAPE_INTENSITY_OFFSET = 60.0      # Насколько фигуры будут ярче (+) или темнее (-) фона

# ==============================================================================
# Cleanup and Setup
# ==============================================================================
if ROOT_DIR.exists():
    print(f"Removing existing dummy dataset at: {ROOT_DIR}")
    shutil.rmtree(ROOT_DIR)

RAW_ROOT = ROOT_DIR / "raw"
print(f"Creating new dummy dataset at: {RAW_ROOT}")

# ==============================================================================
# Generators - Набор "кирпичиков" для создания изображений и масок
# ==============================================================================

# --- Background Generators ---
def generate_gradient_background(h, w):
    """Создает фон с градиентом (может быть диагональным, вертикальным и т.д.)."""
    direction = random.choice(['h', 'v', 'd1', 'd2'])
    if direction == 'h':
        base = np.fromfunction(lambda y, x: x, (h, w))
    elif direction == 'v':
        base = np.fromfunction(lambda y, x: y, (h, w))
    elif direction == 'd1':
        base = np.fromfunction(lambda y, x: x + y, (h, w))
    else: # d2
        base = np.fromfunction(lambda y, x: x - y, (h, w))
    # DEV: Ограничиваем диапазон яркости фона, чтобы фигуры были хорошо видны
    return ((base - base.min()) / (base.max() - base.min()) * 150 + 50).astype(np.uint8)

def generate_noise_background(h, w):
    """Создает фон с Гауссовым шумом."""
    noise = np.random.normal(128, 40, (h, w))
    return np.clip(noise, 0, 255).astype(np.uint8)

# --- Shape Drawing Functions (рисуют на существующей маске) ---
def draw_random_circle(mask):
    h, w = mask.shape
    center = (random.randint(0, w), random.randint(0, h))
    radius = random.randint(min(h, w) // 12, min(h, w) // 5)
    cv2.circle(mask, center, radius, 255, -1)

def draw_random_ellipse(mask):
    h, w = mask.shape
    center = (random.randint(0, w), random.randint(0, h))
    axes = (random.randint(w // 10, w // 4), random.randint(h // 10, h // 4))
    angle = random.randint(0, 360)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

def draw_random_polygon(mask):
    h, w = mask.shape
    num_vertices = random.randint(3, 7)
    points = np.random.randint(0, min(h, w), size=(num_vertices, 2))
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

def draw_random_wavy_line(mask):
    h, w = mask.shape
    num_points = 100
    thickness = random.randint(8, 25)
    amplitude = random.randint(h // 12, h // 5)
    period = random.uniform(0.5, 2)
    
    y_offset = random.randint(amplitude, h - amplitude)
    x = np.linspace(0, w, num_points)
    y = y_offset + amplitude * np.sin(2 * np.pi * x / (w * period))
    
    points = np.vstack((x, y)).astype(np.int32).T
    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness)

# ==============================================================================
# Main generation loop
# ==============================================================================
background_generators = [generate_gradient_background, generate_noise_background]
shape_drawers = [draw_random_circle, draw_random_ellipse, draw_random_polygon, draw_random_wavy_line]

for split_name, num_images in SPLITS.items():
    img_dir = RAW_ROOT / "images" / split_name
    msk_dir = RAW_ROOT / "masks" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} images/masks for '{split_name}' split...")

    for i in range(num_images):
        # 1. Сгенерировать фон
        bg_func = random.choice(background_generators)
        background = bg_func(*IMAGE_SIZE)

        # 2. Создать пустую маску и нарисовать на ней несколько фигур
        mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        num_shapes = random.randint(*SHAPE_COUNT_RANGE)
        for _ in range(num_shapes):
            shape_func = random.choice(shape_drawers)
            shape_func(mask)
        
        # 3. *** КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: "Впечатываем" фигуры в изображение ***
        # DEV: Мы используем маску, чтобы изменить яркость пикселей фона там,
        # где находятся наши фигуры. Это основной фикс.
        image_float = background.astype(np.float32)
        # Добавляем или вычитаем яркость в местах, где маска > 0
        image_float[mask > 0] += SHAPE_INTENSITY_OFFSET
        
        # 4. Добавить немного общего шума поверх всего изображения
        noise = np.random.normal(0, BACKGROUND_NOISE_INTENSITY, IMAGE_SIZE)
        image_float += noise
        
        # 5. Привести финальное изображение к формату uint8
        image = np.clip(image_float, 0, 255).astype(np.uint8)

        # 6. Сохранить файлы
        filename = f"sample_{split_name}_{i:03d}.png"
        cv2.imwrite(str(img_dir / filename), image)
        cv2.imwrite(str(msk_dir / filename), mask)

print("\nDummy dataset created successfully!")
print(f"Raw data path: {RAW_ROOT}")

# ==============================================================================
# Generate a sample config file (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ==============================================================================
# DEV: FIX - Используем .as_posix() для путей, чтобы избежать проблем с
# обратными слэшами на Windows при парсинге YAML.
# Это делает конфиг кросс-платформенным.
prepared_path_posix = (ROOT_DIR / 'prepared').as_posix()
raw_path_posix = RAW_ROOT.as_posix()

config_content = f"""
dataset_id: "dummy_v1_realistic"
prepared_root: "{prepared_path_posix}"

source:
  # Path to the raw data we just created
  extracted_root: "{raw_path_posix}"

preparation:
  preparer: folder
  # Paths relative to extracted_root
  img_base_rel: "images"
  msk_base_rel: "masks"
  splits: {list(SPLITS.keys())}

# Empty pipeline will default to Passthrough
processing_pipeline: []

writer:
  image_format:
    ext: .tif
    dtype: uint16
  mask_format:
    ext: .png
    dtype: uint8
"""

config_path = Path("configs/dummy_dataset_config.yaml")
config_path.parent.mkdir(exist_ok=True)
config_path.write_text(config_content)

print(f"\nSample configuration file created at: {config_path}")
print("You can now run the preparer with:")
print(f"python run_preparation.py --config {config_path}")