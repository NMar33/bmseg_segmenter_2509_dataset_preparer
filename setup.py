from setuptools import setup, find_packages

setup(
    name="dataset_preparer",
    version="0.1.0",
    author="Your Name", # Замени на свое имя
    description="A utility for preparing and preprocessing computer vision datasets.",
    packages=find_packages(), # Автоматически найдет все наши модули в `src/`
    install_requires=[
        "numpy",
        "pyyaml",
        "opencv-python-headless",
        "imageio",
        "tifffile",
        "tqdm",
        "pydantic",
        "requests",
        "gdown",
        "rarfile",
    ],
    entry_points={
        'console_scripts': [
            'prepare-dataset=run_preparation:main',
        ],
    },
    python_requires='>=3.8',
)