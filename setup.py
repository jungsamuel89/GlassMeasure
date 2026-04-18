from setuptools import setup, find_packages

setup(
    name="glassmeasure",
    version="1.0.0",
    description="Glass surface measurement using SAM3 + LiDAR depth",
    author="Samuel Jung",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["templates/*.html", "static/*.css"]},
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.22.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "opencv-python>=4.8.0",
        "flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "samu=app:main",
        ],
    },
    python_requires=">=3.10",
)
