"""
Setup script for LITE++
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="litepp",
    version="1.0.0",
    author="AntVision AI Research",
    description="LITE++: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time MOT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/litepp",
    packages=find_packages(exclude=["experiments", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "scikit-learn>=1.0.0",
        ],
    },
)
