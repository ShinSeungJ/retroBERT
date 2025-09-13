#!/usr/bin/env python3
"""
Setup script for retroBERT package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

# Read version from __init__.py
def get_version():
    version_file = os.path.join("retrobert", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="retrobert",
    version=get_version(),
    author="Your Name",  # Replace with actual author name
    author_email="your.email@institution.edu",  # Replace with actual email
    description="Stress Susceptibility Prediction using BERT-based Deep Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/retroBERT",  # Replace with actual repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "accelerate": [
            "accelerate>=0.20.0",
        ],
        "visualization": [
            "tensorboard>=2.8.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "retrobert=run_retrobert:main",
            "retrobert-main=retrobert.main:main",
            "retrobert-demo-inference=retrobert.demo_inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "retrobert": ["*.py"],
    },
    zip_safe=False,
    keywords="deep learning, BERT, stress prediction, behavioral analysis, time series",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/retroBERT/issues",
        "Source": "https://github.com/yourusername/retroBERT",
        "Documentation": "https://github.com/yourusername/retroBERT#readme",
    },
)
