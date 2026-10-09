#!/usr/bin/env python3
"""
Setup script for the retroBERT package
"""

from setuptools import setup, find_packages
import os


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


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
    author="Seung Jae Shin",
    maintainer="Seung Jae Shin",
    description="Susceptibility prediction from pre-stress pose dynamics using a BERT encoder",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ShinSeungJ/retroBERT",
    packages=find_packages(),
    # run_retrobert.py is a top-level module, not part of the package; it has to be
    # installed explicitly or the `retrobert` console script cannot import it.
    py_modules=["run_retrobert"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "retrobert=run_retrobert:main",
            "retrobert-train=retrobert.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "retrobert": ["*.py"],
    },
    zip_safe=False,
    keywords="deep learning, BERT, stress susceptibility, pose, behavioral analysis, time series",
)
