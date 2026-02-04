"""
GA-Power-Systems: Geometric Algebra Power Theory Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="gapot",
    version="1.0.0",
    author="Francisco G. Montoya, Alfredo Alcayde, Francisco M. Arrabal-Campos",
    author_email="pagilm@ual.es",
    description="Geometric Algebra Power Theory Framework for Power Systems Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fmontoyaual/ga-power-systems",
    project_urls={
        "Bug Tracker": "https://github.com/fmontoyaual/ga-power-systems/issues",
        "Documentation": "https://github.com/fmontoyaual/ga-power-systems/docs",
        "Source": "https://github.com/fmontoyaual/ga-power-systems",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "clifford>=1.4.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "plotly>=5.0.0",
            "pandas>=1.3.0",
        ],
    },
    keywords=[
        "geometric algebra",
        "clifford algebra",
        "power systems",
        "power theory",
        "harmonic analysis",
        "electrical engineering",
        "GAPoT",
        "power quality",
    ],
)
