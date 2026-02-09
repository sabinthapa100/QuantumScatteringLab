"""Setup configuration for QuantumScatteringLab."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantumscatteringlab",
    version="0.1.0",
    author="Sabin Thapa",
    author_email="sthapa3@kent.edu",
    description="Testing quantum simulation framework for lattice field theories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sabinthapa100/QuantumScatteringLab",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "qiskit>=1.0.0,<2.0.0",
        "qiskit-aer>=0.13.0,<1.0.0",
        "matplotlib>=3.7.0,<4.0.0",
    ],
    extras_require={
        "gpu": [
            "quimb>=1.7.0",
            "cupy-cuda12x>=12.0.0",
        ],
        "dev": [
            "pytest>=7.4.0,<8.0.0",
            "pytest-cov>=4.1.0,<5.0.0",
            "pytest-benchmark>=4.0.0,<5.0.0",
            "black>=23.0.0,<24.0.0",
            "flake8>=6.0.0,<7.0.0",
            "mypy>=1.5.0,<2.0.0",
            "isort>=5.12.0,<6.0.0",
        ],
        "ui": [
            "streamlit>=1.28.0,<2.0.0",
            "plotly>=5.17.0,<6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0,<8.0.0",
            "sphinx-rtd-theme>=1.3.0,<2.0.0",
        ],
        "all": [
            "quimb>=1.7.0",
            "pytest>=7.4.0",
            "streamlit>=1.28.0",
            "sphinx>=7.0.0",
        ],
    },
)
