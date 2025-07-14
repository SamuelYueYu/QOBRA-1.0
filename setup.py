#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qobra-pennylane",
    version="1.0.0",
    author="Samuel Yu",
    author_email="samuel.yu@yale.edu",
    description="Quantum Optimization Benchmark Library with PennyLane",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamuelYueYu/QOBRA-PennyLane",
    project_urls={
        "Bug Tracker": "https://github.com/SamuelYueYu/QOBRA-PennyLane/issues",
        "Documentation": "https://github.com/SamuelYueYu/QOBRA-PennyLane/docs",
        "Source Code": "https://github.com/SamuelYueYu/QOBRA-PennyLane",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
        "jax": ["pennylane[jax]", "jax", "jaxlib"],
        "tf": ["pennylane[tf]", "tensorflow>=2.3"],
        "torch": ["pennylane[torch]", "torch>=1.8"],
        "all": [
            "pennylane[jax]", "jax", "jaxlib",
            "pennylane[tf]", "tensorflow>=2.3",
            "pennylane[torch]", "torch>=1.8",
            "gurobipy>=9.0",
            "cplex>=20.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "qobra-benchmark=qobra_pennylane.cli:main",
        ],
    },
    keywords=[
        "quantum computing",
        "optimization",
        "benchmark",
        "pennylane",
        "qaoa",
        "vqe",
        "quantum algorithms",
        "combinatorial optimization",
    ],
    include_package_data=True,
    zip_safe=False,
)