<div align="center">
  <img src="https://github.com/SamuelYueYu/QOBRA-1.0/blob/main/assets/QOBRA_logo_gradient_resized.png">
</div>

# QOBRA 1.0
**Quantum Operator-Based Real-Amplitude autoencoder (QOBRA)** for molecular design and optimization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamuelYueYu/QOBRA-1.0/blob/main/src/QOBRA_demo.ipynb)

## Overview
QOBRA is a quantum machine learning framework designed for molecular design and optimization. This repository demonstrates the algorithm's capabilities through a proof-of-concept implementation focused on generating protein sequences that bind to specific metal ions (metalloproteins). The ultimate goal is to extend this approach for broader molecular design applications.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
    1. [Prerequisites](#prerequisites)
    2. [Steps to Install](#steps-to-install)
3. [Usage](#usage)
    1. [Training](#training)
    2. [Generation](#generation)
4. [Configuration](#configuration)
5. [Dataset](#dataset)
6. [Contact](#contact)
7. [License](#license)

## Features
- **Quantum-enhanced autoencoder**: Leverages quantum computing principles for molecular representation learning
- **Metal ion binding prediction**: Specialized architecture for metalloprotein sequence generation
- **Scalable training**: Support for multi-GPU training with CUDA acceleration
- **Flexible molecular design**: Extensible framework for various molecular design tasks
- **Comprehensive evaluation**: Built-in metrics for assessing generated sequences
- **Interactive demonstrations**: Jupyter notebook tutorials and examples

## Installation
### Prerequisites
- Python 3.11.5 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Steps to Install
```bash
# Clone the repository
git clone https://github.com/SamuelYueYu/QOBRA-1.0.git
cd QOBRA-1.0

# Create a conda environment (recommended)
conda env create -f environment.yml
conda activate qobra

# Alternative: Install with pip
pip install -r requirements.txt
```

### Key Dependencies
- **Quantum Computing**: qiskit==1.4.2, qiskit-algorithms==0.3.1, qiskit-machine-learning==0.8.2
- **Machine Learning**: PyTorch 2.2.0 + cu121, numpy==1.26.4, scipy==1.15.2
- **Bioinformatics**: biopython==1.85
- **High-Performance Computing**: cupy-cuda12x==12.3.0, mpi4py==3.1.5

## Usage

### Training
Train the QOBRA model for specific metal ions:

```bash
# Train for different metal ions
# Parameters: [metal_type] [training_mode] [device_id]

# Calcium (Ca2+)
python train.py 6 1 0

# Magnesium (Mg2+) 
python train.py 2 1 0

# Zinc (Zn2+)
python train.py 8 1 0
```

**Parameters:**
- `metal_type`: Atomic number or identifier for the target metal ion
- `training_mode`: Training configuration (1 for standard training)
- `device_id`: GPU device ID (0 for first GPU, CPU if no GPU available)

### Generation
Generate new molecular sequences using trained models:

```bash
# Generate sequences for trained models
# Parameters: [metal_type] [generation_mode] [num_sequences]

# Generate sequences for Calcium binding
python generate.py 6 1 1

# Generate multiple sequences
python generate.py 6 1 10  # Generate 10 sequences
```

### Batch Processing
Use the provided shell scripts for convenient batch processing:

```bash
# Run training for all supported metals
./run-Ca.sh   # Calcium
./run-Mg.sh   # Magnesium  
./run-Zn.sh   # Zinc
```

## Configuration
The model configuration can be adjusted through:

1. **Command-line parameters**: Metal type, training mode, and device selection
2. **Configuration files**: Modify hyperparameters in the source code
3. **Environment variables**: Set `CUDA_VISIBLE_DEVICES` for GPU selection

## Dataset
The `dataset/` directory contains:
- Training sequences for different metal-binding proteins
- Validation datasets for model evaluation
- Preprocessing scripts for custom datasets

For custom molecular design tasks, prepare your data in the same format as the provided examples.

## Interactive Demo
Explore QOBRA's capabilities through the interactive Jupyter notebook:
```bash
jupyter notebook src/QOBRA_demo.ipynb
```

## Project Structure
```
QOBRA-1.0/
├── src/                    # Source code
├── seq_to_struct/          # Sequence-to-structure utilities
├── dataset/                # Training and validation data
├── assets/                 # Images and documentation assets
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment specification
└── run-*.sh               # Batch processing scripts
```

## Contributing
We welcome contributions to extend QOBRA for broader molecular design applications. Please feel free to submit issues, feature requests, or pull requests.

## Contact
For questions, comments, or support, please contact:

**Yue Yu** (samuel.yu@yale.edu)

**Francesco Calcagno** (francesco.calcagno@unibo.it)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This is a proof-of-concept implementation focused on metalloproteins. The QOBRA framework is designed to be extensible for various molecular design and optimization tasks.*
