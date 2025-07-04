# QOBRA 1.0
Quantum Operator-Based Real Ansatz (QOBRA) for generating protein sequences that bind to specific metal ions

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamuelYueYu/QOBRA-1.0/blob/main/src/QOBRA_demo.ipynb)

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
    1. [Prerequisites](#prerequisites)
    2. [Steps to Install](#steps-to-install)
3. [Usage](#usage)
4. [Configuration](#configuration)
    1. [Configuration File](#configuration-file)
    2. [Environment Variables](#environment-variables)
5. [Contact](#contact)
6. [License](#license)

## Features
- biopython==1.85
- cupy-cuda12x==12.3.0
- mpi4py==3.1.5
- numpy==1.26.4
- qiskit==1.4.2
- qiskit-algorithms==0.3.1
- qiskit-ibm-runtime==0.37.0
- qiskit-machine-learning==0.8.2
- scipy==1.15.2
- PyTorch 2.2.0 + cu121 package

## Installation
### Prerequisites
- Python 3.11.5

### Steps to Install
```bash
# Example installation steps
git clone https://github.com/SamuelYueYu/Metalloprotein.git
cd project
npm install
```

## Usage
```
# Procedure to train & use network to generate new sequences
python train.py 6 1 0
python generate.py 6 1 1
```

## Contact
For questions, comments, or support, please contact:

Yue Yu (samuel.yu@yale.edu)

Francesco Calcagno (francesco.calcagno@unibo.it)
