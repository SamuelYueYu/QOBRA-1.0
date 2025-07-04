# Quantum Protein Sequence Generator - Codebase Documentation

## Overview

This codebase implements a quantum machine learning system for generating novel protein sequences with metal-binding sites. The system uses a quantum autoencoder architecture to learn the distribution of metal-binding proteins and generate new sequences with similar properties.

## Architecture

The system consists of several key components:

### 1. **Quantum Model (`model.py`, `ansatz.py`)**
- **Quantum Circuits**: Defines encoder, decoder, and full autoencoder circuits
- **Ansatz**: Uses RealAmplitudes ansatz with full entanglement for maximum expressivity
- **Parameters**: Manages trainable quantum parameters for the encoder/decoder

### 2. **Sequence Processing (`coding.py`, `inputs.py`)**
- **Encoding**: Converts amino acid sequences into quantum state vectors
- **Decoding**: Reconstructs sequences from quantum states
- **Data Preparation**: Loads and preprocesses protein sequences from PDB data
- **Token Mapping**: Creates numerical representations for amino acids and special tokens

### 3. **Training Pipeline (`train.py`, `cost.py`)**
- **Loss Function**: Uses Maximum Mean Discrepancy (MMD) to match target distribution
- **Optimization**: COBYLA optimizer for quantum parameter updates
- **Evaluation**: Reconstruction accuracy on training and test sets

### 4. **Generation System (`gen.py`, `gen_func.py`)**
- **De Novo Generation**: Creates novel sequences from latent space samples
- **Validation**: Filters sequences based on biological constraints
- **Visualization**: PyMol scripts for 3D structure visualization

### 5. **Analysis Tools (`count.py`)**
- **Statistics**: Comprehensive analysis of sequence properties
- **Visualization**: Distribution comparisons and training progress plots
- **Target Distribution**: Gaussian latent space for structured generation

## Key Features

### Quantum Autoencoder
- **Encoder**: Maps protein sequences to structured latent quantum states
- **Decoder**: Generates sequences from latent representations
- **Latent Space**: Gaussian-distributed quantum states for controllable generation

### Metal-Binding Focus
- **Special Tokens**: '+' markers identify metal-binding amino acids
- **Validation**: Ensures generated sequences have realistic binding patterns
- **Visualization**: Color-coded PyMol scripts highlight binding sites

### Quality Control
- **Novelty**: Generated sequences must differ from training data
- **Uniqueness**: No duplicate sequences in generated set
- **Validity**: Sequences must meet biological constraints (length, structure, etc.)

## File Descriptions

### Core Components
- `model.py`: Quantum circuit definitions for encoder/decoder
- `ansatz.py`: Quantum ansatz and parameter configuration
- `coding.py`: Sequence encoding/decoding functions
- `inputs.py`: Data loading and preprocessing
- `train.py`: Main training loop and optimization
- `cost.py`: Loss functions and training utilities
- `gen.py`: De novo sequence generation
- `gen_func.py`: Generation helper functions and visualization
- `count.py`: Statistical analysis and target distribution

### Data Flow
1. **Input**: Protein sequences with metal-binding annotations
2. **Encoding**: Convert sequences to quantum state vectors
3. **Training**: Optimize encoder to match target distribution
4. **Generation**: Sample latent space and decode to sequences
5. **Validation**: Filter sequences based on quality criteria
6. **Analysis**: Compare generated vs. training distributions

## Usage

The system is designed to be run from command line with the following pattern:
```bash
python script.py [metal_types] [num_qubits] [repetitions] [mode]
```

Where:
- `metal_types`: Target metals (e.g., Zn, Cu, Fe)
- `num_qubits`: Number of qubits in quantum system
- `repetitions`: Layers in quantum ansatz
- `mode`: 0 for fresh training, 1 for resuming

## Key Innovations

### Quantum Encoding
- Maps discrete amino acid sequences to continuous quantum states
- Preserves sequential and structural information
- Enables probabilistic generation with quantum superposition

### Structured Latent Space
- Gaussian target distribution ensures smooth interpolation
- Maximum Mean Discrepancy loss for distribution matching
- Controllable generation through latent space sampling

### Biological Constraints
- Metal-binding site preservation
- Chain length and structure validation
- Chemical property analysis and visualization

## Output Files

### Training Results
- `Results-{experiment}.txt`: Reconstruction accuracy summary
- `R-{experiment}.txt`: Detailed sequence comparisons
- `opt-e-{experiment}.pkl`: Trained encoder parameters

### Generation Results
- `denovo-{seed}.txt`: Generated sequences
- `Bar-{seed}-{experiment}.png`: Amino acid frequency comparisons
- `chains-{seed}-{experiment}.png`: Chain distribution plots
- `plus-{seed}-{experiment}.png`: Metal-binding site distributions

### Visualization
- `{index}.pml`: PyMol visualization scripts
- `{index}.txt`: Raw sequence files
- Individual protein structure folders

## Performance Metrics

### Training Metrics
- **MMD Loss**: Measures how well encoder matches target distribution
- **Reconstruction Rate**: Percentage of perfectly reconstructed sequences
- **Training Time**: Optimization duration

### Generation Metrics
- **Novel (N)**: Fraction of sequences not in training set
- **Unique (U)**: Fraction of non-duplicate sequences
- **Valid (V)**: Fraction meeting biological constraints

## Scientific Impact

This quantum machine learning approach enables:
- **Protein Design**: Generate novel metal-binding proteins
- **Drug Discovery**: Create proteins with specific binding properties
- **Enzyme Engineering**: Design catalysts with metal cofactors
- **Structural Biology**: Understand protein-metal interactions

The combination of quantum computing and protein biology opens new possibilities for computational protein design and understanding the fundamental principles of metal-binding proteins.