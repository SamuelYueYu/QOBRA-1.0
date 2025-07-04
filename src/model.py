"""
QOBRA - Model Module

This module defines the quantum circuit models used in the QOBRA system.
It creates three main quantum circuits:
1. Encoder model - Transforms molecular sequences into latent quantum states
2. Decoder model - Reconstructs sequences from latent states
3. Full model - Complete autoencoder for training (encoder + decoder)

The quantum circuits use parameterized gates that are optimized during training
to learn meaningful representations of molecular sequences and their functional patterns.
"""

import os, pickle
from ansatz import *
from coding import *

from qiskit.visualization import circuit_drawer
from qiskit_algorithms.utils import algorithm_globals

# Set random seed for reproducible quantum circuit behavior
# This ensures consistent results across different runs
algorithm_globals.random_seed = 0

# =============================================
# ENCODER MODEL DEFINITION
# =============================================
# The encoder transforms input molecular sequences into quantum latent representations
# It consists of:
# 1. Input feature map (fm_i) - Encodes classical sequence data as quantum amplitudes
# 2. Parameterized ansatz (e) - Trainable quantum circuit for learning representations

qc_e = QuantumCircuit(num_tot)  # Create quantum circuit with required number of qubits

# Add input feature map layer
# This layer encodes the classical molecular sequence data into quantum amplitudes
qc_e = qc_e.compose(fm_i.assign_parameters(i_params))

# Add barrier for visual clarity (separates encoding from processing)
qc_e.barrier()

# Add encoder ansatz layer
# This parameterized quantum circuit learns to compress sequence information
qc_e = qc_e.compose(e.assign_parameters(e_params))

# =============================================
# FULL AUTOENCODER MODEL DEFINITION
# =============================================
# The full model is used for training and includes both encoder and decoder
# Architecture: Input → Encoder → Latent State → Decoder → Output
# The goal is to reconstruct the input sequence through the quantum latent space

train_qc = QuantumCircuit(num_tot)  # Create training circuit

# ENCODER PATH
# Add input feature map to encode classical sequence data
train_qc = train_qc.compose(fm_i.assign_parameters(i_params))

# Add encoder ansatz to compress information into latent space
train_qc = train_qc.compose(e.assign_parameters(e_params))

# Add barrier to separate encoder from decoder
train_qc.barrier()

# DECODER PATH
# Add inverse encoder ansatz to reconstruct from latent space
# Note: Using the same parameters as encoder but in reverse (inverse operation)
train_qc = train_qc.compose(e.assign_parameters(e_params).inverse())

# =============================================
# DECODER MODEL DEFINITION
# =============================================
# The decoder generates new molecular sequences from quantum latent representations
# It's used for de novo sequence generation after training
# Architecture: Latent State → Decoder → Output Sequence

qc_d = QuantumCircuit(num_tot)  # Create decoder circuit

# Add latent feature map layer
# This encodes latent variables (sampled from learned distribution) as quantum amplitudes
# Only uses the first num_latent qubits for latent representation
qc_d = qc_d.compose(fm_l.assign_parameters(l_params), range(num_latent))

# Add barrier for visual clarity
qc_d.barrier()

# Add inverse encoder ansatz to generate sequences from latent space
# Uses the trained encoder parameters in reverse to decode latent states
qc_d = qc_d.compose(e.assign_parameters(e_params).inverse())