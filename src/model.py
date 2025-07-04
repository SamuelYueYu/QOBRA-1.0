import os, pickle
from ansatz import *
from coding import *

from qiskit.visualization import circuit_drawer
from qiskit_algorithms.utils import algorithm_globals

# Set random seed for reproducible quantum circuit parameter initialization
# This ensures consistent results across different runs during development
algorithm_globals.random_seed = 0

# Encoder model: Quantum circuit that encodes amino acid sequences into latent quantum states
# This circuit takes classical amino acid sequence data and maps it to quantum state space
# The encoder learns to compress protein sequences into a lower-dimensional quantum representation
qc_e = QuantumCircuit(num_tot)
qc_e = qc_e.compose(fm_i.assign_parameters(i_params))  # Feature map for input sequences
qc_e.barrier()  # Visual separator in circuit diagram
qc_e = qc_e.compose(e.assign_parameters(e_params))     # Encoder ansatz with learnable parameters

# Full model: Complete encoder-decoder circuit used during training
# This is the autoencoder architecture that learns to reconstruct input sequences
# The model is trained to minimize reconstruction error between input and output sequences
train_qc = QuantumCircuit(num_tot)
train_qc = train_qc.compose(fm_i.assign_parameters(i_params))  # Input feature map
train_qc = train_qc.compose(e.assign_parameters(e_params))     # Encoder (forward pass)
train_qc.barrier()  # Visual separator between encoder and decoder
train_qc = train_qc.compose(e.assign_parameters(e_params).inverse())  # Decoder (inverse of encoder)

# Decoder model: Quantum circuit that generates new sequences from latent representations
# This circuit takes quantum latent states and decodes them back to amino acid sequences
# Used for generating novel protein sequences with desired properties
qc_d = QuantumCircuit(num_tot)
qc_d = qc_d.compose(fm_l.assign_parameters(l_params), range(num_latent))  # Latent feature map
qc_d.barrier()  # Visual separator in circuit diagram
qc_d = qc_d.compose(e.assign_parameters(e_params).inverse())  # Decoder (inverse encoder)