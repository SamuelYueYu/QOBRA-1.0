"""
QOBRA (Quantum Operator-Based Real-Amplitude autoencoder) - Ansatz Module

This module sets up the quantum circuit structure and parameters for the QOBRA system.
It defines the quantum ansatz (parameterized quantum circuits) used for encoding and
decoding molecular sequences. The current implementation demonstrates on protein sequences
but the framework is designed for general molecular design applications.

Key concepts:
- Ansatz: A parameterized quantum circuit that can be trained
- Feature maps: Quantum circuits that encode classical molecular data into quantum states
- Parameters: Trainable variables in the quantum circuits
"""

import os, difflib, sys
folder_name=os.getcwd()

# Navigate to the training data directory where molecular sequence data is stored
os.chdir('Training data')

# Extract command line arguments for sequence types and quantum circuit parameters
# keys_target: List of sequence types to analyze (e.g., ['Ca', 'Mg', 'Zn'] for metal-binding proteins)
keys_target = sys.argv[1:-3]

# Import quantum computing libraries
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametertable import ParameterView

# Import quantum machine learning components
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import BlueprintCircuit, RealAmplitudes

# Extract quantum circuit configuration from command line arguments
r = int(sys.argv[-2])  # Number of repetitions/layers in the ansatz
num_tot = int(sys.argv[-3])  # Total number of qubits
num_trash = 0  # Number of unused qubits (currently 0)
num_latent = num_tot - num_trash  # Number of qubits used for latent representation

# Calculate dimension sizes for quantum states
dim_tot = 2**num_tot  # Total dimension of quantum state space
dim_latent = 2**num_latent  # Dimension of latent space

# Define parameter counts for different circuit components
num_feature = 2**num_tot  # Number of features for input encoding
num_encode = (r+1)*num_tot  # Number of encoding parameters (depends on repetitions)
num_decode = num_encode  # Number of decoding parameters (symmetric)

# Create feature maps (quantum circuits that encode classical data)
# fm_i: Feature map for input sequences (maps classical sequence to quantum state)
fm_i = RawFeatureVector(num_feature)
# Create named parameters for input feature map (used for sequence encoding)
i_params = [Parameter(fr'$\iota_{{{i}}}$') for i in range(num_feature)]

# fm_l: Feature map for latent space (maps latent variables to quantum state)
fm_l = RawFeatureVector(dim_latent)
# Create named parameters for latent feature map (used for generation)
l_params = [Parameter(fr'$\lambda_{{{i}}}$') for i in range(dim_latent)]

def ansatz(num_qubits, r, prefix):
    """
    Creates a parameterized quantum circuit (ansatz) for training.
    
    This function generates a RealAmplitudes circuit which is a common ansatz
    for variational quantum algorithms. It consists of rotation gates and
    entangling gates repeated in layers.
    
    Parameters:
    - num_qubits: Number of qubits in the circuit
    - r: Number of repetitions/layers in the ansatz
    - prefix: String prefix for parameter names
    
    Returns:
    - RealAmplitudes circuit with full entanglement pattern
    """
    return RealAmplitudes(num_qubits, entanglement="full", reps=r, parameter_prefix=prefix)

# Create encoder and decoder ansatz circuits
# e: Encoder ansatz (transforms input sequences to latent representation)
e = ansatz(num_tot, r, "e")
# d: Decoder ansatz (transforms latent representation back to sequences)
d = ansatz(num_tot, r, "d")

# Create named parameters for encoder and decoder circuits
# e_params: Parameters for the encoder circuit (trainable weights)
e_params = [Parameter(fr'$\epsilon_{{{i}}}$') for i in range(num_encode)]
# d_params: Parameters for the decoder circuit (trainable weights)
d_params = [Parameter(fr'$\delta_{{{i}}}$') for i in range(num_decode)]

# Note: n_params (noise parameters) are commented out - could be used for
# noise modeling in future versions
#n_params = [Parameter(fr'$\nu_{{{i}}}$') for i in range(num_latent)]
