import os, difflib, sys
folder_name=os.getcwd()

# Navigate to the training data directory containing metal-binding protein sequences
# This directory contains processed protein sequences organized by metal type
os.chdir('Training data')

# Extract target metal types from command line arguments
# These specify which metal-binding proteins to focus on (e.g., Zn, Cu, Fe, etc.)
keys_target = sys.argv[1:-3]

from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametertable import ParameterView

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import BlueprintCircuit, RealAmplitudes

# Extract quantum system parameters from command line arguments
# r: Number of repetitions in the quantum ansatz (controls model complexity)
# num_tot: Total number of qubits in the quantum system
r = int(sys.argv[-2])
num_tot,num_trash = int(sys.argv[-3]),0  # num_trash: Reserved qubits (currently unused)
num_latent = num_tot-num_trash  # Number of qubits used for latent representation

# Calculate dimensions of quantum state spaces
# These determine the capacity of the quantum system to represent information
dim_tot = 2**num_tot        # Total dimension of quantum state space
dim_latent = 2**num_latent  # Dimension of latent quantum state space

# Calculate number of parameters for different components
# These determine the expressivity and learning capacity of the quantum model
num_feature = 2**num_tot    # Parameters for input feature encoding
num_encode = (r+1)*num_tot  # Parameters for encoder (depends on ansatz repetitions)
num_decode = num_encode     # Parameters for decoder (same as encoder)

# Define feature maps for encoding classical data into quantum states
# fm_i: Maps input amino acid sequences to quantum states
# fm_l: Maps latent representations to quantum states for generation
fm_i = RawFeatureVector(num_feature)
i_params = [Parameter(fr'$\iota_{{{i}}}$') for i in range(num_feature)]  # Input parameters
fm_l = RawFeatureVector(dim_latent)
l_params = [Parameter(fr'$\lambda_{{{i}}}$') for i in range(dim_latent)]  # Latent parameters

def ansatz(num_qubits, r, prefix):
    """
    Create a variational quantum ansatz for the encoder/decoder
    
    Args:
        num_qubits: Number of qubits in the circuit
        r: Number of repetitions (layers) in the ansatz
        prefix: Prefix for parameter naming
    
    Returns:
        RealAmplitudes ansatz with full entanglement pattern
    
    This ansatz uses RealAmplitudes with full entanglement, which creates
    a highly expressive quantum circuit capable of representing complex
    relationships between amino acids in protein sequences.
    """
    return RealAmplitudes(num_qubits, entanglement="full", reps=r, parameter_prefix = prefix)

# Create encoder and decoder ansatz circuits
# These are the core components that learn to compress and decompress protein sequences
e = ansatz(num_tot, r, "e")  # Encoder ansatz
d = ansatz(num_tot, r, "d")  # Decoder ansatz (currently unused, decoder uses inverse encoder)

# Define trainable parameters for the quantum circuits
# These parameters are optimized during training to learn protein sequence patterns
e_params = [Parameter(fr'$\epsilon_{{{i}}}$') for i in range(num_encode)]  # Encoder parameters
d_params = [Parameter(fr'$\delta_{{{i}}}$') for i in range(num_decode)]   # Decoder parameters
#n_params = [Parameter(fr'$\nu_{{{i}}}$') for i in range(num_latent)]    # Noise parameters (commented out)
