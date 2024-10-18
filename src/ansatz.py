import os, difflib, sys
folder_name=os.getcwd()

# Go to folder
os.chdir('Training data(max)')
keys_target = sys.argv[2:-1]

from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametertable import ParameterView
from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit.circuit.library import BlueprintCircuit, RealAmplitudes
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

re,rd = 1,1
num_latent,num_trash = int(sys.argv[1]),0
dim = 2**num_latent
num_tot = num_latent+num_trash

num_feature = 2**num_tot
num_encode,num_decode = (re+1)*num_tot,(rd+1)*num_tot

fm_i = RawFeatureVector(num_feature)
i_params = [Parameter(fr'$\iota_{{{i}}}$') for i in range(num_feature)]
fm_l = RawFeatureVector(dim)
l_params = [Parameter(fr'$\lambda_{{{i}}}$') for i in range(dim)]

def ansatz(num_qubits, r, prefix):
    return RealAmplitudes(num_qubits, entanglement="full", reps=r, parameter_prefix = prefix)

e = ansatz(num_tot, re, "e")
d = ansatz(num_tot, rd, "d")

e_params = [Parameter(fr'$\epsilon_{{{i}}}$') for i in range(num_encode)]
d_params = [Parameter(fr'$\delta_{{{i}}}$') for i in range(num_decode)]
n_params = [Parameter(fr'$\nu_{{{i}}}$') for i in range(num_latent)]

# Custom style with adjusted parameter font size
style = {
    'fontsize': 12,        # Font size for gate names
    'label_font_size': 8, # Font size for qubit labels
    'subfont_size': 20     # Font size for parameter names (like θ)
}
