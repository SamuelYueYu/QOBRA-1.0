import os, difflib, sys
folder_name=os.getcwd()

# Go to folder
os.chdir('Training data')
keys_target = sys.argv[1:-3]

from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametertable import ParameterView

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import BlueprintCircuit, RealAmplitudes

r = int(sys.argv[-2])
num_tot,num_trash = int(sys.argv[-3]),0
num_latent = num_tot-num_trash

dim_tot = 2**num_tot
dim_latent = 2**num_latent

num_feature = 2**num_tot
num_encode = (r+1)*num_tot
num_decode = num_encode

fm_i = RawFeatureVector(num_feature)
i_params = [Parameter(fr'$\iota_{{{i}}}$') for i in range(num_feature)]
fm_l = RawFeatureVector(dim_latent)
l_params = [Parameter(fr'$\lambda_{{{i}}}$') for i in range(dim_latent)]

def ansatz(num_qubits, r, prefix):
    return RealAmplitudes(num_qubits, entanglement="full", reps=r, parameter_prefix = prefix)

e = ansatz(num_tot, r, "e")
d = ansatz(num_tot, r, "d")

e_params = [Parameter(fr'$\epsilon_{{{i}}}$') for i in range(num_encode)]
d_params = [Parameter(fr'$\delta_{{{i}}}$') for i in range(num_decode)]
#n_params = [Parameter(fr'$\nu_{{{i}}}$') for i in range(num_latent)]