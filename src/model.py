import os, pickle
from ansatz import *
from coding import *

from qiskit.visualization import circuit_drawer
from qiskit_algorithms.utils import algorithm_globals

# Trash training
algorithm_globals.random_seed = 0

# Encoder model
qc_e = QuantumCircuit(num_tot)
qc_e = qc_e.compose(fm_i.assign_parameters(i_params))
qc_e.barrier()
qc_e = qc_e.compose(e.assign_parameters(e_params))

# Full model
train_qc = QuantumCircuit(num_tot)
train_qc = train_qc.compose(fm_i.assign_parameters(i_params))
train_qc = train_qc.compose(e.assign_parameters(e_params))
train_qc.barrier()
train_qc = train_qc.compose(e.assign_parameters(e_params).inverse())

# Decoder model
qc_d = QuantumCircuit(num_tot)
qc_d = qc_d.compose(fm_l.assign_parameters(l_params), range(num_latent))
qc_d.barrier()
qc_d = qc_d.compose(e.assign_parameters(e_params).inverse())