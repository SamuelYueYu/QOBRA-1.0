import os, pickle
from ansatz import *
from coding import *
from qiskit.visualization import circuit_drawer

qc_v = QuantumCircuit(num_tot)
qc_v = qc_v.compose(fm_i.assign_parameters(i_params))
qc_v.barrier()
qc_v = qc_v.compose(e.decompose().assign_parameters(e_params))
    
# Draw the circuit with the custom font sizes
#circuit_drawer(qc_v, output='mpl', style=style, filename=f'qc_v-{num_latent}.png')

# Training model
train_qc = QuantumCircuit(num_tot)
train_qc = train_qc.compose(fm_i.assign_parameters(i_params))
train_qc = train_qc.compose(e.assign_parameters(e_params))
train_qc.barrier()

for i in range(num_latent):
    train_qc.ry(n_params[i], i)
train_qc = train_qc.compose(d.assign_parameters(d_params).inverse())
#circuit_drawer(train_qc, output='mpl', style=style, filename=f'train_qc-{num_latent}.png')

# Decoder model
qc_d = QuantumCircuit(num_tot)
qc_d = qc_d.compose(fm_l.assign_parameters(l_params), range(num_latent))

qc_d.barrier()
for i in range(num_latent):
    qc_d.ry(n_params[i], i)
qc_d = qc_d.compose(d.assign_parameters(d_params).inverse())
#circuit_drawer(qc_d, output='mpl', style=style, filename=f'qc_d-{num_latent}.png')