from cost import *
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 0
opt = COBYLA(maxiter=1000)

# Create an initial set of network parameters
#if int(sys.argv[-1]) == 0:
xe = algorithm_globals.random.random(e.num_parameters)
xd = algorithm_globals.random.random(d.num_parameters)
#else:
    # To restart the optimization from where it left off, load the result
#    with open('opt-e-' + S + '.pkl', 'rb') as F:
#        loaded_result = pickle.load(F)
#    xe = loaded_result.x
#    with open('opt-d-' + S + '.pkl', 'rb') as F:
#        loaded_result = pickle.load(F)
#    xd = loaded_result.x

# optimize the encoder, fit the latent representation
print("Kernel training")
start = time.time()
f = partial(k_loss, train_input=input_states, 
            test_input=test_states, start=start)
opt_result_e = opt.minimize(fun=f, x0=xe)
elapsed_e = (time.time()-start) / 60
print(f"Fit in {elapsed_e:0.2f} min")

# Store the result in a pickle file
with open('opt-e-' + S + '.pkl', 'wb') as F:
    pickle.dump(opt_result_e, F)

start = time.time()
# optimize the decoder, maximize fidelity
print("\nFidelity training")
f = partial(f_loss, pe=opt_result_e.x, 
            train_input=input_states, 
            test_input=test_states, start=start)
opt_result_d = opt.minimize(fun=f, x0=xd)
elapsed_d = (time.time()-start) / 60
print(f"Fit in {elapsed_d:0.2f} min")

start = time.time()
# Store the result in a pickle file
with open(f'opt-d-{S}.pkl', 'wb') as F:
    pickle.dump(opt_result_d, F)

# Write mu & std to a .pkl file
with open(f'std-{S}.pkl', 'wb') as F:
    pickle.dump(std, F)
    
file = open(f"Results-{S}.txt", "w")
file.write("Dataset\tSize\tR\n")
file.close()
    
file = open(f"R-{S}.txt", "w")
file.write("TRAINING SET\n")
file.close()
output(input_states, opt_result_e.x, opt_result_d.x, 
       nstd, head, float_to_char, "Train")

file = open(f"R-{S}.txt", "a")
file.write("TEST SET\n")
file.close()
output(test_states, opt_result_e.x, opt_result_d.x, 
       nstd, head, float_to_char, "Test")

elapsed_p = (time.time()-start)/60
print(f"Printed in {elapsed_p:0.2f} min")
print(f"Total elapsed time: {(elapsed_e + elapsed_d + elapsed_p) / 60:0.2f} h")