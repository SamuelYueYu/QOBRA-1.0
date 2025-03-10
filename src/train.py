from cost import *
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA

# Optimize the encoder
start = time.time()
opt = COBYLA(maxiter=500)
print("Encoder training")

f = partial(e_loss, train_input=train_seqs, 
            test_input=test_seqs)
opt_result = opt.minimize(fun=f, x0=xe)

# Store the result in a pickle file
xe = opt_result.x
with open(f'{S}/opt-e-{S}.pkl', 'wb') as F:
    pickle.dump(xe, F)

elapsed_e = (time.time()-start) / 3600
print(f"Fit in {elapsed_e:0.2f} h")

start = time.time()
file = open(f"{S}/Results-{S}.txt", "w")
file.write("Dataset\tSize\tR\n")
file.close()

file = open(f"{S}/R-{S}.txt", "w")
file.write("TRAINING SET\n")
file.close()
output(train_seqs, head, "Train")

file = open(f"{S}/R-{S}.txt", "a")
file.write("TEST SET\n")
file.close()
output(test_seqs, head, "Test")

elapsed_p = (time.time()-start)/60
print(f"Printed in {elapsed_p:.2f} min")
print(f"Finished in {elapsed_e + elapsed_p/60:.2f} h")