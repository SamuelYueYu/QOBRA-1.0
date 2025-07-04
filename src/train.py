from cost import *
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA

# Main training loop for the quantum encoder
# This optimizes the encoder parameters to minimize MMD loss between
# encoded sequences and target Gaussian distribution

print("Encoder training")
start = time.time()

# Initialize COBYLA optimizer with maximum iterations
# COBYLA (Constrained Optimization BY Linear Approximations) is well-suited
# for quantum optimization problems as it handles noisy objective functions
opt = COBYLA(maxiter=500)

# Create partial function for optimization with fixed training and test data
# This allows the optimizer to call the loss function with only parameter updates
f = partial(e_loss, train_input=train_seqs, 
            test_input=test_seqs)

# Run optimization to find optimal encoder parameters
# The optimizer will iteratively update parameters to minimize MMD loss
opt_result = opt.minimize(fun=f, x0=xe)

# Store optimized parameters for later use
# These parameters represent the learned encoder that maps sequences to latent space
xe = opt_result.x
with open(f'{S}/opt-e-{S}.pkl', 'wb') as F:
    pickle.dump(xe, F)

# Calculate and report training time
elapsed_e = (time.time()-start) / 3600
print(f"Fit in {elapsed_e:0.2f} h")

# Evaluation phase: Test the trained autoencoder on reconstruction tasks
# This phase evaluates how well the model can reconstruct input sequences
# after encoding them to latent space and decoding back

start = time.time()

# Initialize results files for storing reconstruction performance
file = open(f"{S}/Results-{S}.txt", "w")
file.write("Dataset\tSize\tR\n")  # Header: Dataset, Size, Reconstruction Rate
file.close()

# Initialize detailed results file for individual sequence comparisons
file = open(f"{S}/R-{S}.txt", "w")
file.write("TRAINING SET\n")
file.close()

# Evaluate reconstruction performance on training set
# This shows how well the model has learned to reconstruct training sequences
output(train_seqs, head, "Train")

# Evaluate reconstruction performance on test set
# This shows the model's generalization ability on unseen sequences
file = open(f"{S}/R-{S}.txt", "a")
file.write("TEST SET\n")
file.close()
output(test_seqs, head, "Test")

# Calculate and report total evaluation time
elapsed_p = (time.time()-start)/60
print(f"Printed in {elapsed_p:.2f} min")

# Report total experiment time (training + evaluation)
print(f"Finished in {elapsed_e + elapsed_p/60:.2f} h")