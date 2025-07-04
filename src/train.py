"""
QOBRA - Training Module

This module implements the training process for the QOBRA quantum autoencoder.
It optimizes the encoder parameters to learn meaningful representations of molecular
sequences with functional patterns. The current implementation demonstrates on
protein sequences but is designed for general molecular applications.

The training process:
1. Uses COBYLA optimizer to minimize the encoder loss
2. Trains on molecular sequences with functional annotations
3. Evaluates performance on both training and test sets
4. Saves trained parameters and generates performance reports

Key optimization: The encoder is trained to map molecular sequences to a target
latent distribution, enabling generation of novel sequences with similar properties.
"""

from cost import *
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA

# =============================================
# ENCODER TRAINING PROCESS
# =============================================
# Train the quantum encoder to learn optimal representations of molecular sequences

# Record training start time for performance tracking
start = time.time()

# Initialize COBYLA optimizer
# COBYLA (Constrained Optimization BY Linear Approximation) is well-suited for
# quantum parameter optimization as it doesn't require gradient information
opt = COBYLA(maxiter=500)  # Maximum 500 iterations to prevent overtraining
print("Encoder training")

# Create partial function for loss computation
# This binds the training and test data to the loss function, leaving only
# the parameters to be optimized
f = partial(e_loss, train_input=train_seqs, test_input=test_seqs)

# Optimize encoder parameters
# The optimizer minimizes the loss function by adjusting encoder parameters
# xe contains the initial parameter values (random or from previous training)
opt_result = opt.minimize(fun=f, x0=xe)

# =============================================
# SAVE TRAINED PARAMETERS
# =============================================
# Store the optimized encoder parameters for future use

# Extract optimized parameters from the optimization result
xe = opt_result.x

# Save parameters to pickle file for persistence
# This allows the trained model to be reused for generation or further training
with open(f'{S}/opt-e-{S}.pkl', 'wb') as F:
    pickle.dump(xe, F)

# Calculate and report training time
elapsed_e = (time.time() - start) / 3600  # Convert to hours
print(f"Fit in {elapsed_e:0.2f} h")

# =============================================
# PERFORMANCE EVALUATION
# =============================================
# Evaluate the trained encoder on training and test sets

# Record evaluation start time
start = time.time()

# Initialize results file
# This file will contain performance metrics for different datasets
file = open(f"{S}/Results-{S}.txt", "w")
file.write("Dataset\tSize\tR\n")  # Header: Dataset name, size, reconstruction accuracy
file.close()

# =============================================
# TRAINING SET EVALUATION
# =============================================
# Evaluate how well the encoder reconstructs training sequences

# Initialize detailed results file
file = open(f"{S}/R-{S}.txt", "w")
file.write("TRAINING SET\n")
file.close()

# Generate detailed reconstruction results for training set
# This function compares input sequences with their reconstructions
# and calculates similarity metrics
output(train_seqs, head, "Train")

# =============================================
# TEST SET EVALUATION
# =============================================
# Evaluate generalization performance on unseen test sequences

# Add test set section to results file
file = open(f"{S}/R-{S}.txt", "a")
file.write("TEST SET\n")
file.close()

# Generate detailed reconstruction results for test set
# This evaluates how well the model generalizes to new sequences
output(test_seqs, head, "Test")

# =============================================
# FINAL PERFORMANCE REPORTING
# =============================================
# Calculate and report total training and evaluation time

elapsed_p = (time.time() - start) / 60  # Convert to minutes
print(f"Printed in {elapsed_p:.2f} min")

# Report total time for the entire training process
print(f"Finished in {elapsed_e + elapsed_p/60:.2f} h")