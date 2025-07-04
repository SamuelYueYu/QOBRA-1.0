"""
QOBRA - Cost Function Module

This module implements the cost function and loss computation for training the quantum autoencoder.
It uses Maximum Mean Discrepancy (MMD) as the primary loss function to match the encoded
molecular sequences to a target distribution in latent space.

Key components:
- MMD kernel loss computation for distribution matching
- Parallel processing for efficient computation
- Latent space encoding and representation
- Visualization of training progress and distributions
- Target distribution generation for optimal latent space structure

The goal is to train the encoder to map molecular sequences to a well-structured latent
distribution that enables generation of novel sequences with similar properties.
"""

import warnings
import time, torch
from count import *
from qiskit.quantum_info import Statevector, state_fidelity

def SWAP(args):
    """
    Compute the SWAP kernel between a single vector and a matrix of vectors.
    
    This function calculates the squared dot product between one vector and
    all vectors in a matrix. It's used as part of the MMD (Maximum Mean Discrepancy)
    kernel computation for measuring distribution similarity.
    
    Parameters:
    - args: Tuple containing (x, Y) where x is a vector and Y is a matrix
    
    Returns:
    - 1 minus the squared dot product (kernel distance measure)
    """
    x, Y = args
    return 1 - np.dot(Y, x)**2

def compute_kernel_loss(X, Y):
    """
    Compute the kernel-based Maximum Mean Discrepancy (MMD) loss between two distributions.
    
    MMD is a statistical measure that quantifies the difference between two distributions.
    It's particularly useful for training generative models to match target distributions.
    This implementation uses parallel processing for efficient computation.
    
    Parameters:
    - X: First distribution (encoded protein sequences)
    - Y: Second distribution (target latent distribution)
    
    Returns:
    - Mean MMD loss value
    """
    # Create a pool of parallel workers for efficient computation
    with Pool(processes=processes) as pool:
        # Prepare arguments: Each row in X is paired with the entire matrix Y
        args = [(X[i], Y) for i in range(X.shape[0])]  # Pass Y and rows of X for efficient computation
        
        # Map rows of X to the helper function in parallel
        results = pool.map(SWAP, args)
    
    # Return the mean kernel loss across all samples
    return np.mean(results)

def latent_rep(x, p):
    """
    Compute the latent representation of a molecular sequence using the quantum encoder.
    
    This function takes a molecular sequence (encoded as quantum amplitudes) and
    passes it through the encoder circuit to obtain its latent representation.
    The encoder parameters are fixed during this computation.
    
    Parameters:
    - x: Input molecular sequence encoded as quantum amplitudes
    - p: Encoder parameters (fixed during this computation)
    
    Returns:
    - Real-valued latent representation vector
    """
    # Create parameter dictionary for the quantum circuit
    # Map input features to their corresponding amplitudes
    param_dict = {i_params[j]: x[j] for j in range(num_feature)}
    
    # Add encoder parameters to the parameter dictionary
    param_dict.update({e_params[j]: p[j] for j in range(num_encode)})
    
    # Execute the encoder quantum circuit with the given parameters
    q = qc_e.assign_parameters(param_dict)
    psi = Statevector.from_instruction(q)
    
    # Return the real part of the quantum state amplitudes
    return np.real(psi.data)

def latent_encode(p, train_input, test_input):
    """
    Encode training and test molecular sequences into latent representations.
    
    This function processes both training and test datasets through the quantum encoder,
    converting molecular sequences into latent space representations. It uses parallel
    processing for efficient computation of large datasets.
    
    Parameters:
    - p: Encoder parameters
    - train_input: Training molecular sequences
    - test_input: Test molecular sequences
    
    Returns:
    - train_encode: Encoded training sequences
    - test_encode: Encoded test sequences
    """
    # Create partial function for sequence encoding
    # This binds the encoding parameters, leaving only the sequence as input
    f = partial(encode_amino_acid_sequence, ctf=ctf, 
                head=head, max_len=dim_tot-1, 
                vec_len=dim_tot)
    
    # Encode all sequences to quantum amplitudes in parallel
    with Pool(processes=processes) as pool:
        train_states = np.array(pool.map(f, train_input))
        test_states = np.array(pool.map(f, test_input))
    
    # Create partial function for latent representation computation
    # This binds the encoder parameters, leaving only the state as input
    f = partial(latent_rep, p=p)
    
    # Compute latent representations for all sequences in parallel
    with Pool(processes=processes) as pool:
        train_encode = np.array(pool.map(f, train_states))
        test_encode = np.array(pool.map(f, test_states))
    
    return train_encode, test_encode

# Global lists to track training progress
# These store the kernel loss values during training for visualization
trains_k, tests_k = [], []

def plot_hist(train, test, target):
    """
    Plot histograms of latent space distributions for training monitoring.
    
    This function creates visualizations to monitor the training progress by
    showing how well the encoded sequences match the target distribution.
    It plots both the full distributions and a detailed view of the first component.
    
    Parameters:
    - train: Training set latent representations
    - test: Test set latent representations  
    - target: Target distribution samples
    """
    plt.figure(figsize=(5, 5))
    clear_output(wait=True)
    
    # Plot histogram of latent space amplitudes (excluding first component)
    plt.hist(train[:, 1:].flatten(), density=True, bins=dim_latent, 
             color='r', alpha=1, label='Train')
    plt.hist(test[:, 1:].flatten(), density=True, bins=dim_latent, 
             color='g', alpha=.3, label='Test')
    plt.hist(target[:, 1:].flatten(), density=True, bins=dim_latent, 
             color='b', alpha=.2, label='Target')
    
    plt.title("Frequency of state amplitudes")
    plt.xlabel("State amplitudes")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Add inset plot for the first component (most important)
    # [left, bottom, width, height] in normalized figure coordinates
    inset_ax = plt.axes([0.125, 0.679, 0.2, 0.2])
    
    # Plot histograms of the first component (head amplitudes)
    inset_ax.hist(abs(train[:, 0]).flatten(), density=True, bins=dim_latent, 
                  color='r', alpha=1)
    inset_ax.hist(abs(test[:, 0]).flatten(), density=True, bins=dim_latent, 
                  color='g', alpha=.3)
    inset_ax.hist(target[:, 0].flatten(), density=True, bins=dim_latent, 
                  color='b', alpha=.2)
    
    # Format inset plot
    inset_ax.tick_params(axis='both', labelsize=8)
    inset_ax.yaxis.tick_right()  # Set y-axis ticks to the right
    
    # Save the histogram plot
    plt.savefig(f"{S}/{S}-hist.png", dpi=300, bbox_inches='tight')
    
def plot(train, test, target):
    """
    Generate all training progress plots.
    
    This function creates comprehensive visualizations of the training process,
    including distribution histograms and loss curves over training iterations.
    
    Parameters:
    - train: Training set latent representations
    - test: Test set latent representations
    - target: Target distribution samples
    """
    # Plot distribution histograms
    plot_hist(train, test, target)
    
    # Plot training loss curves
    plt.figure(figsize=(5, 5))
    clear_output(wait=True)
    
    # Plot MMD loss evolution during training
    plt.plot(range(len(trains_k)), trains_k, label="Training latent")
    plt.plot(range(len(tests_k)), tests_k, ":", label="Testing latent")
        
    plt.title("m-MMD / iteration")
    plt.xlabel("Iteration")
    plt.ylabel("m-MMD")
    plt.legend()
    
    # Save the loss evolution plot
    plt.savefig(f"{S}/{S}-E.png", dpi=300, bbox_inches='tight')

def e_loss(p, train_input, test_input):
    """
    Compute the encoder loss function for optimization.
    
    This is the main loss function that the optimizer minimizes during training.
    It computes the MMD loss between the encoded sequences and a target distribution,
    encouraging the encoder to map sequences to a well-structured latent space.
    
    Parameters:
    - p: Encoder parameters to optimize
    - train_input: Training protein sequences
    - test_input: Test protein sequences
    
    Returns:
    - Training MMD loss value (scalar)
    """
    s = time.time()  # Record computation start time
    n = len(train_input)  # Number of training samples
    
    # Encode both training and test sequences into latent space
    train_encode, test_encode = latent_encode(p, train_input, test_input)
    
    # Generate target distribution samples
    # This creates the ideal latent distribution we want to match
    target = make_target(n, mu, std)
    
    # Compute MMD kernel loss for both training and test sets
    k_train = compute_kernel_loss(train_encode, target)
    k_test = compute_kernel_loss(test_encode, target)
    
    # Store loss values for plotting
    trains_k.append(k_train)
    tests_k.append(k_test)
    
    # Print progress every 10 iterations
    if (len(trains_k) - 1) % 10 == 0:
        print(len(trains_k) - 1)  # Current iteration
        print(f"m-MMD loss: {k_train:.4f}, {k_test:.4f}")
        
        # Additional diagnostics
        train_eff = train_encode[:, 1:].flatten()
        interval = time.time() - s
        print(f"Interval: {interval:.2f} s")
    
    # Generate training progress plots
    plot(train_encode, test_encode, target)
    
    # Return training loss for optimization
    return k_train