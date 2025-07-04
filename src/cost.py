import warnings
import time, torch
from count import *
from qiskit.quantum_info import Statevector, state_fidelity

def SWAP(args):
    """
    Helper function to compute kernel distances for Maximum Mean Discrepancy (MMD)
    
    Args:
        args: Tuple containing (x, Y) where x is a single data point and Y is the reference set
    
    Returns:
        Distance measure (1 - squared dot product) for MMD calculation
    
    This function calculates the kernel distance between a single point and all points
    in a reference set. It's used in parallel processing to compute MMD efficiently.
    The kernel used is the polynomial kernel: K(x,y) = (x^T y)^2
    """
    x,Y = args
    return 1 - np.dot(Y, x)**2

def compute_kernel_loss(X, Y):
    """
    Compute Maximum Mean Discrepancy (MMD) loss between two distributions
    
    Args:
        X: First distribution (encoded sequences)
        Y: Second distribution (target Gaussian distribution)
    
    Returns:
        Mean MMD loss value
    
    MMD measures the distance between two probability distributions by comparing
    their means in a reproducing kernel Hilbert space. This is used to train
    the encoder to match the target Gaussian distribution in latent space.
    """
    # Create a pool of workers for parallel computation
    with Pool(processes=processes) as pool:
        # Prepare arguments: Each row in X is paired with the entire matrix Y
        args = [(X[i], Y) for i in range(X.shape[0])]  # Pass Y and rows of X for efficient computation
        # Map rows of X to the helper function in parallel
        results = pool.map(SWAP, args)
    return np.mean(results)

def latent_rep(x, p):
    """
    Generate latent representation of an encoded sequence
    
    Args:
        x: Encoded input sequence (quantum state amplitudes)
        p: Encoder parameters
    
    Returns:
        Real-valued latent representation (quantum state amplitudes)
    
    This function applies the encoder to an input sequence to generate its
    latent representation. The latent space is where the model learns to
    compress protein sequences into a structured representation.
    """
    # Create parameter dictionary combining input features and encoder parameters
    param_dict = {i_params[j]: x[j] for j in range(num_feature)}
    param_dict.update({e_params[j]: p[j] for j in range(num_encode)})
    
    # Apply encoder to generate quantum state
    q = qc_e.assign_parameters(param_dict)
    psi = Statevector.from_instruction(q)
    
    # Return real part of amplitudes (imaginary parts are typically zero for this ansatz)
    return np.real(psi.data)

def latent_encode(p, train_input, test_input):
    """
    Encode both training and test sequences into latent space
    
    Args:
        p: Encoder parameters
        train_input: Training protein sequences
        test_input: Test protein sequences
    
    Returns:
        train_encode: Encoded training sequences
        test_encode: Encoded test sequences
    
    This function processes both training and test sets through the complete
    encoding pipeline: sequence -> quantum state -> latent representation
    """
    # Create partial function for sequence encoding with fixed parameters
    f = partial(encode_amino_acid_sequence, ctf=ctf, 
                head=head, max_len=dim_tot-1, 
                vec_len=dim_tot)
    
    # Encode sequences to quantum states in parallel
    with Pool(processes = processes) as pool:
        train_states = np.array(pool.map(f, train_input))
        test_states = np.array(pool.map(f, test_input))
        
    # Apply encoder to generate latent representations
    f = partial(latent_rep, p=p)
    with Pool(processes = processes) as pool:
        train_encode = np.array(pool.map(f, train_states))
        test_encode = np.array(pool.map(f, test_states))
    
    return train_encode,test_encode

# Global lists to track training progress
# These store the MMD losses for training and test sets across iterations
trains_k,tests_k = [],[]

def plot_hist(train, test, target):
    """
    Create histogram visualization of latent space distributions
    
    Args:
        train: Training set latent representations
        test: Test set latent representations  
        target: Target Gaussian distribution
    
    This function visualizes how well the encoder learns to match the target
    distribution by plotting histograms of the latent space amplitudes.
    """
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    
    # Plot histograms of state amplitudes (excluding the first component)
    plt.hist(train[:,1:].flatten(), density=True, bins=dim_latent, color='r', alpha=1, label='Train')
    plt.hist(test[:,1:].flatten(), density=True, bins=dim_latent, color='g', alpha=.3, label='Test')
    plt.hist(target[:,1:].flatten(), density=True, bins=dim_latent, color='b', alpha=.2, label='Target')
    
    plt.title("Frequency of state amplitudes")
    plt.xlabel("State amplitudes")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Add an inset plot focusing on the first component (header)
    # [left, bottom, width, height] in normalized figure coordinates
    inset_ax = plt.axes([0.125, 0.679, 0.2, 0.2])
    
    # Plot histograms of the first component (header values)
    inset_ax.hist(abs(train[:,0]).flatten(), density=True, bins=dim_latent, color='r', alpha=1)
    inset_ax.hist(abs(test[:,0]).flatten(), density=True, bins=dim_latent, color='g', alpha=.3)
    inset_ax.hist(target[:,0].flatten(), density=True, bins=dim_latent, color='b', alpha=.2)
    
    # Customize inset plot appearance
    inset_ax.tick_params(axis='both', labelsize=8)
    inset_ax.yaxis.tick_right()  # Move y-axis ticks to the right
    
    plt.savefig(f"{S}/{S}-hist.png", dpi = 300, bbox_inches='tight')
    
def plot(train, test, target):
    """
    Create comprehensive visualization of training progress
    
    Args:
        train: Training set latent representations
        test: Test set latent representations
        target: Target Gaussian distribution
    
    This function creates both histogram and loss curve visualizations
    to monitor the training progress and convergence.
    """
    # Create histogram plot
    plot_hist(train, test, target)
    
    # Create loss curve plot
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    plt.plot(range(len(trains_k)), trains_k, label="Training latent")
    plt.plot(range(len(tests_k)), tests_k, ":", label="Testing latent")
        
    plt.title("m-MMD / iteration")
    plt.xlabel("Iteration")
    plt.ylabel("m-MMD")
    plt.legend()
    plt.savefig(f"{S}/{S}-E.png",dpi = 300, bbox_inches='tight')

def e_loss(p, train_input, test_input):
    """
    Calculate encoder loss using Maximum Mean Discrepancy (MMD)
    
    Args:
        p: Encoder parameters to optimize
        train_input: Training protein sequences
        test_input: Test protein sequences
    
    Returns:
        Training MMD loss (scalar value for optimization)
    
    This is the main loss function used to train the encoder. It measures
    how well the encoder maps protein sequences to the target Gaussian
    distribution in latent space using MMD.
    """
    s = time.time()
    n = len(train_input)
    
    # Encode sequences into latent space
    train_encode,test_encode = latent_encode(p, train_input, test_input)
    
    # Generate target distribution with same number of samples
    target = make_target(n, mu, std)
    
    # Compute MMD losses for both training and test sets
    k_train = compute_kernel_loss(train_encode, target)
    k_test = compute_kernel_loss(test_encode, target)
    
    # Store losses for progress tracking
    trains_k.append(k_train)
    tests_k.append(k_test)
        
    # Print progress every 10 iterations
    if (len(trains_k)-1)%10 == 0:
        print(len(trains_k)-1)
        print(f"m-MMD loss: {k_train:.4f}, {k_test:.4f}")

        # Calculate some statistics (currently unused)
        train_eff = train_encode[:,1:].flatten()
        interval = time.time()-s
        print(f"Interval: {interval:.2f} s")
    
    # Create visualization of current training state
    plot(train_encode, test_encode, target)
    
    # Return training loss for optimization
    return k_train