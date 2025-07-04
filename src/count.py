"""
QOBRA - Count Module

This module handles token frequency analysis, target distribution generation, and
sequence reconstruction evaluation. It creates the mapping between molecular tokens and
quantum amplitudes, generates target distributions for training, and evaluates
the performance of the trained quantum autoencoder.

Key functionality:
- Token frequency analysis and quantum amplitude mapping
- Target distribution generation for optimal latent space structure
- Sequence reconstruction evaluation and comparison
- Performance metrics calculation and visualization
- Statistical analysis of training results

The module is crucial for both training (creating target distributions) and
evaluation (measuring reconstruction quality). Current implementation uses amino
acids as tokens but is designed for general molecular sequence applications.
"""

from functools import partial
from multiprocessing import Pool

from inputs import *
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Target distribution parameters for latent space
# These create a well-structured Gaussian distribution in latent space
mu, std = 0., 1/np.sqrt(1.5*dim_tot)  # Mean=0, std scaled by quantum dimension

def make_target(n, mu, std):
    """
    Generate target distribution samples for training the quantum encoder.
    
    This function creates samples from a multivariate Gaussian distribution
    that serves as the target for the latent space. The distribution is
    carefully constructed to lie on the unit sphere (valid quantum states).
    
    The first component is always positive (representing the "head" amplitude),
    while the remaining components follow a Gaussian distribution. This structure
    ensures that the generated samples are valid quantum state amplitudes.
    
    Parameters:
    - n: Number of samples to generate
    - mu: Mean of the Gaussian distribution
    - std: Standard deviation of the Gaussian distribution
    
    Returns:
    - Array of target latent space samples (n × dim_latent)
    """
    target = []
    np.random.seed(4)  # Ensure reproducible target generation
    
    # Generate samples until we have enough valid ones
    while len(target) < n:
        # Generate (dim_latent-1) random values from Gaussian distribution
        t = np.random.normal(mu, std, dim_latent-1)
        
        # Calculate the norm of the tail components
        A = np.linalg.norm(t)

        # Ensure the sample lies within the unit sphere
        if A < 1:
            # Calculate the head component to ensure unit norm
            h = np.sqrt(1 - A**2)
            
            # Create complete target vector [head, tail_components]
            t = np.array([h] + list(t))
            target.append(t)
    
    return np.array(target)

# =============================================
# QUANTUM AMPLITUDE MAPPING SYSTEM
# =============================================
# Create mapping between amino acid tokens and quantum amplitudes

# Generate amplitude values using cumulative sum approach
# This creates a non-uniform distribution that reflects amino acid frequencies
keys = [(-1)**i * (.5 + sum(xt[:i])) for i in range(Len)]
keys = sorted(keys)  # Sort to ensure consistent ordering

# Calculate head amplitude (normalization constant)
# This ensures proper quantum state normalization
head = np.max(np.abs(keys)) * (.75*num_tot - 3)

# Create bidirectional mappings between characters and amplitudes
ctf = {ks[i]: keys[i] for i in range(Len)}     # Character to frequency mapping
ftc = {keys[i]: ks[i] for i in range(Len)}     # Frequency to character mapping

# Save the character-to-frequency mapping for reference
F = open(f"{S}/PDBcodes-{S}.txt", "a")
F.write(f"{ctf}")
F.close()

# =============================================
# ENCODER PARAMETER INITIALIZATION
# =============================================
# Initialize or load encoder parameters based on training mode

if switch == 0:
    # Training mode: Initialize with random parameters
    algorithm_globals.random_seed = 42
    xe = algorithm_globals.random.random(e.num_parameters)
else:
    # Inference mode: Load pre-trained parameters
    with open(f'{S}/opt-e-{S}.pkl', 'rb') as F:
        xe = pickle.load(F)

# =============================================
# TOKEN FREQUENCY VISUALIZATION
# =============================================
# Create visualization of token frequencies in Gaussian order

# Extract tokens and their frequencies in Gaussian order
cnt_x3 = list(gauss_sort.keys())
cnt_y3 = [cnt[x] for x in cnt_x3]

# Create bar chart of token frequencies
fig, ax = plt.subplots(figsize=(10, 5))
clear_output(wait=True)
bars = ax.bar(keys, cnt_y3, width=1.8, color='g')

# Add amplitude values as labels on each bar
i = 0
for bar in bars:
    height = bar.get_height()  # Get the height of each bar
    ax.text(
        bar.get_x() + bar.get_width()/2,  # X position at center of bar
        height,                            # Y position slightly above bar
        f'{keys[i]:.1f}',                 # Amplitude value label
        ha='center',                       # Horizontal alignment
        va='bottom',                       # Vertical alignment
        fontsize=4.5                       # Font size
    )
    i += 1

# Format and save the plot
ax.set_title('Frequencies of token numbers', fontsize=16)
ax.set_xticks(keys, cnt_x3)
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel("Token values", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
fig.savefig(f"{S}/tokens-{metals}.png", dpi=300, bbox_inches='tight')

def output(seqs, h, Set):
    """
    Evaluate sequence reconstruction performance of the trained autoencoder.
    
    This function tests how well the quantum autoencoder can reconstruct input
    molecular sequences. It processes sequences through the full autoencoder
    (input → encoder → decoder → output) and measures reconstruction quality.
    
    The evaluation includes:
    - Sequence-by-sequence reconstruction comparison
    - Overall reconstruction success rate
    - Detailed similarity analysis
    
    Parameters:
    - seqs: List of molecular sequences to evaluate
    - h: Head amplitude for normalization
    - Set: Dataset name (e.g., "Train", "Test")
    """
    # Open output files for results
    file = open(f"{S}/Results-{S}.txt", "a")    # Summary results
    file1 = open(f"{S}/R-{S}.txt", "a")         # Detailed results
    
    match, success_rate = 0, 0  # Initialize performance counters
    
    # Create quantum circuits for evaluation
    # Input circuit (just the feature map)
    qc_i = QuantumCircuit(num_tot)
    qc_i = qc_i.compose(fm_i.assign_parameters(i_params))
    
    # Full autoencoder circuit (input → encoder → decoder → output)
    qc_o = QuantumCircuit(num_tot)
    qc_o = qc_o.compose(fm_i.assign_parameters(i_params))      # Input encoding
    qc_o = qc_o.compose(e.assign_parameters(e_params))         # Encoder
    qc_o = qc_o.compose(e.assign_parameters(e_params).inverse()) # Decoder
    
    # Prepare encoding function for parallel processing
    f = partial(encode_amino_acid_sequence, ctf=ctf, 
                head=h, max_len=max_len, vec_len=dim_tot)
    
    # Encode all sequences in parallel
    with Pool(processes=processes) as pool:
        states = np.array(pool.map(f, seqs))
    
    # Evaluate each sequence individually
    for i in states:
        # ============================================
        # PROCESS INPUT SEQUENCE
        # ============================================
        # Create parameter dictionary for input
        param_dict = {i_params[j]: i[j] for j in range(num_feature)}
        
        # Execute input circuit (ground truth)
        original = qc_i.assign_parameters(param_dict)
        i_sv = Statevector(original).data**2  # Get probability amplitudes
        
        # ============================================
        # PROCESS THROUGH AUTOENCODER
        # ============================================
        # Add encoder parameters to parameter dictionary
        param_dict.update({e_params[j]: xe[j] for j in range(num_encode)})
        
        # Execute full autoencoder circuit
        output = qc_o.assign_parameters(param_dict)
        o_sv = Statevector(output).data**2  # Get output probability amplitudes
        
        # ============================================
        # DECODE AND COMPARE SEQUENCES
        # ============================================
        # Decode both input and output back to amino acid sequences
        original_seq = decode_amino_acid_sequence(np.sqrt(i_sv), ftc, h)
        output_seq = decode_amino_acid_sequence(np.sqrt(o_sv), ftc, h)
        
        # Find sequence terminators for proper comparison
        idx1 = original_seq.find("X")
        idx2 = output_seq.find("X")
        
        # Write sequences to detailed results file
        if idx1 != -1:
            file1.write(f"I: {original_seq[:idx1]}\n")  # Input (truncated at X)
        else:
            file1.write(f"I: {original_seq}\n")         # Input (full sequence)
            
        if idx2 != -1:
            file1.write(f"O: {output_seq[:idx2]}\n")    # Output (truncated at X)
        else:
            file1.write(f"O: {output_seq}\n")           # Output (full sequence)
        
        # ============================================
        # CALCULATE SIMILARITY METRICS
        # ============================================
        # Measure similarity between input and output sequences
        ratio = seq_match_ratio(original_seq, output_seq)
        file1.write(f"Sequence ratio of correct matches: {ratio}\n")
        file1.write("*" * dim_tot + '\n')  # Add separator
        
        # Count perfect reconstructions
        if ratio == 1:
            success_rate += 1
    
    # Calculate overall success rate
    success_rate /= len(seqs)
    
    # Write summary results
    file.write(f"{Set}\t{len(seqs)}\t{success_rate:.3f}\n")
    
    # Close output files
    file.close()
    file1.close()