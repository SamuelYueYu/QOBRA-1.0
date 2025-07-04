from functools import partial
from multiprocessing import Pool

from inputs import *
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Parameters for target Gaussian distribution in latent space
# This distribution represents the ideal latent space that the encoder should learn
mu,std = 0.,1/np.sqrt(1.5*dim_tot)  # Mean=0, std scaled by system dimension

def make_target(n, mu, std):
    """
    Generate target quantum states following a Gaussian distribution
    
    Args:
        n: Number of target states to generate
        mu: Mean of the Gaussian distribution
        std: Standard deviation of the Gaussian distribution
    
    Returns:
        Array of normalized quantum states representing target distribution
    
    This function creates the target distribution that the encoder learns to match.
    The states are normalized to ensure they represent valid quantum states.
    """
    target = []
    np.random.seed(4)  # Fixed seed for reproducible target generation
    
    # Generate states until we have enough valid ones
    while len(target) < n:
        # Generate random vector from Gaussian distribution
        t = np.random.normal(mu, std, dim_latent-1)
        A = np.linalg.norm(t)  # Calculate magnitude
        
        # Only accept states that can be normalized (A < 1)
        if A < 1:
            h = np.sqrt(1 - A**2)  # Calculate header component for normalization
            t = np.array([h] + list(t))  # Prepend header to create valid quantum state
            target.append(t)
    
    return np.array(target)

# Create numerical encoding values for amino acid tokens
# This maps each amino acid to a unique numerical value for quantum encoding
keys = [(-1)**i * (.5 + sum(xt[:i])) for i in range(Len)]
keys = sorted(keys)  # Sort to create ordered mapping

# Scale the keys to fit within the quantum state space
# The scaling factor ensures proper distribution across the available space
head = np.max(np.abs(keys))*(.75*num_tot - 3)

# Create bidirectional mappings between characters and numerical values
ctf = {ks[i]:keys[i] for i in range(Len)}      # Character to frequency mapping
ftc = {keys[i]:ks[i] for i in range(Len)}      # Frequency to character mapping

# Save the character-to-frequency mapping for later reference
F = open(f"{S}/PDBcodes-{S}.txt", "a")
F.write(f"{ctf}")
F.close()

# Initialize encoder parameters based on operation mode
if switch == 0:  # Fresh start - use random parameters
    algorithm_globals.random_seed = 42
    xe = algorithm_globals.random.random(e.num_parameters)
else:  # Resume training - load pre-trained parameters
    with open(f'{S}/opt-e-{S}.pkl', 'rb') as F:
        xe = pickle.load(F)

# Prepare data for amino acid frequency visualization
# Sort amino acids by their frequency in Gaussian order
cnt_x3 = list(gauss_sort.keys())
cnt_y3 = [cnt[x] for x in cnt_x3]

# Create visualization of amino acid token frequencies
fig,ax = plt.subplots(figsize=(10,5))
clear_output(wait=True)
bars = ax.bar(keys, cnt_y3, width=1.8, color='g')

# Add numerical labels to each bar showing the token value
i = 0
for bar in bars:
    height = bar.get_height()  # Get frequency value
    ax.text(
        bar.get_x() + bar.get_width()/2,  # Center horizontally
        height,                           # Position above bar
        f'{keys[i]:.1f}',                # Token value label
        ha='center',                     # Horizontal alignment
        va='bottom',                     # Vertical alignment
        fontsize=4.5                     # Font size
    )
    i += 1

ax.set_title('Frequencies of token numbers', fontsize=16)
ax.set_xticks(keys, cnt_x3)  # Set amino acid labels
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel("Token values", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
fig.savefig(f"{S}/tokens-{metals}.png", dpi = 300, bbox_inches='tight')

def output(seqs, h, Set):
    """
    Process sequences through the quantum autoencoder and evaluate performance
    
    Args:
        seqs: List of protein sequences to process
        h: Header value for encoding/decoding
        Set: Dataset identifier ("Train" or "Test")
    
    This function runs sequences through the complete encode-decode pipeline,
    compares input and output sequences, and calculates reconstruction accuracy.
    It's used to evaluate how well the model learns to reconstruct protein sequences.
    """
    # Open output files for results
    file = open(f"{S}/Results-{S}.txt", "a")
    file1 = open(f"{S}/R-{S}.txt", "a")
    match,success_rate = 0,0
    
    # Create quantum circuits for input and output processing
    # Input circuit: just the feature map
    qc_i = QuantumCircuit(num_tot)
    qc_i = qc_i.compose(fm_i.assign_parameters(i_params))
    
    # Output circuit: full autoencoder (encode then decode)
    qc_o = QuantumCircuit(num_tot)
    qc_o = qc_o.compose(fm_i.assign_parameters(i_params))        # Feature map
    qc_o = qc_o.compose(e.assign_parameters(e_params))           # Encoder
    qc_o = qc_o.compose(e.assign_parameters(e_params).inverse()) # Decoder
    
    # Encode all sequences in parallel for efficiency
    f = partial(encode_amino_acid_sequence, ctf=ctf, 
                head=h, max_len=max_len, vec_len=dim_tot)
    with Pool(processes = processes) as pool:
        states = np.array(pool.map(f, seqs))
    
    # Process each encoded sequence through the autoencoder
    for i in states:
        # Create parameter dictionary for input encoding
        param_dict = {i_params[j]: i[j] for j in range(num_feature)}
        
        # Generate input quantum state
        original = qc_i.assign_parameters(param_dict)
        i_sv = Statevector(original).data**2  # Get state probabilities
        
        # Add encoder parameters for output generation
        param_dict.update({e_params[j]: xe[j] for j in range(num_encode)})
        
        # Generate output quantum state (after encode-decode)
        output = qc_o.assign_parameters(param_dict)
        o_sv = Statevector(output).data**2  # Get state probabilities
        
        # Decode both input and output states back to sequences
        original_seq = decode_amino_acid_sequence(np.sqrt(i_sv), ftc, h)
        output_seq = decode_amino_acid_sequence(np.sqrt(o_sv), ftc, h)
        
        # Find termination markers for sequence truncation
        idx1 = original_seq.find("X")
        idx2 = output_seq.find("X")
        
        # Write sequences to output file (truncate at termination)
        if idx1 != -1:
            file1.write(f"I: {original_seq[:idx1]}\n")
        else:
            file1.write(f"I: {original_seq}\n")
        
        if idx2 != -1:
            file1.write(f"O: {output_seq[:idx2]}\n")
        else:
            file1.write(f"O: {output_seq}\n")
        
        # Calculate sequence similarity ratio
        ratio = seq_match_ratio(original_seq, output_seq)
        file1.write(f"Sequence ratio of correct matches: {ratio}\n")
        file1.write("*"*dim_tot + '\n')
        
        # Count perfect reconstructions
        if ratio == 1:
            success_rate += 1
    
    # Calculate and record overall success rate
    success_rate /= len(seqs)
    file.write(f"{Set}\t{len(seqs)}\t{success_rate:.3f}\n")
    
    # Close output files
    file.close()
    file1.close()