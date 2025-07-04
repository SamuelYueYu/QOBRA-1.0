from model import *
from collections import Counter
from scipy.special import gamma

import matplotlib.pyplot as plt
from IPython.display import clear_output

from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals

# Extract operation mode from command line arguments
# switch: Controls whether to use fresh parameters (0) or load pre-trained ones (1)
switch = int(sys.argv[-1])

# Create unique experiment identifier based on target metals and repetition count
# This helps organize results and maintain reproducibility across different runs
S = f"{metals}-{r}"

# Create output directory for this experiment if it doesn't exist
# All results, plots, and intermediate files will be stored here
if not os.path.exists(f"{S}"):
    os.makedirs(f"{S}")

# Convert metal list to string for consistent naming
metals = "".join(keys_target)

def cnts(seqs, folder = '', seed = 0, is_denovo = True):
    """
    Count and analyze various statistics of protein sequences
    
    Args:
        seqs: List of protein sequences to analyze
        folder: Output folder for saving results
        seed: Random seed for reproducibility
        is_denovo: Whether sequences are de novo generated or training data
    
    Returns:
        cnt: Dictionary of amino acid frequencies
        dn_cnts: Chain number counts
        len_cnts: Length distribution
        plus_cnts: Metal-binding site counts
    
    This function provides comprehensive statistical analysis of protein sequences,
    including amino acid frequencies, chain distributions, and metal-binding patterns.
    """
    cnt,dn_cnts = {},[]
    len_cnts,plus_cnts = [],[]
    
    # Process each sequence to extract various statistics
    for s in seqs:
        idx = s.find("X")  # Find termination marker
        s1 = s[:idx+1] if idx != -1 else s  # Truncate at termination
        
        # Calculate actual sequence length (excluding special tokens)
        len_s1 = len(''.join([c for c in s1 if c != "+"]))
        
        # Save de novo sequences to file for later analysis
        if is_denovo:
            file = open(f"{folder}/denovo-{seed}.txt", "a")
            file.write(f"{s1[:-1]}\n")
            file.write("*"*dim_tot + '\n')
            file.close()
        
        # Count occurrences of special characters
        char_cnts = Counter(s1)
        len_cnts.append(len_s1)                    # Sequence length
        dn_cnts.append(char_cnts[':'])             # Number of chains
        plus_cnts.append(char_cnts['+'])           # Number of metal-binding sites
        
        # Count amino acid frequencies (normalized by sequence length)
        single_cnt = {}
        for i in range(len(s1)):
            if i < len(s1)-1 and (s1[i+1] == "+" or s1[i+1] == "X"):
                # Handle two-character tokens (metal-binding AAs or termination)
                single_cnt[s1[i:i+2]] = single_cnt.get(s1[i:i+2],0) + 1
            elif s1[i] != "+" and s1[i] != "X":
                # Handle single amino acids
                single_cnt[s1[i]] = single_cnt.get(s1[i],0) + 1
        
        # Normalize counts by sequence length to get frequencies
        for k in single_cnt.keys():
            cnt[k] = cnt.get(k,0) + single_cnt[k]/len_s1
        
    # Normalize frequencies across all sequences
    for k in cnt.keys():
        cnt[k] /= len(seqs)
    
    return cnt, dn_cnts, len_cnts, plus_cnts

# Define parameters for target distribution generation
# These parameters control the shape of the Gaussian distribution used for latent space
mu,std = 0.,1/np.sqrt(1.5*dim_tot)  # Mean and standard deviation for target distribution

# Dictionary to store metal-binding sequences organized by metal type
# This allows the model to learn patterns specific to different metals
metal_dct = {}

# Process each target metal to create comprehensive training dataset
for k in keys_target:
    lst,PDBcodes = prep(k)  # Load and preprocess sequences for this metal
    for i in lst:
        if "++" not in i:  # Exclude sequences with multiple consecutive metal sites
            l = metal_dct.get(i, [])
            l.append(k)
            metal_dct[i] = l

# Set processing parameters for dataset creation
max_len = dim_tot-1     # Maximum sequence length (reserve one position for termination)
cap,processes = 6000,6  # Training set size cap and number of parallel processes

# Extract all unique protein sequences from the metal dictionary
seqs = list(metal_dct.keys())
all_input_seqs,all_test_seqs = [],[]

# Split sequences into training and test sets using 80:20 ratio
# This ensures the model is evaluated on unseen data
for i in range(len(seqs)):
    s = seqs[i].replace("\n", ":")  # Replace newlines with chain separators
    if i%5 > 0:  # 80% for training
        all_input_seqs.append(s)
    else:        # 20% for testing
        all_test_seqs.append(s)

# Initialize final training and test sequence lists
train_seqs,test_seqs = [],[]

# Create file to track PDB codes for traceability
F = open(f"{S}/PDBcodes-{S}.txt", "w")

# Process training sequences with quality control
for s in all_input_seqs:
    seg = crop(s, max_len)  # Crop sequence to fit model constraints
    char_cnts = Counter(seg)
    
    # Only include sequences with metal-binding sites and within size limits
    if 0 < char_cnts['+'] and len(train_seqs) < cap:
        train_seqs.append(seg)
        code = PDBcodes[s.replace(':', '\n')]  # Get original PDB code
        
        # Record PDB code and processed sequence for traceability
        F.write(f"{code}\n")
        if seg.find('X') > -1:
            F.write(f"{seg[:seg.find('X') + 1]}\n")
        else: 
            F.write(f"{seg}\n")
        F.write("-"*max_len + '\n')
F.close()

# Calculate dataset sizes
train_size = len(train_seqs)
test_size = train_size//5  # Test set is 20% of training set size

# Process test sequences with same quality control
for s in all_test_seqs:
    seg = crop(s, max_len)
    char_cnts = Counter(seg)
    if 0 < char_cnts['+'] and len(test_seqs) < test_size:
        test_seqs.append(seg)
    
# Analyze training sequences to understand amino acid distribution
cnt,dn_cnts_real,len_cnts_real,plus_cnts_real = cnts(train_seqs, is_denovo=False)
max_dn = np.max(dn_cnts_real)  # Maximum number of chains in training data

# Add small random noise to frequency counts to break ties
# This ensures stable sorting and prevents edge cases in token mapping
srtd = sorted(list(cnt.values()), reverse=True)
for k in cnt.keys():
    cnt[k] += random.random()*srtd[-1]*1e-3
srtd = sorted(list(cnt.values()), reverse=True)

# Create reverse mapping from frequencies to amino acids
rev_cnt = {cnt[k]:k for k in cnt.keys()}

# Create Gaussian-ordered token mapping for optimal quantum encoding
# This ordering places most frequent tokens at the center of the distribution
# and least frequent at the extremes, matching the target Gaussian distribution
gsrtd = [srtd[0]]
for i in range(1, len(srtd)):
    if i%2 == 0:
        gsrtd.append(srtd[i])    # Add to end (positive tail)
    else:
        gsrtd = [srtd[i]] + gsrtd  # Add to beginning (negative tail)

# Create mappings between frequencies and amino acids
dec_sort = {rev_cnt[k]: k for k in srtd}      # Decreasing order mapping
gauss_sort = {rev_cnt[k]: k for k in gsrtd}   # Gaussian order mapping

# Extract ordered amino acid keys for token mapping
ks = list(gauss_sort.keys())
Len = len(ks)

# Create array for token position calculations
# This is used to generate the numerical encoding values
xt = np.array([1. for i in range(Len-1)])

# Create frequency array for visualization and comparison
# This shows the expected frequencies for each amino acid token
cnt_y1 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]