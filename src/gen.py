"""
QOBRA - Generation Module

This module generates de novo molecular sequences using the trained quantum autoencoder.
It samples from the learned latent distribution and decodes the samples into novel
molecular sequences with desired functional properties. The current implementation
demonstrates on protein sequences but is designed for general molecular applications.

Key functionality:
- Generation of novel molecular sequences from latent space
- Validation of generated sequences (novelty, uniqueness, validity)
- Quality assessment and filtering
- Statistical analysis and comparison with training data
- Visualization of generation results
- Visualization script generation (current example: PyMOL for proteins)

The generation process:
1. Sample from target latent distribution
2. Decode samples to molecular sequences
3. Validate sequences for functional plausibility
4. Filter sequences based on quality criteria
5. Analyze and compare with training data
"""

import time
import re as Re
from gen_func import *

# Initialize lists to store generation quality metrics
# N = Novel sequences, U = Unique sequences, V = Valid sequences
Nlist, Ulist, Vlist = [], [], []

# Record total generation start time
begin = time.time()

# Define regex patterns for binding site validation
# These patterns detect problematic binding site arrangements
pattern3 = r".\+.\+.\+"      # 3 consecutive binding sites (too dense)
pattern5 = r".\+.\+.\+.\+.\+"  # 5 consecutive binding sites (extremely dense)

# =============================================
# MAIN GENERATION LOOP
# =============================================
# Generate sequences for multiple random seeds (currently set to 1)

for seed in range(1):
    # Initialize quality counters for this seed
    N, U, V = 0, 0, 0  # Novel, Unique, Valid counters
    valid = []          # Store valid sequences
    
    # Record generation start time for this seed
    start = time.time()
    
    # =============================================
    # LATENT SPACE SAMPLING
    # =============================================
    # Generate latent space samples for decoding
    
    # Sample twice the training size for better coverage
    n = int(train_size * 2)
    
    # Sample from the target distribution (same as used for training)
    # This ensures generated sequences have similar latent properties
    denovos = make_target(n, mu, std)
    
    # Create output directory for this seed
    folder = f"{S}/{seed}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # =============================================
    # SEQUENCE GENERATION
    # =============================================
    # Decode latent samples into protein sequences
    
    # Generate sequences in parallel using the quantum decoder
    with Pool(processes=processes) as pool:
        gen = pool.map(generate, denovos)
    
    # =============================================
    # SEQUENCE VALIDATION AND FILTERING
    # =============================================
    # Validate each generated sequence for quality and novelty
    
    n = len(gen)
    for i in range(len(gen)):
        novel, unique = True, False  # Initialize flags
        s = gen[i]
        
        # Find sequence terminator and truncate if present
        idx = s.find("X")
        if idx != -1:
            s2 = s[:idx]    # Truncate at terminator
        else: 
            s2 = s          # Use full sequence
        
        print(s2)  # Display generated sequence
        
        # =============================================
        # NOVELTY CHECK
        # =============================================
        # Check if sequence is novel (not in training set)
        
        N += 1  # Count as potential novel sequence
        
        # Compare with all training sequences for similarity
        f = partial(issame, str2=s2)
        with Pool(processes=processes) as pool:
            ismatch = pool.map(f, seqs)
        
        # If sequence matches any training sequence, it's not novel
        if any(ismatch):
            N -= 1
            novel = False
            print("Is match")
        
        # =============================================
        # UNIQUENESS CHECK
        # =============================================
        # Check if sequence is unique (not already generated)
        
        if s2 not in valid: 
            U += 1
            unique = True
        
        # =============================================
        # VALIDITY CHECK
        # =============================================
        # Check if sequence meets biological validity criteria
        
        # Find problematic binding site patterns
        match3 = Re.findall(pattern3, s2)  # 3 consecutive binding sites
        match5 = Re.findall(pattern5, s2)  # 5 consecutive binding sites
        
        # Count special characters in sequence
        Cnts = Counter(s2)
        
        # Validity criteria:
        # - Must have binding sites ('+' present)
        # - No consecutive chain separators ('::')
        # - No extremely dense binding sites (pattern5)
        # - Limited moderately dense binding sites (pattern3 < 2)
        if "+" in s2 and '::' not in s2 and len(match5) == 0 and len(match3) < 2:
            
            # =============================================
            # CHAIN LENGTH VALIDATION
            # =============================================
            # Ensure all protein chains meet minimum length requirements
            
            # Remove binding site markers for length calculation
            s3 = ''.join([c for c in s2 if c != "+"])
            
            # Split into chains at ':' separators
            s4 = s2.split(':')  # With binding sites
            s5 = s3.split(':')  # Without binding sites
            
            # Remove empty chains
            if s5[-1] == '': 
                s5 = s5[:-1]
            
            # Check chain length requirements
            # Either all chains ≥ threshold length OR single chain ≥ threshold length
            if all(len(l) >= threshold for l in s5) or \
               (':' not in s3 and len(s3) >= threshold):
                
                V += 1  # Count as valid sequence
                
                # Store sequence if it's both novel and unique
                if novel and unique:
                    valid.append(s)
                    
                # =============================================
                # BINDING SITE ANALYSIS
                # =============================================
                # Analyze binding sites for PyMOL visualization
                
                # Create dictionary of binding site positions by chain
                HL = {}
                for j in range(len(s4)):
                    chain, chainID = s4[j], alphabets[j]
                    
                    # Find binding sites in this chain
                    shift = 0  # Track position shift due to '+' markers
                    for k in range(len(chain)):
                        if chain[k] == "+":
                            # Record binding site position (adjusted for markers)
                            if chainID in HL.keys():
                                HL[chainID].append(k - shift)
                            else: 
                                HL[chainID] = [k - shift]
                            shift += 1
                
                # =============================================
                # OUTPUT GENERATION
                # =============================================
                # Generate output files for this valid sequence
                
                pdb_name = f"{i}.pdb"
                prot_folder = f'{folder}/Samples/{i}'
                
                # Generate PyMOL script and sequence file
                pml(s2, HL, i, prot_folder)
        
        print("*" * dim_tot)  # Separator for output readability
        
        # Stop when we have enough valid sequences
        if len(valid) == train_size:
            n = i
            break
    
    # =============================================
    # GENERATION RESULTS ANALYSIS
    # =============================================
    # Analyze the quality of generated sequences
    
    size = min(n, len(gen))
    print(size)
    
    # Initialize results file for this seed
    file = open(f"{folder}/denovo-{seed}.txt", "w")
    file.write("De novo sequences\n")
    file.close()
    
    # Analyze generated sequences for statistical properties
    cnt, dn_cnts_gen, len_cnts_gen, plus_cnts_gen = cnts(valid, folder, seed)
    
    # Report generation time
    elapsed = (time.time() - start) / 3600
    print(f"Generated in {elapsed:0.2f} h")
    
    # =============================================
    # AMINO ACID FREQUENCY ANALYSIS
    # =============================================
    # Compare amino acid frequencies between training and generated sequences
    
    # Get frequency vectors for comparison
    cnt_y2 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]
    
    # Format amino acid labels for visualization
    cnt_x2 = []
    for i in range(len(cnt_x)):
        if cnt_x[i][-1] == "+":
            cnt_x2.append(cnt_x[i][0] + "\n+")  # Binding site marker
        else:
            cnt_x2.append(cnt_x[i])
    
    # Calculate relative ratios between generated and training frequencies
    cnt_y1 = np.array(cnt_y1)
    cnt_y2 = np.array(cnt_y2)
    dif = cnt_y2 / cnt_y1
    
    # Calculate statistical measures
    mean_relative_ratio = np.mean(dif)
    std_relative_ratio = np.std(dif)
    
    # Store amino acid frequency analysis results
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-AA-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)
    
    # =============================================
    # QUALITY METRICS STORAGE
    # =============================================
    # Store generation quality metrics
    
    # Calculate and store ratios
    Nlist.append(N / size)  # Novel sequence ratio
    Ulist.append(U / size)  # Unique sequence ratio
    Vlist.append(V / size)  # Valid sequence ratio
    
    # =============================================
    # VISUALIZATION GENERATION
    # =============================================
    # Create bar chart comparing amino acid frequencies
    
    # Number of amino acid categories
    x = np.arange(len(cnt_x2))
    width = 0.36  # Bar width
    
    # Create main comparison plot
    fig, ax = plt.subplots(figsize=(12, 4))
    clear_output(wait=True)
    
    # Plot training vs generated frequencies
    ax.bar(x - width/2, cnt_y1, width, label='Training')
    ax.bar(x + width/2, cnt_y2, width, label='De novo')
    ax.set_title('Frequency of occurence of each AA')
    
    # Create inset plot with logarithmic scale
    ax_inset = inset_axes(ax, width="30%", height="50%", loc='right')
    ax_inset.bar(x - width/2, cnt_y1, width)
    ax_inset.bar(x + width/2, cnt_y2, width)
    ax_inset.xaxis.set_visible(False)
    ax_inset.set_yscale('log')  # Log scale for better visibility
    ax_inset.tick_params(axis='y', labelsize=8)
    
    # Format main plot
    ax.set_xticks(x, cnt_x2)
    ax.set_xlabel("AA/AA+/\\n")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    # Save the comparison plot
    fig.savefig(f"{folder}/Bar-{seed}-{S}.png", dpi=300, bbox_inches='tight')
    
    # =============================================
    # COMPREHENSIVE PROPERTY ANALYSIS
    # =============================================
    # Compare various sequence properties between training and generated data
    
    # Compare chain number distributions
    Plot(dn_cnts_real, dn_cnts_gen, 'chain numbers', folder, seed)
    
    # Compare binding site distributions
    Plot(plus_cnts_real, plus_cnts_gen, 'binding sites', folder, seed)
    
    # Compare sequence length distributions
    Plot(len_cnts_real, len_cnts_gen, 'length', folder, seed)

# =============================================
# FINAL RESULTS SUMMARY
# =============================================
# Compile and save overall generation statistics

# Open results file for writing final metrics
file = open(f"{filename}", "a")

# Write mean values for all quality metrics
file.write(f"{np.mean(Nlist):.3f}\t{np.mean(Ulist):.3f}\t{np.mean(Vlist):.3f}\n")

# Write standard deviations for all quality metrics
file.write(f"{np.std(Nlist):.3f}\t{np.std(Ulist):.3f}\t{np.std(Vlist):.3f}\n")

file.close()

# Report total generation time
elapsed = (time.time() - begin) / 3600
print(f"Total time: {elapsed:0.2f} h")