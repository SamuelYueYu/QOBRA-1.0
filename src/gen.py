import time
import re as Re
from gen_func import *

# Initialize tracking lists for generation statistics
# N: Novel sequences (not in training set)
# U: Unique sequences (no internal duplicates)
# V: Valid sequences (meet all quality criteria)
Nlist,Ulist,Vlist = [],[],[]

begin = time.time()

# Define regular expression patterns for sequence validation
# These patterns identify problematic metal-binding site arrangements
pattern3 = r".\+.\+.\+"      # Pattern for 3 consecutive metal-binding sites
pattern5 = r".\+.\+.\+.\+.\+"  # Pattern for 5 consecutive metal-binding sites

# Main generation loop (can be extended for multiple seeds)
for seed in range(1):
    N,U,V = 0,0,0  # Initialize counters for this seed
    valid = []     # Store valid generated sequences
    start = time.time()
    
    # Generate target latent states for de novo sequence generation
    # These serve as starting points in latent space for the decoder
    n = int(train_size*2)  # Generate more candidates than needed
    denovos = make_target(n, mu, std)
        
    # Create output directory for this seed
    folder = f"{S}/{seed}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Generate sequences from latent states in parallel
    # Each latent state is decoded into a protein sequence
    with Pool(processes=processes) as pool:
        gen = pool.map(generate, denovos)
    
    # Process and validate each generated sequence
    n = len(gen)
    for i in range(len(gen)):
        novel,unique = True,False
        s = gen[i]
        
        # Find termination marker and truncate sequence
        idx = s.find("X")
        if idx != -1:
            s2 = s[:idx]  # Truncate at termination marker
        else: 
            s2 = s
        
        print(s2)
        N += 1  # Count as initially novel
        
        # Check if sequence is novel (not in training set)
        # This parallel comparison prevents generating known sequences
        f = partial(issame, str2=s2)
        with Pool(processes=processes) as pool:
            ismatch = pool.map(f, seqs)
        
        # If sequence matches training data, it's not novel
        if any(ismatch):
            N -= 1
            novel = False
            print("Is match")
        
        # Check if sequence is unique (not already generated)
        if s2 not in valid: 
            U += 1
            unique = True
        
        # Validate sequence structure using regex patterns
        match3 = Re.findall(pattern3, s2)  # Find 3-consecutive metal sites
        match5 = Re.findall(pattern5, s2)  # Find 5-consecutive metal sites
        Cnts = Counter(s2)  # Count character occurrences
        
        # Apply quality filters for valid sequences
        # Requirements: has metal sites, no double colons, no 5-consecutive sites, max 1 triple site
        if "+" in s2 and '::' not in s2 and len(match5) == 0 and len(match3) < 2:
            # Create versions with and without metal-binding markers
            s3 = ''.join([c for c in s2 if c != "+"])  # Remove metal markers
            s4 = s2.split(':')   # Split at chain separators (with markers)
            s5 = s3.split(':')   # Split at chain separators (without markers)
            
            # Clean up empty chain segments
            if s5[-1] == '': 
                s5 = s5[:-1]
            
            # Apply length thresholds: each chain must be long enough
            if all(len(l) >= threshold for l in s5) or \
            (':' not in s3 and len(s3) >= threshold):
                V += 1  # Count as valid
                
                # Store sequence if it's both novel and unique
                if novel and unique:
                    valid.append(s)
                    
                # Create dictionary of metal-binding site positions for each chain
                HL = {}
                for j in range(len(s4)):
                    chain,chainID = s4[j],alphabets[j]
                    
                    # Find positions of metal-binding sites within each chain
                    shift = 0
                    for k in range(len(chain)):
                        if chain[k] == "+":
                            if chainID in HL.keys():
                                HL[chainID].append(k-shift)
                            else: 
                                HL[chainID] = [k-shift]
                            shift += 1  # Account for removed '+' markers
                
                # Generate PyMol visualization files for valid sequences
                pdb_name = f"{i}.pdb"
                prot_folder = f'{folder}/Samples/{i}'
                pml(s2, HL, i, prot_folder)
        
        print("*"*dim_tot)
        
        # Stop when we have enough valid sequences
        if len(valid) == train_size:
            n = i
            break
    
    # Calculate actual number of sequences processed
    size = min(n, len(gen))
    print(size)
    
    # Initialize output file for de novo sequences
    file = open(f"{folder}/denovo-{seed}.txt", "w")
    file.write("De novo sequences\n")
    file.close()
    
    # Analyze generated sequences and create statistics
    cnt,dn_cnts_gen,len_cnts_gen,plus_cnts_gen = cnts(valid, folder, seed)
    
    # Report generation time
    elapsed = (time.time()-start) / 3600
    print(f"Generated in {elapsed:0.2f} h")
    
    # Prepare data for amino acid frequency comparison
    cnt_y2 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]
    
    # Format amino acid labels for visualization
    cnt_x2 = []
    for i in range(len(cnt_x)):
        if cnt_x[i][-1] == "+":
            cnt_x2.append(cnt_x[i][0] + "\n+")  # Split metal-binding AA labels
        else:
            cnt_x2.append(cnt_x[i])
    
    # Calculate relative ratios between generated and training frequencies
    cnt_y1 = np.array(cnt_y1)
    cnt_y2 = np.array(cnt_y2)
    dif = cnt_y2/cnt_y1  # Ratio of generated to training frequencies
    
    # Calculate statistics of frequency ratios
    mean_relative_ratio = np.mean(dif)
    std_relative_ratio = np.std(dif)
    
    # Store relative ratio statistics
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-AA-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)
    
    # Store generation statistics (as fractions of total processed)
    Nlist.append(N/size)  # Fraction of novel sequences
    Ulist.append(U/size)  # Fraction of unique sequences
    Vlist.append(V/size)  # Fraction of valid sequences
    
    # Create amino acid frequency comparison plot
    x = np.arange(len(cnt_x2))  # Number of amino acid categories
    width = 0.36  # Width of each bar
    
    fig,ax = plt.subplots(figsize=(12,4))
    clear_output(wait=True)
    
    # Create side-by-side bar chart comparing training vs generated frequencies
    ax.bar(x - width/2, cnt_y1, width, label='Training')
    ax.bar(x + width/2, cnt_y2, width, label='De novo')
    ax.set_title('Frequency of occurence of each AA')
    
    # Create inset plot with logarithmic scale for better visibility
    ax_inset = inset_axes(ax, width="30%", height="50%", loc='right')
    ax_inset.bar(x - width/2, cnt_y1, width)
    ax_inset.bar(x + width/2, cnt_y2, width)
    ax_inset.xaxis.set_visible(False)
    ax_inset.set_yscale('log')  # Log scale for small frequencies
    ax_inset.tick_params(axis='y', labelsize=8)
    
    # Customize main plot
    ax.set_xticks(x, cnt_x2)  # Set amino acid labels
    ax.set_xlabel("AA/AA+/\\n")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"{folder}/Bar-{seed}-{S}.png", dpi = 300, bbox_inches='tight')
    
    # Generate additional distribution comparison plots
    Plot(dn_cnts_real, dn_cnts_gen, 'chain numbers', folder, seed)      # Chain count distribution
    Plot(plus_cnts_real, plus_cnts_gen, 'binding sites', folder, seed)  # Metal-binding site distribution
    Plot(len_cnts_real, len_cnts_gen, 'length', folder, seed)           # Sequence length distribution

# Save overall generation statistics
file = open(f"{filename}", "a")
file.write(f"{np.mean(Nlist):.3f}\t{np.mean(Ulist):.3f}\t{np.mean(Vlist):.3f}\n")  # Mean values
file.write(f"{np.std(Nlist):.3f}\t{np.std(Ulist):.3f}\t{np.std(Vlist):.3f}\n")   # Standard deviations
file.close()

# Report total generation time
elapsed = (time.time()-begin) / 3600
print(f"Total time: {elapsed:0.2f} h")