import string, urllib3
import sys, pickle, os
from count import *

# Initialize results file for tracking generation statistics
# This file will store N(Novel), U(Unique), V(Valid) statistics for each run
filename = f"{S}/NUV-{S}.txt"
file = open(f"{filename}", "w")
file.write("N\tU\tV\n")  # Header: Novel, Unique, Valid
file.close()

# Get alphabet letters for chain identification in multi-chain proteins
# Each chain in a protein complex gets a unique letter identifier (A, B, C, etc.)
alphabets = list(string.ascii_uppercase)

# Define color palette for PyMol visualization
# These colors will be used to highlight different chains and binding sites
colors = ["red", "green", "yellow", "blue", 
          "salmon", "marine", "grey", "black"]

def generate(lcode):
    """
    Generate a protein sequence from a latent code using the trained decoder
    
    Args:
        lcode: Latent space representation (quantum state vector)
    
    Returns:
        Decoded protein sequence string
    
    This function takes a point in latent space and uses the trained decoder
    to generate a corresponding protein sequence. It's the core of the
    de novo sequence generation process.
    """
    # Create parameter dictionary for the decoder
    # Combine latent parameters with trained encoder parameters
    param_dict = {l_params[j]: lcode[j] for j in range(dim_tot)}
    param_dict.update({e_params[j]: xe[j] for j in range(num_decode)})
    
    # Apply decoder circuit to generate quantum state
    o_qc = qc_d.assign_parameters(param_dict)
    o_sv = Statevector(o_qc).data**2  # Get probability amplitudes
    
    # Decode quantum state back to amino acid sequence
    return decode_amino_acid_sequence(np.sqrt(o_sv), ftc, head)

def pml(s, HL, idx, prot_folder):
    """
    Generate PyMol script files for protein visualization
    
    Args:
        s: Protein sequence string
        HL: Dictionary mapping chain IDs to metal-binding site positions
        idx: Sequence index for file naming
        prot_folder: Output folder for visualization files
    
    This function creates PyMol command files that can be used to visualize
    the generated protein sequences with metal-binding sites highlighted.
    It includes color-coding for different amino acid types and chains.
    """
    # Ensure the output directory exists
    if not os.path.exists(prot_folder):
        os.makedirs(prot_folder)
    
    # Save the raw sequence to a text file
    file = open(f"{prot_folder}/{idx}.txt", "w")
    file.write(s)
    file.close()
        
    # Generate PyMol script for visualization
    i = 0
    file = open(f"{prot_folder}/{idx}.pml", "w")
    
    # Process each chain in the protein
    for ID in HL.keys():
        # Set chain color (cycling through available colors)
        file.write(f"color {colors[int(i%len(colors))]}, chain {ID} and elem C\n")
        
        # Create selection string for metal-binding residues
        resi = f'{HL[ID][0]}'  # First binding site position
        for res in HL[ID][1:]:  # Add remaining binding sites
            resi += f"+{res}"
        
        # Create PyMol selection and display commands
        file.write(f"select res{ID}, chain {ID} and resi {resi}\n")
        file.write(f"show sticks, res{ID}\n")  # Show binding sites as sticks
        
        # Color-code amino acids by their chemical properties
        # Hydrophobic amino acids (nonpolar, water-repelling)
        file.write(f"color lime, res{ID} and resn ALA+VAL+LEU+ILE+MET+PHE+TRP+PRO+GLY and elem C\n")
        
        # Polar amino acids (water-attracting, but uncharged)
        file.write(f"color orange, res{ID} and resn SER+THR+CYS+ASN+GLN+TYR and elem C\n")
        
        # Positively charged amino acids (basic)
        file.write(f"color magenta, res{ID} and resn LYS+ARG+HIS and elem C\n")
        
        # Negatively charged amino acids (acidic)
        file.write(f"color cyan, res{ID} and resn ASP+GLU and elem C\n")
        
        # Add residue labels to show amino acid types
        file.write(f"label res{ID}, resn\n")
        i += 1
    
    # Set label color for visibility
    file.write(f"set label_color, yellow")
    file.close()
    
    print(f"PDB sequence saved to {prot_folder}")

# Dictionary mapping plot types to file naming conventions
# This helps maintain consistent naming across different analysis plots
mp = {'binding sites':'plus', 'chain numbers':'chains', 'length':'len'}

def Plot(real, gen, s, folder, seed):
    """
    Create distribution comparison plots between real and generated sequences
    
    Args:
        real: Distribution from real training data
        gen: Distribution from generated sequences
        s: Type of distribution ('binding sites', 'chain numbers', or 'length')
        folder: Output folder for plots
        seed: Random seed for file naming
    
    This function creates side-by-side bar plots comparing the distributions
    of various sequence properties between training data and generated sequences.
    It helps evaluate how well the model captures important structural features.
    """
    # Count the occurrences of each element in both distributions
    counts_real = Counter(real)
    counts_gen = Counter(gen)
    
    # Initialize list to store relative ratios for statistical analysis
    relative_ratios = []
    
    # Special handling for sequence length distributions
    if s == 'length':
        # Create binned distribution for sequence lengths
        # Use 8-amino acid bins for better visualization
        bins = [i*8 for i in range(dim_tot//8+1)]
        
        # Initialize bin counts
        counts_real_bin = [0]*(len(bins)-1)
        counts_gen_bin = [0]*(len(bins)-1)
        
        # Populate bins with sequence length counts
        for i in range(len(bins)-1):
            # Count sequences falling in each bin for real data
            for k in counts_real.keys():
                if bins[i] < k < bins[i+1]:
                    counts_real_bin[i] += counts_real[k]
            # Count sequences falling in each bin for generated data
            for k in counts_gen.keys():
                if bins[i] < k < bins[i+1]:
                    counts_gen_bin[i] += counts_gen[k]
        
        # Normalize counts to get frequencies
        sum_real = sum(counts_real_bin)
        sum_gen = sum(counts_gen_bin)
        norm_counts_real = [i/sum_real for i in counts_real_bin]
        norm_counts_gen = [i/sum_gen for i in counts_gen_bin]
    
        # Calculate relative ratios for statistical analysis
        for i in range(len(norm_counts_real)):
            real_freq = norm_counts_real[i]
            gen_freq = norm_counts_gen[i]
            
            if real_freq != 0:  # Avoid division by zero
                relative_ratios.append(gen_freq/real_freq)
        
        # Calculate mean and standard deviation of relative ratios
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        # Calculate bin widths for bar plot
        bin_widths = np.diff(bins)
        
        # Create bar plot for length distribution
        plt.figure(figsize=(4,4))
        clear_output(wait=True)
        plt.bar(bins[:-1], norm_counts_real, log = True, alpha=1, 
                width=bin_widths, align='edge', label='Training')
        plt.bar(bins[:-1], norm_counts_gen, log = True, alpha=.6, 
                width=bin_widths, align='edge', label='De novo')
        plt.xlabel("Length")
    
    else:
        # Handle discrete count distributions (binding sites, chain numbers)
        # Normalize counts to get frequency distributions
        sum_real = sum(list(counts_real.values()))
        sum_gen = sum(list(counts_gen.values()))
        
        norm_counts_real = {k: v/sum_real for k, v in counts_real.items()}
        norm_counts_gen = {k: v/sum_gen for k, v in counts_gen.items()}
        
        # Extract keys and values for plotting
        x_real,y_real = list(norm_counts_real.keys()), list(norm_counts_real.values())
        x_gen,y_gen = list(norm_counts_gen.keys()), list(norm_counts_gen.values())
        
        x_real,y_real = np.array(x_real),np.array(y_real)
        x_gen,y_gen = np.array(x_gen),np.array(y_gen)
    
        # Calculate relative ratios for each category
        for k in x_real:
            real_freq = norm_counts_real[k]
            gen_freq = norm_counts_gen.get(k,0)  # Default to 0 if not present
            relative_ratios.append(gen_freq/real_freq)
        
        # Calculate statistics of relative ratios
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        # Create bar plot for discrete distributions
        plt.figure(figsize=(4,4))
        clear_output(wait=True)
        plt.bar(x_real, y_real, log = True, alpha=1, label='Training')
        plt.bar(x_gen, y_gen, log = True, alpha=.6, label='De novo')
        plt.xlabel("Count")
    
    # Customize plot appearance
    plt.title(f"Distribution of {s}")
    plt.yscale('log')  # Logarithmic scale for better visualization
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save plot with appropriate filename
    plt.savefig(f"{folder}/{mp[s]}-{seed}-{S}.png", dpi = 300, bbox_inches='tight')
    
    # Store relative ratio statistics for later analysis
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-{mp[s]}-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)