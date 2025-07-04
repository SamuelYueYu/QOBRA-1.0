"""
QOBRA - Generation Functions Module

This module contains utility functions for de novo molecular sequence generation,
visualization, and statistical analysis. It provides tools for:

- Quantum sequence generation from latent space
- Visualization script generation (current example: PyMOL for proteins)
- Statistical analysis and comparison of generated vs training sequences
- Data visualization and plotting utilities
- File I/O operations for sequence storage

These functions support the main generation pipeline by providing specialized
tools for sequence analysis, visualization, and validation across different
molecular domains.
"""

import string, urllib3
import sys, pickle, os
from count import *

# =============================================
# INITIALIZATION AND SETUP
# =============================================

# Initialize output file for generation metrics
# This file will store novelty, uniqueness, and validity statistics
filename = f"{S}/NUV-{S}.txt"
file = open(f"{filename}", "w")
file.write("N\tU\tV\n")  # Header: Novel, Unique, Valid
file.close()

# Get alphabet for chain naming (A, B, C, ..., T)
# Used to assign unique identifiers to protein chains
alphabets = list(string.ascii_uppercase)

# PyMOL color palette for visualization
# These colors will be used to highlight different chains and binding sites
colors = ["red", "green", "yellow", "blue", 
          "salmon", "marine", "grey", "black"]

def generate(lcode):
    """
    Generate a molecular sequence from a latent space code using the quantum decoder.
    
    This function takes a latent space vector and converts it into a molecular sequence
    using the trained quantum decoder. The latent code is combined with the trained
    encoder parameters to generate the quantum state, which is then decoded into
    a molecular sequence.
    
    Parameters:
    - lcode: Latent space vector (numpy array)
    
    Returns:
    - Decoded molecular sequence string
    """
    # Create parameter dictionary for the quantum decoder
    # Map latent parameters to their corresponding values
    param_dict = {l_params[j]: lcode[j] for j in range(dim_tot)}
    
    # Add trained encoder parameters (used in inverse for decoding)
    param_dict.update({e_params[j]: xe[j] for j in range(num_decode)})
    
    # Execute the quantum decoder circuit
    o_qc = qc_d.assign_parameters(param_dict)
    
    # Get the quantum state amplitudes (squared for probabilities)
    o_sv = Statevector(o_qc).data**2
    
    # Decode the quantum state back to amino acid sequence
    return decode_amino_acid_sequence(np.sqrt(o_sv), ftc, head)

def pml(s, HL, idx, prot_folder):
    """
    Generate visualization script and sequence files for structure visualization.
    
    This function creates visualization commands (current example: PyMOL for proteins)
    to visualize molecular structures with highlighted functional sites. It generates
    both a sequence file and a visualization script that can be used to visualize
    the molecular structure with color-coded elements and functional sites.
    
    Parameters:
    - s: Molecular sequence string
    - HL: Dictionary mapping segment IDs to functional site positions
    - idx: Sequence index for file naming
    - prot_folder: Output directory for generated files
    """
    # Ensure the output directory exists
    if not os.path.exists(prot_folder):
        os.makedirs(prot_folder)
    
    # =============================================
    # SEQUENCE FILE GENERATION
    # =============================================
    # Save the protein sequence to a text file
    
    file = open(f"{prot_folder}/{idx}.txt", "w")
    file.write(s)
    file.close()
    
    # =============================================
    # PYMOL SCRIPT GENERATION
    # =============================================
    # Generate PyMOL commands for structure visualization
    
    i = 0
    file = open(f"{prot_folder}/{idx}.pml", "w")
    
    # Process each protein chain
    for ID in HL.keys():
        # Set base color for this chain (cycling through color palette)
        file.write(f"color {colors[int(i%len(colors))]}, chain {ID} and elem C\n")
        
        # Create residue selection string for binding sites
        resi = f'{HL[ID][0]}'  # First binding site
        for res in HL[ID][1:]:
            resi += f"+{res}"   # Add additional binding sites
        
        # Create PyMOL selection for binding sites in this chain
        file.write(f"select res{ID}, chain {ID} and resi {resi}\n")
        
        # Show binding sites as stick representation
        file.write(f"show sticks, res{ID}\n")
        
        # =============================================
        # AMINO ACID TYPE COLOR CODING
        # =============================================
        # Color amino acids by their chemical properties
        
        # Hydrophobic amino acids (non-polar)
        file.write(f"color lime, res{ID} and resn ALA+VAL+LEU+ILE+MET+PHE+TRP+PRO+GLY and elem C\n")
        
        # Polar amino acids (uncharged)
        file.write(f"color orange, res{ID} and resn SER+THR+CYS+ASN+GLN+TYR and elem C\n")
        
        # Positively charged amino acids (basic)
        file.write(f"color magenta, res{ID} and resn LYS+ARG+HIS and elem C\n")
        
        # Negatively charged amino acids (acidic)
        file.write(f"color cyan, res{ID} and resn ASP+GLU and elem C\n")
        
        # Add residue labels for identification
        file.write(f"label res{ID}, resn\n")
        i += 1
    
    # Set label color for visibility
    file.write(f"set label_color, yellow")
    file.close()
    
    print(f"PDB sequence saved to {prot_folder}")

# Mapping dictionary for plot property names
# Maps internal property names to more descriptive labels
mp = {'binding sites': 'plus', 'chain numbers': 'chains', 'length': 'len'}

def Plot(real, gen, s, folder, seed):
    """
    Generate comparative plots between training and generated sequence properties.
    
    This function creates statistical comparison plots showing how well the
    generated sequences match the training data distribution for various
    properties like sequence length, functional site count, and segment numbers.
    
    Parameters:
    - real: Training data values
    - gen: Generated data values  
    - s: Property name string ('length', 'binding sites', 'chain numbers')
    - folder: Output directory for plots
    - seed: Random seed for file naming
    """
    # Count occurrences of each property value
    counts_real = Counter(real)
    counts_gen = Counter(gen)
    
    # Initialize list for relative ratio calculations
    relative_ratios = []
    
    # =============================================
    # SEQUENCE LENGTH ANALYSIS
    # =============================================
    # Special handling for sequence length (continuous variable)
    
    if s == 'length':
        # Create bins for length distribution
        # Use 8-amino acid bins for grouping
        bins = [i*8 for i in range(dim_tot//8+1)]
        
        # Initialize bin counts
        counts_real_bin = [0] * (len(bins)-1)
        counts_gen_bin = [0] * (len(bins)-1)
        
        # Distribute counts into bins
        for i in range(len(bins)-1):
            # Count training sequences in this bin
            for k in counts_real.keys():
                if bins[i] < k < bins[i+1]:
                    counts_real_bin[i] += counts_real[k]
            
            # Count generated sequences in this bin
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
            
            if real_freq != 0:
                relative_ratios.append(gen_freq/real_freq)
        
        # Calculate statistical measures
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        # Calculate bin widths for plotting
        bin_widths = np.diff(bins)
        
        # Create the length distribution plot
        plt.figure(figsize=(4, 4))
        clear_output(wait=True)
        
        # Plot training and generated distributions
        plt.bar(bins[:-1], norm_counts_real, log=True, alpha=1, 
                width=bin_widths, align='edge', label='Training')
        plt.bar(bins[:-1], norm_counts_gen, log=True, alpha=.6, 
                width=bin_widths, align='edge', label='De novo')
        plt.xlabel("Length")
    
    # =============================================
    # DISCRETE PROPERTY ANALYSIS
    # =============================================
    # Handle discrete properties (binding sites, chain numbers)
    
    else:
        # Normalize counts to get frequencies
        sum_real = sum(list(counts_real.values()))
        sum_gen = sum(list(counts_gen.values()))
        
        norm_counts_real = {k: v/sum_real for k, v in counts_real.items()}
        norm_counts_gen = {k: v/sum_gen for k, v in counts_gen.items()}
        
        # Extract keys and values for plotting
        x_real, y_real = list(norm_counts_real.keys()), list(norm_counts_real.values())
        x_gen, y_gen = list(norm_counts_gen.keys()), list(norm_counts_gen.values())
        
        x_real, y_real = np.array(x_real), np.array(y_real)
        x_gen, y_gen = np.array(x_gen), np.array(y_gen)
    
        # Calculate relative ratios for each property value
        for k in x_real:
            real_freq = norm_counts_real[k]
            gen_freq = norm_counts_gen.get(k, 0)  # Default to 0 if not present
            relative_ratios.append(gen_freq/real_freq)
        
        # Calculate statistical measures
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        # Create the discrete property plot
        plt.figure(figsize=(4, 4))
        clear_output(wait=True)
        
        # Plot training and generated distributions
        plt.bar(x_real, y_real, log=True, alpha=1, label='Training')
        plt.bar(x_gen, y_gen, log=True, alpha=.6, label='De novo')
        plt.xlabel("Count")
    
    # =============================================
    # PLOT FORMATTING AND SAVING
    # =============================================
    # Format and save the comparison plot
    
    plt.title(f"Distribution of {s}")
    plt.yscale('log')  # Log scale for better visibility of differences
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save the plot with descriptive filename
    plt.savefig(f"{folder}/{mp[s]}-{seed}-{S}.png", dpi=300, bbox_inches='tight')
    
    # =============================================
    # STATISTICAL RESULTS STORAGE
    # =============================================
    # Store statistical comparison results
    
    # Save relative ratio statistics for later analysis
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-{mp[s]}-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)