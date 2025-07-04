import numpy as np
import random
from ansatz import *

# Minimum length threshold for valid protein chain segments
# Chains shorter than this are considered too small for meaningful analysis
threshold = 4

# Create a string representation of target metals for file naming
metals = ''.join(keys_target)

# Define standard 20 amino acids using single-letter codes
# This represents the fundamental building blocks of proteins
keys_aa = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']

# Create comprehensive token vocabulary for amino acid sequence encoding
# This includes both regular amino acids and metal-binding variants (marked with '+')
cnt_x = []
# Add regular amino acids
for aa in keys_aa:
    cnt_x.append(aa)
# Add metal-binding amino acids (marked with '+')
for aa in keys_aa:
    cnt_x.append(aa + "+")

# Add special tokens for sequence structure
cnt_x.append(':')   # Chain separator
cnt_x.append(':X')  # Termination token

def LCS(str1, str2):
    """
    Calculate Longest Common Subsequence between two protein sequences
    
    Args:
        str1, str2: Protein sequences to compare
    
    Returns:
        Length of longest common subsequence
    
    This function is used to identify similar protein sequences and avoid
    redundancy in the training dataset.
    """
    same = False
    s = difflib.SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    return match.size

def issame(str1, str2):
    """
    Determine if two protein sequences are substantially similar
    
    Args:
        str1, str2: Protein sequences to compare
    
    Returns:
        Boolean indicating if sequences are similar (>10% overlap)
    
    This prevents training on highly similar sequences which could lead
    to overfitting on redundant patterns.
    """
    same = False
    match = LCS(str1, str2)
    # Consider sequences similar if they have >10% overlap in both directions
    if match/len(str1) > .1 and match/len(str2) > .1:
        same = True
    return same

def prep(metal):
    """
    Prepare protein sequence data for a specific metal type
    
    Args:
        metal: Metal type to process (e.g., 'Zn', 'Cu', 'Fe')
    
    Returns:
        ret: List of unique protein sequences
        PDBcodes: Dictionary mapping sequences to PDB codes
    
    This function loads protein sequences from files, removes duplicates,
    and maintains mapping to original PDB structure codes for traceability.
    """
    ret,temp,PDBcodes = [],[],{}
    folder_name = f"{metal}_bind"
    
    # Load all sequence files for this metal type
    for filename in os.listdir(folder_name):
        if filename[0] != '.':  # Skip hidden files
            f = f"{folder_name}/{filename}"
            
            with open(f,'r') as file:
                lines = file.readlines()
            
            # Concatenate all lines into a single sequence string
            s = ''
            for i in range(len(lines)):
                if len(lines[i]) > 0:
                    s += lines[i]
            temp.append(s)
            PDBcodes[s] = filename[:4]  # Extract PDB code from filename
    
    # Remove highly similar sequences to avoid redundancy
    i = 0
    while i < len(temp)-1:
        lcs = LCS(temp[i], temp[i+1])
        if len(temp[i]) == 0 or len(temp[i+1]) == 0:
            i += 1
        elif lcs/len(temp[i]) < .1 and lcs/len(temp[i+1]) < .1:
            ret.append(temp[i])  # Keep sequence if sufficiently different
            i += 1
        else:
            # Remove the longer sequence if they're too similar
            if len(temp[i+1]) < len(temp[i]): 
                del temp[i]
            else: 
                del temp[i+1]
    return ret,PDBcodes

def crop(sequence, max_len):
    """
    Intelligently crop protein sequences to fit within maximum length constraints
    
    Args:
        sequence: Original protein sequence
        max_len: Maximum allowed length
    
    Returns:
        Cropped sequence that preserves important structural features
    
    This function prioritizes keeping metal-binding sites (marked with '+')
    and maintains chain boundaries while fitting within length limits.
    """
    random.seed(1)  # Ensure reproducible cropping
    seq = sequence + "X"  # Add termination marker
    
    # Count actual amino acids (excluding special tokens)
    t = len(''.join([c for c in sequence if (c != "+" and c != "X")]))
    
    # If sequence is short enough, extend it by repeating
    if t < max_len:
        s = seq
        t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        
        # Repeat sequence until we reach max_len
        while t < max_len:
            s += seq[:min(len(seq), max_len-t)]
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        return s
    
    # For longer sequences, intelligently select important regions
    # Find positions of metal-binding sites (+) and chain separators (:)
    idx = seq.find("+")      # First metal-binding site
    idx_n = seq.find(":")    # First chain separator
    
    # Generate random distances for cropping boundaries
    d1 = random.randint(threshold, dim_tot//2)
    d2 = random.randint(threshold, dim_tot//2)
    
    # Adjust cropping strategy based on relative positions
    if idx_n > idx:
        idx = max(0, idx-random.randint(1,9))
    else:
        idx = max(0, idx-d1)
        idx = min(idx, idx_n-threshold)
        if seq[idx] == "+":
            idx -= 1
        idx_n = seq.find(":", idx_n+1)
    
    # Set end boundary
    idx_n = min(idx_n, len(seq)-2) + 1
    if idx_n == len(seq)-1:
        idx_n += 1
    
    # Extract initial crop
    s = seq[max(0, idx):idx_n]
    t = len(''.join([c for c in s if (c != "+" and c != "X")]))
    
    # Extend crop if needed by adding context from both sides
    if t < max_len:
        # Add sequence from the left
        i = max(0, idx)-1
        while t < max_len-d2 and i > -1:
            if seq[i] != "+":
                s = seq[i] + s
                i -= 1
            else:
                s = seq[i-1:i+1] + s  # Include amino acid with its binding marker
                i -= 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        
        # Add sequence from the right
        j = idx_n
        while t < max_len and j < len(seq):
            if seq[j+1] != "+" and seq[j+1] != "X":
                s = s + seq[j]
                j += 1
            else:
                s = s + seq[j:j+2]  # Include amino acid with its marker
                j += 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
            
        # Continue adding from left if still too short
        while t < max_len and i > -1:
            if seq[i] != "+":
                s = seq[i] + s
                i -= 1
            else:
                s = seq[i-1:i+1] + s
                i -= 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        return s
    
    # If sequence is still too long, truncate while preserving token structure
    n,i = len(s),0
    for j in range(max_len):
        if i >= n: break
        elif i < n-1 and (s[i+1] == "+" or s[i+1] == "X"):
            i += 2  # Skip over token pairs
        else:
            i += 1
    t = len(''.join([c for c in s[:i] if (c != "+" and c != "X")]))
    return s[:i]
  
def amps(sequence, ctf, max_len):
    """
    Convert protein sequence to amplitude values for quantum encoding
    
    Args:
        sequence: Protein sequence string
        ctf: Character-to-frequency mapping dictionary
        max_len: Maximum sequence length
    
    Returns:
        List of amplitude values corresponding to sequence tokens
    
    This function maps each amino acid (and special tokens) to numerical
    values that can be used as quantum state amplitudes.
    """
    prob_vector,n,i = [],len(sequence),0
    
    # Process sequence token by token
    while i < n:
        if i < n-1 and (sequence[i+1] == "+" or sequence[i+1] == "X"):
            # Handle two-character tokens (e.g., "H+", ":X")
            prob_vector.append(ctf[sequence[i:i+2]])
            i += 2
        else:
            # Handle single-character tokens
            prob_vector.append(ctf[sequence[i]])
            i += 1
    return prob_vector

def encode_amino_acid_sequence(sequence, ctf, head, max_len, vec_len):
    """
    Encode a protein sequence into a quantum state vector
    
    Args:
        sequence: Protein sequence string
        ctf: Character-to-frequency mapping dictionary
        head: Header value for normalization
        max_len: Maximum sequence length
        vec_len: Vector length for quantum state
    
    Returns:
        Normalized quantum state vector representing the protein sequence
    
    This is the key function that converts classical protein sequences
    into quantum state representations suitable for quantum machine learning.
    """
    # Convert sequence to amplitude values
    state_vector = amps(sequence, ctf, max_len)
    s = len(state_vector)
    
    # Add header value and normalize to create valid quantum state
    state_vector = [head] + state_vector
    state_vector = np.array(state_vector)
    state_vector /= np.linalg.norm(state_vector)  # Normalize for quantum state validity
    return state_vector

def decode_amino_acid_sequence(code, ftc, head):
    """
    Decode quantum state amplitudes back into protein sequence
    
    Args:
        code: Quantum state amplitude vector
        ftc: Frequency-to-character mapping dictionary
        head: Header value for scaling
    
    Returns:
        Decoded protein sequence string
    
    This function reverses the encoding process, converting quantum states
    back to classical amino acid sequences for sequence generation.
    """
    ret = ""
    factor = head/code[0].real  # Calculate scaling factor from header
    
    # Decode each amplitude back to amino acid token
    for i in range(1,dim_tot):
        diff = 10**6
        key = code[i].real*factor
        c = key
        
        # Find closest matching token in frequency dictionary
        for k in ftc.keys():
            if abs(abs(k) - c) < diff:
                diff = abs(abs(k) - c)
                key = k
        
        # Return empty string if no valid token found
        if key not in ftc.keys():
            return ""
        ret += ftc[key]
    return ret

def seq_match_ratio(s1, s2):
    """
    Calculate the similarity ratio between two protein sequences
    
    Args:
        s1, s2: Protein sequences to compare
    
    Returns:
        Average match ratio (0-1) indicating sequence similarity
    
    This function evaluates how well the model reconstructs input sequences
    by comparing original and decoded sequences character by character.
    """
    score,i = 0,0
    # Count matching characters at each position
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            score += 1
    
    # Return average of bidirectional match ratios
    return np.mean(np.array([score/len(s1), score/len(s2)]))