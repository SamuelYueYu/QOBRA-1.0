import numpy as np
import random
from ansatz import *

threshold = 4
metals = ''.join(keys_target)

keys_aa = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']
cnt_x = []
for aa in keys_aa:
    cnt_x.append(aa)
for aa in keys_aa:
    cnt_x.append(aa + "+")

cnt_x.append(':')
cnt_x.append(':X')

def LCS(str1, str2):
    same = False
    s = difflib.SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    return match.size

def issame(str1, str2):
    same = False
    match = LCS(str1, str2)
    if match/len(str1) > .1 and match/len(str2) > .1:
        same = True
    return same

def prep(metal):
    ret,temp,PDBcodes = [],[],{}
    folder_name = f"{metal}_bind"
    
    for filename in os.listdir(folder_name):
        if filename[0] != '.':
            f = f"{folder_name}/{filename}"
            
            with open(f,'r') as file:
                # Read the contents line by line
                lines = file.readlines()
            
            s = ''
            for i in range(len(lines)):
                if len(lines[i]) > 0:
                    s += lines[i]
            temp.append(s)
            PDBcodes[s] = filename[:4]
    
    i = 0
    while i < len(temp)-1:
        lcs = LCS(temp[i], temp[i+1])
        if len(temp[i]) == 0 or len(temp[i+1]) == 0:
            i += 1
        elif lcs/len(temp[i]) < .1 and lcs/len(temp[i+1]) < .1:
            ret.append(temp[i])
            i += 1
        else:
            if len(temp[i+1]) < len(temp[i]): del temp[i]
            else: del temp[i+1]
    return ret,PDBcodes

def crop(sequence, max_len):
    random.seed(1)
    seq = sequence + "X"
    t = len(''.join([c for c in sequence if (c != "+" and c != "X")]))
    if t < max_len:
        s = seq
        t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        
        while t < max_len:
            s += seq[:min(len(seq), max_len-t)]
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        return s
    
    # Find the first occurrence of the character
    idx = seq.find("+")
    idx_n = seq.find(":")
    d1 = random.randint(threshold, dim_tot//2)
    d2 = random.randint(threshold, dim_tot//2)
    
    if idx_n > idx:
        idx = max(0, idx-random.randint(1,9))
    else:
        idx = max(0, idx-d1)
        idx = min(idx, idx_n-threshold)
        if seq[idx] == "+":
            idx -= 1
        idx_n = seq.find(":", idx_n+1)
    
    idx_n = min(idx_n, len(seq)-2) + 1
    if idx_n == len(seq)-1:
        idx_n += 1
    s = seq[max(0, idx):idx_n]
    t = len(''.join([c for c in s if (c != "+" and c != "X")]))
    
    if t < max_len:
        i = max(0, idx)-1
        while t < max_len-d2 and i > -1:
            if seq[i] != "+":
                s = seq[i] + s
                i -= 1
            else:
                s = seq[i-1:i+1] + s
                i -= 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        
        j = idx_n
        while t < max_len and j < len(seq):
            if seq[j+1] != "+" and seq[j+1] != "X":
                s = s + seq[j]
                j += 1
            else:
                s = s + seq[j:j+2]
                j += 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
            
        while t < max_len and i > -1:
            if seq[i] != "+":
                s = seq[i] + s
                i -= 1
            else:
                s = seq[i-1:i+1] + s
                i -= 2
            t = len(''.join([c for c in s if (c != "+" and c != "X")]))
        return s
    
    n,i = len(s),0
    for j in range(max_len):
        if i >= n: break
        elif i < n-1 and (s[i+1] == "+" or s[i+1] == "X"):
            i += 2
        else:
            i += 1
    t = len(''.join([c for c in s[:i] if (c != "+" and c != "X")]))
    return s[:i]
  
def amps(sequence, ctf, max_len):
    # Create a vector representing the amino acid sequence
    prob_vector,n,i = [],len(sequence),0
    
    while i < n:
        if i < n-1 and (sequence[i+1] == "+" or sequence[i+1] == "X"):
            prob_vector.append(ctf[sequence[i:i+2]])
            i += 2
        else:
            prob_vector.append(ctf[sequence[i]])
            i += 1
    return prob_vector

def encode_amino_acid_sequence(sequence, ctf, head, max_len, vec_len):
    # Create a binary string representing the amino acid sequence
    state_vector = amps(sequence, ctf, max_len)
    s = len(state_vector)
    
    state_vector = [head] + state_vector
    state_vector = np.array(state_vector)
    state_vector /= np.linalg.norm(state_vector)
    return state_vector

# Define a function to decode a vector of qubit state amplitudes into aa sequence
def decode_amino_acid_sequence(code, ftc, head):
    ret = ""
    factor = head/code[0].real
    
    for i in range(1,dim_tot):
        diff = 10**6
        key = code[i].real*factor
        c = key
        
        for k in ftc.keys():
            if abs(abs(k) - c) < diff:
                diff = abs(abs(k) - c)
                key = k
        
        if key not in ftc.keys():
            return ""
        ret += ftc[key]
    return ret

def seq_match_ratio(s1, s2):
    score,i = 0,0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            score += 1
    return np.mean(np.array([score/len(s1), score/len(s2)]))