import numpy as np
import random
from ansatz import *

heads,metals = 1,''.join(keys_target)
keys_aa = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']
cnt_x = []
for aa in keys_aa:
    cnt_x.append(aa)
for aa in keys_aa:
    cnt_x.append(aa + "+")
cnt_x.append('X')
cnt_x.append('\n')

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
    ret,temp = [],[]
    m = metal
    
    for filename in os.listdir(f"{m}_bind-1"):
        if filename[0] != '.':
            f = f"{m}_bind-1/{filename}"
            with open(f,'r') as file:
                # Read the contents line by line
                lines = file.readlines()
            s = ''
            for i in range(len(lines)):
                if len(lines[i]) > 0:
                    s += lines[i]
            temp.append(s)
    
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
    return ret

def crop(sequence, max_len):
    # Find the first occurrence of the character
    idx = sequence.find("+")
    idx = max(0, idx-random.randint(1,17))
    
    s = sequence + "X"
    s = s[idx:]
    n = len(s)
    t = len(''.join([c for c in s if c != "+"]))
    
    if t >= max_len:
        i = 0
        for j in range(max_len):
            if i >= n: break
            elif i < n-1 and s[i+1] == "+":
                i += 2
            else:
                i += 1
        return s[:i]
    
    while t < max_len:
        s += sequence[:min(len(sequence), max_len-t)]
        t = len(''.join([c for c in s if c != "+"]))
    return s
  
def amps(sequence, ctf, max_len):
    # Create a binary string representing the amino acid sequence
    state_vector = []
    seg = crop(sequence, max_len)
    n = len(seg)
    
    i = 0
    while i < n:
        if i < n-1 and seg[i+1] == "+":
            state_vector.append(ctf[seg[i:i+2]])
            i += 2
        else:
            state_vector.append(ctf[seg[i]])
            i += 1
    return np.linalg.norm(state_vector), state_vector

def encode_amino_acid_sequence(sequence, ctf, head, max_len, vec_len):
    # Create a binary string representing the amino acid sequence
    amp, state_vector = amps(sequence, ctf, max_len)
    s = len(state_vector)
    
    state_vector = [head]*heads + state_vector
    state_vector = np.array(state_vector)
    state_vector /= np.linalg.norm(state_vector)
    return state_vector, state_vector[0]

# Define a function to decode a vector of qubit state amplitudes into aa sequence
def decode_amino_acid_sequence(code, ftc, head):
    ret = ""
    factor = head/code[0].real
    for i in range(dim):
        key,diff = code[i].real*factor,1e3
        if i < heads:
            continue
        else:
            for k in ftc.keys():
                if abs(k - code[i].real*factor) < diff:
                    diff = abs(k - code[i].real*factor)
                    key = k
            ret += ftc[key]
    return ret

def seq_match_ratio(s1, s2):
    score,i = 0,0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            score += 1
    return np.mean(np.array([score/len(s1), score/len(s2)]))