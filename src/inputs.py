from model import *
from collections import Counter
from scipy.special import gamma

import matplotlib.pyplot as plt
from IPython.display import clear_output

from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals

switch = int(sys.argv[-1])
S = f"{metals}-{r}"

if not os.path.exists(f"{S}"):
    os.makedirs(f"{S}")
metals = "".join(keys_target)

def cnts(seqs, folder = '', seed = 0, is_denovo = True):
    cnt,dn_cnts = {},[]
    len_cnts,plus_cnts = [],[]
    
    for s in seqs:
        idx = s.find("X")
        s1 = s[:idx+1] if idx != -1 else s
        len_s1 = len(''.join([c for c in s1 if c != "+"]))
        
        if is_denovo:
            file = open(f"{folder}/denovo-{seed}.txt", "a")
            file.write(f"{s1[:-1]}\n")
            file.write("*"*dim_tot + '\n')
            file.close()
        
        char_cnts = Counter(s1)
        len_cnts.append(len_s1)
        dn_cnts.append(char_cnts[':'])
        plus_cnts.append(char_cnts['+'])
        
        single_cnt = {}
        for i in range(len(s1)):
            if i < len(s1)-1 and (s1[i+1] == "+" or s1[i+1] == "X"):
                single_cnt[s1[i:i+2]] = single_cnt.get(s1[i:i+2],0) + 1
            elif s1[i] != "+" and s1[i] != "X":
                single_cnt[s1[i]] = single_cnt.get(s1[i],0) + 1
        
        for k in single_cnt.keys():
            cnt[k] = cnt.get(k,0) + single_cnt[k]/len_s1
        
    for k in cnt.keys():
        cnt[k] /= len(seqs)
    return cnt, dn_cnts, len_cnts, plus_cnts

# Now we want to create the right metal codes as supplied
metal_dct = {} # For multi-ion in the future
for k in keys_target:
    lst,PDBcodes = prep(k)
    for i in lst:
        if "++" not in i:
            l = metal_dct.get(i, [])
            l.append(k)
            metal_dct[i] = l

max_len = dim_tot-1
cap,processes = 6000,6

# create matching training input/output sets
seqs = list(metal_dct.keys())
all_input_seqs,all_test_seqs = [],[]

#random.shuffle(seqs)
for i in range(len(seqs)):
    s = seqs[i].replace("\n", ":")
    if i%5 > 0:
        all_input_seqs.append(s)
    else:
        all_test_seqs.append(s)

train_seqs,test_seqs = [],[]
F = open(f"{S}/PDBcodes-{S}.txt", "w")

for s in all_input_seqs:
    seg = crop(s, max_len)
    char_cnts = Counter(seg)
    
    if 0 < char_cnts['+'] and len(train_seqs) < cap:
        train_seqs.append(seg)
        code = PDBcodes[s.replace(':', '\n')]
        
        F.write(f"{code}\n")
        if seg.find('X') > -1:
            F.write(f"{seg[:seg.find('X') + 1]}\n")
        else: F.write(f"{seg}\n")
        F.write("-"*max_len + '\n')
F.close()

train_size = len(train_seqs)
test_size = train_size//5

for s in all_test_seqs:
    seg = crop(s, max_len)
    char_cnts = Counter(seg)
    if 0 < char_cnts['+'] and len(test_seqs) < test_size:
        test_seqs.append(seg)
    
cnt,dn_cnts_real,len_cnts_real,plus_cnts_real = cnts(train_seqs, is_denovo=False)
max_dn = np.max(dn_cnts_real)

srtd = sorted(list(cnt.values()), reverse=True)
for k in cnt.keys():
    cnt[k] += random.random()*srtd[-1]*1e-3
srtd = sorted(list(cnt.values()), reverse=True)

rev_cnt = {cnt[k]:k for k in cnt.keys()}
gsrtd = [srtd[0]]
for i in range(1, len(srtd)):
    if i%2 == 0:
        gsrtd.append(srtd[i])
    else:
        gsrtd = [srtd[i]] + gsrtd
dec_sort = {rev_cnt[k]: k for k in srtd}

gauss_sort = {rev_cnt[k]: k for k in gsrtd}
ks = list(gauss_sort.keys())
Len = len(ks)

xt = np.array([1. for i in range(Len-1)])
cnt_y1 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]