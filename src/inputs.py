from model import *
from collections import Counter
from qiskit.quantum_info import Statevector
from qiskit_algorithms.utils import algorithm_globals

random.seed(100)
# now we want to create the right metal codes as supplied
metal_dct = {}
for k in keys_target:
    lst = prep(k)
    for i in lst:
        if "++" not in i:
            l = metal_dct.get(i, [])
            l.append(k)
            metal_dct[i] = l
    
# Define a function to encode the amino acid sequence into a vector of qubit state
max_len,vec_len = dim-heads,2**num_tot
cap = 6000

# Padding head range factors
scale_up = 1
shrink,neg_lim = 1e-2,.5
# m-MMD difference Gaussian Sigma
Sigma = 2e-4
# Number of multiprocesses
processes = 16

seqs = []
for s in metal_dct.keys():
    seqs.append(s)

# create matching training input/output sets
input_states,all_input_seqs = [],[]
test_states,all_test_seqs = [],[]

random.shuffle(seqs)
for i in range(len(seqs)):
    if i%3 > 0:
        all_input_seqs.append(seqs[i])
    else:
        all_test_seqs.append(seqs[i])

Fld = "Count"
if not os.path.exists(Fld):
    os.makedirs(Fld)

train_seqs,test_seqs = [],[]
for s in all_input_seqs:
    seg = crop(s, max_len)
    if "+" in seg and len(train_seqs) < cap:
        train_seqs.append(seg)

train_size = len(train_seqs)        
test_size = round(train_size/5)

for s in all_test_seqs:
    seg = crop(s, max_len)
    if "+" in seg and len(test_seqs) < test_size:
        test_seqs.append(seg)

times = 20
metals = "".join(keys_target)
S = f"{metals}-{neg_lim}-{scale_up}-d"

def cnts(seqs, folder = '', seed = 0, is_denovo = True):
    cnt = {}
    dn_cnts = []
    len_cnts = []
    plus_cnts = []
    
    for s in seqs:
        idx = s.find("X")
        if idx != -1:
            s2 = s[:idx+1]
            s1 = s2[:-1]
        else:
            s1 = s
            s2 = s
        
        if is_denovo:
            file = open(f"{folder}/denovo-{seed}.txt", "a")
            file.write(f"{s1}\n")
            file.write("*"*dim + '\n')
            file.close()
        
        char_cnts = Counter(s1)
        len_cnts.append(len(s1))
        dn_cnts.append(char_cnts['\n'])
        plus_cnts.append(char_cnts['+'])
        
        single_cnt = {}
        for i in range(len(s)):
            if i < len(s)-1 and s[i+1] == "+":
                single_cnt[s[i:i+2]] = single_cnt.get(s[i:i+2],0) + 1
            elif s[i] != "+":
                single_cnt[s[i]] = single_cnt.get(s[i],0) + 1
        
        for k in single_cnt.keys():
            single_cnt[k] /= len(s)
            cnt[k] = cnt.get(k,0) + single_cnt[k]
        
    for k in cnt.keys():
        cnt[k] /= len(seqs)
    return cnt, dn_cnts, len_cnts, plus_cnts

def output(input_states, xe, xd, nstd, head, ftc, Set):
    # Open the file in add mode
    file = open(f"Results-{S}.txt", "a")
    file1 = open(f"R-{S}.txt", "a")
    match,success_rate = 0,0
    
    qc_i = QuantumCircuit(num_tot)
    qc_i = qc_i.compose(fm_i.assign_parameters(i_params))
    
    qc_o = QuantumCircuit(num_tot)
    qc_o = qc_o.compose(fm_i.assign_parameters(i_params))
    qc_o = qc_o.compose(e.assign_parameters(e_params))
    for i in range(num_latent):
        qc_o.ry(n_params[i], i)
    qc_o = qc_o.compose(d.assign_parameters(d_params).inverse())
    
    for i in input_states:
        # Input
        param_dict = {i_params[j]: i[j] for j in range(num_feature)}
        original = qc_i.assign_parameters(param_dict)
        i_sv = Statevector(original).data
        
        # Output
        noise = np.random.normal(0, nstd, num_latent)
        param_dict.update({e_params[j]: xe[j] for j in range(num_encode)})
        param_dict.update({n_params[j]: noise[j] for j in range(num_latent)})
        param_dict.update({d_params[j]: xd[j] for j in range(num_decode)})
        
        output = qc_o.assign_parameters(param_dict)
        o_sv = Statevector(output).data
        
        # Decode input & output
        original_seq = decode_amino_acid_sequence(i_sv, ftc, head)
        output_seq = decode_amino_acid_sequence(o_sv, ftc, head)
        
        idx1 = original_seq.find("X")
        idx2 = output_seq.find("X")
        
        if idx1 != -1:
            file1.write(f"I: {original_seq[:idx1]}\n")
        else:
            file1.write(f"I: {original_seq}\n")
        if idx2 != -1:
            file1.write(f"O: {output_seq[:idx2]}\n")
        else:
            file1.write(f"O: {output_seq}\n")
        
        # Measure i/o match
        ratio = seq_match_ratio(original_seq, output_seq)
        file1.write(f"Sequence ratio of correct matches: {ratio}\n")
        file1.write("*"*dim + '\n')
        if ratio == 1:
            success_rate += 1
        
    # Write each number to the file
    success_rate /= len(input_states)
    file.write(f"{Set}\t{len(input_states)}\t{success_rate:.3f}\n")
    # Close the files
    file.close()
    file1.close()
