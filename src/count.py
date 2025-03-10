from functools import partial
from multiprocessing import Pool

from inputs import *
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mu,std = 0.,1/np.sqrt(1.5*dim_tot)
def make_target(n, mu, std):
    target = []
    np.random.seed(4)
    
    while len(target) < n:
        t = np.random.normal(mu, std, dim_latent-1)
        A = np.linalg.norm(t)

        if A < 1:
            h = np.sqrt(1 - A**2)
            t = np.array([h] + list(t))
            target.append(t)
    return np.array(target)

keys = [(-1)**i * (.5 + sum(xt[:i])) for i in range(Len)]
keys = sorted(keys)
head = np.max(np.abs(keys))*(.75*num_tot - 3)

ctf = {ks[i]:keys[i] for i in range(Len)}
ftc = {keys[i]:ks[i] for i in range(Len)}

F = open(f"{S}/PDBcodes-{S}.txt", "a")
F.write(f"{ctf}")
F.close()

if switch == 0:
    algorithm_globals.random_seed = 42
    xe = algorithm_globals.random.random(e.num_parameters)
else:
    with open(f'{S}/opt-e-{S}.pkl', 'rb') as F:
        xe = pickle.load(F)

# Token frequencies in Gaussian order
cnt_x3 = list(gauss_sort.keys())
cnt_y3 = [cnt[x] for x in cnt_x3]

fig,ax = plt.subplots(figsize=(10,5))
clear_output(wait=True)
bars = ax.bar(keys, cnt_y3, width=1.8, color='g')

# Loop over each bar and add a text label
i = 0
for bar in bars:
    height = bar.get_height() # Get the height of each bar
    ax.text(
        bar.get_x() + bar.get_width()/2,  # X position at the center of the bar
        height,  # Y position slightly above the bar
        f'{keys[i]:.1f}',  # Text label (here, just the height)
        ha='center',  # Horizontal alignment center
        va='bottom',   # Vertical alignment bottom
        fontsize=4.5  # Adjust font size
    )
    i += 1
ax.set_title('Frequencies of token numbers', fontsize=16)

ax.set_xticks(keys, cnt_x3)
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel("Token values", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
fig.savefig(f"{S}/tokens-{metals}.png", dpi = 300, bbox_inches='tight')

def output(seqs, h, Set):
    # Open the file in add mode
    file = open(f"{S}/Results-{S}.txt", "a")
    file1 = open(f"{S}/R-{S}.txt", "a")
    match,success_rate = 0,0
    
    qc_i = QuantumCircuit(num_tot)
    qc_i = qc_i.compose(fm_i.assign_parameters(i_params))
    
    qc_o = QuantumCircuit(num_tot)
    qc_o = qc_o.compose(fm_i.assign_parameters(i_params))
    qc_o = qc_o.compose(e.assign_parameters(e_params))
    qc_o = qc_o.compose(e.assign_parameters(e_params).inverse())
    
    f = partial(encode_amino_acid_sequence, ctf=ctf, 
                head=h, max_len=max_len, vec_len=dim_tot)
    with Pool(processes = processes) as pool:
        states = np.array(pool.map(f, seqs))
    
    for i in states:
        # Input
        param_dict = {i_params[j]: i[j] for j in range(num_feature)}
        original = qc_i.assign_parameters(param_dict)
        i_sv = Statevector(original).data**2
        
        # Output
        param_dict.update({e_params[j]: xe[j] for j in range(num_encode)})
        output = qc_o.assign_parameters(param_dict)
        o_sv = Statevector(output).data**2
        
        # Decode input & output
        original_seq = decode_amino_acid_sequence(np.sqrt(i_sv), ftc, h)
        output_seq = decode_amino_acid_sequence(np.sqrt(o_sv), ftc, h)
        
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
        file1.write("*"*dim_tot + '\n')
        if ratio == 1:
            success_rate += 1
    # Write each number to the file
    success_rate /= len(seqs)
    file.write(f"{Set}\t{len(seqs)}\t{success_rate:.3f}\n")
    # Close the files
    file.close()
    file1.close()