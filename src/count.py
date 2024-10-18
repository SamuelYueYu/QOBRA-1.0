import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool

from inputs import *
from scipy.stats import norm
from IPython.display import clear_output
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def in_range(arr, low, high):
    return sum(1 for elem in arr if low < elem < high)

cnt,dn_cnts_real,len_cnts_real,plus_cnts_real = cnts(train_seqs, is_denovo=False)
rev_cnt = {cnt[k]:k for k in cnt.keys()}
srtd = sorted(list(cnt.values()), reverse=True)

gsrtd = [srtd[0]]
for i in range(1, len(srtd)):
    if i%2 == 1:
        gsrtd = gsrtd + [srtd[i]]
    else:
        gsrtd = [srtd[i]] + gsrtd
dec_sort = {rev_cnt[k]: k for k in srtd}
gauss_sort = {rev_cnt[k]: k for k in gsrtd}

freq_diff = [max(gsrtd[i], gsrtd[i+1])/min(gsrtd[i], gsrtd[i+1]) 
             for i in range(len(gsrtd)-1)]
min_fd = np.min(freq_diff)
token_diff = [0.] + [d/min_fd for d in freq_diff]
sum_token_diff = sum(token_diff)

ks = list(gauss_sort.keys())
char_to_float,float_to_char = {},{}

for i in range(len(ks)):
    char_to_float[ks[i]] = round(sum(token_diff[:i+1])-sum_token_diff*neg_lim, 2)
head = max(list(char_to_float.values())) * scale_up
for k in ks:
    float_to_char[char_to_float[k]] = k

sv0s = []
sv_mus,sv_sigs = [],[]

for s in train_seqs:
    sv,sv0 = encode_amino_acid_sequence(s, char_to_float, head, max_len, vec_len)
    if len(sv) > 0:
        input_states.append(sv)
        sv0s.append(sv0)
        sv_mus.append(np.mean( sv[heads:] ))

mu = np.mean(sv_mus)
sv0_mean,sv0_std = np.mean(sv0s),np.std(sv0s)

in_ranges = {}
l,r = -15,10
for i in range(l,r):
    in_ranges[i] = in_range(sv0s, sv0_mean+i*sv0_std, sv0_mean+(i+1)*sv0_std)

for k in list(in_ranges.keys())[1:]:
    if k < 0 and in_ranges[k] > 0 and in_ranges[k-1] == 0:
        l = k
        break

for k in range(list(in_ranges.keys())[-1]):
    if k > 0 and in_ranges[k] == 0 and in_ranges[k-1] > 0:
        r = k

def head_list(trial_std):
    hs = []
    for j in range(10**4):
        arr = np.random.normal(mu, trial_std, dim-heads)
        a = np.linalg.norm(arr)
        if a < 1:
            hs.append(np.sqrt((1-a**2)/heads))
    return np.array(hs)

if int(sys.argv[-1]) == 0:
    std,min_dif = 0,1
    for i in range(75,150):
        trial_std = i*1e-3
        hs = head_list(trial_std)
        
        if abs(np.mean(hs) - sv0_mean)/sv0_std < min_dif:
            min_dif = abs(np.mean(hs) - sv0_mean)/sv0_std
            std = trial_std
else:
    with open(f'std-{S}.pkl', 'rb') as F:
        std = pickle.load(F)

for s in test_seqs:
    sv,_ = encode_amino_acid_sequence(s, char_to_float, head, max_len, vec_len)
    if len(sv) > 0:
        test_states.append(sv)

input_states = np.array(input_states)
test_states = np.array(test_states)

# define a normal distribution noise variance
nstd = std/times
# Token frequencies in token order
cnt_y1 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]

# Token frequencies in Gaussian order
cnt_x3 = list(gauss_sort.keys())
cnt_y3 = [cnt[x] for x in cnt_x3]
for i in range(len(cnt_x3)):
    if len(cnt_x3[i]) > 1:
        cnt_x3[i] = cnt_x3[i][0] + '\n+'
    elif cnt_x3[i] == '\n':
        cnt_x3[i] = '\\n'

def make_target(n):
    target = []
    while len(target) < n:
        t = np.random.normal(mu, std, dim-heads)
        A = np.linalg.norm(t)
        h = np.sqrt(abs(1 - A**2)/heads)
        
        while A > 1 or not (l < (h-sv0_mean)/sv0_std < r):
            t = np.random.normal(mu, std, dim-heads)
            A = np.linalg.norm(t)
            h = np.sqrt(abs(1 - A**2)/heads)
            
        t = np.array([h]*heads + list(t))
        t /= np.linalg.norm(t)
        target.append(t)
    return np.array(target)
    
plt.figure(figsize=(5,5))
clear_output(wait=True)
plt.bar(list(in_ranges.keys()), list(in_ranges.values()), 
        width=1., color='m', align='edge')
plt.yscale('log')
    
plt.title(f"Number of heads in each range")
plt.xlabel("Range")
plt.ylabel("Count")
plt.savefig(f"{Fld}/Heads-{metals}.png", dpi = 300, bbox_inches='tight')

#fig,ax = plt.subplots(figsize=(8,4))
#clear_output(wait=True)
#ax.bar(cnt_x3, cnt_y3, color='g')
#ax.set_title('Frequencies of AA/AA+/\\n in Gaussian-like form')

# Create inset plot
#ax_inset = inset_axes(ax, width="30%", height="50%", loc='upper right')
#ax_inset.bar(cnt_x3, cnt_y3, color='g')
#ax_inset.xaxis.set_visible(False)
#ax_inset.set_yscale('log')

#ax_inset.tick_params(axis='y', labelsize=8)
#ax.set_xlabel("AA/AA+/\\n")
#ax.set_ylabel("Frequency")
#fig.savefig(f"{Fld}/shape-{metals}.png", dpi = 300, bbox_inches='tight')

cnt_x4 = [char_to_float[k] for k in gauss_sort.keys()]
fig,ax = plt.subplots(figsize=(8,4))
clear_output(wait=True)
ax.bar(cnt_x4, cnt_y3, color='g')
ax.set_title('Frequencies of token numbers')

# Create right inset plot
ax_inset = inset_axes(ax, width="30%", height="50%", loc='upper right')
ax_inset.bar(cnt_x4, cnt_y3, color='g')
ax_inset.xaxis.set_visible(False)
ax_inset.set_yscale('log')
ax_inset.tick_params(axis='y', labelsize=8)

# Create left inset plot
inset_ax = inset_axes(ax, width="30%", height="50%", loc='upper left')
inset_ax.bar(cnt_x3, cnt_y3, color='g')
inset_ax.xaxis.set_visible(False)
inset_ax.yaxis.set_visible(False)

ax.set_xticks(cnt_x4, cnt_x3)
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel("Token values")
ax.set_ylabel("Frequency")
fig.savefig(f"{Fld}/tokens-{metals}.png", dpi = 300, bbox_inches='tight')
