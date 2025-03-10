import string, urllib3
import sys, pickle, os
from count import *

# Open the file
filename = f"{S}/NUV-{S}.txt"
file = open(f"{filename}", "w")
file.write("N\tU\tV\n")
file.close()

# Get the first 20 uppercase letters
alphabets = list(string.ascii_uppercase)
# PyMol colors
colors = ["red", "green", "yellow", "blue", 
          "salmon", "marine", "grey", "black"]

def generate(lcode):
    param_dict = {l_params[j]: lcode[j] for j in range(dim_tot)}
    param_dict.update({e_params[j]: xe[j] for j in range(num_decode)})
    
    o_qc = qc_d.assign_parameters(param_dict)
    o_sv = Statevector(o_qc).data**2
    return decode_amino_acid_sequence(np.sqrt(o_sv), ftc, head)

def pml(s, HL, idx, prot_folder):
    # Ensure the output directory exists
    if not os.path.exists(prot_folder):
        os.makedirs(prot_folder)
    
    # TXT sequence
    file = open(f"{prot_folder}/{idx}.txt", "w")
    file.write(s)
    file.close()
        
    # PyMol commands for highlighting bonding AAs
    i = 0
    file = open(f"{prot_folder}/{idx}.pml", "w")
    for ID in HL.keys():
        file.write(f"color {colors[int(i%len(colors))]}, chain {ID} and elem C\n")
        resi = f'{HL[ID][0]}'
        for res in HL[ID][1:]:
            resi += f"+{res}"
        file.write(f"select res{ID}, chain {ID} and resi {resi}\n")
        file.write(f"show sticks, res{ID}\n")
        
        # Hydrophobic AAs
        file.write(f"color lime, res{ID} and resn ALA+VAL+LEU+ILE+MET+PHE+TRP+PRO+GLY and elem C\n")
        # Polar AAs
        file.write(f"color orange, res{ID} and resn SER+THR+CYS+ASN+GLN+TYR and elem C\n")
        # Positive charged AAs
        file.write(f"color magenta, res{ID} and resn LYS+ARG+HIS and elem C\n")
        # Negative charged AAs
        file.write(f"color cyan, res{ID} and resn ASP+GLU and elem C\n")
        
        file.write(f"label res{ID}, resn\n")
        i += 1
    file.write(f"set label_color, yellow")
    file.close()
    print(f"PDB sequence saved to {prot_folder}")

mp = {'binding sites':'plus', 'chain numbers':'chains', 'length':'len'}
def Plot(real, gen, s, folder, seed):
    # Count the occurrences of each element
    counts_real = Counter(real)
    counts_gen = Counter(gen)
    
    # Mean relative ratio
    relative_ratios = []
    
    if s == 'length':
        # Use numpy to create bins and calculate the counts
        bins = [i*8 for i in range(dim_tot//8+1)]
        
        counts_real_bin = [0]*(len(bins)-1)
        counts_gen_bin = [0]*(len(bins)-1)
        
        for i in range(len(bins)-1):
            for k in counts_real.keys():
                if bins[i] < k < bins[i+1]:
                    counts_real_bin[i] += counts_real[k]
            for k in counts_gen.keys():
                if bins[i] < k < bins[i+1]:
                    counts_gen_bin[i] += counts_gen[k]
        
        # Sum
        sum_real = sum(counts_real_bin)
        sum_gen = sum(counts_gen_bin)
        
        # Normalize
        norm_counts_real = [i/sum_real for i in counts_real_bin]
        norm_counts_gen = [i/sum_gen for i in counts_gen_bin]
    
        for i in range(len(norm_counts_real)):
            real_freq = norm_counts_real[i]
            gen_freq = norm_counts_gen[i]
            
            if real_freq != 0:
                relative_ratios.append(gen_freq/real_freq)
        
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        # Calculate the width of each bin for the bar plot
        bin_widths = np.diff(bins)
        
        # Plot using plt.bar, centered at the bin edges
        plt.figure(figsize=(4,4))
        clear_output(wait=True)
        plt.bar(bins[:-1], norm_counts_real, log = True, alpha=1, 
                width=bin_widths, align='edge', label='Training')
        plt.bar(bins[:-1], norm_counts_gen, log = True, alpha=.6, 
                width=bin_widths, align='edge', label='De novo')
        plt.xlabel("Length")
    else:
        # Sum
        sum_real = sum(list( counts_real.values() ))
        sum_gen = sum(list( counts_gen.values() ))
        
        # Normalize the counts by dividing by the total number of elements
        norm_counts_real = {k: v/sum_real for k, v in counts_real.items()}
        norm_counts_gen = {k: v/sum_gen for k, v in counts_gen.items()}
        x_real,y_real = list(norm_counts_real.keys()), list(norm_counts_real.values())
        x_gen,y_gen = list(norm_counts_gen.keys()), list(norm_counts_gen.values())
        
        x_real,y_real = np.array(x_real),np.array(y_real)
        x_gen,y_gen = np.array(x_gen),np.array(y_gen)
    
        for k in x_real:
            real_freq = norm_counts_real[k]
            gen_freq = norm_counts_gen.get(k,0)
            relative_ratios.append(gen_freq/real_freq)
        mean_relative_ratio = np.mean(relative_ratios)
        std_relative_ratio = np.std(relative_ratios)
        
        plt.figure(figsize=(4,4))
        clear_output(wait=True)
        plt.bar(x_real, y_real, log = True, alpha=1, label='Training')
        plt.bar(x_gen, y_gen, log = True, alpha=.6, label='De novo')
        plt.xlabel("Count")
    
    plt.title(f"Distribution of {s}")
    plt.yscale('log')
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{folder}/{mp[s]}-{seed}-{S}.png", dpi = 300, bbox_inches='tight')
    
    # Store the result in a pickle file
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-{mp[s]}-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)