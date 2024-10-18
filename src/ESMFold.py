import sys, pickle, os
import string, urllib3
import requests, py3Dmol
from count import *

threshold = 4
# Get the first 20 uppercase letters
alphabets = list(string.ascii_uppercase)
# PyMol colors
colors = ["red", "green", "yellow", "blue", "salmon", "marine", "grey", "black"]

# Now we want to get the decoder parameters ...
with open('opt-d-' + S + '.pkl', 'rb') as F:
    loaded_result = pickle.load(F)
xd = loaded_result.x

# Create a folder to store the generated results & stats
if not os.path.exists(S):
    os.makedirs(S)

# Open the file
filename = f"{S}/NUV-{S}.txt"
if not os.path.exists(filename):
    file = open(f"{filename}", "w")
    file.write("N\tU\tV\n")
    file.close()

file = open(f"{filename}", "w")
file.write(f"N\tU\tV\n")
file.close()

urllib3.disable_warnings()
def predict_structure(sequence, filename="predicted_structure.pdb", output_dir="predicted_structures"):
    """
    Predicts the protein structure for a given sequence using the ESMFold API 
    and saves the result to a specified filename within the provided output directory.
    """
    # ESMFold API endpoint
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    # Read the protein sequences from the txt file
    s = sequence.strip()  # This will read all sequences

    # The sequences must be formatted as a single string, with each chain separated by newlines
    # POST request to the ESMFold endpoint
    response = requests.post(url, data=s, verify=False)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct the full path for the file
    filepath = os.path.join(output_dir, filename)
    
    # Save the PDB content to the file
    with open(filepath, 'w') as f:
        f.write(response.text)
    print(f"PDB structure saved to {filepath}")
    
def generate(lcode, seed):
    np.random.seed(seed)
    noise = np.random.normal(0, nstd, num_latent)
    
    param_dict = {l_params[j]: lcode[j] for j in range(dim)}
    param_dict.update({n_params[j]: noise[j] for j in range(num_latent)})
    param_dict.update({d_params[j]: xd[j] for j in range(num_decode)})
    
    output_qc = qc_d.assign_parameters(param_dict)
    output_sv = Statevector(output_qc).data
    
    return decode_amino_acid_sequence(output_sv, float_to_char, head)

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
        file.write(f"color {colors[i]}, chain {ID} and elem C\n")
        resi = f'{HL[ID][0]}'
        for res in HL[ID][1:]:
            resi += f"+{res}"
        file.write(f"select res{ID}, chain {ID} and resi {resi}\n")
        file.write(f"show sticks, res{ID}\n")
        file.write(f"color lime, res{ID} and resn ALA+VAL+LEU+ILE+MET+PHE+TRP+PRO+GLY and elem C\n")
        file.write(f"color orange, res{ID} and resn SER+THR+CYS+ASN+GLN+TYR and elem C\n")
        file.write(f"color magenta, res{ID} and resn LYS+ARG+HIS and elem C\n")
        file.write(f"color cyan, res{ID} and resn ASP+GLU and elem C\n")
        file.write(f"label res{ID}, resn\n")
        i += 1
    file.write(f"set label_color, yellow")
    file.close()
    print(f"PDB sequence saved to {prot_folder}")

mp = {"+":'plus', '\\n':'bksh', 'length':'len'}
def Plot(real, gen, s, folder, seed):
    # Count the occurrences of each element
    counts_real = Counter(real)
    sum_real = sum(list( counts_real.values() ))
    counts_gen = Counter(gen)
    sum_gen = sum(list( counts_gen.values() ))

    # Normalize the counts by dividing by the total number of elements
    norm_counts_real = {k: v/sum_real for k, v in counts_real.items()}
    norm_counts_gen = {k: v/sum_gen for k, v in counts_gen.items()}
    
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    plt.bar(norm_counts_real.keys(), norm_counts_real.values(),
            log = True, alpha=1, label='Training')
    plt.bar(norm_counts_gen.keys(), norm_counts_gen.values(),
            log = True, alpha=.6, label='De novo')
    
    plt.title(f"Distribution of {s}")
    if s == 'length':
        plt.xlabel("Length")
    else: plt.xlabel("Count")
    
    plt.yscale('log')
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{folder}/{mp[s]}-{seed}-{S}.png", dpi = 300, bbox_inches='tight')
