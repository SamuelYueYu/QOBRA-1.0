import os, pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Go to folder
os.chdir('Training data(max)')

keys_aa = ['A', 'C', 'D', 'E', 'F', 
           'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 
           'S', 'T', 'V', 'W', 'Y']
cnt_x = []
for aa in keys_aa:
    cnt_x.append(aa)
for aa in keys_aa:
    cnt_x.append(aa + "\n+")
cnt_x.append('X')
cnt_x.append('\\n')

i = 0
color = ['r','m','y','g','c','b']
line = ["^-",'d--','v:','s-.','o-','D--']

plt.figure(figsize=(8,4))
clear_output(wait=True)
# Loop through all files in the directory
for file_name in os.listdir('.'):
    # Check if it's a file and ends with .pkl
    if os.path.isfile(file_name) and file_name.endswith('.pkl'):
        # Get the relative differences
        with open(file_name, 'rb') as f:
            dif = pickle.load(f)
        
        plt.plot(cnt_x, abs(dif), f'{color[i]}{line[i]}', label=file_name[4:-4])
        i += 1
    
plt.title("Relative difference of occurrence frequency")
plt.xlabel("AA/AA+/\\n")
plt.ylabel(r'$\frac{Predicted}{Original} - 1$')
plt.legend(loc = 'upper left')
plt.savefig(f"dif.png", dpi = 300, bbox_inches='tight')