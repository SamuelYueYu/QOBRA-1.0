# make a modeller usage file
import os
import glob
import csv
import sys
import matplotlib.pyplot as plt
from modeller import *
from modeller.scripts import complete_pdb

# Get folder name and subfolder number from command-line arguments or user input
if len(sys.argv) > 2:
    folder_name = sys.argv[1]
    subfolder_num = int(sys.argv[2])
elif len(sys.argv) > 1:
    folder_name = sys.argv[1]
    subfolder_num = int(input("Enter subfolder number (1, 2, 3, or 4): ").strip())
else:
    folder_name = input("Enter folder name (Ca, Mg, or Zn): ").strip()
    subfolder_num = int(input("Enter subfolder number (1, 2, 3, or 4): ").strip())

# Validate folder name
valid_folders = ['Ca', 'Mg', 'Zn']
if folder_name not in valid_folders:
    print(f"Error: Folder must be one of {valid_folders}")
    sys.exit(1)

# Validate subfolder number
valid_subfolders = [1, 2, 3, 4]
if subfolder_num not in valid_subfolders:
    print(f"Error: Subfolder number must be one of {valid_subfolders}")
    sys.exit(1)

# Construct subfolder path (e.g., Ca/Ca_1/)
subfolder_name = f"{folder_name}_{subfolder_num}"
subfolder_path = os.path.join(folder_name, subfolder_name)

if not os.path.isdir(subfolder_path):
    print(f"Error: Subfolder {subfolder_path} does not exist")
    sys.exit(1)

# Set up Modeller environment
env = Environ()
env.libs.topology.read(file='$(LIB)/top_heav.lib')
env.libs.parameters.read(file='$(LIB)/par.lib')

# Find all PDB files in the specified subfolder (each in its own numbered directory)
# Only include main PDB files (exclude solvated-* and temp-* files)
pdb_pattern = os.path.join(subfolder_path, '**', '*.pdb')
all_pdb_files = glob.glob(pdb_pattern, recursive=True)
pdb_files = sorted([f for f in all_pdb_files 
                    if not os.path.basename(f).startswith(('solvated-', 'temp-'))])

if not pdb_files:
    print(f"No PDB files found in {subfolder_path}/ folder")
    sys.exit(1)

print(f"Processing {len(pdb_files)} PDB files in {subfolder_path}/")

# Process each PDB file and collect DOPE scores
results = []
dope_scores = []
for pdb_file in pdb_files:
    pdb_name = os.path.basename(pdb_file)
    try:
        # Read the model
        mdl = complete_pdb(env, pdb_file)
        
        # Calculate normalized DOPE score (Z-score)
        score = mdl.assess_normalized_dope()
        
        results.append({'PDB_File': pdb_name, 'DOPE_Score': score})
        dope_scores.append(score)
        print(f"Processed {pdb_name}: normalized DOPE score = {score}")
    except Exception as e:
        print(f"Error processing {pdb_name}: {e}")
        results.append({'PDB_File': pdb_name, 'DOPE_Score': 'Error'})

# Write results to CSV file in the subfolder
output_csv = os.path.join(subfolder_path, f'{subfolder_name}_DOPE.csv')
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['PDB_File', 'DOPE_Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"\nResults saved to {output_csv}")

# Create histogram plot of DOPE scores
if dope_scores:
    fig, ax = plt.subplots(figsize=(4, 4))

    # Fix x,y-axis limits
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(0, 1.02)
    
    # Add translucent colored regions for N-DOPE zones
    # Green zone: N-DOPE < -1.5 (near-native structures)
    ax.axvspan(-10, -1.5, alpha=0.2, color='green', zorder=0)
    # Yellow zone: -1.5 <= N-DOPE < 1 (ambiguous)
    ax.axvspan(-1.5, 1, alpha=0.2, color='yellow', zorder=0)
    # Red zone: N-DOPE >= 1 (inaccurate)
    ax.axvspan(1, 10, alpha=0.2, color='red', zorder=0)
    
    # Add dashed vertical lines to separate zones (black for high contrast)
    ax.axvline(-1.5, color='black', linestyle='--', alpha=1, linewidth=1.5, zorder=2)
    ax.axvline(1, color='black', linestyle='--', alpha=1, linewidth=1.5, zorder=2)
    
    # Add text labels for threshold values
    ax.text(-1.5, 0.95, '-1.5', fontsize=8, ha='left', va='bottom', 
            color='black', zorder=3, bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', alpha=0, edgecolor='none'))
    ax.text(1, 0.95, '1.0', fontsize=8, ha='left', va='bottom', 
            color='black', zorder=3, bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', alpha=0, edgecolor='none'))
    
    # Create histogram with automatic binning; normalize frequencies (area = 1)
    ax.hist(dope_scores, bins='auto', edgecolor='none', alpha=.7,
            color='gray', density=True, zorder=1)
    
    # Set custom tick locations
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    
    # Y-axis label and ticks only for Ca_1, Mg_1, Zn_1
    if subfolder_num == 1:
        ax.set_ylabel('Frequency', fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

    # X-axis label and ticks only for Zn_1, Zn_2, Zn_3
    if folder_name == 'Zn' and subfolder_num in (1, 2, 3):
        ax.set_xlabel('DOPE Score', fontsize=12)
        ax.tick_params(axis='x', labelbottom=True)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    
    plt.tight_layout()
    
    # Save histogram to subfolder
    output_plot = os.path.join(subfolder_path, f'{subfolder_name}_DOPE.png')
    plt.savefig(output_plot, dpi=500, transparent=True)
    plt.close()
    
    print(f"Histogram saved to {output_plot}")
else:
    print("No valid DOPE scores to plot")