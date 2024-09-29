import time
from ESMFold import *

Nlist,Ulist,Vlist = [],[],[]
begin = time.time()
for seed in range(1):
    N,U,V = 0,0,0
    valid = []
    start = time.time()
    denovos = make_target(int(6e3))
        
    folder = f"{S}/{seed}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    f = partial(generate, seed=seed)
    with Pool(processes=processes) as pool:
        gen = pool.map(f, denovos)
    
    for i in range(len(gen)):
        s = gen[i]
        idx = s.find("X")
        
        # Termination at X
        if idx != -1: s2 = s[:idx]
        else: s2 = s
        
        print(s2)
        N += 1
        f = partial(issame, str2=s2)
        with Pool(processes=processes) as pool:
            ismatch = pool.map(f, seqs)
        
        if any(ismatch):
            N -= 1    
            print("Is match")
        else:
            if s2 not in valid:
                U += 1
                if "+" in s2 and '\n\n' not in s2:
                    # Sequence without +
                    s3 = ''.join([c for c in s2 if c != "+"])
                    # Split equence w/ + at every '\n' character
                    s4 = s2.split('\n')
                    # Split equence w/o + at every '\n' character
                    s5 = s3.split('\n')
                    
                    if all(len(line) >= threshold for line in s5) or \
                    ('\n' not in s3 and len(s3) >= threshold):
                        V += 1
                        valid.append(s)
                        
                        if len(valid) < 200:
                            # Dictionary of highlighted metal-binding AAs
                            HL = {}
                            for j in range(len(s4)):
                                chain,chainID = s4[j],alphabets[j]
                                
                                shift = 0
                                for k in range(len(chain)):
                                    if chain[k] == "+":
                                        if chainID in HL.keys():
                                            HL[chainID].append(k-shift)
                                        else: HL[chainID] = [k-shift]
                                        shift += 1
                            
                            pdb_name = f"{i}.pdb"
                            prot_folder = f'{folder}/Samples/{i}'
                            # predict_structure(s3, pdb_name, prot_folder)
                            pml(s2, HL, i, prot_folder)
        print("*"*dim)
    size = len(gen)
    
    file = open(f"{folder}/denovo-{seed}.txt", "w")
    file.write("De novo sequences\n")
    file.close()
    
    #file = open(f"{folder}/table-{seed}.txt", "w")
    #file.write("Real\tGen\n")
    #file.close()
    
    cnt,dn_cnts_gen,len_cnts_gen,plus_cnts_gen = cnts(valid, folder, seed)
    elapsed = (time.time()-start) / 60
    print(f"Generated in {elapsed:0.2f} min")
    
    cnt_y2 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]
    cnt_x2 = []
    for i in range(len(cnt_x)):
        if len(cnt_x[i]) == 2:
            cnt_x2.append(cnt_x[i][0] + "\n+")
        else:
            if cnt_x[i] == "\n":
                cnt_x2.append("\\n")
            else:
                cnt_x2.append(cnt_x[i])
    
    cnt_y1 = np.array(cnt_y1)
    cnt_y2 = np.array(cnt_y2)
    dif = cnt_y2/cnt_y1 - 1
    
    Nlist.append(N/size)
    Ulist.append(U/size)
    Vlist.append(V/size)
    
    # Store the result in a pickle file
    with open(f'{folder}/dif-{num_tot}-{S}.pkl', 'wb') as F:
        pickle.dump(dif, F)
    
    # Number of categories
    x = np.arange(len(cnt_x2))
    # Width of each bar
    width = 0.36
    
    fig,ax = plt.subplots(figsize=(8,4))
    clear_output(wait=True)
    
    ax.bar(x - width/2, cnt_y1, width, label='Training')
    ax.bar(x + width/2, cnt_y2, width, label='De novo')
    ax.set_title('Frequency of occurence of each AA')
    
    # Create inset plot
    ax_inset = inset_axes(ax, width="30%", height="50%", loc='right')
    ax_inset.bar(x - width/2, cnt_y1, width)
    ax_inset.bar(x + width/2, cnt_y2, width)
    ax_inset.xaxis.set_visible(False)
    ax_inset.set_yscale('log')
    
    ax_inset.tick_params(axis='y', labelsize=8)
    ax.set_xticks(x, cnt_x2)  # Set the x-ticks to the category names
    ax.set_xlabel("AA/AA+/\\n")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(f"{folder}/Bar-{seed}-{S}.png", dpi = 300, bbox_inches='tight')
    
    Plot(dn_cnts_real, dn_cnts_gen, '\\n', folder, seed)
    Plot(plus_cnts_real, plus_cnts_gen, '+', folder, seed)
    Plot(len_cnts_real, len_cnts_gen, 'length', folder, seed)

# Open the file in add mode
file = open(f"{filename}", "a")
file.write(f"{np.mean(Nlist):.3f}\t{np.mean(Ulist):.3f}\t{np.mean(Vlist):.3f}\n")
file.write(f"{np.std(Nlist):.3f}\t{np.std(Ulist):.3f}\t{np.std(Vlist):.3f}\n")
file.close()

elapsed = (time.time()-begin) / 60
print(f"Total time: {elapsed:0.2f} min")