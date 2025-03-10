import time
import re as Re
from gen_func import *

Nlist,Ulist,Vlist = [],[],[]
begin = time.time()
# Pattern to search for
pattern3 = r".\+.\+.\+"
pattern5 = r".\+.\+.\+.\+.\+"

for seed in range(1):
    N,U,V = 0,0,0
    valid = []
    start = time.time()
    
    n = int(train_size*2)
    denovos = make_target(n, mu, std)
        
    folder = f"{S}/{seed}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with Pool(processes=processes) as pool:
        gen = pool.map(generate, denovos)
    
    n = len(gen)
    for i in range(len(gen)):
        novel,unique = True,False
        s = gen[i]
        idx = s.find("X")
        
        # Termination at X
        if idx != -1:
            s2 = s[:idx]
        else: s2 = s
        
        print(s2)
        N += 1
        f = partial(issame, str2=s2)
        with Pool(processes=processes) as pool:
            ismatch = pool.map(f, seqs)
        
        if any(ismatch):
            N -= 1
            novel = False
            print("Is match")
#        else:
        if s2 not in valid: 
            U += 1
            unique = True
        
        # Check if the pattern exists in the input string
        match3 = Re.findall(pattern3, s2)
        match5 = Re.findall(pattern5, s2)
        Cnts = Counter(s2)
        
        # and Cnts[':'] <= max_dn
        if "+" in s2 and '::' not in s2 and len(match5) == 0 and len(match3) < 2:
            # Sequence without + 
            s3 = ''.join([c for c in s2 if c != "+"])
            # Split equence w/ + at every ':' character
            s4 = s2.split(':')
            # Split equence w/o + at every ':' character
            s5 = s3.split(':')
            if s5[-1] == '': s5 = s5[:-1]
            
            if all(len(l) >= threshold for l in s5) or \
            (':' not in s3 and len(s3) >= threshold):
                V += 1
                if novel and unique:
                    valid.append(s)
                    
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
                pml(s2, HL, i, prot_folder)
        
        print("*"*dim_tot)
        if len(valid) == train_size:
            n = i
            break
    size = min(n, len(gen))
    print(size)
    
    file = open(f"{folder}/denovo-{seed}.txt", "w")
    file.write("De novo sequences\n")
    file.close()
    
    cnt,dn_cnts_gen,len_cnts_gen,plus_cnts_gen = cnts(valid, folder, seed)
    elapsed = (time.time()-start) / 3600
    print(f"Generated in {elapsed:0.2f} h")
    
    cnt_y2 = [cnt[x] if x in cnt.keys() else 0 for x in cnt_x]
    cnt_x2 = []
    for i in range(len(cnt_x)):
        if cnt_x[i][-1] == "+":
            cnt_x2.append(cnt_x[i][0] + "\n+")
        else:
            cnt_x2.append(cnt_x[i])
    
    cnt_y1 = np.array(cnt_y1)
    cnt_y2 = np.array(cnt_y2)
    dif = cnt_y2/cnt_y1
    
    mean_relative_ratio = np.mean(dif)
    std_relative_ratio = np.std(dif)
    
    # Store the result in a pickle file
    data_to_store = {'mean': mean_relative_ratio, 'std': std_relative_ratio}
    with open(f'{folder}/RR-AA-{seed}-{S}.pkl', 'wb') as F:
        pickle.dump(data_to_store, F)
    
    Nlist.append(N/size)
    Ulist.append(U/size)
    Vlist.append(V/size)
    
    # Number of categories
    x = np.arange(len(cnt_x2))
    # Width of each bar
    width = 0.36
    
    fig,ax = plt.subplots(figsize=(12,4))
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
    
    Plot(dn_cnts_real, dn_cnts_gen, 'chain numbers', folder, seed)
    Plot(plus_cnts_real, plus_cnts_gen, 'binding sites', folder, seed)
    Plot(len_cnts_real, len_cnts_gen, 'length', folder, seed)

# Open the file in add mode
file = open(f"{filename}", "a")
file.write(f"{np.mean(Nlist):.3f}\t{np.mean(Ulist):.3f}\t{np.mean(Vlist):.3f}\n")
file.write(f"{np.std(Nlist):.3f}\t{np.std(Ulist):.3f}\t{np.std(Vlist):.3f}\n")
file.close()

elapsed = (time.time()-begin) / 3600
print(f"Total time: {elapsed:0.2f} h")