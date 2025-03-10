import warnings
import time, torch
from count import *
from qiskit.quantum_info import Statevector, state_fidelity

# Helper function to compute dot products for one row in X against all rows in Y
def SWAP(args):
    x,Y = args
    return 1 - np.dot(Y, x)**2

def compute_kernel_loss(X, Y):
    # Create a pool of workers
    with Pool(processes=processes) as pool:
        # Prepare arguments: Each row in X is paired with the entire matrix Y
        args = [(X[i], Y) for i in range(X.shape[0])] # Pass Y and rows of X for efficient dot
        # Map rows of X to the helper function
        results = pool.map(SWAP, args)
    return np.mean(results)

def latent_rep(x, p):
    # Latent space representation
    param_dict = {i_params[j]: x[j] for j in range(num_feature)}
    param_dict.update({e_params[j]: p[j] for j in range(num_encode)})
    
    q = qc_e.assign_parameters(param_dict)
    psi = Statevector.from_instruction(q)
    return np.real(psi.data)

def latent_encode(p, train_input, test_input):
    f = partial(encode_amino_acid_sequence, ctf=ctf, 
                head=head, max_len=dim_tot-1, 
                vec_len=dim_tot)
    
    with Pool(processes = processes) as pool:
        train_states = np.array(pool.map(f, train_input))
        test_states = np.array(pool.map(f, test_input))
        
    f = partial(latent_rep, p=p)
    with Pool(processes = processes) as pool:
        train_encode = np.array(pool.map(f, train_states))
        test_encode = np.array(pool.map(f, test_states))
    return train_encode,test_encode

trains_k,tests_k = [],[]
def plot_hist(train, test, target):
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    
    # Plot the histogram of the data hist
    plt.hist(train[:,1:].flatten(), density=True, bins=dim_latent, color='r', alpha=1, label='Train')
    plt.hist(test[:,1:].flatten(), density=True, bins=dim_latent, color='g', alpha=.3, label='Test')
    plt.hist(target[:,1:].flatten(), density=True, bins=dim_latent, color='b', alpha=.2, label='Target')
    
    plt.title("Frequency of state amplitudes")
    plt.xlabel("State amplitudes")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Add an inset plot
    # [left, bottom, width, height] in normalized figure coordinates
    inset_ax = plt.axes([0.125, 0.679, 0.2, 0.2])
    
    inset_ax.hist(abs(train[:,0]).flatten(), density=True, bins=dim_latent, color='r', alpha=1)
    inset_ax.hist(abs(test[:,0]).flatten(), density=True, bins=dim_latent, color='g', alpha=.3)
    inset_ax.hist(target[:,0].flatten(), density=True, bins=dim_latent, color='b', alpha=.2)
    
    # Reduce tick label size
    inset_ax.tick_params(axis='both', labelsize=8)
    # Set y-axis ticks to the right in the inset plot
    inset_ax.yaxis.tick_right()
    plt.savefig(f"{S}/{S}-hist.png", dpi = 300, bbox_inches='tight')
    
def plot(train, test, target):
    plot_hist(train, test, target)
    
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    plt.plot(range(len(trains_k)), trains_k, label="Training latent")
    plt.plot(range(len(tests_k)), tests_k, ":", label="Testing latent")
        
    plt.title("m-MMD / iteration")
    plt.xlabel("Iteration")
    plt.ylabel("m-MMD")
    plt.legend()
    plt.savefig(f"{S}/{S}-E.png",dpi = 300, bbox_inches='tight')

def e_loss(p, train_input, test_input):
    s = time.time()
    n = len(train_input)
    
    train_encode,test_encode = latent_encode(p, train_input, test_input)
    target = make_target(n, mu, std)
    
    k_train = compute_kernel_loss(train_encode, target)
    k_test = compute_kernel_loss(test_encode, target)
    trains_k.append(k_train)
    tests_k.append(k_test)
        
    if (len(trains_k)-1)%10 == 0:
        print(len(trains_k)-1)
        print(f"m-MMD loss: {k_train:.4f}, {k_test:.4f}")

        train_eff = train_encode[:,1:].flatten()
        interval = time.time()-s
        print(f"Interval: {interval:.2f} s")
    
    # plotting part
    plot(train_encode, test_encode, target)
    return k_train